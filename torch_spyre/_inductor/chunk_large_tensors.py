# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split oversized pointwise and reduction ops into memory-safe chunks.

Runs after ``propagate_spyre_tensor_layouts`` / ``insert_restickify`` and
before ``span_reduction``.  Each chunk becomes a normal ComputedBuffer that
work-division handles without special-casing.

Pointwise ops:
    Controlling-dim-based split dim selection.  The first chunk is modified
    in-place; secondary chunks are new ComputedBuffers scattered back into
    the original buffer via a chained MutationLayoutSHOULDREMOVE.

Reduction ops:
    Symbol-aware split dim selection.  Only output (non-reduction) dims are
    split — each chunk is a fully independent reduction over its slice, so
    the scatter-back pattern is identical to the pointwise case.  Reduction
    dims are never split (that would require a separate accumulation pass).
"""

import math
from dataclasses import dataclass

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    Scatter,
)
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout

from . import config
from .ir import FixedTiledLayout, SpyreReduction
from .logging_utils import get_inductor_logger
from .pass_utils import concretize_expr, device_coordinates, host_coordinates
from .views import matching_dim
from .work_division import MAX_SPAN_BYTES

logger = get_inductor_logger("chunk_large_tensors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingInfo:
    total_bytes: int
    per_core_span: int
    best_split: int
    dev_dim_size: int
    dev_dim_stride: int
    host_dim: int
    stick_elems: int


def _find_best_split(dim_size: int, max_cores: int) -> int:
    """Return largest divisor of dim_size that is <= max_cores."""
    for i in range(min(max_cores, dim_size), 0, -1):
        if dim_size % i == 0:
            return i
    return 1


def _find_controlling_dim(
    layout: FixedTiledLayout,
    max_cores: int,
) -> tuple[int, int, int, int] | None:
    """Find the outermost splittable device dim and its best core split.

    Walks device dims outer-to-inner (skipping stick dim). For each
    device dim, uses stride_map to find the corresponding host dim.
    Returns the first host dim with size > 1.

    The controlling dim determines per-core memory span.
    Splitting inner dims increases parallelism but does NOT reduce span.
    Splitting the stick dim is not supported (atomic memory unit).

    Returns
    -------
    (host_dim_idx, dev_dim_size, dev_dim_stride, best_split)
        where best_split is the largest divisor of dev_dim_size
        that is <= max_cores.  Returns None if no valid splittable
        dim is found.

    Example::

        host_size   = [32,   8193,  1740]
        device_size = [8193, 28,    32,  64]
        stride_map  = [1740, 64, 14255820,  1]
        device_dim 0: stride_map=1740 -> host_dim 1 (M=8193), size>1
        8193 = 3 * 2731 -> best_split = 3 (largest divisor <= 32)
        dev_dim_stride = 28 * 32 * 64 = 57344
        -> returns (host_dim=1, dev_dim_size=8193,
                    dev_dim_stride=57344, best_split=3)
    """
    stl = layout.device_layout
    device_size = [int(s) for s in stl.device_size]
    host_size = [int(s) for s in layout.size]
    host_stride = [int(s) for s in layout.stride]
    stick_elems = stl.elems_per_stick()

    for device_dim in range(len(device_size) - 1):  # skip last=stick
        sm = int(stl.stride_map[device_dim])
        if sm <= 0:
            continue
        # Collect all host dims matching this stride with size > 1.
        # In practice stride_map values map to unique host strides so
        # matching_dims has at most one element.  The list handles the
        # theoretical edge case where a size-1 dim shares a stride
        # with a valid dim.
        matching_dims = [
            d for d, s in enumerate(host_stride) if s == sm and host_size[d] > 1
        ]
        if not matching_dims:
            continue
        host_dim = matching_dims[0]
        dev_dim_size = device_size[device_dim]
        dev_dim_stride = math.prod(device_size[device_dim + 1 :])
        # Skip dims smaller than one stick — cannot produce
        # stick-aligned chunks for dims with fewer elements
        # than stick_elems (e.g. host_dim size=32 < 64).
        if dev_dim_size < stick_elems:
            logger.debug(
                "Skipping device_dim %d: host_dim %d size=%d "
                "< stick_elems=%d, cannot chunk",
                device_dim,
                host_dim,
                host_size[host_dim],
                stick_elems,
            )
            continue
        best_split = _find_best_split(dev_dim_size, max_cores)
        return host_dim, dev_dim_size, dev_dim_stride, best_split
    return None


def _compute_num_chunks(
    chunking_info: ChunkingInfo,
    max_cores: int,
) -> int:
    """Compute number of chunks needed.

    Uses two estimates and picks the best:
    - num_from_span:  per_core_span fits in 256MB
    - num_from_total: total fits in max_cores × 256MB

    If chunking by num_from_total improves dim divisibility
    (chunk dim gets better core split), total formula is used.
    Otherwise takes max of both.
    """
    # Fallback path: no controlling dim found
    if chunking_info.dev_dim_size == 0:
        return max(
            1, math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))
        )

    num_from_span = math.ceil(chunking_info.per_core_span / MAX_SPAN_BYTES)
    num_from_total = math.ceil(chunking_info.total_bytes / (MAX_SPAN_BYTES * max_cores))

    # Check if chunking by num_from_total improves divisibility
    total_sticks_preview = math.ceil(
        chunking_info.dev_dim_size / chunking_info.stick_elems
    )
    chunk_sticks_preview = math.ceil(total_sticks_preview / max(num_from_total, 1))
    chunk_dim_preview = chunk_sticks_preview * chunking_info.stick_elems
    chunk_best_split = _find_best_split(chunk_dim_preview, max_cores)
    if chunk_best_split > chunking_info.best_split and num_from_total > 1:
        return num_from_total
    return max(num_from_span, num_from_total)


def _needs_chunking(
    layout: FixedTiledLayout,
    max_cores: int,
    controlling: tuple[int, int, int, int] | None,
) -> ChunkingInfo | None:
    """Return ChunkingInfo if this op needs chunking, else None.

    Uses the controlling dim's best_split (already computed in
    ``_find_controlling_dim``) to simulate work_division's per-core span.

    Two cases trigger chunking:
    1. ``per_core_span > 256 MB`` after best split on controlling dim
       -- catches small tensors with prime-like dims,
       e.g. [32, 8193, 1740]: best_split=3, per_core=313 MB > 256 MB
    2. ``total_bytes > 256 MB * max_cores``
       -- catches large tensors regardless of divisibility,
       e.g. [32, 8192, 17408]: total=9.17 GB > 8 GB

    Falls back to total_bytes threshold if no controlling dim found.
    """
    device_size = [int(s) for s in layout.device_layout.device_size]
    itemsize = layout.dtype.itemsize
    total_bytes = math.prod(device_size) * itemsize
    stick_elems = layout.device_layout.elems_per_stick()

    if controlling is None:
        if total_bytes > MAX_SPAN_BYTES * max_cores:
            host_size = [int(s) for s in layout.size]
            fallback_host_dim = max(range(len(host_size)), key=lambda d: host_size[d])
            return ChunkingInfo(
                total_bytes=total_bytes,
                per_core_span=total_bytes,
                best_split=1,
                dev_dim_size=0,
                dev_dim_stride=0,
                host_dim=fallback_host_dim,
                stick_elems=stick_elems,
            )
        return None

    host_dim, dev_dim_size, dev_dim_stride, best_split = controlling
    per_core_span = math.ceil(dev_dim_size / best_split) * dev_dim_stride * itemsize
    needs_chunk_for_span = per_core_span > MAX_SPAN_BYTES
    needs_chunk_for_total = total_bytes > MAX_SPAN_BYTES * max_cores

    if needs_chunk_for_span or needs_chunk_for_total:
        logger.info(
            "Op needs chunking: dev_dim_size=%d best_split=%d "
            "per_core_span=%.2fMB total=%.2fGB "
            "(span_limit=256MB total_limit=%.2fGB)",
            dev_dim_size,
            best_split,
            per_core_span / (1024**2),
            total_bytes / (1024**3),
            (MAX_SPAN_BYTES * max_cores) / (1024**3),
        )
        return ChunkingInfo(
            total_bytes=total_bytes,
            per_core_span=per_core_span,
            best_split=best_split,
            dev_dim_size=dev_dim_size,
            dev_dim_stride=dev_dim_stride,
            host_dim=host_dim,
            stick_elems=stick_elems,
        )
    return None


def _make_chunk_layout(
    original_ftl: FixedTiledLayout,
    split_dim_idx: int,
    chunk_size: int,
) -> FixedTiledLayout:
    """Build a ``FixedTiledLayout`` for a single chunk."""
    host_size = [int(s) for s in original_ftl.size]
    host_size[split_dim_idx] = chunk_size

    host_stride = [1] * len(host_size)
    for d in range(len(host_size) - 2, -1, -1):
        host_stride[d] = host_stride[d + 1] * host_size[d + 1]

    stl = SpyreTensorLayout(host_size, original_ftl.dtype)
    return FixedTiledLayout(
        original_ftl.device,
        original_ftl.dtype,
        host_size,
        host_stride,
        stl,
    )


def _make_chunk_fn(orig_fn, dim: int, offset: int):
    """Return an ``inner_fn`` that shifts the split dim by *offset*."""

    def inner_fn(index):
        idx = list(index)
        idx[dim] = idx[dim] + offset
        return orig_fn(idx)

    return inner_fn


def _make_output_indexer(offset: int, split_dim: int):
    """Return a scatter output indexer shifted by *offset* on *split_dim*."""

    def output_indexer(index):
        out = list(index)
        out[split_dim] += offset
        return out

    return output_indexer


def _register_and_insert(
    buf: ComputedBuffer,
    op: ComputedBuffer,
    operations: list[Operation],
    insert_pos: int,
) -> int:
    """Register *buf* in the graph and insert it at *insert_pos*.

    ``V.graph.register_operation`` appends to the same ``operations`` list,
    so the duplicate is removed before the positioned insert.

    Returns the next insert position.
    """
    buf.name = V.graph.register_buffer(buf)
    V.graph.register_operation(buf)
    buf.origins = op.origins
    if buf in operations:
        operations.remove(buf)
    operations.insert(insert_pos, buf)
    return insert_pos + 1


# ---------------------------------------------------------------------------
# Pointwise chunking
# ---------------------------------------------------------------------------


def _chunk_op(
    op: ComputedBuffer,
    max_cores: int,
    operations: list[Operation],
    op_index: int,
    chunking_info: ChunkingInfo,
    original_ftl: FixedTiledLayout,
) -> int:
    """Split *op* into memory-safe chunks along the controlling dim.

    Chunk 0 is the original op shrunk in-place (ranges only; layout
    stays full-size so the scheduler finds it by name).  Chunks 1..N-1
    are new ComputedBuffer + Scatter overwrite pairs.

    Chunk sizes are stick-aligned (multiples of ``stick_elems``) so the
    hardware scheduler always finds valid chunk-parameter candidates.
    """
    original_ranges = list(op.data.ranges)
    original_inner_fn = op.data.inner_fn

    split_dim_idx = chunking_info.host_dim
    if split_dim_idx >= len(original_ranges):
        logger.warning(
            "%s: controlling host_dim=%d out of range for %d-D ranges "
            "(layout.size ndim mismatch with op.data.ranges); skipping chunking",
            op.get_name(),
            split_dim_idx,
            len(original_ranges),
        )
        return 0

    split_dim_full_size = int(original_ranges[split_dim_idx])
    stick_elems = chunking_info.stick_elems

    # -- Step 1: decide number of chunks --
    num_chunks = _compute_num_chunks(chunking_info, max_cores)

    # -- Step 2: stick-aligned chunk size --
    # Chunk at stick level so every chunk is a multiple of stick_elems.
    # SpyreTensorLayout pads the last stick, so reading slightly beyond
    # split_dim_full_size is safe.
    total_sticks = math.ceil(split_dim_full_size / stick_elems)
    sticks_per_chunk = math.ceil(total_sticks / num_chunks)
    chunk_size = sticks_per_chunk * stick_elems
    num_chunks = math.ceil(total_sticks / sticks_per_chunk)

    logger.info(
        "Chunking %s: split_dim=%d full_size=%d "
        "sticks=%d sticks_per_chunk=%d "
        "chunk_size=%d num_chunks=%d total=%.2fGB",
        op.get_name(),
        split_dim_idx,
        split_dim_full_size,
        total_sticks,
        sticks_per_chunk,
        chunk_size,
        num_chunks,
        chunking_info.total_bytes / (1024**3),
    )

    # -- Chunk 0: shrink original op in-place --
    chunk0_size = min(chunk_size, split_dim_full_size)
    chunk0_ranges = list(original_ranges)
    chunk0_ranges[split_dim_idx] = chunk0_size
    object.__setattr__(op.data, "ranges", chunk0_ranges)

    insert_pos = op_index + 1
    n_inserted = 0

    # -- Chunks 1..N-1: direct scatter mutations into the original output --
    for chunk_idx in range(1, num_chunks):
        chunk_offset = chunk_idx * chunk_size
        remaining_elems = max(0, split_dim_full_size - chunk_offset)
        remaining_sticks = math.ceil(remaining_elems / stick_elems)
        this_chunk_size = min(remaining_sticks * stick_elems, chunk_size)

        chunk_ranges = list(original_ranges)
        chunk_ranges[split_dim_idx] = this_chunk_size

        mutation_data = Scatter(
            device=op.data.device,
            dtype=op.data.dtype,
            inner_fn=_make_chunk_fn(original_inner_fn, split_dim_idx, chunk_offset),
            ranges=chunk_ranges,
            output_indexer=_make_output_indexer(chunk_offset, split_dim_idx),
        )
        mutation_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=mutation_data,
        )
        insert_pos = _register_and_insert(mutation_buf, op, operations, insert_pos)
        n_inserted += 1

    return n_inserted


# ---------------------------------------------------------------------------
# Reduction chunking — same scatter-back pattern as pointwise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    dep: MemoryDep
    layout: FixedTiledLayout
    total_bytes: int


def _device_bytes(layout: FixedTiledLayout) -> int:
    """Total device-layout bytes including stick padding."""
    return (
        math.prod(concretize_expr(s) for s in layout.device_layout.device_size)
        * layout.dtype.itemsize
    )


def _collect_tensor_infos(op: ComputedBuffer) -> list[TensorInfo]:
    """Collect TensorInfo for all FixedTiledLayout reads and the write of op.

    Reads whose buffers do not have FixedTiledLayout (constants, etc.) are
    skipped rather than raising.
    """
    rw = op.get_read_writes()
    infos: list[TensorInfo] = []
    for dep in rw.reads:
        if not isinstance(dep, MemoryDep):
            continue
        buf = V.graph.get_buffer(dep.name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            continue
        infos.append(TensorInfo(dep, layout, _device_bytes(layout)))
    write_dep = next(iter(rw.writes))
    if not isinstance(op.layout, FixedTiledLayout):
        raise RuntimeError(f"{op.get_name()} output does not have FixedTiledLayout")
    infos.append(TensorInfo(write_dep, op.layout, _device_bytes(op.layout)))
    return infos


def _contiguous_stride(size: list[int]) -> list[int]:
    stride = [1] * len(size)
    for dim in range(len(size) - 2, -1, -1):
        stride[dim] = stride[dim + 1] * size[dim + 1]
    return stride


def _infer_dim_order(op: ComputedBuffer, layout: FixedTiledLayout) -> list[int]:
    """Fallback: infer device dim order from host↔device coordinate matching."""
    write_dep = next(iter(op.get_read_writes().writes))
    out_coords = host_coordinates(layout, write_dep)
    dev_coords = device_coordinates(layout.device_layout, write_dep)
    stick_dim = matching_dim(out_coords, dev_coords[-1])

    dim_order: list[int] = []
    for expr in dev_coords[:-1]:
        dim = matching_dim(out_coords, expr)
        if dim is not None and dim != stick_dim and dim not in dim_order:
            dim_order.append(dim)

    for dim, expr in enumerate(out_coords):
        if dim == stick_dim or dim in dim_order:
            continue
        if expr != 0:
            dim_order.append(dim)

    for dim in range(len(layout.size)):
        if dim != stick_dim and dim not in dim_order:
            dim_order.append(dim)

    if stick_dim is not None and stick_dim not in dim_order:
        dim_order.append(stick_dim)

    if not dim_order:
        logger.debug(
            "%s: _infer_dim_order found no matching device coords; "
            "falling back to identity dim order",
            op.get_name(),
        )
        return list(range(len(layout.size)))
    return dim_order


def _make_reduction_chunk_layout(
    op: ComputedBuffer, chunk_host_size: list[int], split_dim: int
) -> FixedTiledLayout:
    """Build a FixedTiledLayout for a reduction chunk, preserving device dim order.

    Scales only the split dimension's device size proportionally (device_size is
    in sticks, not elements).  Preserving the original dim_map and stride_map
    prevents dim reordering and inflating the per-core address span.
    """
    orig_dl = op.layout.device_layout
    chunk_stride = _contiguous_stride(chunk_host_size)

    device_split_dim: int | None = None
    try:
        write_dep = next(iter(op.get_read_writes().writes))
        out_coords = host_coordinates(op.layout, write_dep)
        dev_coords = device_coordinates(op.layout.device_layout, write_dep)
        split_host_coord = out_coords[split_dim]
        device_split_dim = matching_dim(list(dev_coords[:-1]), split_host_coord)
    except Exception:
        pass

    if device_split_dim is not None:
        new_device_size = [concretize_expr(s) for s in orig_dl.device_size]
        orig_device_dim_size = concretize_expr(orig_dl.device_size[device_split_dim])
        orig_host_dim_size = concretize_expr(op.layout.size[split_dim])
        # Use ceil so a chunk smaller than one stick-group still gets 1 stick.
        new_device_size[device_split_dim] = max(
            1,
            math.ceil(
                orig_device_dim_size * chunk_host_size[split_dim] / orig_host_dim_size
            ),
        )
        if hasattr(orig_dl, "dim_map"):
            chunk_stl = SpyreTensorLayout(
                new_device_size,
                list(orig_dl.dim_map),
                list(orig_dl.stride_map),
                orig_dl.device_dtype,
            )
        else:
            # Fallback for older builds that lack dim_map (3-arg device constructor).
            chunk_stl = SpyreTensorLayout(
                new_device_size, list(orig_dl.stride_map), orig_dl.device_dtype
            )
    else:
        dim_order = _infer_dim_order(op, op.layout)
        c_size = [concretize_expr(s) for s in chunk_host_size]
        c_stride = [concretize_expr(s) for s in chunk_stride]
        chunk_stl = SpyreTensorLayout(c_size, c_stride, op.layout.dtype, dim_order)

    return FixedTiledLayout(
        op.layout.device,
        op.layout.dtype,
        chunk_host_size,
        chunk_stride,
        chunk_stl,
    )


def _copy_loop_metadata(src, dst) -> None:
    if hasattr(src, "origins"):
        object.__setattr__(dst, "origins", src.origins)
    if hasattr(src, "traceback"):
        object.__setattr__(dst, "traceback", src.traceback)


def _make_shifted_reduction_fn(orig_fn, split_dim: int, offset: int):
    """Shifted inner_fn for Reduction chunks — preserves reduction_index arg."""

    def inner_fn(index, reduction_index):
        shifted = list(index)
        shifted[split_dim] += offset
        return orig_fn(shifted, reduction_index)

    return inner_fn


def _make_chunk_reduction_data(
    original_data, split_dim: int, offset: int, chunk_ranges: list[int]
):
    """Create a fresh Reduction/SpyreReduction node for one chunk.

    Takes *original_data* (captured before chunk-0 mutation) so it is safe to
    call for all chunks uniformly.
    """
    fn = (
        original_data.inner_fn
        if offset == 0
        else _make_shifted_reduction_fn(original_data.inner_fn, split_dim, offset)
    )
    kwargs = dict(
        device=original_data.device,
        dtype=original_data.dtype,
        inner_fn=fn,
        ranges=chunk_ranges,
        reduction_ranges=list(original_data.reduction_ranges),
        reduction_type=original_data.reduction_type,
        src_dtype=original_data.src_dtype,
        reduction_hint=original_data.reduction_hint,
    )
    chunk_data = (
        SpyreReduction(op_info=original_data.op_info, **kwargs)
        if isinstance(original_data, SpyreReduction)
        else Reduction(**kwargs)
    )
    _copy_loop_metadata(original_data, chunk_data)
    return chunk_data




def _choose_split_dim(
    op: ComputedBuffer,
    tensor_infos: list[TensorInfo],
    total_limit_bytes: int,
) -> tuple[int, sympy.Symbol] | None:
    """Return (dim, symbol) identifying the best op.data.ranges dimension to split.

    Two correctness hazards are avoided:

    1. Symbol alignment: Inductor drops size-1 dims from write_dep.ranges, so a
       leading batch dim of 1 shifts every symbol one slot relative to
       op.data.ranges.  We skip size-1 ranges when consuming symbols instead of
       using zip().

    2. Stick-dim exclusion: splitting the innermost (stick) dimension would not
       reduce the per-core device address span for outer dimensions and would
       produce chunks misaligned with the hardware stick boundary.  The stick
       dim is excluded entirely from candidate selection.
    """
    write_dep = next(iter(op.get_read_writes().writes))
    output_symbols = list(write_dep.ranges.keys())
    output_ranges = [concretize_expr(size) for size in op.data.ranges]

    stick_host_dim: int | None = None
    if isinstance(op.layout, FixedTiledLayout):
        try:
            out_coords = host_coordinates(op.layout, write_dep)
            dev_coords = device_coordinates(op.layout.device_layout, write_dep)
            stick_host_dim = matching_dim(out_coords, dev_coords[-1])
        except Exception:
            pass

    offending = [info for info in tensor_infos if info.total_bytes > total_limit_bytes]

    sym_idx = 0
    best: tuple[bool, int, int, int] | None = None
    best_dim: int | None = None
    best_sym: sympy.Symbol | None = None

    for dim, size in enumerate(output_ranges):
        if size <= 1:
            continue
        if sym_idx >= len(output_symbols):
            logger.warning(
                "%s: op.data.ranges has more non-trivial dims than write_dep "
                "symbols (%d); dims at index %d and beyond are not eligible for "
                "splitting",
                op.get_name(),
                len(output_symbols),
                dim,
            )
            break
        sym = output_symbols[sym_idx]
        sym_idx += 1

        # Exclude the stick dim — splitting it requires special handling.
        if dim == stick_host_dim:
            continue

        affected = [info for info in tensor_infos if sym in info.dep.index.free_symbols]
        if not affected:
            continue
        covers_all = all(sym in info.dep.index.free_symbols for info in offending)
        score = sum(info.total_bytes for info in affected)
        rank = (covers_all, score, size, -dim)
        if best is None or rank > best:
            best = rank
            best_dim = dim
            best_sym = sym

    if best_dim is None or best_sym is None:
        return None
    return best_dim, best_sym


def _required_chunks(
    tensor_infos: list[TensorInfo],
    split_sym: sympy.Symbol,
    total_limit_bytes: int,
    full_size: int,
) -> int:
    affected = [
        info.total_bytes
        for info in tensor_infos
        if split_sym in info.dep.index.free_symbols
    ]
    if not affected or full_size <= 0:
        return 1
    # ceil(total_bytes / limit) underestimates when chunk_size = ceil(S/n) pushes
    # per-chunk bytes over the limit.  Compute max_chunk_size via floor division
    # so each chunk's bytes are guaranteed to be <= total_limit_bytes.
    max_bytes = max(affected)
    bytes_per_element = max_bytes / full_size
    if bytes_per_element <= 0:
        return 1
    max_chunk_size = max(1, int(total_limit_bytes / bytes_per_element))
    return max(1, math.ceil(full_size / max_chunk_size))


def _chunk_reduction_op(
    op: ComputedBuffer,
    operations: list[Operation],
    op_index: int,
    tensor_infos: list[TensorInfo],
    total_limit_bytes: int,
) -> int:
    """Split a reduction op's output space into chunks — same scatter pattern as pointwise.

    Only op.data.ranges (output dims) are split.  op.data.reduction_ranges are
    never touched — each chunk independently reduces its own slice of the output
    space, so no partial-reduction merging is needed.

    Returns the number of new operations inserted so the caller can skip them.
    """
    result = _choose_split_dim(op, tensor_infos, total_limit_bytes)
    if result is None:
        return 0
    split_dim, split_sym = result

    uncovered = [
        info.dep.name
        for info in tensor_infos
        if info.total_bytes > total_limit_bytes
        and split_sym not in info.dep.index.free_symbols
    ]
    if uncovered:
        logger.warning(
            "%s: split dim %d does not cover over-limit tensors %s; chunking may "
            "only partially reduce the aggregate tensor size",
            op.get_name(),
            split_dim,
            uncovered,
        )

    # Capture original data before any mutation.
    original_data = op.data
    original_ranges = [concretize_expr(r) for r in original_data.ranges]
    full_size = original_ranges[split_dim]

    required_chunks = _required_chunks(
        tensor_infos, split_sym, total_limit_bytes, full_size
    )
    if required_chunks <= 1 or full_size <= 1:
        return 0

    # Align chunk_size to stick boundary so device memory access is aligned.
    stick_elems = op.layout.device_layout.elems_per_stick()
    chunk_size = math.ceil(full_size / required_chunks)
    if stick_elems > 1:
        chunk_size = math.ceil(chunk_size / stick_elems) * stick_elems
    num_chunks = math.ceil(full_size / chunk_size)
    if num_chunks <= 1:
        return 0

    logger.info(
        "Chunking reduction %s: dim=%d, full_size=%d, chunk_size=%d, num_chunks=%d, "
        "limit=%.2fMB, max_tensor=%.2fMB",
        op.get_name(),
        split_dim,
        full_size,
        chunk_size,
        num_chunks,
        total_limit_bytes / (1024 * 1024),
        max(info.total_bytes for info in tensor_infos) / (1024 * 1024),
    )

    # Chunk 0: replace op.data with a fresh node to clear @cache_on_self caches
    # (e.g. get_default_sizes_body) that would otherwise carry the original
    # un-chunked iteration space into span_reduction / work_distribution.
    first_ranges = list(original_ranges)
    first_ranges[split_dim] = min(chunk_size, full_size)
    object.__setattr__(
        op,
        "data",
        _make_chunk_reduction_data(original_data, split_dim, 0, first_ranges),
    )
    ComputedBuffer.get_default_sizes_body.clear_cache(op)

    # Chunks 1..N-1: new compute buffer + scatter-back into op's buffer.
    insert_pos = op_index + 1
    n_inserted = 0
    for chunk_idx in range(1, num_chunks):
        offset = chunk_idx * chunk_size
        remaining = full_size - offset
        this_chunk_size = min(chunk_size, remaining)
        # Align last chunk to stick boundary.
        if stick_elems > 1:
            this_chunk_size = math.ceil(this_chunk_size / stick_elems) * stick_elems
        chunk_ranges = list(original_ranges)
        chunk_ranges[split_dim] = this_chunk_size
        chunk_layout_size = [concretize_expr(s) for s in op.layout.size]
        chunk_layout_size[split_dim] = this_chunk_size

        chunk_buf = ComputedBuffer(
            name=None,
            layout=_make_reduction_chunk_layout(op, chunk_layout_size, split_dim),
            data=_make_chunk_reduction_data(
                original_data, split_dim, offset, chunk_ranges
            ),
        )
        chunk_buf.origins = op.origins
        if hasattr(op, "origin_node"):
            chunk_buf.origin_node = op.origin_node
        insert_pos = _register_and_insert(chunk_buf, op, operations, insert_pos)
        n_inserted += 1

        loader = chunk_buf.make_loader()

        def _scatter_inner(index, _loader=loader):
            return _loader(index)

        overwrite_buf = ComputedBuffer(
            name=None,
            layout=MutationLayoutSHOULDREMOVE(op),
            data=Scatter(
                device=original_data.device,
                dtype=original_data.dtype,
                inner_fn=_scatter_inner,
                ranges=chunk_ranges,
                output_indexer=_make_output_indexer(offset, split_dim),
            ),
        )
        _copy_loop_metadata(original_data, overwrite_buf.data)
        overwrite_buf.origins = op.origins
        insert_pos = _register_and_insert(overwrite_buf, op, operations, insert_pos)
        n_inserted += 1

    # Each secondary chunk inserts exactly 2 operations (chunk_buf + overwrite_buf).
    return n_inserted


# ---------------------------------------------------------------------------
# Public pass
# ---------------------------------------------------------------------------


def chunk_large_tensors(operations: list[Operation]) -> None:
    """Split Pointwise and Reduction ops whose device footprint exceeds the limit.

    Must run **after** ``propagate_spyre_tensor_layouts`` /
    ``insert_restickify`` and before ``span_reduction``.
    """
    max_cores = config.sencores
    total_limit_bytes = MAX_SPAN_BYTES * max_cores
    i = 0
    while i < len(operations):
        op = operations[i]

        if isinstance(op, ComputedBuffer) and isinstance(op.layout, FixedTiledLayout):
            # --- Pointwise ---
            if (
                isinstance(op.data, Pointwise)
                # Note: ir.Pointwise is broader than torch.Tag.pointwise.
                # inner_fn can in theory access non-corresponding input
                # indices making chunking unsafe.
                # TODO: Use OpsHandler to verify output[i] only uses
                # input[i] before chunking.
                and len(op.data.ranges) >= 2
            ):
                controlling = _find_controlling_dim(op.layout, max_cores)
                chunking_info = _needs_chunking(op.layout, max_cores, controlling)
                if chunking_info is not None:
                    n_inserted = _chunk_op(
                        op, max_cores, operations, i, chunking_info, op.layout
                    )
                    i += n_inserted

            # --- Reduction ---
            elif isinstance(op.data, Reduction):
                try:
                    tensor_infos = _collect_tensor_infos(op)
                except RuntimeError as e:
                    logger.debug(
                        "%s: _collect_tensor_infos failed (%s); skipping",
                        op.get_name(),
                        e,
                    )
                    i += 1
                    continue
                if tensor_infos:
                    max_bytes = max(info.total_bytes for info in tensor_infos)
                    if max_bytes > total_limit_bytes:
                        n_inserted = _chunk_reduction_op(
                            op, operations, i, tensor_infos, total_limit_bytes
                        )
                        i += n_inserted

        i += 1

