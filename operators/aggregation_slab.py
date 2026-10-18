"""Fixed workspaces for pass-2 aggregation.

The pass-2 input tick is only a feed unit. Accumulated state and, when useful,
several feed ticks live in operator-owned byte slabs. Changing the pipeline
chunk size therefore changes how a fixed aggregation batch is filled without
forcing the accumulated representation to grow and reallocate per tick.

``DenseAggregationSlab`` stores:

* one byte of occupancy per possible key; and
* one reduction-identity-initialized segment per aggregate value.

Value segments may have trailing fields, e.g. AVG's ``[sum, count]`` state.
All segments are typed views into one ``uint8`` allocation.  This keeps the
workspace contiguous and prevents per-tick allocator fragmentation.

``FixedColumnSlab`` is the generic counterpart used by sorted runs and input
batching. It stores arbitrary row-aligned typed columns in one raw allocation
and tracks the populated prefix when used as an input batch.
"""

from typing import List, Sequence, Tuple

import torch


ValueSpec = Tuple[torch.dtype, Tuple[int, ...]]


def _align_up(value: int, alignment: int) -> int:
  return ((value + alignment - 1) // alignment) * alignment


def _element_size(dtype: torch.dtype) -> int:
  return torch.empty((), dtype=dtype).element_size()


def _reduction_identity(dtype: torch.dtype, op: str):
  if op == "sum":
    return 0
  if dtype == torch.bool:
    if op == "amax":
      return False
    if op == "amin":
      return True
  if dtype.is_floating_point:
    if op == "amax":
      return -torch.inf
    if op == "amin":
      return torch.inf
  else:
    limits = torch.iinfo(dtype)
    if op == "amax":
      return limits.min
    if op == "amin":
      return limits.max
  raise ValueError(f"unsupported slab reduction {op!r} for {dtype}")


def _column_layout(capacity: int, specs: Sequence[ValueSpec],
                   min_alignment: int = 8,
                   column_alignment: int = 1):
  if capacity < 0:
    raise ValueError(f"capacity must be non-negative, got {capacity}")
  max_alignment = max(
      [min_alignment] + [_element_size(dtype) for dtype, _ in specs])
  cursor = 0
  segments = []
  for dtype, trailing_shape in specs:
    elem_size = _element_size(dtype)
    cursor = _align_up(cursor, max(column_alignment, elem_size))
    width = 1
    for extent in trailing_shape:
      width *= int(extent)
    size_b = capacity * width * elem_size
    segments.append((cursor, size_b, dtype, tuple(trailing_shape)))
    cursor += size_b
  return segments, _align_up(cursor, max_alignment)


def _allocate_columns(capacity: int, specs: Sequence[ValueSpec], device,
                      min_alignment: int = 8,
                      column_alignment: int = 1):
  segments, total_b = _column_layout(
      capacity, specs, min_alignment, column_alignment)
  storage = torch.empty(total_b, dtype=torch.uint8, device=device)
  columns = []
  for offset, size_b, dtype, trailing_shape in segments:
    shape = (capacity,) + trailing_shape
    columns.append(
        storage[offset:offset + size_b].view(dtype).view(shape))
  return storage, columns


class FixedColumnSlab:
  """One raw allocation exposed as fixed-capacity typed columns.

  ``length`` tracks the populated prefix for both a resident sorted run and
  an append-only input batch.
  """

  def __init__(self, capacity: int, specs: Sequence[ValueSpec], device):
    self.capacity = int(capacity)
    # A run column may be handed directly to nvCOMP at final emission.
    # Independent torch allocations are strongly aligned; preserve that
    # contract for typed views carved from the shared uint8 allocation.
    self.storage, self.columns = _allocate_columns(
        self.capacity, specs, device,
        min_alignment=256, column_alignment=256)
    self.length = 0

  @staticmethod
  def required_bytes(capacity: int,
                     specs: Sequence[ValueSpec]) -> int:
    return _column_layout(
        capacity, specs, min_alignment=256, column_alignment=256)[1]

  @staticmethod
  def row_bytes(specs: Sequence[ValueSpec]) -> int:
    total = 0
    for dtype, trailing_shape in specs:
      width = 1
      for extent in trailing_shape:
        width *= int(extent)
      total += width * _element_size(dtype)
    return total

  @property
  def nbytes(self) -> int:
    return self.storage.numel() * self.storage.element_size()

  @property
  def remaining(self) -> int:
    return self.capacity - self.length

  def can_append(self, tensors: Sequence[torch.Tensor]) -> bool:
    if len(tensors) != len(self.columns):
      raise ValueError("input tensor count does not match slab column count")
    rows = tensors[0].shape[0] if tensors else 0
    if any(t.shape[0] != rows for t in tensors):
      raise ValueError("all staged input columns must have equal row counts")
    return rows <= self.remaining

  def append(self, tensors: Sequence[torch.Tensor]) -> None:
    if not self.can_append(tensors):
      rows = tensors[0].shape[0] if tensors else 0
      raise ValueError(
          f"input batch has {rows} rows but slab has only "
          f"{self.remaining} rows free")
    rows = tensors[0].shape[0] if tensors else 0
    lo, hi = self.length, self.length + rows
    for dst, src in zip(self.columns, tensors):
      if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
        raise ValueError(
            f"input trailing shape {tuple(src.shape[1:])} does not match "
            f"slab row shape {tuple(dst.shape[1:])}")
      if src.dtype != dst.dtype:
        raise ValueError(
            f"input dtype {src.dtype} does not match slab dtype {dst.dtype}")
      dst[lo:hi].copy_(src)
    self.length = hi

  def tensors(self) -> List[torch.Tensor]:
    return [column[:self.length] for column in self.columns]

  def reset(self) -> None:
    self.length = 0


class DenseAggregationSlab:
  """One fixed, direct-addressed aggregation workspace.

  ``key_min..key_max`` defines the address space.  The caller admits this
  representation only when the range is compact enough for the arena.
  """

  def __init__(self, key_min: int, key_max: int, key_dtype: torch.dtype,
               value_specs: Sequence[ValueSpec], ops: Sequence[str],
               device):
    if key_max < key_min:
      raise ValueError(
          f"invalid key range [{key_min}, {key_max}]")
    if len(value_specs) != len(ops):
      raise ValueError("value_specs and ops must have the same length")

    self.key_min = int(key_min)
    self.key_max = int(key_max)
    self.key_dtype = key_dtype
    self.capacity = self.key_max - self.key_min + 1
    self.ops = tuple(ops)

    specs = [(torch.bool, ())] + list(value_specs)
    self.storage, columns = _allocate_columns(
        self.capacity, specs, device)
    self.occupied = columns[0]
    self.occupied.fill_(False)

    self.values = columns[1:]
    for value, op in zip(self.values, self.ops):
      value.fill_(_reduction_identity(value.dtype, op))

  @staticmethod
  def required_bytes(capacity: int,
                     value_specs: Sequence[ValueSpec]) -> int:
    """Exact size of the single raw allocation, including alignment."""
    specs = [(torch.bool, ())] + list(value_specs)
    return _column_layout(capacity, specs)[1]

  @property
  def nbytes(self) -> int:
    return self.storage.numel() * self.storage.element_size()

  def add(self, keys: torch.Tensor,
          values: Sequence[torch.Tensor]) -> None:
    """Fold one pass-2 tick into the slab without growing running state."""
    if keys.dim() != 1:
      raise ValueError(f"slab keys must be 1-D, got {tuple(keys.shape)}")
    if len(values) != len(self.values):
      raise ValueError("tick value count does not match slab value count")
    if keys.numel() == 0:
      return

    # Pass 1 tracked the global bounds used to construct this slab, so these
    # offsets are in range by construction.  Avoid a per-tick min/max host
    # synchronization merely to re-check that invariant.
    offsets = keys.to(torch.int64) - self.key_min
    self.occupied[offsets] = True

    for dst, src, op in zip(self.values, values, self.ops):
      if src.shape[0] != keys.numel() or src.shape[1:] != dst.shape[1:]:
        raise ValueError(
            f"tick value shape {tuple(src.shape)} does not match slab row "
            f"shape {tuple(dst.shape[1:])}")
      index = offsets
      if src.dim() > 1:
        index = offsets.view(
            (offsets.numel(),) + (1,) * (src.dim() - 1)).expand_as(src)
      dst.scatter_reduce_(
          0, index, src, reduce=op, include_self=True)

  def finish(self):
    """Materialize ascending populated keys and their aggregate states."""
    offsets = torch.nonzero(self.occupied, as_tuple=False).flatten()
    keys = (offsets + self.key_min).to(self.key_dtype)
    values = [value.index_select(0, offsets) for value in self.values]
    return keys, values
