import torch
import numpy as np
import constants
from typing import NamedTuple
from utility.tensor_utils import dtype_size
from IO.pages.record_table import RecordTableBuilder, Selection
from IO.pages.region import DEVICE, Region


class UvaSource(NamedTuple):
    """A zero-copy read of a column's host rows (``TableVariable.uva_source``):
    ``seg_ptrs`` and ``seg_rows``, int64 GPU tensors (views of one
    allocation), give each segment's device address and first row, plus the
    end, in a row count of the column's segments; the read covers rows
    ``[row_start, row_stop)`` of that count, of ``dtype`` and ``width``."""
    seg_ptrs: torch.Tensor
    seg_rows: torch.Tensor
    row_start: int
    row_stop: int
    dtype: torch.dtype
    width: int


def _uva_source(segs: np.ndarray, start, stop, dtype, width) -> UvaSource:
    """A ``UvaSource`` from ``segs`` (``n`` device addresses, then ``n + 1``
    row starts), copied to the GPU as one tensor.  The rows ``[start,
    stop)`` must lie inside the segments: the kernels do not look past
    them."""
    n = (len(segs) - 1) // 2
    assert n > 0 and int(segs[n]) <= start <= stop <= int(segs[-1]), (
        f"rows [{start}, {stop}) outside the segments' rows "
        f"[{int(segs[n]) if n else 0}, {int(segs[-1])})")
    dev = torch.from_numpy(segs).to("cuda")
    return UvaSource(dev[:n], dev[n:], start, stop, dtype, width)


class Variable:
    """A column with no records to read: an empty column, or the GPU tensor
    a read produced (``_contiguous_data``, set by ``DirectIn.phase_finalize``
    for a RAW transfer or ``DirectIn.decompress``), trimmed to ``row_range``.
    A column held as records is a ``TableVariable``."""

    def __init__(self, tensor_type: str, row_range: slice = None,
                 selectivity: float = None):
        self.tensor_type = tensor_type
        # Optional row range of ``_contiguous_data`` that ``data()`` keeps
        self.row_range = row_range
        # Pre-computed selectivity (mask_sum / total_rows) for mask columns
        self.selectivity = selectivity
        # When set, data() returns this directly.  Variable itself does no
        # work.
        self._contiguous_data = None

    def __repr__(self):
        return f"Variable (type='{self.tensor_type}')"

    def orig_elem_size(self):
        return dtype_size(type_to_torch(self.tensor_type))

    def raw_elem_size(self):
        return 0

    def col_width(self):
        return 0

    def col_length(self):
        return 0

    def raw_numel(self):
        return 0

    def data_to_transfer(self):
        return 0

    def data_original(self):
        return self.col_length() * self.col_width() * self.orig_elem_size()

    def orig_numel(self):
        return self.col_width() * self.col_length()

    # A ``TableVariable`` overrides this with its records' codecs.
    is_compressed = False

    def data(self):
        """The column as a single normalized tensor (squeeze width-1 columns,
        unsqueeze scalars); an empty tensor for an empty column."""
        if self._contiguous_data is None:
            return torch.empty(0, dtype=type_to_torch(self.tensor_type), device="cuda")
        tensor = self._contiguous_data
        if self.row_range is not None:
            tensor = tensor[self.row_range.start:self.row_range.stop]
        if tensor.dim() == 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return tensor

    def col_to_gpu(self, device='cuda:0'):
        return torch.empty(0, dtype=type_to_torch(self.tensor_type), device=device)

    def get_sliced_rep(self, slice_in: slice) -> "Variable":
        """A slice of a column with no rows is empty."""
        return Variable(tensor_type=self.tensor_type,
                        selectivity=self.selectivity)

    def free_underlying_mem(self, cpu_pool, gpu_pool):
        pass

    def get_type_torch(self):
        return type_to_torch(self.tensor_type)

    # A ``TableVariable`` is read through its record table.
    table_backed = False

    def reads_nothing(self) -> bool:
        """True when the column has no records to read."""
        return True

    def first_codec_name(self):
        """The codec of the first compressed record (``None``: no compressed
        record)."""
        return None

    def leading_codec_name(self):
        """The codec of the first record (``None`` for RAW or no record):
        O(1), for callers that take a rep's first record as speaking for
        it."""
        return None


class TableVariable(Variable):
    """A column held as a record table of regions (``IO/pages/
    record_table.py``): what the scan and the region producer
    (``IO/codec/region_producer.py``) write.

    The Variable reads rows ``[lo, hi)`` of the table's row space (its
    streams concatenated in order, the partition-major order of a sort
    output).  ``get_sliced_rep`` gives another ``TableVariable`` over the same
    table, holding no reference; the column's own Variable (the full range)
    frees the table, slices free nothing.  The tick reader selects records
    through ``record_selection``.  It is never empty: a producer that wrote
    no row hands out an empty Variable instead.
    """

    table_backed = True

    def __init__(self, table, tensor_type: str, selectivity: float = None,
                 lo: int = 0, hi: int = None, records=None):
        super().__init__(tensor_type=tensor_type, selectivity=selectivity)
        self._table = table
        self._lo = int(lo)
        self._hi = int(table.num_rows if hi is None else hi)
        assert 0 <= self._lo < self._hi <= table.num_rows, (
            f"rows [{self._lo}, {self._hi}) outside a table of "
            f"{table.num_rows}")
        self._owns_table = self._lo == 0 and self._hi == table.num_rows \
            and hi is None
        # ``(first, stop)``: the rows are exactly records ``[first, stop)``
        # (``whole_records``), read whole, with no search and no trim.
        self._records = records

    def __repr__(self):
        return (f"TableVariable ({self.col_length()} x {self.col_width()}, "
                f"type='{self.tensor_type}', regions="
                f"{len(self._table.regions)})")

    def reads_nothing(self) -> bool:
        return False

    def raw_elem_size(self):
        return self._table.stored_dtype.itemsize

    def col_width(self):
        return int(self._table.width)

    def col_length(self):
        return self._hi - self._lo

    def raw_numel(self):
        return self.col_length() * self.col_width()

    def data_to_transfer(self):
        return self.record_selection().stored_bytes()

    @property
    def is_compressed(self):
        return self._table.is_compressed

    def first_codec_name(self):
        from compression.compression_algorithms import _ALGO_NAMES
        codecs = self._table.codec
        comp = codecs[codecs != 0]
        return _ALGO_NAMES[int(comp[0])] if len(comp) else None

    def leading_codec_name(self):
        from compression.compression_algorithms import _ALGO_NAMES
        cid = int(self._table.codec[0]) if self._table.num_records else 0
        return _ALGO_NAMES[cid] if cid else None

    def data(self):
        assert self._contiguous_data is not None, (
            "a TableVariable is a host column: read it through a DirectIn")
        return super().data()

    def col_to_gpu(self, device='cuda:0'):
        """The column's rows as a GPU tensor, read through a one-stage
        DirectIn on the current stream."""
        from IO.transfer.in_stage import DirectIn
        from IO.transfer.tick_layout import pin_cached_reads
        stage = DirectIn(self)
        DirectIn.install_compiled_tick_layout([stage])
        # The slot pieces the bind resolves stay where they are through the
        # buffer's allocation (which may lend the cache region's lowest
        # blocks) until the copies are issued.
        unpin = pin_cached_reads(self)
        try:
            agg_buf, plan = DirectIn.plan_input_data_movement([stage])
            if plan is not None:
                DirectIn.launch_h2d(agg_buf, plan)
                DirectIn.launch_d2d(agg_buf, plan)
        finally:
            unpin()
        if plan is not None:
            stage.phase_finalize(None)
            stage.decompress()
            plan.release()
        return stage.gpu_col.data()

    def record_table(self):
        return self._table

    def is_host_raw(self) -> bool:
        """True when every record is RAW and every region in pinned host
        memory: the column can be read zero-copy (``uva_source``)."""
        from IO.pages.region import HOST
        return (not self._table.is_compressed
                and all(r.location == HOST for r in self._table.regions))

    def is_uva_readable(self) -> bool:
        """True when every record is RAW and every region in pinned host
        memory or on the GPU: a kernel can read the rows by address
        (``uva_source``), as the rearrange gather does.  Under the
        free-list baseline an intermediate's regions are on the GPU until
        an eviction moves them home."""
        from IO.pages.region import DEVICE, HOST
        return (not self._table.is_compressed
                and all(r.location in (HOST, DEVICE)
                        for r in self._table.regions))

    def uva_source(self) -> "UvaSource":
        """What a zero-copy (UVA) read of this column takes
        (``IO/zero_copy_reader.cu``): the table's segments
        (``RecordTable.uva_segments``, one per stretch of records adjacent
        in one region, at the region's device alias, built once per table),
        with rows counted in the table's row space, and this Variable's rows
        ``[lo, hi)``."""
        if self._records is not None:
            # Whole records: their own segments, rows from 0.
            return _uva_source(self._table.record_segments(*self._records),
                               0, self._hi - self._lo,
                               self._table.stored_dtype,
                               int(self._table.width))
        return _uva_source(self._table.uva_segments(), self._lo, self._hi,
                           self._table.stored_dtype, int(self._table.width))

    def record_selection(self) -> Selection:
        """The records holding this Variable's rows, trimmed to them; a
        slice of whole records (``whole_records``) selects them as they
        are."""
        if self._records is not None:
            first, stop = self._records
            return Selection(self._table,
                             np.arange(first, stop, dtype=np.int64), 0,
                             self._hi - self._lo)
        return self._table.select_rows(self._lo, self._hi)

    def record_bounds(self) -> np.ndarray:
        """The record boundaries inside this Variable's rows, counted from
        its first row, with 0 and its row count."""
        g = self._table._global_start
        a = int(np.searchsorted(g, self._lo, side="right"))
        b = int(np.searchsorted(g, self._hi, side="left"))
        return np.concatenate(([0], g[a:b] - self._lo,
                               [self._hi - self._lo])).astype(np.int64)

    def whole_records(self, starts, ends):
        """Slices ``[starts[t], ends[t])`` of this Variable's rows, one per
        tick: a slice whose two ends are record boundaries reads its
        records whole (``record_selection``, ``uva_source``); None for the
        others and for empty ticks."""
        g = self._table._global_start
        starts = np.asarray(starts, dtype=np.int64) + self._lo
        ends = np.asarray(ends, dtype=np.int64) + self._lo
        r0 = np.minimum(np.searchsorted(g, starts), len(g) - 1)
        r1 = np.minimum(np.searchsorted(g, ends), len(g) - 1)
        ok = (g[r0] == starts) & (g[r1] == ends) & (ends > starts)
        out = []
        for t in range(len(starts)):
            if not ok[t]:
                out.append(None)
                continue
            out.append(TableVariable(
                self._table, self.tensor_type, selectivity=self.selectivity,
                lo=int(starts[t]), hi=int(ends[t]),
                records=(int(r0[t]), int(r1[t]))))
        return out

    def get_sliced_rep(self, slice_in: slice) -> Variable:
        start, stop = slice_in.start, slice_in.stop
        if start >= stop:
            return Variable(tensor_type=self.tensor_type,
                            selectivity=self.selectivity)
        assert 0 <= start and self._lo + stop <= self._hi, (
            f"slice [{start}, {stop}) outside {self.col_length()} rows")
        return TableVariable(self._table, self.tensor_type,
                             selectivity=self.selectivity,
                             lo=self._lo + start, hi=self._lo + stop)

    def free_underlying_mem(self, cpu_pool, gpu_pool):
        """The column's own Variable frees every region of the table (once
        no stream holds it); a slice frees nothing."""
        if not self._owns_table:
            return
        self._table.free(cpu_pool, gpu_pool)


def gpu_region_column(dst, tensor_type):
    """A GPU-resident column: the 2-D GPU tensor ``dst`` as one RAW record
    of one GPU region (an arena tensor, freed with its last reference),
    read through ``data()`` or a ``DirectIn`` (D2D).  The zero-copy stages'
    outputs and the final sort's are such columns.  No rows give an empty
    Variable over ``dst``."""
    if dst.shape[0] == 0:
        col = Variable(tensor_type=tensor_type)
    else:
        nbytes = dst.numel() * dst.element_size()
        builder = RecordTableBuilder(dst.dtype, dst.shape[1])
        k = builder.add_region(Region(
            nbytes, DEVICE, storage=dst.reshape(-1).view(torch.uint8)))
        builder.append(k, 0, nbytes, nbytes)
        col = TableVariable(builder.freeze(), tensor_type)
    col._contiguous_data = dst
    return col


def torch_to_type(dtype):
    assert isinstance(dtype, torch.dtype)
    if dtype == constants.float_dtype:
        return 'float'
    if dtype == constants.date_dtype:
        return 'date'
    if dtype == constants.int_dtype:
        return 'int'
    if dtype == torch.uint32:
        return 'uint32'
    if dtype == torch.bool:
        return 'bool'
    return 'string'

def type_to_torch(dtype):
    if dtype == 'float':
        return constants.float_dtype
    if dtype == 'int':
        return constants.int_dtype
    if dtype == 'date':
        return constants.date_dtype
    if dtype == 'uint32':
        return torch.uint32
    if dtype == 'bool':
        return torch.bool
    return constants.string_dtype
