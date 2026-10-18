"""TPC-H parquet columns in TQP-Vortex's encoding, decoded on the GPU.

The datagen parquet of one scale factor is ``<root>/sf<sf>/parquet/<table>/``
(``root``: ``config.json``'s ``"tables path"``, ``TQP_TABLES_PATH``).
``ParquetTable`` reads one table's part files with cuDF's chunked reader
(``pylibcudf``) and converts every chunk on the GPU, with torch, into these
tensors, the encoding the engine reads:

  keys (bigint) and the int columns   int64 [rows]
  l_quantity DECIMAL(11,2)            int64 [rows], unscaled // 100
  other DECIMAL(11,2)                 float64 [rows], unscaled / 100.0
  dates                               int32 [rows], days since 1990-01-01
  O_ORDERSTATUS, L_RETURNFLAG,
  L_LINESTATUS                        int8 [rows], the character code
  other strings                       int8 [rows, width], the bytes then NUL
                                      padding (``WIDTH``)

The same 54 columns: L_LINENUMBER, L_COMMENT, O_CLERK, P_COMMENT, PS_COMMENT,
N_COMMENT and R_COMMENT are skipped.  Row order is the parquet's: a table's
part files in sorted name order (empty files included), row groups and rows
as stored.

**Footers.**  The constructor reads every part file's footer once: the row
count per file and in total (``rows``, known before any data is read), and
each column's type as cuDF maps it, checked against the encoding: an
integer, a DECIMAL32 or DECIMAL64 of scale 2 (``l_quantity`` must be one), a
date, or a string, which every column of ``WIDTH`` and ``CHAR1`` and no
other must be.

**Guards.**  A null, a string longer than its width, a
non-ASCII string, an empty O_ORDERSTATUS / L_RETURNFLAG / L_LINESTATUS, a
fractional ``l_quantity``, a date outside 1992-1998; and a table with no part
files or no rows, or a row count other than the footers'.  Each raises
``ParquetEncodingError``.  The device-side guards of a chunk are read with
one host sync, before the chunk is handed out.

**Memory and streams.**  cuDF allocates through RMM, which must already be
the engine's arena (``utility/rmm_arena.install``).  ``batches`` takes torch's
current stream at its first ``next()`` as the reader's stream: the reader,
its string padding and the torch conversion run on it, and every later
``next()`` must run under it (it raises ``RuntimeError`` otherwise).  An
int64 column and a padded string are zero-copy views of the cuDF buffer,
which the tensor keeps alive; the other columns are new torch tensors.
Either way a batch's memory goes back to the arena ordered on the reader's
stream only, when its tensor dies (``record_stream`` does nothing for it),
and the next chunk's decode may reuse it.  So a consumer on another stream
must make that stream wait on the reader's before it uses a batch, and keep
the batch's tensors alive until its work on them has completed.

**I/O.**  The file reads run on libkvikio's thread pool, sized by
``KVIKIO_NTHREADS`` at its first use; importing this module sets it to 32
unless the environment sets it.  At the default limits (one chunk per pass
of ``PASS_READ_LIMIT_B``) the SF100 parquet decodes at about 16 GB/s with
the page cache hot.
"""
from __future__ import annotations

import datetime
import glob
import os

os.environ.setdefault("KVIKIO_NTHREADS", "32")

# Imported with the module, never from a call: importing pylibcudf leaves a
# frame alive, which keeps its callers' frames and their locals (a scan's
# chunks, a query's pools) alive past the arena reset after the warmup.
import pylibcudf as plc  # noqa: E402
import rmm.pylibrmm.stream as rmm_stream  # noqa: E402
import torch  # noqa: E402

TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp",
          "region", "supplier"]
SKIP = {"l_linenumber", "l_comment", "o_clerk", "p_comment", "ps_comment",
        "n_comment", "r_comment", "ignore"}
CHAR1 = {"o_orderstatus", "l_returnflag", "l_linestatus"}
WIDTH = {"c_name": 25, "c_address": 40, "c_phone": 15, "c_mktsegment": 10,
         "c_comment": 117, "o_orderpriority": 15, "o_comment": 79,
         "l_shipinstruct": 25, "l_shipmode": 10, "p_name": 55, "p_mfgr": 25,
         "p_brand": 10, "p_type": 25, "p_container": 10, "s_name": 25,
         "s_address": 40, "s_phone": 15, "s_comment": 101, "n_name": 25,
         "r_name": 25}
_UNIX = datetime.date(1970, 1, 1)
EPOCH = (datetime.date(1990, 1, 1) - _UNIX).days       # date32 -> TQP days
DATE_LO = (datetime.date(1992, 1, 1) - _UNIX).days
DATE_HI = (datetime.date(1998, 12, 31) - _UNIX).days

PASS_READ_LIMIT_B = 4096 * 1024 ** 2
CHUNK_READ_LIMIT_B = 0

_INTS = {plc.TypeId.INT8: ("|i1", 1), plc.TypeId.INT16: ("<i2", 2),
         plc.TypeId.INT32: ("<i4", 4), plc.TypeId.INT64: ("<i8", 8)}
_DECIMALS = (plc.TypeId.DECIMAL32, plc.TypeId.DECIMAL64)


class ParquetEncodingError(ValueError):
    """The parquet does not hold what the encoding expects."""


def table_dir(root: str, sf: int, table: str) -> str:
    return os.path.join(root, f"sf{sf}", "parquet", table)


class _TorchStream:
    """A torch stream in the CUDA stream protocol, for RMM's ``Stream``."""

    def __init__(self, stream: torch.cuda.Stream):
        self._stream = stream

    def __cuda_stream__(self):
        return (0, self._stream.cuda_stream)


class _DeviceArray:
    """A device buffer in the CUDA array interface, keeping its owner alive
    for as long as a tensor viewing it lives."""

    def __init__(self, ptr: int, shape: tuple, typestr: str, owner):
        self._owner = owner
        self.__cuda_array_interface__ = {
            "shape": shape, "typestr": typestr, "data": (ptr, False),
            "version": 3, "strides": None, "stream": None}


def _view(col, typestr: str, itemsize: int) -> torch.Tensor:
    """A fixed-width cuDF column's values (at least one) as a 1-D tensor,
    zero-copy."""
    buf = col.data()
    return torch.as_tensor(
        _DeviceArray(buf.ptr + col.offset() * itemsize, (col.size(),),
                     typestr, (col, buf)), device="cuda")


def _check_type(name: str, dtype) -> None:
    """A footer's (cuDF-mapped) column type against the encoding."""
    tid = dtype.id()
    string = name in CHAR1 or name in WIDTH
    if string or tid == plc.TypeId.STRING:
        if not (string and tid == plc.TypeId.STRING):
            raise ParquetEncodingError(f"{name}: unexpected type {tid.name}")
        return
    if tid in _DECIMALS:
        if dtype.scale() != -2:
            raise ParquetEncodingError(
                f"{name}: expected scale 2, got {-dtype.scale()}")
        return
    if name == "l_quantity" or not (
            tid in _INTS or tid == plc.TypeId.TIMESTAMP_DAYS):
        raise ParquetEncodingError(f"{name}: unexpected type {tid.name}")


class ParquetTable:
    """One TPC-H table's datagen parquet, read in TQP-Vortex's encoding.

    ``files`` is ``[(path, rows)]`` in read order, ``rows`` the table's row
    count, ``columns`` its encoded columns (upper case),
    all from the footers.  ``batches`` streams the columns."""

    def __init__(self, root: str, sf: int, table: str):
        self.table = table
        src = table_dir(root, sf, table)
        paths = sorted(glob.glob(os.path.join(src, "*.parquet")))
        if not paths:
            raise ParquetEncodingError(f"no parquet files in {src}")
        meta = plc.io.parquet_metadata.read_parquet_metadata(
            plc.io.SourceInfo(paths))
        groups = iter(meta.rowgroup_metadata())
        per_file = [sum(next(groups)["num_rows"] for _ in range(k))
                    for k in meta.num_rowgroups_per_file()]
        self.files = list(zip(paths, per_file))
        self.rows = sum(per_file)
        if not self.rows:
            raise ParquetEncodingError(f"{table}: no rows")
        self._types = {}
        for field in meta.schema().root().children():
            name = field.name()
            if name not in SKIP:
                _check_type(name, field.cudf_type())
                self._types[name.upper()] = field.cudf_type()
        self.columns = list(self._types)

    def encoding(self, column: str) -> tuple:
        """``(dtype, width)`` of a column's encoding: ``width`` is the bytes
        per row of an int8 [rows, width] string, 1 for a 1-D column."""
        name = column.lower()
        tid = self._types[column].id()
        if tid == plc.TypeId.STRING:
            return torch.int8, WIDTH.get(name, 1)
        if tid == plc.TypeId.TIMESTAMP_DAYS:
            return torch.int32, 1
        if tid in _DECIMALS and name != "l_quantity":
            return torch.float64, 1
        return torch.int64, 1

    def batches(self, columns=None, *,
                chunk_read_limit: int = CHUNK_READ_LIMIT_B,
                pass_read_limit: int = PASS_READ_LIMIT_B):
        """Yield ``(first_row, {COLUMN: tensor})`` per chunk, in row order,
        covering rows ``0 .. rows`` without gaps: ``columns`` (default all)
        in the encoding, on the GPU, decoded on the reader's stream
        (module docstring)."""
        columns = self.columns if columns is None else columns
        torch_stream = torch.cuda.current_stream()
        stream = rmm_stream.Stream(_TorchStream(torch_stream))
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo([p for p, _ in self.files])
        ).column_names([c.lower() for c in columns]).build()
        reader = plc.io.parquet.ChunkedParquetReader(
            options, stream=stream, chunk_read_limit=chunk_read_limit,
            pass_read_limit=pass_read_limit)
        first = 0
        while reader.has_next():
            chunk = reader.read_chunk()
            n = chunk.tbl.num_rows()
            if first + n > self.rows:
                raise ParquetEncodingError(
                    f"{self.table}: more than {self.rows} rows")
            if not n:
                continue
            out, guards = {}, []
            for name, col in zip(chunk.column_names(), chunk.tbl.columns()):
                out[name.upper()] = _encode(name, col, stream, guards)
            del chunk
            _raise_on(guards)
            yield first, out
            if torch.cuda.current_stream() != torch_stream:
                raise RuntimeError(
                    f"{self.table}: batches resumed on another stream than "
                    f"the reader's")
            first += n
        if first != self.rows:
            raise ParquetEncodingError(
                f"{self.table}: {first} rows, expected {self.rows}")


def _cents_to_float(cents: torch.Tensor) -> torch.Tensor:
    """``unscaled / 100.0`` as numpy computes it, a true division: torch
    turns a host scalar divisor into a multiply by its reciprocal, which
    rounds differently, so the divisor is a 0-dim GPU tensor.  It is made
    per call: one kept across calls would be freed under it by the arena
    reset ``main.py`` makes after the warmup."""
    hundred = torch.full((), 100.0, dtype=torch.float64, device=cents.device)
    return cents.to(torch.float64) / hundred


def _encode(name: str, col, stream, guards: list) -> torch.Tensor:
    """One cuDF column of a chunk in the encoding.  Appends the
    chunk's device-side guards to ``guards`` as ``(flag, message)``."""
    if col.null_count():
        raise ParquetEncodingError(f"{name}: {col.null_count()} nulls")
    tid = col.type().id()
    if tid in _INTS:
        values = _view(col, *_INTS[tid])
        return values if tid == plc.TypeId.INT64 else values.to(torch.int64)
    if tid in _DECIMALS:
        cents = (_view(col, "<i8", 8) if tid == plc.TypeId.DECIMAL64
                 else _view(col, "<i4", 4).to(torch.int64))
        if name == "l_quantity":
            guards.append(((cents % 100).ne(0).any(),
                           "l_quantity has a fractional value"))
            return torch.div(cents, 100, rounding_mode="floor")
        return _cents_to_float(cents)
    if tid == plc.TypeId.TIMESTAMP_DAYS:
        days = _view(col, "<i4", 4)
        guards.append(((days < DATE_LO).any() | (days > DATE_HI).any(),
                       f"{name}: date outside 1992-1998"))
        return days - EPOCH
    # A string: the footers admit no other type.
    codes = _fixed_width(name, col, stream)
    if name in CHAR1:
        guards.append((codes.eq(0).any(), f"{name}: empty value"))
    return codes


def _fixed_width(name: str, col, stream) -> torch.Tensor:
    """strncpy(dst, s, width) for every value: cuDF pads
    each string to ``width`` characters with NUL.  The padded characters are
    ``rows x width`` bytes exactly when every value is ASCII and at most
    ``width`` bytes, and are then the [rows, width] array (a CHAR1 column's
    [rows] codes), viewed zero-copy."""
    n = col.size()
    width = 1 if name in CHAR1 else WIDTH[name]
    padded = plc.strings.padding.pad(
        col, width, plc.strings.side_type.SideType.RIGHT, "\x00", stream)
    chars = padded.data()
    if chars.nbytes != n * width:
        offsets = col.child(0)
        wide = offsets.type().id() == plc.TypeId.INT64
        ends = _view(offsets, "<i8" if wide else "<i4", 8 if wide else 4)
        longest = int((ends[1:n + 1] - ends[:n]).max())
        if longest > width:
            raise ParquetEncodingError(
                f"{name}: value of {longest} bytes exceeds width {width}")
        raise ParquetEncodingError(f"{name}: non-ASCII value")
    return torch.as_tensor(
        _DeviceArray(chars.ptr, (n, width) if name not in CHAR1 else (n,),
                     "|i1", (padded, chars)), device="cuda")


def _raise_on(guards: list) -> None:
    """Raise the first guard that fired: one host sync for the chunk."""
    if not guards:
        return
    fired = torch.stack([flag for flag, _ in guards]).cpu().tolist()
    for hit, (_, message) in zip(fired, guards):
        if hit:
            raise ParquetEncodingError(message)
