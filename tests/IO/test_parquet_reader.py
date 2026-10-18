"""The parquet reader (``IO/parquet_reader.py``): the datagen parquet in
TQP-Vortex's encoding, decoded on the GPU.

A ``next()`` under another stream than the reader's raises (SF1 data).  Small
parquet files written with cuDF cover each type's encoding, empty part files,
the footer type checks and every input guard.
"""
from __future__ import annotations

import os
from collections import defaultdict

import pylibcudf as plc
import pytest
import torch
from pylibcudf.gpumemoryview import gpumemoryview

from IO import parquet_reader
from IO.parquet_reader import ParquetEncodingError, ParquetTable, table_dir

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the reader decodes on the GPU")

MB = 1024 ** 2


def _read(table: ParquetTable, **limits) -> tuple:
    """Every column of ``table`` on the host, and the chunk count."""
    parts, chunks, expect = defaultdict(list), 0, 0
    for first, columns in table.batches(**limits):
        assert first == expect
        rows = {t.shape[0] for t in columns.values()}
        assert len(rows) == 1
        expect += rows.pop()
        chunks += 1
        for name, tensor in columns.items():
            parts[name].append(tensor.cpu())
    assert expect == table.rows
    return {name: torch.cat(p) for name, p in parts.items()}, chunks


@requires_cuda
def test_resumed_on_another_stream_raises(sf1_parquet):
    """The reader runs on the stream current at the first ``next()``; a
    ``next()`` under another stream raises."""
    pt = ParquetTable(sf1_parquet, 1, "lineitem")
    batches = pt.batches(["L_ORDERKEY"], chunk_read_limit=8 * MB,
                         pass_read_limit=32 * MB)
    reader = torch.cuda.Stream()
    with torch.cuda.stream(reader):
        next(batches)
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="another stream"):
            next(batches)
    torch.cuda.synchronize()


# ── Small files written with cuDF ─────────────────────────────────────────

def _fixed(values, type_id, scale=0, dtype=torch.int64) -> "plc.Column":
    data = torch.tensor(values, dtype=dtype, device="cuda")
    return plc.Column(plc.DataType(type_id, scale), len(values),
                      gpumemoryview(data), None, 0, 0, [])


def _int64(values):
    return _fixed(values, plc.TypeId.INT64)


def _int32(values):
    return _fixed(values, plc.TypeId.INT32, dtype=torch.int32)


def _decimal(unscaled, scale=-2):
    return _fixed(unscaled, plc.TypeId.DECIMAL64, scale)


def _dates(days):
    return _fixed(days, plc.TypeId.TIMESTAMP_DAYS, dtype=torch.int32)


def _strings(values):
    return plc.Column.from_iterable_of_py(values)


def _with_nulls(col):
    mask = plc.null_mask.create_null_mask(col.size(),
                                          plc.null_mask.MaskState.ALL_NULL)
    return col.with_mask(mask, col.size())


def _write(root, table, files):
    """``files``: one ``{name: Column}`` per part file (``{}`` writes an
    empty file with the first file's columns), named in order."""
    src = table_dir(str(root), 1, table)
    os.makedirs(src, exist_ok=True)
    template = files[0]
    for i, columns in enumerate(files):
        if not columns:
            columns = {n: plc.copying.slice(c, [0, 0])[0]
                       for n, c in template.items()}
        tbl = plc.Table(list(columns.values()))
        meta = plc.io.types.TableInputMetadata(tbl)
        for cm, (name, col) in zip(meta.column_metadata, columns.items()):
            cm.set_name(name)
            if col.type().id() == plc.TypeId.DECIMAL64:
                cm.set_decimal_precision(11)
        options = plc.io.parquet.ParquetWriterOptions.builder(
            plc.io.SinkInfo([os.path.join(src, f"part-{i:05d}.parquet")]),
            tbl).metadata(meta).build()
        plc.io.parquet.write_parquet(options)


def _lineitem(**override):
    columns = {
        "l_orderkey": _int64([7, 1 << 40]),
        "l_suppkey": _int32([3, -4]),
        "l_linenumber": _int32([1, 2]),
        "l_quantity": _decimal([100, 5000]),
        "l_discount": _decimal([5, 10]),
        "l_extendedprice": _decimal([3, 123456789]),
        "l_returnflag": _strings(["A", "N"]),
        "l_shipdate": _dates([8035, 10591]),
        "l_shipmode": _strings(["MAIL", "TRUCK"]),
        "l_comment": _strings(["skipped", "too"]),
    }
    columns.update(override)
    return columns


@requires_cuda
def test_encodings(tmp_path, cuda_allocator):
    second = _lineitem(l_orderkey=_int64([9, 10]),
                       l_shipmode=_strings(["SHIP", "AIR"]))
    _write(tmp_path, "lineitem", [_lineitem(), {}, second])
    pt = ParquetTable(str(tmp_path), 1, "lineitem")
    assert [r for _, r in pt.files] == [2, 0, 2]
    assert pt.rows == 4
    assert pt.columns == ["L_ORDERKEY", "L_SUPPKEY", "L_QUANTITY", "L_DISCOUNT",
                          "L_EXTENDEDPRICE", "L_RETURNFLAG", "L_SHIPDATE",
                          "L_SHIPMODE"]
    got, _ = _read(pt)
    assert got["L_ORDERKEY"].tolist() == [7, 1 << 40, 9, 10]
    assert got["L_SUPPKEY"].dtype == torch.int64
    assert got["L_SUPPKEY"].tolist() == [3, -4] * 2
    assert got["L_QUANTITY"].dtype == torch.int64
    assert got["L_QUANTITY"].tolist() == [1, 50] * 2
    assert got["L_DISCOUNT"].dtype == torch.float64
    assert got["L_DISCOUNT"].tolist() == [5 / 100.0, 10 / 100.0] * 2
    assert got["L_EXTENDEDPRICE"].tolist() == [3 / 100.0, 123456789 / 100.0] * 2
    assert got["L_RETURNFLAG"].dtype == torch.int8
    assert got["L_RETURNFLAG"].tolist() == [ord("A"), ord("N")] * 2
    assert got["L_SHIPDATE"].dtype == torch.int32
    assert got["L_SHIPDATE"].tolist() == [8035 - 7305, 10591 - 7305] * 2
    shipmode = got["L_SHIPMODE"]
    assert shipmode.shape == (4, 10) and shipmode.dtype == torch.int8
    assert [bytes(r.tolist()) for r in shipmode] == [
        s.encode().ljust(10, b"\0") for s in ["MAIL", "TRUCK", "SHIP", "AIR"]]
    assert pt.encoding("L_SHIPMODE") == (torch.int8, 10)
    assert pt.encoding("L_RETURNFLAG") == (torch.int8, 1)
    assert pt.encoding("L_QUANTITY") == (torch.int64, 1)
    assert pt.encoding("L_DISCOUNT") == (torch.float64, 1)
    assert pt.encoding("L_SHIPDATE") == (torch.int32, 1)


_GUARDS = {
    "null": (lambda: {"l_orderkey": _with_nulls(_int64([1, 2]))}, "2 nulls"),
    "too long": (lambda: {"l_shipmode": _strings(["MAIL", "REGULAR AIR"])},
                 "value of 11 bytes exceeds width 10"),
    "non-ASCII": (lambda: {"l_shipmode": _strings(["MAIL", "TRÜCK"])},
                  "non-ASCII value"),
    "empty flag": (lambda: {"l_returnflag": _strings(["A", ""])},
                   "l_returnflag: empty value"),
    "wide flag": (lambda: {"l_returnflag": _strings(["A", "NO"])},
                  "value of 2 bytes exceeds width 1"),
    "fractional quantity": (lambda: {"l_quantity": _decimal([100, 150])},
                            "l_quantity has a fractional value"),
    "date before 1992": (lambda: {"l_shipdate": _dates([8034, 9000])},
                         "date outside 1992-1998"),
    "date after 1998": (lambda: {"l_shipdate": _dates([9000, 10592])},
                        "date outside 1992-1998"),
}


@requires_cuda
@pytest.mark.parametrize("case", list(_GUARDS))
def test_guards(tmp_path, cuda_allocator, case):
    override, message = _GUARDS[case]
    _write(tmp_path, "lineitem", [_lineitem(), _lineitem(**override())])
    pt = ParquetTable(str(tmp_path), 1, "lineitem")
    with pytest.raises(ParquetEncodingError, match=message):
        _read(pt)


_FOOTER = {
    "string as int": ({"l_shipmode": _int64}, "l_shipmode: unexpected type"),
    "int as string": ({"l_orderkey": _strings}, "l_orderkey: unexpected type"),
    "quantity as int": ({"l_quantity": _int64}, "l_quantity: unexpected type"),
    "scale 3": ({"l_discount": lambda v: _decimal(v, -3)},
                "expected scale 2, got 3"),
}


@requires_cuda
@pytest.mark.parametrize("case", list(_FOOTER))
def test_footer_types(tmp_path, cuda_allocator, case):
    override, message = _FOOTER[case]
    (column, make), = override.items()
    values = ["1", "2"] if make is _strings else [1, 2]
    _write(tmp_path, "lineitem", [_lineitem(**{column: make(values)})])
    with pytest.raises(ParquetEncodingError, match=message):
        ParquetTable(str(tmp_path), 1, "lineitem")


@requires_cuda
def test_no_files_and_no_rows(tmp_path, cuda_allocator):
    with pytest.raises(ParquetEncodingError, match="no parquet files"):
        ParquetTable(str(tmp_path), 1, "lineitem")
    _write(tmp_path, "lineitem", [_lineitem(), {}])
    os.remove(os.path.join(table_dir(str(tmp_path), 1, "lineitem"),
                           "part-00000.parquet"))
    with pytest.raises(ParquetEncodingError, match="lineitem: no rows"):
        ParquetTable(str(tmp_path), 1, "lineitem")
