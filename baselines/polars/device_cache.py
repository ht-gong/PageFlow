"""Device-resident input tables for cudf-polars (resident mode, see AGENTS.md).

cudf-polars has no API that keeps a frame on the GPU across `collect` calls: a
LazyFrame over a host DataFrame is a `DataFrameScan`, whose `do_evaluate` copies
the projected columns host->device on every collect (`plc.Table.from_arrow`).
This module replaces that method so each column is uploaded once and then served
from GPU memory by every later collect in the process.

Only the in-memory executor evaluates `DataFrameScan` through `do_evaluate` on
the whole frame; the streaming executor slices the frame into partitions on its
own path, so resident mode runs `executor="in-memory"`.

A column is keyed by (name, height, dtype): TPC-H column names are unique across
tables, and the SF1 warm data and the target data differ in height. Only one
scale factor's columns are held at a time (`clear()` between them).

    import device_cache            # patches on import
    device_cache.fill()            # misses are uploaded and kept
    ...collect...                  # e.g. the SF1 warm pass, then the target's fill
    device_cache.freeze()          # misses are uploaded for that collect only
    device_cache.reset_counters(); ...timed collect...; device_cache.misses

Tied to cudf-polars 26.06: the version and `do_evaluate`'s signature are checked
on import, so an upgrade fails loudly instead of silently timing the copy.
"""
import inspect

import polars as pl
import cudf_polars
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import DataFrameScan

_VERSION = "26.06"
if not cudf_polars.__version__.startswith(_VERSION):
    raise ImportError(f"device_cache patches cudf-polars {_VERSION}.x, found {cudf_polars.__version__}")
_ORIG = DataFrameScan.do_evaluate
if list(inspect.signature(_ORIG).parameters) != ["schema", "df", "projection", "context"]:
    raise ImportError(f"DataFrameScan.do_evaluate changed: {inspect.signature(_ORIG)}")

_cache = {}         # (name, height, dtype) -> cudf_polars Column on the device
_filling = False
misses = []         # column names uploaded since reset_counters()
hits = 0


def fill():
    global _filling
    _filling = True


def freeze():
    global _filling
    _filling = False


def clear():
    _cache.clear()
    freeze()
    reset_counters()


def reset_counters():
    global hits
    misses.clear()
    hits = 0


def resident_bytes():
    return sum(c.obj.device_buffer_size() for c in _cache.values())


def _do_evaluate(cls, schema, df, projection, *, context):
    global hits
    if len(schema) == 0:
        return _ORIG(schema, df, projection, context=context)
    pl_df = pl.DataFrame._from_pydf(df) if not isinstance(df, pl.DataFrame) else df
    stream = context.get_cuda_stream()
    cols = []
    for name in (projection if projection is not None else pl_df.columns):
        key = (name, pl_df.height, str(pl_df.schema[name]))
        col = _cache.get(key)
        if col is None:
            misses.append(name)
            col = DataFrame.from_polars(pl_df.select(name), stream=stream).columns[0]
            if _filling:
                _cache[key] = col
        else:
            hits += 1
        # Shallow copy: Column.set_sorted mutates in place, and the cached
        # column must stay as uploaded.
        cols.append(col.copy())
    out = DataFrame(cols, stream=stream)
    assert all(c.obj.type() == dtype.plc_type
               for c, dtype in zip(out.columns, schema.values(), strict=True))
    return out


DataFrameScan.do_evaluate = classmethod(_do_evaluate)
