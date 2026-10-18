#!/usr/bin/env python
"""polars_gpu column probes (see probe_common.py). Same GPUEngine and table scans
as the TPC-H runner (polars/run_tpch_polars.py), then a warm-up. The engine's
fallback_mode is "warn", so a probe that cudf-polars cannot run on the GPU
raises a warning and is recorded as FALLBACK."""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "polars"))
import probe_common as pc  # noqa: E402
import polars as pl  # noqa: E402
from run_tpch_polars import TABLES, build_engine, scan  # noqa: E402


def main():
    parquet, cols, out_csv = pc.args()
    print(f"Polars {pl.__version__} | engine=gpu", flush=True)
    engine = build_engine()
    lf = {t: scan(parquet, t) for t in TABLES}
    t0 = time.time()
    lf["lineitem"].select(pl.len()).collect(engine=engine)
    print(f"warm-up done in {time.time() - t0:.1f}s", flush=True)

    def probe(table, column):
        q = lf[table].select(pl.col(column).min().alias("mn"), pl.col(column).max().alias("mx"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.time()
            q.collect(engine=engine)
            dt = time.time() - t0
        if caught:
            return "FALLBACK", dt, str(caught[0].message).splitlines()[0][:160]
        return "OK", dt, ""

    pc.run(cols, out_csv, probe)


if __name__ == "__main__":
    main()
