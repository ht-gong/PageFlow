#!/usr/bin/env python3
"""Compare an ndsgen dataset with one of the same SF from the retired dbgen + Spark
pipeline (datagen/gen_tpch.sh, removed after this check passed at SF1/10/100/1000;
its datasets stay valid).

  compare.py <reference parquet dir> <ndsgen parquet dir> [--tables t1,t2] [--no-rows]

Checks, per table:
  rows     the same multiset of rows: count and an order-independent sum of
           per-row hashes (DuckDB hash() over every column but `ignore`)
  schema   column names, Arrow types and parquet physical types
  layout   part-files, row groups, rows per file
  format   per column: the share of files whose chunk is dictionary-encoded,
           and the compressed bytes (ndsgen / reference)
  order    the key column within files: share of ascending neighbours and the
           position-key correlation (Spark's shuffle: ~0.5 and ~+0.7)
Exits non-zero if rows or schema differ.
"""
import argparse
import glob
import os
import sys

import duckdb
import numpy as np
import pyarrow.parquet as pq

TABLES = ["lineitem", "orders", "partsupp", "part", "customer", "supplier", "nation", "region"]
KEY = {"lineitem": "l_orderkey", "orders": "o_orderkey", "partsupp": "ps_partkey", "part": "p_partkey",
       "customer": "c_custkey", "supplier": "s_suppkey"}
DICT = {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}


def files(d, t):
    return sorted(glob.glob(os.path.join(d, t, "*.parquet")))


def fingerprint(con, d, t):
    cols = [n for n in pq.read_schema(files(d, t)[0]).names if n != "ignore"]
    q = (f"SELECT count(*), sum(hash({', '.join(cols)}))::HUGEINT, count(ignore) "
         f"FROM read_parquet('{d}/{t}/*.parquet')")
    return con.execute(q).fetchone()


def chunk_stats(fs):
    """per column: [files, dict files, compressed bytes]; plus files, row groups, rows per file"""
    cols, rgs, rows = {}, 0, []
    for f in fs:
        md = pq.ParquetFile(f).metadata
        rgs += md.num_row_groups
        rows.append(md.num_rows)
        for r in range(md.num_row_groups):
            rg = md.row_group(r)
            for i in range(rg.num_columns):
                c = rg.column(i)
                s = cols.setdefault(c.path_in_schema, [0, 0, 0])
                s[0] += 1
                s[1] += bool(DICT & set(c.encodings))
                s[2] += c.total_compressed_size
    return cols, rgs, rows


def order(fs, key):
    asc, corr = [], []
    for f in fs[:: max(1, len(fs) // 8)][:8]:
        x = pq.read_table(f, columns=[key])[key].to_numpy()
        if len(x) > 2:
            asc.append(np.mean(np.diff(x) > 0))
            corr.append(np.corrcoef(np.arange(len(x)), x)[0, 1])
    return np.mean(asc), np.mean(corr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("new")
    p.add_argument("--tables", default=",".join(TABLES))
    p.add_argument("--no-rows", action="store_true")
    a = p.parse_args()
    con = duckdb.connect()
    bad = []
    for t in a.tables.split(","):
        rf, nf = files(a.ref, t), files(a.new, t)
        print(f"== {t}")
        if not a.no_rows:
            fr, fn = fingerprint(con, a.ref, t), fingerprint(con, a.new, t)
            ok = fr == fn
            print(f"  rows    {'OK ' if ok else 'DIFF'} count {fr[0]:,} vs {fn[0]:,}, row-hash sum "
                  f"{'equal' if fr[1] == fn[1] else 'DIFFERENT'}, ignore non-null {fr[2]} vs {fn[2]}")
            bad += [] if ok else [f"{t} rows"]
        sr, sn = pq.read_schema(rf[0]), pq.read_schema(nf[0])
        pr = [pq.ParquetFile(rf[0]).schema.column(i).physical_type for i in range(len(sr))]
        pn = [pq.ParquetFile(nf[0]).schema.column(i).physical_type for i in range(len(sn))]
        same = sr.names == sn.names and [str(f.type) for f in sr] == [str(f.type) for f in sn] and pr == pn
        print(f"  schema  {'OK' if same else 'DIFF'}")
        if not same:
            bad.append(f"{t} schema")
            for f, g, x, y in zip(sr, sn, pr, pn):
                if (f.name, str(f.type), x) != (g.name, str(g.type), y):
                    print(f"     {f.name} {f.type} {x}  vs  {g.name} {g.type} {y}")
        cr, rgr, rr = chunk_stats(rf)
        cn, rgn, rn = chunk_stats(nf)
        print(f"  layout  files {len(rf)} vs {len(nf)}, row groups {rgr} vs {rgn}, rows/file median "
              f"{int(np.median(rr))} vs {int(np.median(rn))} (max {max(rr)} vs {max(rn)})")
        tot_r = sum(v[2] for v in cr.values())
        tot_n = sum(v[2] for v in cn.values())
        print(f"  bytes   {tot_r / 1e6:,.1f} MB vs {tot_n / 1e6:,.1f} MB ({tot_n / tot_r - 1:+.1%})")
        print(f"  {'column':18s} {'dict files ref':>15s} {'ndsgen':>8s} {'bytes ratio':>12s}")
        for c in cr:
            r, n = cr[c], cn.get(c, [0, 0, 0])
            flag = "" if (r[1] > 0) == (n[1] > 0) and abs(n[2] / max(r[2], 1) - 1) < 0.05 else "   <--"
            print(f"  {c:18s} {r[1] / r[0]:15.0%} {n[1] / max(n[0], 1):8.0%} {n[2] / max(r[2], 1):12.3f}{flag}")
        if t in KEY:
            ar, kr = order(rf, KEY[t])
            an, kn = order(nf, KEY[t])
            print(f"  order   {KEY[t]}: ascending neighbours {ar:.3f} vs {an:.3f}, corr(position, key) {kr:+.3f} vs {kn:+.3f}")
    print("RESULT:", "OK" if not bad else "DIFF " + ", ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
