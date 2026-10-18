#!/usr/bin/env python3
"""Validate that a TPC-H parquet dataset is a CLEAN, EXTERNAL NDS-H build.

  validate_dataset.py <parquet_dir> <scale_factor>

<parquet_dir> holds one sub-dir per table (customer/ lineitem/ ... supplier/),
the layout every runner reads (`{dir}/{table}/*.parquet`). Exits 0 if the dataset
passes every check, non-zero (with a diagnosis) on the first failure.

Why this exists
---------------
All baselines MUST run on parquet from the SAME generator: datagen/ndsgen,
which writes the NDS-H dataset (TPC-H dbgen 2.17.3 rows, the Spark transcode's
schema, layout and encodings). Datasets written by the retired dbgen + Spark
pipeline (datagen/gen_tpch.sh, writer parquet-mr) hold the same rows and stay
valid. See results/GENERATOR.md.

Two classes of bad dataset have polluted results before, and this guard rejects
both:

  1. Wrong generator -- parquet written by DuckDB's own `dbgen` (16-col, no
     trailing `ignore`) or by cudf. Detected via the parquet `created_by`
     writer string: parquet-rs with ndsgen's NDSGEN.json next to the parquet
     dir, or parquet-mr (the retired Spark transcode).

  2. Corrupt values -- some legacy NDS-H SF100/SF300 builds have an out-of-spec
     `part` table: `p_brand` takes 80 contiguous values (Brand#11..Brand#90)
     instead of the 25 the TPC-H spec allows (Brand#<1-5><1-5>), inflating the
     distinct (brand,type,size) group count from 187,500 to ~600,000 and
     silently changing q16/q17/q19 answers. Detected by checking p_brand against
     the spec alphabet.

Checks: writer is ndsgen (or the retired parquet-mr transcode); part.p_brand within the 25-value spec set;
distinct(p_brand,p_type,p_size) <= 187,500; deterministic per-SF row counts.
"""
import glob
import os
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq

# TPC-H p_brand spec: 'Brand#' + M + N, M in 1..5, N in 1..5  ->  exactly 25.
SPEC_BRANDS = {f"Brand#{m}{n}" for m in range(1, 6) for n in range(1, 6)}
MAX_BRAND_TYPE_SIZE = 25 * 150 * 50  # 187,500 -- fully saturated part catalogue

# Deterministic per-SF-unit row counts (nation/region are fixed).
PER_SF = {"customer": 150_000, "orders": 1_500_000, "part": 200_000,
          "partsupp": 800_000, "supplier": 10_000}
FIXED = {"nation": 25, "region": 5}
# lineitem averages ~6.0M/SF but is data-dependent; bound it loosely.
LINEITEM_PER_SF = 6_000_000


class Fail(Exception):
    pass


def files(parquet_dir, table):
    fs = sorted(glob.glob(os.path.join(parquet_dir, table, "*.parquet")))
    if not fs:
        raise Fail(f"{table}: no parquet files under {parquet_dir}/{table}/")
    return fs


def table_rows(fs):
    return sum(pq.ParquetFile(f).metadata.num_rows for f in fs)


def check_writer(parquet_dir):
    fs = files(parquet_dir, "part")
    writer = pq.ParquetFile(fs[0]).metadata.created_by or ""
    # datagen/ndsgen writes with parquet-rs and leaves NDSGEN.json next to parquet/.
    ndsgen = os.path.join(os.path.dirname(os.path.abspath(parquet_dir)), "NDSGEN.json")
    if writer.lower().startswith("parquet-rs") and os.path.exists(ndsgen):
        print(f"  writer      OK  ({writer}, datagen/ndsgen)")
        return
    # The retired dbgen + Spark transcode (datagen/ndsgen/compare.py: the same rows).
    if not writer.lower().startswith("parquet-mr"):
        raise Fail(
            f"wrong generator: part parquet written by {writer!r}, expected datagen/ndsgen "
            f"(parquet-rs, with NDSGEN.json beside the parquet dir) or the retired NDS-H "
            f"Spark transcode (parquet-mr). DuckDB- or cudf-written parquet is "
            f"self-generated and MUST NOT be used -- see results/GENERATOR.md.")
    print(f"  writer      OK  ({writer}, the retired dbgen + Spark transcode)")


def check_part_values(parquet_dir):
    fs = files(parquet_dir, "part")
    import pyarrow as pa
    t = pa.concat_tables(
        [pq.read_table(f, columns=["p_brand", "p_type", "p_size"]) for f in fs])
    brands = set(pc.unique(t["p_brand"]).to_pylist())
    bad = sorted(brands - SPEC_BRANDS)
    if bad:
        raise Fail(
            f"corrupt part.p_brand: {len(brands)} distinct values, "
            f"{len(bad)} out of TPC-H spec (e.g. {bad[:6]}). Spec allows only the "
            f"25 Brand#<1-5><1-5> values. This is the legacy NDS-H SF100/SF300 "
            f"corruption -- regenerate with the current dbgen.")
    groups = t.group_by(["p_brand", "p_type", "p_size"]).aggregate([]).num_rows
    if groups > MAX_BRAND_TYPE_SIZE:
        raise Fail(f"corrupt part: distinct(brand,type,size)={groups:,} exceeds "
                   f"spec max {MAX_BRAND_TYPE_SIZE:,}")
    print(f"  part.p_brand OK  ({len(brands)} brands, {groups:,} brand/type/size groups)")


def check_row_counts(parquet_dir, sf):
    for table, fixed in FIXED.items():
        n = table_rows(files(parquet_dir, table))
        if n != fixed:
            raise Fail(f"{table}: {n:,} rows, expected {fixed}")
    for table, per in PER_SF.items():
        n = table_rows(files(parquet_dir, table))
        want = per * sf
        if n != want:
            raise Fail(f"{table}: {n:,} rows, expected {want:,} at SF{sf}")
    n = table_rows(files(parquet_dir, "lineitem"))
    lo, hi = int(LINEITEM_PER_SF * sf * 0.98), int(LINEITEM_PER_SF * sf * 1.02)
    if not (lo <= n <= hi):
        raise Fail(f"lineitem: {n:,} rows, expected ~{LINEITEM_PER_SF * sf:,} "
                   f"(+/-2%) at SF{sf}")
    print(f"  row counts  OK  (lineitem {n:,})")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    parquet_dir, sf = sys.argv[1], int(sys.argv[2])
    print(f"validating {parquet_dir} as clean external NDS-H SF{sf} ...")
    try:
        check_writer(parquet_dir)
        check_part_values(parquet_dir)
        check_row_counts(parquet_dir, sf)
    except Fail as e:
        print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"VALID: {parquet_dir} is a clean external NDS-H SF{sf} dataset.")


if __name__ == "__main__":
    main()
