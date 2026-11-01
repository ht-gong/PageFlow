#!/usr/bin/env python3
"""One baseline engine's datasets in its format map, for a suite's loop
(AGENTS.md, "Before every benchmark"): the disk guard, generation and
deletion, one (engine, SF) at a time.  Experiments only read what this made.

    engine_data.py status                  # every dataset under $DATA_DIR, free disk
    engine_data.py need <engine> <SF>      # the guard alone (exit 1: no room)
    engine_data.py make <engine> <SF>      # guard, then $DATA_DIR/<engine>/sf<SF>
                                           # and its sf1 (the warm pass's)
    engine_data.py drop <engine> <SF>      # delete $DATA_DIR/<engine>/sf<SF>

Env: DATA_DIR (the shared root; an engine's is $DATA_DIR/<engine>), MIN_FREE_GB
(the runners' floor, default 30), and for make what datagen/dataset.sh reads
(PY, NDSGEN, GEN_PARALLEL, GEN_BATCH), so make runs through the launcher
(docker/run.sh, or a box's native stand-in).  --map-key picks the profiling SF
in a map that holds more than one.

The estimate: for each column, the compressed bytes of the map's pick in
results/format_profile.csv at the profiling SF, scaled by SF / that SF (a
column the map leaves null gets parquet-mr's default, dict + snappy, as ndsgen
writes it).  make stops unless the disk holding DATA_DIR has the estimate plus
25% and then MIN_FREE_GB free.  A dataset already there is validated, not
regenerated, and must be this engine's: its NDSGEN.json names the map and the
engine.  drop never touches the shared $DATA_DIR/sf<SF>.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAP = os.path.join(ROOT, "results", "format_map.json")
PROFILE = os.path.join(ROOT, "results", "format_profile.csv")
ENGINES = ("rapids", "polars_gpu", "duckdb_cpu", "sirius")
DEFAULT_FORMAT = {"encoding": "dict", "compression": "snappy"}
GB = 1e9


def data_dir() -> str:
    path = os.environ.get("DATA_DIR")
    if not path:
        sys.exit("DATA_DIR is not set")
    return path


def picks(engine: str, map_key: str | None) -> tuple[dict, int]:
    """(``{table: {column: format | None}}``, the profiling SF) of ``engine``."""
    with open(MAP) as f:
        maps = json.load(f)
    if map_key is None:
        if len(maps) != 1:
            sys.exit(f"{MAP} holds {sorted(maps)}: pass --map-key")
        map_key = next(iter(maps))
    m = maps[map_key]
    if engine not in m["engines"]:
        sys.exit(f"{MAP} [{map_key}] has no {engine}: run the profiling pass "
                 f"(make ablation) for it first")
    return m["engines"][engine], int(m["scale_factor"])


def estimate(engine: str, sf: int, map_key: str | None) -> float:
    """Bytes of ``engine``'s dataset at ``sf``, from the profile (above)."""
    cols, at = picks(engine, map_key)
    size = {}
    with open(PROFILE, newline="") as f:
        for r in csv.DictReader(f):
            if r["engine"] == engine and int(r["scale_factor"]) == at and r["bytes"]:
                size[(r["table"], r["column"], r["encoding"], r["compression"])] = \
                    int(r["bytes"])
    total = 0
    for table, cs in cols.items():
        for col, fmt in cs.items():
            fmt = fmt or DEFAULT_FORMAT
            key = (table, col, fmt["encoding"], fmt["compression"])
            if key not in size:
                sys.exit(f"{PROFILE} has no bytes for {engine} {key} at SF{at}")
            total += size[key]
    return total * sf / at


def du(path: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(path) for f in fs)


def free_bytes(path: str) -> int:
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free


def owner(sf_dir: str) -> str:
    """The format NDSGEN.json records for ``sf_dir`` ("map <path> <engine>"
    for a format-map dataset), or why there is none."""
    try:
        with open(os.path.join(sf_dir, "NDSGEN.json")) as f:
            return json.load(f).get("format", "?")
    except FileNotFoundError:
        return "no NDSGEN.json (not an ndsgen dataset)"


def is_engines(sf_dir: str, engine: str) -> bool:
    fmt = owner(sf_dir)
    return fmt.startswith("map ") and fmt.split()[-1] == engine


def guard(engine: str, sfs: list[int], map_key: str | None) -> bool:
    """Print the guard's arithmetic for the datasets of ``sfs`` not yet there;
    True when they fit."""
    root = os.path.join(data_dir(), engine)
    todo = [s for s in sfs if not os.path.isdir(os.path.join(root, f"sf{s}", "parquet"))]
    est = sum(estimate(engine, s, map_key) for s in todo)
    floor = float(os.environ.get("MIN_FREE_GB") or 30) * GB
    need, free = est * 1.25 + floor, free_bytes(root)
    ok = not todo or free >= need     # nothing to generate: nothing to fit
    print(f"{engine} SF{'/SF'.join(map(str, todo)) or ' (all present)'}: estimate "
          f"{est / GB:.1f} GB; need {need / GB:.1f} GB (+25%, +{floor / GB:.0f} GB "
          f"floor), free {free / GB:.1f} GB -> {'OK' if ok else 'NO ROOM'}")
    return ok


def cmd_status(_a) -> int:
    top = data_dir()
    rows = []
    for engine in ("", *ENGINES):
        root = os.path.join(top, engine)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if name.startswith("sf") and os.path.isdir(d):
                rows.append((engine or "(shared)", name, du(d) / GB, owner(d)))
    for engine, name, size, fmt in rows:
        print(f"  {engine:<11} {name:<7} {size:8.1f} GB  {fmt}")
    if not rows:
        print("  no datasets")
    print(f"free on {top}: {free_bytes(top) / GB:.1f} GB")
    return 0


def cmd_need(a) -> int:
    return 0 if guard(a.engine, sorted({1, a.sf}), a.map_key) else 1


def cmd_make(a) -> int:
    root = os.path.join(data_dir(), a.engine)
    for s in sorted({1, a.sf}):
        d = os.path.join(root, f"sf{s}")
        if os.path.isdir(os.path.join(d, "parquet")) and not is_engines(d, a.engine):
            print(f"{d} is not {a.engine}'s format-map dataset ({owner(d)}): "
                  f"drop it first")
            return 1
    if not guard(a.engine, sorted({1, a.sf}), a.map_key):
        return 1
    opts = f"--format-map {MAP} --engine {a.engine}"
    if a.map_key:
        opts += f" --map-key {a.map_key}"
    env = dict(os.environ, DATA_DIR=root, GEN_OPTS=opts)
    for s in sorted({1, a.sf}):
        # Generates a missing dataset; validates one already there.
        rc = subprocess.call([os.path.join(HERE, "dataset.sh"), str(s)], env=env)
        if rc:
            print(f"dataset.sh {s} failed (exit {rc})")
            return rc
    return 0


def cmd_drop(a) -> int:
    d = os.path.join(data_dir(), a.engine, f"sf{a.sf}")
    if not os.path.isdir(d):
        print(f"{d}: not there")
        return 0
    size = du(d)
    shutil.rmtree(d)
    print(f"dropped {d} ({size / GB:.1f} GB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map-key", help="the profiling SF in format_map.json, e.g. sf5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    for name, fn in (("need", cmd_need), ("make", cmd_make), ("drop", cmd_drop)):
        p = sub.add_parser(name)
        p.add_argument("engine", choices=ENGINES)
        p.add_argument("sf", type=int)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
