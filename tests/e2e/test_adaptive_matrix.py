"""SF1 end-to-end correctness over the adaptive-flag matrix.

Every rung drives ``python main.py`` as a subprocess — the real CLI wiring,
warmup included — over all 22 TPC-H queries at SF1, then checks the
results against the checked-in DuckDB reference.  A subprocess per rung is
required: the engine's allocator arena is swapped in once per process, so
two engine configurations cannot share the pytest process (the session
arena from ``tests/conftest.py`` stays resident; budget ~22 GB for it plus
~22 GB for the child).

Rungs (``main.py`` flags), under ``--compression adaptive`` throughout — a
ladder where each rung keeps the previous one's settings and adds one thing:

1. everything off (``--buffer_strategy cpu``: no GPU page cache)
2. + ``--compression_search read-and-write`` (read-leg search plus
   write-leg probe windows)
3. + ``--sizing_search mid_stream`` (sectioned per-pipe sizing plus the
   rearrange gather-window search)
4. + ``--buffer_strategy lru``      — then, each replacing the previous
5. + ``--buffer_strategy dataflow``   policy on top of rung 3:
6. + ``--buffer_strategy mru``
7. + ``--buffer_strategy cpu``      — the same configuration as rung 3, so
   this rung is an A/A repeat of the all-on configuration.

The cache rungs (4-6) sit on the ``mid_stream`` rung: the granule cache
under the sizing search, whose slot governor lends blocks of the cache's
region to a gather window step, a producer's slab or dense buffer, or pass
2's merge that asks.  The cache also
runs in :func:`test_native_core_rung`, search-off with ``lru`` and
``dataflow``, with the compile-time read plan.

One more run sits beside the ladder: rung 1's configuration with
``--compression_search fixed`` replaying the checked-in
``tests/verification/SF1_fixed_search.json`` (per-column codec pins for Q3
and Q9).  It must match rung 1 digit for digit,
run no search, and report the pins as applied.

Geometry.  At the default 256 MB chunk, SF1 pipes have too few ticks to host
a search — every rung would be an identity check of the wiring.  The ladder
therefore runs at a 4 MB chunk with small sizing steps, and each rung must
show what it adds actually ran.  4 MB is the measured sweet spot: at
8 MB only one SF1 pipe searches.  The partition budget is half the chunk,
so most joins split into buckets, up to 256, and the larger keyed
aggregations into 2 to 64, most of these in several pass-2 rounds, as they
do at scale.  B
and the rounds follow from the budget, the chunk and the rows pass 1 wrote,
never from a codec pick, so the rungs with sizing off and the fixed replay
partition alike.  Two anchors apply:

* every rung matches the reference at :data:`SIGFIGS_GEOMETRY` (6): a
  different chunk geometry — or live sizing, which re-chunks mid-run —
  reorders float partial sums past 10 significant figures;
* rungs with sizing off additionally match the same-geometry search-off run
  at :data:`SIGFIGS_THRESHOLD` (10): codec picks are a pure performance
  lever and must not move a digit.

Both compare every row of each result, in any order.  Every rung, and the
fixed replay, also reports rung 1's row counts in ``query_results.csv``,
the counts a bench reads.

The shipped defaults at the default geometry are covered in-process by
``tests/smoke``.

    python test.py tests/e2e          # ~2.5 min on an H100 (8 engine runs)
    python test.py -m "not e2e"       # everything else
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pathlib
import subprocess
import sys

import pytest

from tests.sigfig_compare import (
    SIGFIGS_GEOMETRY, SIGFIGS_THRESHOLD, compare_results)

pytestmark = pytest.mark.e2e

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REFERENCE = _REPO_ROOT / "tests" / "verification" / "SF1_reference.out"
_FIXED_SEARCH_JSON = _REPO_ROOT / "tests" / "verification" / "SF1_fixed_search.json"

# Each engine run's arena: it fits beside the pytest process's own 22 GB
# arena (tests/conftest.py), which stays resident on the same GPU.
CMP_POOL_GB = 22
PINNED_POOL_GB = 6
LIVE_CHUNK_MB = 4
# The partition budget at the live geometry, half the chunk: SF1 joins split
# into 1 to 256 buckets (48 of the 64 into more than one), and twelve keyed
# aggregations into 2 to 256 (Q1, Q2, Q9, Q10, Q13's two, Q15's two, Q16's
# two, Q18's first two), six of them in several rounds (Q18's two 64 rounds
# of a bucket each).  This ladder asserts no B: the aggregations' B at this
# geometry, and the facts the engine chooses it from, are checked in-process
# by tests/operators/test_agg_partitioned_sf1.py (against the cells of
# tests/operators/test_partitions.py), which also covers the packed and
# dense routes with B forced.  At the default 512 MiB no SF1 join or
# aggregation partitions.
LIVE_PARTITION_MB = LIVE_CHUNK_MB // 2
# Sizing plane at the live geometry: both legs start at the chunk (the
# gather leg's config default is 1 GB, not --chunk_size_mb) with a matching
# floor; the tick leg's re-cuttable pipes walk and jump from there on their
# own, and the gather leg marches in small steps (sense δ 2 MB, march Δ
# 8 MB).
LIVE_SIZING_CONFIG = {
    "march": {"delta_mb": 8, "sense_delta_mb": 2},
    "tick": {"c0_mb": LIVE_CHUNK_MB, "c_min_mb": LIVE_CHUNK_MB},
    "rearrange": {"c0_mb": LIVE_CHUNK_MB, "c_min_mb": LIVE_CHUNK_MB},
}

# (compression_search, sizing_search, buffer_strategy), in ladder order.
RUNGS = [
    ("off", "off", "cpu"),
    ("read-and-write", "off", "cpu"),
    ("read-and-write", "mid_stream", "cpu"),
    ("read-and-write", "mid_stream", "lru"),
    ("read-and-write", "mid_stream", "dataflow"),
    ("read-and-write", "mid_stream", "mru"),
    ("read-and-write", "mid_stream", "cpu"),
]
BASELINE = RUNGS[0]


class EngineRun:
    """One ``main.py`` invocation: its results text and per-query perf JSON."""

    def __init__(self, perf_dir: pathlib.Path):
        self.perf_dir = perf_dir
        self.results = (perf_dir / "results.out").read_text(encoding="utf-8")

    def _sections(self, name: str):
        for f in sorted(glob.glob(str(self.perf_dir / "SF1_*.json"))):
            with open(f, encoding="utf-8") as fh:
                yield json.load(fh).get(name) or {}

    def converged_searches(self) -> int:
        """Pipes whose read or write search leg converged (perf 'compression')."""
        n = 0
        for sec in self._sections("compression"):
            for p in sec.get("pipes", []):
                read, write = p.get("read") or {}, p.get("write") or {}
                n += bool(read.get("converged") or write.get("converged"))
        return n

    def sizing_samples(self) -> int:
        """Predicted-vs-observed sizing samples (perf 'sizing_model')."""
        return sum(len(sec.get("samples", [])) for sec in self._sections("sizing_model"))

    def cache_admissions(self) -> int:
        """Pages admitted to the GPU page cache (the movement ledger's
        ``cache_fill``); zero with no cache."""
        return sum(sec["kinds"]["cache_fill"]["pages"]
                   for sec in self._sections("movement"))

    def searched_pipes(self) -> int:
        """Pipes registered with any search driver (perf 'compression')."""
        return sum(len(sec.get("pipes", [])) for sec in self._sections("compression"))

    def fixed_pins_applied(self) -> dict[int, int]:
        """{qid: pins applied} from the fixed-mode stdout lines."""
        out: dict[int, int] = {}
        for ln in (self.perf_dir / "stdout.log").read_text(encoding="utf-8").splitlines():
            if ln.startswith("[compression_search] fixed: Q"):
                head, _, rest = ln.partition(": Q")
                qid, _, tail = rest.partition(":")
                out[int(qid)] = int(tail.split()[0])
        return out

    def row_counts(self) -> dict[int, int]:
        """{qid: result rows} from ``query_results.csv``; all 22 must be OK."""
        with open(self.perf_dir / "query_results.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["status"] for r in rows] == ["OK"] * 22, rows
        return {int(r["qid"]): int(r["rows_or_error"]) for r in rows}


def run_main(perf_dir: pathlib.Path, argv, env=None) -> EngineRun:
    """One ``python main.py <argv>`` subprocess (``argv`` the full flag list,
    its ``--perf_report`` included) with ``env`` over the environment: its
    output saved in ``perf_dir``, a zero exit and no ``[QFAIL]`` line
    asserted."""
    perf_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", *argv]
    env = dict(os.environ, TQP_QFAIL_TRACE="1", **(env or {}))
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, text=True,
                          capture_output=True, timeout=1800)
    (perf_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (perf_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
    assert proc.returncode == 0, f"{' '.join(cmd)} exited {proc.returncode}:\n{tail}"
    # A failed query is reported per line and does not fail the process.
    failed = [ln for ln in proc.stdout.splitlines() if "[QFAIL]" in ln]
    assert not failed, f"{' '.join(argv)}: query failures:\n" + "\n".join(failed)
    return EngineRun(perf_dir)


def _run_engine(perf_dir: pathlib.Path, *flags: str, extra_env=None) -> EngineRun:
    # No warmup pass and a fixed 6 GB host pin: a correctness run does not
    # need the timing-stability warmup (33 s of the 67 s a full run took).
    # The movement ledger rides along in every report (experiment telemetry).
    return run_main(
        perf_dir,
        ["--SF", "1", "--no_debug", "--no_warmup",
         "--pinned_pool_gb", str(PINNED_POOL_GB),
         "--cmp_pool_gb", str(CMP_POOL_GB), "--perf_report", str(perf_dir),
         "--partition_budget_mb", str(LIVE_PARTITION_MB), *flags],
        env={"TQP_EXPERIMENT_TELEMETRY": "1", **(extra_env or {})})


@pytest.fixture(scope="session")
def ladder(sf1_parquet, tmp_path_factory):
    """``run(rung)`` at the live geometry, memoised so the baseline rung is
    run once and shared by every comparison."""
    root = tmp_path_factory.mktemp("e2e")
    sizing_cfg = root / "sizing.json"
    sizing_cfg.write_text(json.dumps(LIVE_SIZING_CONFIG), encoding="utf-8")
    runs: dict[int, EngineRun] = {}

    def run(rung: int) -> EngineRun:
        if rung not in runs:
            search, sizing, buffer = RUNGS[rung]
            runs[rung] = _run_engine(
                root / f"rung{rung + 1}-{search}-{sizing}-{buffer}",
                "--chunk_size_mb", str(LIVE_CHUNK_MB),
                "--compression", "adaptive",
                "--compression_search", search,
                "--sizing_search", sizing,
                "--sizing_config", str(sizing_cfg),
                "--buffer_strategy", buffer)
        return runs[rung]

    return run


@pytest.mark.parametrize(
    "rung", range(len(RUNGS)),
    ids=[f"rung{i + 1}-" + "-".join(c) for i, c in enumerate(RUNGS)])
def test_rung(ladder, rung):
    search, sizing, buffer = RUNGS[rung]
    run = ladder(rung)

    compare_results(run.results, _REFERENCE.read_text(encoding="utf-8"),
                    SIGFIGS_GEOMETRY)
    if sizing == "off":
        compare_results(run.results, ladder(0).results, SIGFIGS_THRESHOLD)
    # The row counts a bench reads (query_results.csv).
    assert run.row_counts() == ladder(0).row_counts()

    # The rung must exercise what it claims to: a search that never went
    # live, a sizing plane that never sampled, or a page cache that never
    # admitted a page proves nothing.
    if search != "off":
        assert run.converged_searches() > 0, (
            f"{RUNGS[rung]}: no compression search converged — the searched "
            "path did not run (see perf 'compression')")
    if sizing != "off":
        assert run.sizing_samples() > 0, (
            f"{RUNGS[rung]}: the sizing plane recorded no samples (see perf "
            "'sizing_model')")
    if buffer != "cpu":
        assert run.cache_admissions() > 0, (
            f"{RUNGS[rung]}: the {buffer} page cache admitted no page (see "
            "perf 'movement', kind cache_fill)")


@pytest.mark.parametrize("buffer", ["lru", "dataflow"])
def test_native_core_rung(ladder, sf1_parquet, tmp_path_factory, buffer):
    """The granule cache (``IO.cache.config.CORE`` ``native``, named
    explicitly, with the compile-time read plan; sizing off): the search-off
    geometry of rung 1 with the ``buffer`` cache, every query matching the
    reference and the search-off run, and the cache admitting pages."""
    pytest.importorskip("TQPlib.tqpcache")
    run = _run_engine(
        tmp_path_factory.mktemp("e2e") / f"native-{buffer}",
        "--chunk_size_mb", str(LIVE_CHUNK_MB),
        "--compression", "adaptive",
        "--compression_search", "off",
        "--sizing_search", "off",
        "--buffer_strategy", buffer,
        extra_env={"TQP_CACHE_CORE": "native", "TQP_CACHE_PLAN": "1",
                   "TQP_CACHE_FLOOR": "0"})
    compare_results(run.results, _REFERENCE.read_text(encoding="utf-8"),
                    SIGFIGS_GEOMETRY)
    compare_results(run.results, ladder(0).results, SIGFIGS_THRESHOLD)
    assert run.row_counts() == ladder(0).row_counts()
    assert run.cache_admissions() > 0, (
        f"native {buffer}: the cache admitted no page (perf 'movement')")


def test_fixed_search_replay(ladder, sf1_parquet, tmp_path_factory):
    """``--compression_search fixed``: the checked-in pins replay with no
    search, every query matches the search-off run at 10 significant
    figures (a codec pin is a pure performance lever), and the engine
    reports the pins as applied for exactly the queries the file names."""
    with open(_FIXED_SEARCH_JSON, encoding="utf-8") as fh:
        expected = {int(q): len(pins) for q, pins in json.load(fh).items()}
    run = _run_engine(
        tmp_path_factory.mktemp("e2e") / "fixed-search",
        "--chunk_size_mb", str(LIVE_CHUNK_MB),
        "--compression", "adaptive",
        "--compression_search", "fixed",
        "--fixed_search_json", str(_FIXED_SEARCH_JSON),
        "--sizing_search", "off",
        "--buffer_strategy", "cpu")
    compare_results(run.results, _REFERENCE.read_text(encoding="utf-8"),
                    SIGFIGS_GEOMETRY)
    compare_results(run.results, ladder(0).results, SIGFIGS_THRESHOLD)
    assert run.row_counts() == ladder(0).row_counts()
    assert run.searched_pipes() == 0, "fixed mode must register no search"
    applied = run.fixed_pins_applied()
    assert {q: n for q, n in applied.items() if n} == expected, (
        f"pins applied {applied} != file {expected}")
