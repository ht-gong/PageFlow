"""SF1 end-to-end correctness of the UVM mode (``main.py --uvm``).

Each cell drives ``python main.py --uvm`` as a subprocess over all 22 TPC-H
queries at SF1, with no warmup, through the ladder's runner
(``test_adaptive_matrix.run_main``): the managed arena is installed once
per process, so it cannot share the pytest process.  The GPU budget is the
ladder's :data:`CMP_POOL_GB` where a cell does not set its own: the default
(the card less 2 GiB) is more than the card has free beside the pytest
arena, which stops a ``--uvm`` run at startup.

Cells:

1. the default chunk and partition budget: every row equals
   ``tests/verification/SF1_reference.out`` at :data:`SIGFIGS_THRESHOLD`
   (10), as the smoke reads the streaming engine;
2. ``--chunk_size_mb 4 --partition_budget_mb 2``, the ladder's geometry:
   many morsels, most joins and twelve keyed aggregations partitioned (B up
   to 256), several core calls and pass-2 rounds; at
   :data:`SIGFIGS_GEOMETRY` (6), since morsels reorder float sums.  Its
   message log stays on, and its saved output must show a join with B > 1
   over more than one core call and a keyed aggregation's partitioned
   pass 2;
3. ``--cmp_pool_gb 1``: a 1 GiB GPU budget, so the driver evicts managed
   pages on the larger queries; at :data:`SIGFIGS_GEOMETRY`.

Each must exit 0 with no ``[QFAIL]`` line and all 22 queries OK in
``query_results.csv``.  At SF1 no cell meets the known-wrong compound
aggregates (Q14 is one morsel at 4 MiB, as the ladder runs it).

    python test.py tests/e2e/test_uvm.py     # ~40 s on an H100 (3 engine runs)
"""
from __future__ import annotations

import pathlib
import re

import pytest

from tests.e2e.test_adaptive_matrix import CMP_POOL_GB, EngineRun, run_main
from tests.sigfig_compare import (
    SIGFIGS_GEOMETRY, SIGFIGS_THRESHOLD, compare_results)

pytestmark = pytest.mark.e2e

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REFERENCE = _REPO_ROOT / "tests" / "verification" / "SF1_reference.out"


def _run_uvm(perf_dir: pathlib.Path, *flags: str, debug: bool = False) -> EngineRun:
    """One ``main.py --uvm`` run; with ``debug`` its message log (the
    partitioning lines) stays on and lands in the saved ``stderr.log``."""
    if "--cmp_pool_gb" not in flags:
        flags += ("--cmp_pool_gb", str(CMP_POOL_GB))
    return run_main(perf_dir, ["--uvm", "--SF", "1",
                               *([] if debug else ["--no_debug"]),
                               "--no_warmup", "--perf_report", str(perf_dir),
                               *flags])


_JOIN_LINE = re.compile(r"partitioned join: B=(\d+) calls=(\d+)")
_AGG_LINE = re.compile(r"partitioned pass 2: B=(\d+) rounds=(\d+)")


@pytest.mark.parametrize("flags,sigfigs", [
    ((), SIGFIGS_THRESHOLD),
    (("--chunk_size_mb", "4", "--partition_budget_mb", "2"), SIGFIGS_GEOMETRY),
    (("--cmp_pool_gb", "1"), SIGFIGS_GEOMETRY),
], ids=["defaults", "small-geometry", "budget-1gib"])
def test_uvm_sf1(sf1_parquet, tmp_path, flags, sigfigs):
    small = "--partition_budget_mb" in flags
    run = _run_uvm(tmp_path / "uvm", *flags, debug=small)
    assert len(run.row_counts()) == 22
    compare_results(run.results, _REFERENCE.read_text(encoding="utf-8"),
                    sigfigs)
    if small:
        # The small geometry partitions: some join takes B > 1 over more
        # than one core call, and some keyed aggregation a partitioned
        # pass 2.
        log = "".join((run.perf_dir / f).read_text(encoding="utf-8")
                      for f in ("stdout.log", "stderr.log"))
        joins = [(int(b), int(c)) for b, c in _JOIN_LINE.findall(log)]
        aggs = [int(b) for b, _ in _AGG_LINE.findall(log)]
        assert any(b > 1 and c > 1 for b, c in joins), joins
        assert any(b > 1 for b in aggs), aggs
