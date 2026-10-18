"""SF1 end-to-end correctness with the arena lending the granule cache's
region (``memory/README.md``, "Lending").

Each cell drives ``python main.py`` as a subprocess over all 22 TPC-H
queries at SF1, with no warmup, through the ladder's runner
(``test_adaptive_matrix.run_main``), on a 2 GiB arena whose cache region
takes 1.75 GiB after each scan: the operators get 0.25 GiB of their own,
so their allocations miss and the allocator lends them the region's
lowest blocks (in C++, inside the allocation), which come back at the next
pipeline start.  One cell per policy, each under another sizing-search mode
(lending does not depend on either).

Each must exit 0 with no ``[QFAIL]`` line, all 22 queries OK, every row
equal to ``tests/verification/SF1_reference.out`` at
:data:`SIGFIGS_GEOMETRY` (6), and its perf reports' ``lend`` sections must
show the misses lent for (none short or failed) and blocks returned.  The
region does not fill at SF1, so these lends move no granule; eviction and
relocation under a lend are ``tests/allocator/test_arena_lend.py``'s.

    python test.py tests/e2e/test_cache_lend.py     # ~40 s on an H100 (3 engine runs)
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tests.e2e.test_adaptive_matrix import run_main
from tests.sigfig_compare import SIGFIGS_GEOMETRY, compare_results

pytestmark = pytest.mark.e2e

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REFERENCE = _REPO_ROOT / "tests" / "verification" / "SF1_reference.out"


def _lends(run) -> dict[int, dict]:
    """{qid: its perf report's ``lend`` section}."""
    out = {}
    for f in sorted(run.perf_dir.glob("SF1_*.json")):
        report = json.loads(f.read_text(encoding="utf-8"))
        out[int(f.stem.split("_")[1])] = report["lend"]
    return out


@pytest.mark.parametrize("policy,sizing", [
    ("lru", "off"), ("dataflow", "per_pipe"), ("mru", "mid_stream"),
])
def test_cache_lend_sf1(sf1_parquet, tmp_path, policy, sizing):
    perf_dir = tmp_path / f"lend-{policy}"
    run = run_main(
        perf_dir,
        ["--SF", "1", "--no_debug", "--no_warmup", "--pinned_pool_gb", "6",
         "--buffer_strategy", policy, "--sizing_search", sizing,
         "--cmp_pool_gb", "2", "--slot_pool_gb", "1.75",
         "--perf_report", str(perf_dir)])
    assert len(run.row_counts()) == 22
    compare_results(run.results, _REFERENCE.read_text(encoding="utf-8"),
                    SIGFIGS_GEOMETRY)
    lends = _lends(run)
    assert len(lends) == 22
    lent = {q: s for q, s in lends.items() if s["lent"]}
    assert lent, "no query lent: the operators never missed"
    for q, s in lends.items():
        assert s["short"] == 0 and s["failed"] == 0, (q, s)
        assert s["misses"] == s["lent"], (q, s)
        # Every block lent came back or was still out at the query's end.
        assert s["blocks_lent"] == s["blocks_returned"] + s["lent_now"], (
            q, s)
    assert sum(s["blocks_returned"] for s in lent.values()) > 0, lent
    stdout = (perf_dir / "stdout.log").read_text(encoding="utf-8")
    assert stdout.count("[lend] Q") == 22
