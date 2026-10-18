"""SF1 end-to-end correctness of the free-list baseline
(``--buffer_strategy freelist``, ``IO/pages/device_regions.py``).

Each cell drives ``python main.py --buffer_strategy freelist`` as a
subprocess over all 22 TPC-H queries at SF1, with no warmup, through the
ladder's runner (``test_adaptive_matrix.run_main``): intermediates stay on
the GPU as regions in ``--compression``'s format (adaptive by default), and
an arena miss evicts the least recently used of them to their homes, one at
a time, until the allocation fits.

Cells:

1. the ladder's 22 GB arena at the default geometry: nothing is evicted;
2. a 4 GiB arena at 64 MiB chunks (32 MiB partition budget), RAW, with
   ``TQP_FREELIST_SQUEEZE_GB=3.45`` holding most of it through each timed
   run (the scan, before the timer, has the whole arena): some queries
   miss mid-run, evict the least recently used regions until the
   allocation fits, read them back from their homes and complete; under
   experiment telemetry, each one's movement ledger files its evictions as
   ``spill_evict`` (logical bytes equal to stored);
3. as 1 with ``TQP_ARENA_BLOCK_B=512``: a byte-granular arena (512-B
   aligned), the scan's small cuDF buffers unbinned;
4. as 2 with ``--freelist_reserve_gb 0.25`` (of the 0.55 GiB the squeeze
   leaves): some allocations find no unpinned region left, draw on the
   reserve and complete, and the reserve refills;
5. as 2 with ``--compression BITCOMP``, and with the default adaptive
   codecs, at a 3.0 GiB squeeze with a 0.45 GiB reserve (compressed ticks
   need their worst-case slabs beside the operator's tensors, and at SF1 the
   regions are too small for evicting them to undo the arena's
   fragmentation: without the reserve the queries that evict fail): the
   regions evicted are the producer's packed records, read back and
   decoded, their logical bytes above their stored ones.

Each must exit 0 with no ``[QFAIL]`` line, all 22 queries OK, and every row
equal to ``tests/verification/SF1_reference.out`` at
:data:`SIGFIGS_GEOMETRY` (6): the intermediates' format and the evictions do
not change the results.

    python test.py tests/e2e/test_freelist.py     # ~2 min on an H100 (6 engine runs)
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tests.e2e.test_adaptive_matrix import CMP_POOL_GB, EngineRun, run_main
from tests.sigfig_compare import SIGFIGS_GEOMETRY, compare_results

pytestmark = pytest.mark.e2e

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REFERENCE = _REPO_ROOT / "tests" / "verification" / "SF1_reference.out"


def _reserve_draws(run: EngineRun) -> dict[int, tuple]:
    """{qid: (draws, bytes drawn, bytes refilled)} of the queries that drew
    on the reserve (perf 'freelist')."""
    out = {}
    for f in sorted(run.perf_dir.glob("SF1_*.json")):
        sec = json.loads(f.read_text(encoding="utf-8"))["freelist"]
        if sec["reserve_draws"]:
            out[int(f.stem.split("_")[1])] = (
                sec["reserve_draws"], sec["reserve_drawn_b"],
                sec["reserve_refilled_b"])
    return out


def _evictions(run: EngineRun) -> dict[int, tuple]:
    """{qid: (evictions, regions, bytes, logical bytes, ledger)} of the
    queries that evicted (perf 'freelist'), ``ledger`` the movement ledger's
    ``spill_evict`` (pages, logical, wire) when the report has one."""
    out = {}
    for f in sorted(run.perf_dir.glob("SF1_*.json")):
        report = json.loads(f.read_text(encoding="utf-8"))
        sec = report["freelist"]
        if sec["evictions"]:
            kinds = (report.get("movement") or {}).get("kinds") or {}
            ev = kinds.get("spill_evict")
            out[int(f.stem.split("_")[1])] = (
                sec["evictions"], sec["regions"], sec["bytes"],
                sec["logical_bytes"], None if ev is None else
                (ev["pages"], ev["logical_b"], ev["wire_b"]))
    return out


_SQUEEZED = ("--chunk_size_mb", "64", "--partition_budget_mb", "32")
_RAW = ("--compression", "RAW")
_COMPRESSED_RESERVE = ("--freelist_reserve_gb", "0.45")


@pytest.mark.parametrize("cmp_gb,flags,squeeze_gb,block_b", [
    (CMP_POOL_GB, (), None, None),
    (4, _SQUEEZED + _RAW, 3.45, None),
    (CMP_POOL_GB, (), None, 512),
    (4, _SQUEEZED + _RAW + ("--freelist_reserve_gb", "0.25"), 3.45, None),
    (4, _SQUEEZED + ("--compression", "BITCOMP") + _COMPRESSED_RESERVE, 3.0,
     None),
    (4, _SQUEEZED + _COMPRESSED_RESERVE, 3.0, None),
], ids=["no-pressure", "squeezed", "512B-blocks", "reserve",
        "squeezed-bitcomp", "squeezed-adaptive"])
def test_freelist_sf1(sf1_parquet, tmp_path, cmp_gb, flags, squeeze_gb,
                      block_b):
    perf_dir = tmp_path / "freelist"
    env = {}
    if squeeze_gb:
        env.update(TQP_FREELIST_SQUEEZE_GB=str(squeeze_gb),
                   TQP_EXPERIMENT_TELEMETRY="1")
    if block_b:
        env["TQP_ARENA_BLOCK_B"] = str(block_b)
    run = run_main(
        perf_dir,
        ["--SF", "1", "--no_debug", "--no_warmup", "--pinned_pool_gb", "6",
         "--buffer_strategy", "freelist", "--cmp_pool_gb", str(cmp_gb),
         "--perf_report", str(perf_dir), *flags],
        env=env or None)
    if block_b:
        stdout = (perf_dir / "stdout.log").read_text(encoding="utf-8")
        assert f"[arena] {block_b} B blocks" in stdout, stdout[-2000:]
    assert len(run.row_counts()) == 22
    compare_results(run.results, _REFERENCE.read_text(encoding="utf-8"),
                    SIGFIGS_GEOMETRY)
    evicted = _evictions(run)
    if squeeze_gb:
        assert evicted, "the squeezed arena evicted nothing: the path did not run"
        # Each query's evictions are its ledger's spill_evict: one page per
        # region, the bytes moved its wire bytes, what they decode to its
        # logical ones (equal for RAW regions).
        for qid, (_, regions, nbytes, logical, ledger) in evicted.items():
            assert ledger == (regions, logical, nbytes), (qid, evicted[qid])
            if "RAW" in flags:
                assert logical == nbytes, (qid, evicted[qid])
        if "RAW" not in flags:
            # Compressed regions decode to more than they store.
            assert any(lg > nb for _, _, nb, lg, _ in evicted.values()), \
                evicted
    else:
        assert not evicted, f"evictions with no pressure: {evicted}"
    drawn = _reserve_draws(run)
    if "--freelist_reserve_gb" in flags:
        assert drawn, "no query drew on the reserve: the path did not run"
        # A refill takes back drawn blocks once they are free again, never
        # more (blocks still in use at the query's end come back with the
        # rest of the reserve there).
        for qid, (_, gave, refilled) in drawn.items():
            assert 0 < gave and refilled <= gave, (qid, drawn[qid])
    else:
        assert not drawn, f"reserve draws without a reserve: {drawn}"
