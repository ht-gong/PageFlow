"""The cache region is the operators', not the scan's (``main.py``).

Each query occurrence scans with no cache region, so the scan has the whole
arena, then sizes the region for its operators before its timer starts, and
empties it at its end.  The scan never uses the cache and the cache is empty
between queries, so nothing is lost: a 14 GiB region of a 30 GiB arena failed
Q13's SF100 scan (a 2.19 GB padded ``o_comment`` chunk against a 2.05 GB
largest free run) while the region sat empty.

One in-process SF1 run of Q6 and Q13 with an LRU cache of 1 GiB records the
region's blocks when each scan starts and ends and when the timed run begins
(``_begin_timed_gc_guard``, which the timer follows): none during the scan,
the whole region from the timer on, none after the last query.  Every result
matches ``tests/verification/SF1_reference.out``.
"""
from __future__ import annotations

import pathlib

import torch

_REPO = pathlib.Path(__file__).resolve().parents[2]
QIDS = [6, 13]
REGION_GB = 1


def test_the_scan_runs_without_the_region_and_the_timer_with_it(
        sf1_parquet, monkeypatch, capsys):
    import TQPlib.tqpmemory as cmp
    import main
    from IO.cache.config import BufferStrategy, set_global_buffer_strategy
    from IO.transfer.data_exchange import set_exchange_to_naive
    from adaptive.instage_optimization import set_adaptive_compression
    from tests.sigfig_compare import (SIGFIGS_THRESHOLD, compare_results,
                                      results_of)
    from utility.logger import set_message_logger

    seen = []
    real_scan, real_guard = main.scan_to_ram, main._begin_timed_gc_guard

    def scan(*args, **kwargs):
        seen.append(("scan", cmp.region_blocks()))
        out = real_scan(*args, **kwargs)
        seen.append(("scanned", cmp.region_blocks()))
        return out

    def guard():
        seen.append(("timer", cmp.region_blocks()))
        return real_guard()

    monkeypatch.setattr(main, "scan_to_ram", scan)
    monkeypatch.setattr(main, "_begin_timed_gc_guard", guard)

    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.LRU)
    set_exchange_to_naive()
    set_adaptive_compression(True)
    try:
        main.exec(1, QIDS, None, None, True, None, slot_pool_gb=REGION_GB)
    finally:
        set_adaptive_compression(False)
        set_global_buffer_strategy(BufferStrategy.CPU)
        set_message_logger(None, False)
    torch.cuda.synchronize()

    out = capsys.readouterr().out
    failed = [ln for ln in out.splitlines() if "[QFAIL]" in ln]
    assert not failed, "query failures:\n" + "\n".join(failed)
    region = (REGION_GB << 30) // cmp.block_size()
    assert seen == [("scan", 0), ("scanned", 0), ("timer", region)] * len(QIDS)
    assert cmp.region_blocks() == 0
    # The region size is shown once, when it first changes.
    resize = [ln for ln in out.splitlines() if ln.startswith("[resize]")]
    assert len(resize) == 1 and resize[0].startswith(
        f"[resize] Q{QIDS[0]} cache region -> {REGION_GB} GB"), resize

    res = (_REPO / "results.out").read_text(encoding="utf-8")
    ref = (_REPO / "tests" / "verification" / "SF1_reference.out").read_text(
        encoding="utf-8")
    compare_results(res, results_of(ref, QIDS), SIGFIGS_THRESHOLD)
