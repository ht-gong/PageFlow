"""Standalone gate: the just-in-time read-leg rebuild obeys a forced
per-chunk pick schedule (GPU, SF1).

Runs Q6 three times — bare baseline, then with a **forced alternating**
observer (compress-all on even sample ticks, all-RAW on odd) attached through
the adaptive-driver factory, cache off and again under the LRU granule cache
with the rebuild's debug epoch asserts armed — and asserts:

* results are byte-identical across all runs (picks are a pure
  performance lever);
* the scan-fed pipe's **per-tick wire bytes** (the movement ledger's
  `scan_h2d` wire bytes, intercepted at its per-bind record) alternate with
  the schedule: RAW ticks move materially more wire bytes than their
  compressed neighbors (the pipeline demonstrably changes read
  representation mid-run on command);
* a zero-copy-staged pipe keeps reading its columns zero-copy (the ledger's
  `scan_uva`, same bytes as the baseline) while they carry compressed
  picks — structure preempts the pick;
* under the LRU cache with `TQP_PICK_DEBUG=1` the build-epoch asserts hold
  across the run (the rebuild lands before the pre-plan that walks the
  rebuilt tick);
* `find_rep_by_algorithm` raises on a non-materialized codec under strict
  and takes no partial action (duck-typed CCR).

Uses a small chunk size (4 MB) so SF1 pipes host tens of ticks.

Standalone (not nested under the pytest arena — same reasoning as the
per-cell subprocesses in tests/e2e):

    python tests/adaptive/pick_flip_probe.py
"""
import pathlib
import sys
from collections import defaultdict

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

QID = 6
CHUNK_B = 4 * 1024 ** 2
MIN_TICKS = 12          # only force-flip pipes long enough to show a pattern
RAW_MARGIN = 1.25       # RAW tick wire bytes vs compressed neighbor


def _strict_selector_check():
    """The strict-selector raise, exercised directly on a duck-typed CCR."""
    from adaptive.instage_optimization import find_rep_by_algorithm

    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    raw_rep = Obj(is_compressed=False)
    comp_rep = Obj(is_compressed=True, first_codec_name=lambda: "ANS",
                   leading_codec_name=lambda: "ANS")
    ccr = Obj(reps=[raw_rep, comp_rep], tag="strict-raise")

    assert find_rep_by_algorithm(ccr, "ANS", strict=True) is comp_rep
    assert find_rep_by_algorithm(ccr, "RAW", strict=True) is raw_rep
    try:
        find_rep_by_algorithm(ccr, "CASCADED", strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "strict selector returned instead of raising for a "
            "non-materialized codec")
    # Non-strict path falls back (the applier-less pipes) — no partial state
    # either way: the CCR is untouched by the raise.
    assert find_rep_by_algorithm(ccr, "CASCADED", strict=False) is raw_rep
    assert [r.is_compressed for r in ccr.reps] == [False, True]
    print("[pick_flip_probe] strict-selector raise: OK")


def main():
    import torch  # noqa: F401
    from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
    import TQPlib.tqpmemory as cmp
    alloc = CUDAPluggableAllocator(
        path_to_so_file="libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc", free_fn_name="pa_cuda_free")
    change_current_allocator(alloc)
    cmp.init(22 * 1024 ** 3, 5 * 1024 ** 2)

    import constants
    import main
    from IO.cache.config import set_global_buffer_strategy, BufferStrategy
    from IO.transfer.data_exchange import set_exchange_to_naive
    import IO.transfer.in_stage as in_stage
    from IO.telemetry import gates, movement
    from utility.logger import perf_logger, set_message_logger
    from adaptive.instage_optimization import set_adaptive_compression
    from adaptive.adaptive_driver import set_tick_observer_factory
    from adaptive.compression_controller import CompressionController

    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.CPU)
    set_exchange_to_naive()
    set_adaptive_compression(True)
    constants.set_chunk_size_b(CHUNK_B)

    _strict_selector_check()

    # ── per-tick byte spy on the movement ledger: the scan_h2d wire bytes
    # each tick's input bind adds (one bind per tick, in tick order), and
    # the UVA bytes a zero-copy stage reads, filed under the open pipe ──
    gates.EXPERIMENT = True
    per_tick = defaultdict(list)   # pipe tag -> [scan_h2d wire bytes]
    uva = defaultdict(int)         # pipe tag -> scan_uva bytes read

    def pipe_tag():
        return next(reversed(perf_logger()._start_times), None)

    def scan_wire():
        return movement.report()["kinds"]["scan_h2d"]["wire_b"]

    real_bind, real_uva = in_stage.bind_tick_sources, movement.record_uva_read

    def bind_spy(*a, **k):
        before = scan_wire()
        plan = real_bind(*a, **k)
        per_tick[pipe_tag()].append(scan_wire() - before)
        return plan

    def uva_spy(col, nbytes):
        if movement.uva_kind(col) == "scan_uva":
            uva[pipe_tag()] += nbytes
        return real_uva(col, nbytes)

    in_stage.bind_tick_sources = bind_spy
    movement.record_uva_read = uva_spy
    results = _REPO_ROOT / "results.out"

    # ── run 1: bare baseline ────────────────────────────────────────────
    main.exec(1, [QID], None, None, True, None)
    baseline_out = results.read_text(encoding="utf-8")
    baseline_ticks = {t: list(v) for t, v in per_tick.items()}
    baseline_uva = dict(uva)
    per_tick.clear()
    uva.clear()

    # ── the forced alternating factory ──────────────────────────────────
    flipped = {}     # pipe name -> controller (pipes we attached to)

    def factory(pipe_name, menu):
        ctrl = CompressionController(menu)
        if ctrl.decision_count() == 0 or menu.total_ticks < MIN_TICKS:
            return None
        n = ctrl.decision_count()
        flipped[pipe_name] = ctrl

        def observer(view):
            # Sample tick t actuates tick t+2 (same parity): even ticks run
            # compressed, odd ticks RAW, from tick 2 on.
            return ctrl.pick_vector(n if view.last_tick % 2 == 0 else 0)

        return observer

    # ── run 2: forced flips, cache off ──────────────────────────────────
    set_tick_observer_factory(factory)
    try:
        main.exec(1, [QID], None, None, True, None)
    finally:
        set_tick_observer_factory(None)
    flip_out = results.read_text(encoding="utf-8")
    flip_ticks = {t: list(v) for t, v in per_tick.items()}
    flip_uva = dict(uva)
    per_tick.clear()
    uva.clear()

    assert flip_out == baseline_out, (
        "forced pick schedule changed query results")
    assert flipped, "no pipe was long enough to force-flip"

    # The direct-staged flipped pipe's wire bytes track the schedule.
    checked_alternation = False
    for pipe, ctrl in flipped.items():
        series = flip_ticks.get(pipe)
        if not series or len(series) < MIN_TICKS:
            continue
        if flip_uva.get(pipe, 0) > 0:
            # Zero-copy-staged pipe (checked below) — picks are moot here.
            continue
        # Steady full-chunk window: skip warm ticks 0-1 (static build) and
        # the possibly-partial last real chunk (an empty epilogue tick
        # binds no page, so it records nothing).
        hi = len(series) - 2
        pairs = 0
        for t in range(2, hi - 1):
            raw_even = series[t] if t % 2 == 0 else series[t + 1]
            raw_odd = series[t + 1] if t % 2 == 0 else series[t]
            assert raw_odd > RAW_MARGIN * raw_even, (
                f"{pipe}: tick {t}/{t+1} wire bytes did not alternate with "
                f"the forced schedule (odd/RAW={raw_odd}, "
                f"even/compressed={raw_even})")
            pairs += 1
        assert pairs >= 4, f"{pipe}: too few steady pairs ({pairs})"
        # And the flipped run moved MORE wire bytes than the all-compressed
        # baseline overall (half its ticks ran RAW).
        base_total = sum(baseline_ticks[pipe])
        flip_total = sum(series)
        assert flip_total > 1.1 * base_total, (pipe, flip_total, base_total)
        checked_alternation = True
        print(f"[pick_flip_probe] {pipe}: {pairs} steady pairs "
              f"alternate; wire bytes {base_total/1e6:.1f} MB (static) -> "
              f"{flip_total/1e6:.1f} MB (alternating)")
    assert checked_alternation, (
        "no direct-staged pipe verified the alternation")

    # A zero-copy-staged pipe carried compressed picks yet kept its
    # zero-copy path (structure preempts).
    zc_pipes = [p for p in flipped if flip_uva.get(p, 0) > 0]
    assert zc_pipes, (
        "no flipped pipe exercised zero-copy under compressed picks "
        "(check mask selectivity)")
    for p in zc_pipes:
        base_zc = baseline_uva.get(p, 0)
        flip_zc = flip_uva[p]
        assert flip_zc == base_zc, (
            f"{p}: zero-copy bytes changed under forced picks "
            f"({base_zc} -> {flip_zc}) — pick was not moot")
    print(f"[pick_flip_probe] zero-copy preempts picks on: {zc_pipes}")

    # ── run 3: forced flips under the LRU cache + debug epoch asserts ───
    from IO.telemetry import gates
    gates.PICK_DEBUG = True      # TQP_PICK_DEBUG, read once at import
    set_global_buffer_strategy(BufferStrategy.LRU)
    set_tick_observer_factory(factory)
    try:
        main.exec(1, [QID], None, None, True, None)
    finally:
        set_tick_observer_factory(None)
        set_global_buffer_strategy(BufferStrategy.CPU)
        gates.PICK_DEBUG = False
    cache_out = results.read_text(encoding="utf-8")
    assert cache_out == baseline_out, (
        "forced pick schedule under LRU cache changed results")
    print("[pick_flip_probe] flips under LRU cache + epoch asserts: OK")

    in_stage.bind_tick_sources = real_bind
    movement.record_uva_read = real_uva
    print("[pick_flip_probe] PASS: read flips, strict raise, LRU flips, zero-copy")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[pick_flip_probe] FAIL: {e}")
        raise
