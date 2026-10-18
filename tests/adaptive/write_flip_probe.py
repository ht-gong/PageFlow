"""Standalone gate: sanctioned mid-run write flips (the write leg of
`--compression_search read-and-write`) on the live SF1 engine.

Forced-schedule probes — the observer IS the schedule, no search involved:

* **A pipe other than a sort**: a forced flip to all-RAW writes on a long
  aggregation pipe (Q1).  Regions before/after the commanded tick carry
  old/new codec (the flipped pipe's result mixes compressed and RAW
  records); full-suite results match the unflipped baseline (sig-fig).
* **Partition sort**: the same forced flip on a sort pipe (Q3): the
  downstream join core decodes the mixed column end to end.  Results
  identical.
* **Consumer slice spanning a flip, cache on**: the sort schedule again
  under the LRU granule cache — mixed records cross the slot and home
  pieces.  Results identical.
* **Null boundaries + spans**: an A-A schedule (seal-only
  boundaries every 4th sample, no codec change) and an A-B-A-B schedule
  (RAW <-> candidate alternation, exercising column re-registration after
  RAW parking) both leave results identical; on every observed tick the
  span the write search reads nests as it assumes: ``compress_out <=
  compute``.

    python tests/adaptive/write_flip_probe.py
"""
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FLIP_AT_SAMPLE = 5      # valid (compute-carrying) samples before the flip
MIN_TICKS = 10          # only pipes with a real mid-run


def main():
    import torch  # noqa: F401
    from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
    import TQPlib.tqpmemory as cmp
    alloc = CUDAPluggableAllocator(
        path_to_so_file="libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc", free_fn_name="pa_cuda_free")
    change_current_allocator(alloc)
    cmp.init(22 * 1024 ** 3, 5 * 1024 ** 2)

    import main
    import constants
    from IO.cache.config import set_global_buffer_strategy, BufferStrategy
    from IO.transfer.data_exchange import set_exchange_to_naive
    from utility.logger import set_message_logger
    from adaptive.adaptive_driver import set_tick_observer_factory
    from adaptive.decision_menu import PickPair, RAW
    from adaptive.instage_optimization import (
        set_adaptive_compression, set_fixed_compression,
        set_mixed_rep_acceptance)

    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.CPU)
    set_exchange_to_naive()
    set_adaptive_compression(True)
    set_fixed_compression("RAW")
    # Small chunks so SF1 pipes have enough ticks to host a mid-run flip.
    constants.set_chunk_size_b(1 * 1024 ** 2)

    results = _REPO_ROOT / "results.out"
    from tests.sigfig_compare import SIGFIGS_THRESHOLD, _compare_lines

    def run(qids):
        # The e2e ladder's 6 GiB pool: the SF1 default (1 GiB) runs out under
        # RAW outputs at 1 MiB chunks.
        main.exec(1, list(qids), None, None, True, None, pinned_pool_gb=6)
        return results.read_text(encoding="utf-8")

    def compare(a, b, what):
        la, lb = a.splitlines(), b.splitlines()
        assert len(la) == len(lb), f"{what}: line counts differ"
        for i, (x, y) in enumerate(zip(la, lb), 1):
            _compare_lines(x, y, i, fixed_sigfigs=SIGFIGS_THRESHOLD)

    # Mixed-codec consumer acceptance on for every flip run, exactly as
    # --compression_search read-and-write arms it.
    set_mixed_rep_acceptance(True)

    span_violations = []

    def make_flip_factory(role_pred, schedule, log, flips):
        """An observer factory flipping write decisions per ``schedule`` on
        pipes matching ``role_pred``.  ``schedule(n_sample, cur_is_raw)``
        returns 'flip-raw' / 'flip-cand' / 'seal' / None."""

        def factory(pipe_name, menu):
            wdec = [i for i, c in enumerate(menu.outputs) if c.is_decision]
            if (menu.total_ticks < MIN_TICKS or not wdec
                    or not role_pred(pipe_name)):
                return None
            state = {"n": 0, "raw": False}

            def vec(to_raw):
                out = []
                for i, c in enumerate(menu.outputs):
                    if c.is_decision:
                        out.append(RAW if to_raw else c.candidate)
                    else:
                        out.append(c.choices[0])
                return tuple(out)

            def observer(view):
                lc = view.last_compute
                if lc is not None:
                    # compress_out is recorded on the compute stream inside
                    # the compute span; the write search's encode gate reads
                    # compute - compress_out.
                    if lc.compress_out > lc.compute + 0.01:
                        span_violations.append(
                            (pipe_name, view.last_tick, "compress_out",
                             lc.compress_out, lc.compute))
                if lc is None or view.last_tick > menu.total_ticks - 3:
                    return None
                state["n"] += 1
                act = schedule(state["n"], state["raw"])
                if act == "flip-raw" and not state["raw"]:
                    state["raw"] = True
                    flips.append((pipe_name, view.last_tick, "->RAW", state))
                    log(f"    flip {pipe_name!r} tick {view.last_tick} "
                        f"-> RAW ({len(wdec)} write decisions, "
                        f"{menu.total_ticks} ticks)")
                    return PickPair(outputs=vec(True), seal=True)
                if act == "flip-cand" and state["raw"]:
                    state["raw"] = False
                    flips.append((pipe_name, view.last_tick, "->cand", state))
                    return PickPair(outputs=vec(False), seal=True)
                if act == "seal":
                    flips.append((pipe_name, view.last_tick, "seal", state))
                    return PickPair(outputs=vec(state["raw"]), seal=True)
                return None

            return observer

        return factory

    log = lambda s: print(s, flush=True)
    is_sort = lambda name: name.endswith("sort pipeline")
    not_sort = lambda name: not is_sort(name)

    # ── baselines (no factory) ──────────────────────────────────────────
    print("[write_flip_probe] baselines ...", flush=True)
    base_q1 = run([1])
    base_q3 = run([3])

    # ── a flip on a pipe other than a sort (Q1) ─────────────────────────
    flips = []
    set_tick_observer_factory(make_flip_factory(
        not_sort, lambda n, raw: "flip-raw" if n == FLIP_AT_SAMPLE else None,
        log, flips))
    try:
        flip_q1 = run([1])
    finally:
        set_tick_observer_factory(None)
    assert any(f[2] == "->RAW" for f in flips), (
        "no pipe hosted a write flip on Q1")
    compare(base_q1, flip_q1, "Q1 flipped vs baseline")
    print(f"[write_flip_probe] {len(flips)} flip(s), results identical",
          flush=True)

    # ── sort pipe flip (Q3) ─────────────────────────────────────────────
    flips = []
    set_tick_observer_factory(make_flip_factory(
        is_sort, lambda n, raw: "flip-raw" if n == FLIP_AT_SAMPLE else None,
        log, flips))
    try:
        flip_q3 = run([3])
    finally:
        set_tick_observer_factory(None)
    sort_flips = [f for f in flips if f[2] == "->RAW"]
    assert sort_flips, "no sort pipe hosted a write flip on Q3"
    compare(base_q3, flip_q3, "Q3 flipped vs baseline")
    print(f"[write_flip_probe] {len(sort_flips)} sort-pipe flip(s), "
          f"results identical", flush=True)

    # ── the same flip under the LRU granule cache ───────────────────────
    flips = []
    set_global_buffer_strategy(BufferStrategy.LRU)
    set_tick_observer_factory(make_flip_factory(
        is_sort, lambda n, raw: "flip-raw" if n == FLIP_AT_SAMPLE else None,
        log, flips))
    try:
        flip_q3_lru = run([3])
    finally:
        set_tick_observer_factory(None)
        set_global_buffer_strategy(BufferStrategy.CPU)
    assert any(f[2] == "->RAW" for f in flips), "no flip under LRU"
    compare(base_q3, flip_q3_lru, "Q3 flipped under LRU vs baseline")
    print("[write_flip_probe] flip under LRU cache, results identical",
          flush=True)

    # ── A-A null boundaries (seal-only, no codec change) ────────────────
    flips = []
    set_tick_observer_factory(make_flip_factory(
        lambda n: True, lambda n, raw: "seal" if n % 4 == 0 else None,
        log, flips))
    try:
        aa_q3 = run([3])
    finally:
        set_tick_observer_factory(None)
    assert any(f[2] == "seal" for f in flips), "no null boundary ran"
    compare(base_q3, aa_q3, "Q3 null boundaries vs baseline")
    print(f"[write_flip_probe] "
          f"{sum(1 for f in flips if f[2] == 'seal')} null boundaries, "
          f"results identical", flush=True)

    # ── A-B-A-B alternation (RAW <-> candidate) ─────────────────────────
    flips = []
    set_tick_observer_factory(make_flip_factory(
        lambda n: True,
        lambda n, raw: ("flip-raw" if not raw else "flip-cand")
        if n % 4 == 0 else None,
        log, flips))
    try:
        abab_q3 = run([3])
    finally:
        set_tick_observer_factory(None)
    n_ab = sum(1 for f in flips if f[2] in ("->RAW", "->cand"))
    assert n_ab >= 2, f"alternation never re-flipped ({flips})"
    compare(base_q3, abab_q3, "Q3 A-B-A-B vs baseline")
    print(f"[write_flip_probe] {n_ab} alternating flips, "
          f"results identical", flush=True)

    assert not span_violations, (
        f"compute span nesting violations: {span_violations[:5]}")
    print("[write_flip_probe] span nesting: compress_out<=compute held on "
          "every observed tick", flush=True)
    print("[write_flip_probe] PASS: Q1, sort, LRU, A-A, A-B-A-B flips")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[write_flip_probe] FAIL: {e}")
        raise
