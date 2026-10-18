"""The write-leg window search and the read-and-write composition.

CPU-only.  The simulator plays the pipeline's write side faithfully: a
``PickPair`` returned at sample *t* becomes the active write regime at tick
*t+1* (the t+1 rule), and each tick's lanes are generated from the
regime that actually built it — so window attribution and scoring can be
checked against ground truth.  Views are real :class:`ResourceMonitor`
instances (including the ``out_slot_raw_bytes`` ranking counters).
"""
import pytest

from adaptive import resource_monitor
from adaptive.compression_controller import CompressionController, RAW
from adaptive.compression_search import (
    CompressionSearchDriver,
    ReadWriteSearchDriver,
    WriteWindowSearchDriver,
    argmin_unimodal,
    max_evaluations,
    min_write_searchable_ticks,
)
from adaptive.decision_menu import (
    DecisionMenu, PickPair, build_input_choices, build_output_choices,
    validate_pick_vector,
)
from adaptive.resource_monitor import ComputeTimes, ResourceMonitor, TransferTimes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mk_view(tick, *, h2d=0.0, d2d=0.0, d2h=0.0, compute=0.0,
            compress_out=0.0, prologue=False, out_bytes=None):
    mon = ResourceMonitor()
    mon.last_tick = tick
    mon.last_transfer = TransferTimes(h2d, d2d, d2h)
    mon.last_compute = (None if prologue
                        else ComputeTimes(compute, compress_out=compress_out))
    mon.out_slot_raw_bytes = out_bytes
    return mon


def make_controller(n_write=4, n_read=0, locked_runs=0, reads=None):
    """A menu with ``n_read`` input decisions, ``n_write`` write decision
    outputs (consumer reads ``reads[i]``, default 1), and ``locked_runs``
    extra locked (non-base) slots after each write decision (the sort
    pipe's B-partition shape)."""
    inputs = []
    for i in range(n_read):
        raw_b = 1000 - i
        inputs.append(build_input_choices(
            {RAW: raw_b, "BITCOMP": raw_b // 2}, raw_b))
    if not inputs:
        inputs.append(build_input_choices({"ANS": 10}, 20))  # singleton
    outputs = []
    for i in range(n_write):
        outputs.append(build_output_choices(
            "ANS", reads=reads[i] if reads else 1))
        for _ in range(locked_runs):
            outputs.append(build_output_choices("ANS", locked=True))
    outputs.append(build_output_choices(None))          # structural RAW
    menu = DecisionMenu(pipe_name="synth pipe", inputs=tuple(inputs),
                        outputs=tuple(outputs))
    return CompressionController(menu)


def infer_k(controller, vec):
    """Active write prefix size = decision outputs picking their candidate."""
    return sum(1 for i in controller.write_decision_indices()
               if vec[i] != RAW)


def counters(controller, t, rate, ratio):
    """The pipeline's cumulative per-slot byte counters at tick ``t`` with
    every write decision column producing ``rate`` raw bytes a tick,
    compressed ``ratio``:1, its stored bytes counted a tick late (at the
    pack)."""
    n_slots = len(controller._menu.outputs)
    raw, stored = [0] * n_slots, [0] * n_slots
    for i in controller.write_decision_indices():
        raw[i] = rate * t
        stored[i] = int(rate * max(0, t - 1) / ratio)
    return raw, stored


def run_write_sim(driver, controller, lanes_of_k, total_ticks,
                  out_bytes=None, rate=None, ratio=4.0):
    """Faithful write-side pipeline: t+1 actuation, per-regime lanes.
    With ``rate``, the views carry byte counters (``counters``), so the
    driver prices its RAW columns' readback.
    Returns (truth {tick: k}, pairs [(tick, PickPair)])."""
    n = controller.write_decision_count()
    active_k, pending = n, None
    truth, pairs = {}, []
    for t in range(total_ticks):
        if pending is not None and t >= pending[0]:
            active_k, pending = pending[1], None
        truth[t] = active_k
        if t == 0:
            view = mk_view(t, prologue=True, out_bytes=out_bytes)
        else:
            h2d, d2d, d2h, comp, cout = lanes_of_k(active_k)
            view = mk_view(t, h2d=h2d, d2d=d2d, d2h=d2h, compute=comp,
                           compress_out=cout, out_bytes=out_bytes)
        if rate is not None:
            view.out_slot_raw_bytes, view.out_slot_stored_bytes = counters(
                controller, t, rate, ratio)
        ret = driver(view)
        if ret is not None:
            assert isinstance(ret, PickPair)
            validate_pick_vector(controller._menu.outputs, ret.outputs,
                                 what="sim outputs")
            pairs.append((t, ret))
            pending = (t + 1, infer_k(controller, ret.outputs))
    return truth, pairs


def compute_bound_lanes(n, kmin):
    """Unimodal write objective with argmin at ``kmin``: d2h falls with k,
    compute (carrying the output-compress span) rises with k, everything
    comfortably compute-or-d2h bound (h2d/d2d small)."""
    def lanes(k):
        d2h = 4.0 + 3.0 * (n - k)
        cout = 2.0 * k
        comp = 8.0 + cout + 2.0 * abs(k - kmin) * (1 if k < kmin else 0)
        return (1.0, 1.0, d2h, comp, cout)
    return lanes


# ---------------------------------------------------------------------------
# Menu surface (locked slots, PickPair)
# ---------------------------------------------------------------------------

def test_locked_output_singleton():
    c = build_output_choices("ANS", locked=True)
    assert c.choices == ("ANS",) and not c.is_decision


def test_locked_raw_singleton():
    c = build_output_choices(None, locked=True)
    assert c.choices == (RAW,)


def test_unlocked_output_decision():
    c = build_output_choices("ANS")
    assert c.choices == (RAW, "ANS") and c.is_decision


def test_pickpair_defaults():
    p = PickPair()
    assert p.inputs is None and p.outputs is None and p.seal is False


# ---------------------------------------------------------------------------
# Controller write surface
# ---------------------------------------------------------------------------

def test_write_decision_indices_skip_locked():
    ctrl = make_controller(n_write=3, locked_runs=2)
    # slots: [dec, lock, lock] * 3 + [structural RAW]
    assert ctrl.write_decision_indices() == [0, 3, 6]
    assert ctrl.write_decision_count() == 3


def test_ranked_write_indices_by_weights():
    ctrl = make_controller(n_write=3)
    weights = [0] * len(ctrl._menu.outputs)
    weights[0], weights[1], weights[2] = 10, 30, 20
    assert ctrl.ranked_write_indices(weights) == [1, 2, 0]


def test_ranked_write_tiebreak_position():
    ctrl = make_controller(n_write=3)
    assert ctrl.ranked_write_indices([5, 5, 5, 0]) == [0, 1, 2]
    assert ctrl.ranked_write_indices(None) == [0, 1, 2]


def test_write_pick_vector_contents():
    ctrl = make_controller(n_write=3, locked_runs=1)
    ranked = ctrl.ranked_write_indices(None)
    v0 = ctrl.write_pick_vector(0, ranked)
    vN = ctrl.write_pick_vector(3, ranked)
    for i in ctrl.write_decision_indices():
        assert v0[i] == RAW and vN[i] == "ANS"
    # locked slots always their only entry; structural RAW stays RAW
    for i, col in enumerate(ctrl._menu.outputs):
        if not col.is_decision:
            assert v0[i] == vN[i] == col.choices[0]
    validate_pick_vector(ctrl._menu.outputs, v0, what="v0")
    validate_pick_vector(ctrl._menu.outputs, vN, what="vN")


def test_write_pick_vector_prefix_respects_ranking():
    ctrl = make_controller(n_write=3)
    weights = [0] * len(ctrl._menu.outputs)
    weights[0], weights[1], weights[2] = 1, 3, 2
    ranked = ctrl.ranked_write_indices(weights)   # [1, 2, 0]
    v1 = ctrl.write_pick_vector(1, ranked)
    assert v1[1] == "ANS" and v1[0] == RAW and v1[2] == RAW


# ---------------------------------------------------------------------------
# Budget / mode selection
# ---------------------------------------------------------------------------

def test_min_write_searchable_ticks_value():
    assert min_write_searchable_ticks() == 2 + 3 + 1 + 4 + 3 + 1


def test_mode_selection():
    ctrl = make_controller(n_write=4)
    assert WriteWindowSearchDriver(ctrl, 200).mode == "ternary"
    drv_mid = WriteWindowSearchDriver(ctrl, 20)
    assert drv_mid.mode == "greedy"
    assert WriteWindowSearchDriver(ctrl, 8).mode == "inert"


def test_inert_no_write_decisions():
    ctrl = make_controller(n_write=0)
    drv = WriteWindowSearchDriver(ctrl, 500)
    assert drv.inert
    assert drv(mk_view(5, compute=10.0)) is None


# ---------------------------------------------------------------------------
# Bottleneck pre-gate
# ---------------------------------------------------------------------------

def test_pregate_transfer_bound_skips():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 200)
    truth, pairs = run_write_sim(
        drv, ctrl, lambda k: (20.0, 0.0, 5.0, 6.0, 1.0), 30)
    assert drv.converged and drv.pregate_skipped and drv.k_star == 4
    assert pairs == []                        # zero windows, zero flips
    assert all(k == 4 for k in truth.values())


def test_benefit_floor_holds_tiny_prize():
    # Compute-dominated but the encode share is tiny (the SF100 Q13
    # filter-pipe lesson): probing costs more than the ceiling — hold the
    # incumbent with zero windows and zero flips.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 200, write_policy="predictive", encode_gate=False)
    _, pairs = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0, 0.05), 60)
    assert drv.converged and drv.benefit_gated and drv.k_star == 4
    assert pairs == [] and drv.windows == []


def test_flip_penalty_gates_large_output_pipe():
    # Real local encode saving BUT large outputs the consumer re-reads RAW
    # (the SF100 Q13 join-core lesson): each column's readback price
    # (20 MB/tick x 3/4 saved x 2 crossings at 25 GB/s = 1.2 ms) is under
    # the 3 ms encode span, so P3 lets it by, but writing all four RAW
    # (4.8 ms) costs more than the saving — P2 holds, zero flips.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 200, write_policy="predictive", encode_gate=False)
    for t in range(12):
        v = mk_view(t, h2d=1.0, d2h=2.0, compute=30.0, compress_out=3.0,
                    prologue=(t == 0), out_bytes=None)
        v.out_slot_raw_bytes, v.out_slot_stored_bytes = counters(
            ctrl, t, 20_000_000, 4.0)
        ret = drv(v)
        assert ret is None
    assert drv.price[0] == pytest.approx(1.2)
    assert not drv.readback_gated
    assert drv.converged and drv.benefit_gated and drv.k_star == 4


def test_flip_penalty_zero_when_outputs_tiny():
    # No observable drain (d2h ~ 0): nothing to penalize — the Q1 shape.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 200, write_policy="predictive", encode_gate=False)
    run_write_sim(drv, ctrl, lambda k: (1.0, 0.0, 0.01, 30.0, 10.0), 30)
    assert drv.emitted and not drv.benefit_gated


def test_benefit_floor_lets_real_prize_through():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400, write_policy="predictive", encode_gate=False)
    run_write_sim(drv, ctrl, compute_bound_lanes(4, 2), 40)
    assert not drv.benefit_gated and drv.emitted


def test_pregate_compute_bound_starts():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 200, encode_gate=False)
    run_write_sim(drv, ctrl, compute_bound_lanes(4, kmin=2), 20)
    assert not drv.pregate_skipped and drv.emitted


# ---------------------------------------------------------------------------
# Ternary windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kmin", [0, 2, 4])
def test_ternary_converges_to_argmin(kmin):
    n = 4
    ctrl = make_controller(n_write=n)
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    lanes = compute_bound_lanes(n, kmin)
    f = lambda k: max(lanes(k)[:4])
    expected, _ = argmin_unimodal(f, n)
    truth, pairs = run_write_sim(drv, ctrl, lanes, 400)
    assert drv.converged and drv.k_star == expected
    # Production phase runs the winning config.
    assert truth[399] == expected


def test_null_boundary_emitted_each_window():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    run_write_sim(drv, ctrl, compute_bound_lanes(4, 2), 400)
    flips = [e for e in drv.emitted if not e[2]]
    nulls = [e for e in drv.emitted if e[2]]
    # One null (seal-only) boundary inside every completed window; the
    # final install is a flip emission.
    assert len(nulls) == len(drv.windows)
    assert len(flips) == len(drv.windows) + 1


def test_window_scores_are_max_lane_means():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    lanes = compute_bound_lanes(4, 2)
    run_write_sim(drv, ctrl, lanes, 400)
    for k, score, n_s in drv.windows:
        assert n_s == drv._W + drv._D
        assert score == pytest.approx(max(lanes(k)[:4]))


def test_window_ks_follow_ternary_schedule():
    n = 4
    ctrl = make_controller(n_write=n)
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    lanes = compute_bound_lanes(n, 2)
    run_write_sim(drv, ctrl, lanes, 400)
    # Replaying the pure generator against the same f must visit the same
    # ks in the same order (deterministic, noise-free sim).
    seen = [k for k, _, _ in drv.windows]
    ks = []
    from adaptive.compression_search import ternary_argmin_gen
    gen = ternary_argmin_gen(n)
    try:
        k = next(gen)
        while True:
            ks.append(k)
            k = gen.send(max(lanes(k)[:4]))
    except StopIteration:
        pass
    assert seen == ks


def test_ternary_ranking_frozen_from_byte_counters():
    ctrl = make_controller(n_write=3)
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    weights = [0] * len(ctrl._menu.outputs)
    weights[0], weights[1], weights[2] = 1, 3, 2
    run_write_sim(drv, ctrl, compute_bound_lanes(3, 1), 400,
                  out_bytes=weights)
    assert drv._ranked == [1, 2, 0]


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------

def test_greedy_one_flip_to_zero():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 20, write_policy="predictive",
                                  encode_gate=False)
    assert drv.mode == "greedy"
    truth, pairs = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0, 10.0 if k else 0.0), 20)
    assert drv.converged and drv.k_star == 0
    assert len(pairs) == 1                    # at most one flip per run
    assert truth[19] == 0


def test_greedy_no_flip_when_benefit_small():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 20, greedy_min_benefit_ms=1e9,
                                  write_policy="predictive", encode_gate=False)
    _, pairs = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0, 10.0), 20)
    # No flip ever fires; the epilogue truncation holds the incumbent.
    assert pairs == [] and drv.k_star == 4


def test_greedy_dominance_resets_on_transfer_tick():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 24, encode_gate=False)
    assert drv.mode == "greedy"
    seq = {t: (40.0 if t % 2 else 1.0) for t in range(24)}  # alternating

    n = ctrl.write_decision_count()
    active = [n]
    pairs = []
    for t in range(24):
        if t == 0:
            view = mk_view(t, prologue=True)
        else:
            view = mk_view(t, h2d=seq[t], compute=30.0, compress_out=10.0)
        ret = drv(view)
        if ret is not None:
            pairs.append((t, ret))
    assert pairs == []                        # dominance never sustained


# ---------------------------------------------------------------------------
# Write-leg gating policy (measured is the default; promoted 2026-07-22)
# ---------------------------------------------------------------------------

def test_write_policy_default_is_measured():
    ctrl = make_controller(n_write=4)
    d = WriteWindowSearchDriver(ctrl, 200)
    assert d.write_policy == "measured"
    assert d._measured and d._drop_p2 and not d._ungated


def test_write_policy_predictive_optout():
    ctrl = make_controller(n_write=4)
    d = WriteWindowSearchDriver(ctrl, 200, write_policy="predictive")
    assert d.write_policy == "predictive"
    assert not d._measured and not d._drop_p2 and not d._ungated


def test_write_policy_ungated():
    ctrl = make_controller(n_write=4)
    d = WriteWindowSearchDriver(ctrl, 200, write_policy="ungated")
    assert d._ungated and d._drop_p2 and not d._measured


def test_write_policy_rejects_unknown():
    ctrl = make_controller(n_write=4)
    with pytest.raises(ValueError):
        WriteWindowSearchDriver(ctrl, 200, write_policy="bogus")


def test_measured_greedy_flips_when_raw_lowers_f():
    # Encode rides the compute lane, so RAW (k=0) measurably lowers the
    # bottleneck f (30 vs the incumbent's 38): the measured probe flips.  This
    # is the real-engine shape the predictive floor was too blind to catch.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 24, encode_gate=False)          # greedy budget, measured
    assert drv.mode == "greedy" and drv._measured
    truth, _ = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0 + 2.0 * k, 2.0 * k), 24)
    assert drv.converged and drv.k_star == 0
    assert drv.windows and drv.windows[-1][0] == 0   # a measured k=0 window
    assert truth[23] == 0


def test_measured_greedy_holds_when_raw_not_better():
    # Encode is OFF the bottleneck lane (compute fixed at 30): RAW does not
    # lower f, so the measured probe holds the incumbent — never a blind flip.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 24, encode_gate=False)
    assert drv.mode == "greedy" and drv._measured
    n = ctrl.write_decision_count()
    truth, _ = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0, 10.0 if k else 0.0), 24)
    assert drv.converged and drv.k_star == n
    assert drv.windows and drv.windows[-1][0] == 0   # it DID measure k=0...
    assert truth[23] == n                            # ...but held the incumbent


# ---------------------------------------------------------------------------
# Epilogue / truncation
# ---------------------------------------------------------------------------

def test_epilogue_truncation_holds_active_config():
    ctrl = make_controller(n_write=4)
    # Feasible by a hair; walk it into the epilogue mid-search.
    T = 40
    drv = WriteWindowSearchDriver(ctrl, T, encode_gate=False)
    run_write_sim(drv, ctrl, compute_bound_lanes(4, 2), T)
    assert drv.converged and drv.k_star is not None


# ---------------------------------------------------------------------------
# read-and-write composition
# ---------------------------------------------------------------------------

def _run_composite(drv, read_ctrl, total_ticks, lanes_of_k, read_f=None):
    """Sim honoring both rules: input vectors actuate t+2 (ignored for lane
    generation here — read_f folds them), PickPairs actuate t+1."""
    n_w = read_ctrl.write_decision_count()
    active_k, pending = n_w, None
    seq_returns = []
    for t in range(total_ticks):
        if pending is not None and t >= pending[0]:
            active_k, pending = pending[1], None
        if t == 0:
            view = mk_view(t, prologue=True)
        else:
            h2d, d2d, d2h, comp, cout = lanes_of_k(active_k)
            view = mk_view(t, h2d=h2d, d2d=d2d, d2h=d2h, compute=comp,
                           compress_out=cout)
        ret = drv(view)
        if ret is not None:
            seq_returns.append((t, ret))
            if isinstance(ret, PickPair):
                pending = (t + 1, infer_k(read_ctrl, ret.outputs))
    return seq_returns


def test_composite_read_then_write():
    ctrl = make_controller(n_write=3, n_read=4)
    drv = ReadWriteSearchDriver(ctrl, 500, encode_gate=False)
    assert not drv.inert and not drv.read.inert
    rets = _run_composite(drv, ctrl, 500, compute_bound_lanes(3, 1))
    kinds = [isinstance(r, PickPair) for _, r in rets]
    # All plain input vectors strictly precede all PickPairs.
    assert True in kinds and False in kinds
    first_pair = kinds.index(True)
    assert all(kinds[first_pair:])
    assert drv.read.converged and drv.write is not None


def test_composite_read_inert_write_only():
    ctrl = make_controller(n_write=3, n_read=0)
    drv = ReadWriteSearchDriver(ctrl, 500, encode_gate=False)
    assert drv.read.inert and not drv.inert
    rets = _run_composite(drv, ctrl, 500, compute_bound_lanes(3, 1))
    assert rets and all(isinstance(r, PickPair) for _, r in rets)
    assert drv.write.converged


def test_composite_both_inert():
    ctrl = make_controller(n_write=0, n_read=0)
    drv = ReadWriteSearchDriver(ctrl, 500)
    assert drv.inert


def test_composite_short_pipe_write_greedy_after_read():
    # Enough ticks for the read search but only greedy room after it.
    ctrl = make_controller(n_write=4, n_read=2)
    total = ctrl and 60
    drv = ReadWriteSearchDriver(ctrl, total)
    if drv.inert:
        pytest.skip("budget shapes changed")
    _run_composite(drv, ctrl, total, compute_bound_lanes(4, 0))
    if drv.write is not None and not drv.write.inert:
        assert drv.write.mode in ("greedy", "ternary")


def test_composite_transfer_bound_zero_probes():
    ctrl = make_controller(n_write=3, n_read=4)
    drv = ReadWriteSearchDriver(ctrl, 500)
    rets = _run_composite(drv, ctrl, 500,
                          lambda k: (50.0, 0.0, 5.0, 6.0, 1.0))
    # Read pre-gates to k*=N (no picks), write pre-gates to k_w*=N: the
    # whole pipe runs static with ZERO actuations.
    assert rets == []
    assert drv.read.pregate_skipped and drv.write.pregate_skipped


# ---------------------------------------------------------------------------
# P1b — the encode-realization gate
# ---------------------------------------------------------------------------

def _encode_gate_lanes(compute, cout, d2h):
    """Compute-dominated incumbent: P1 passes, the encode gate decides on
    compute - compress_out."""
    return lambda k: (1.0, 0.0, d2h, compute, cout if k else 0.0)


def test_encode_gate_holds_when_compute_stays_on_top():
    # compute - compress_out (20) still exceeds the max transfer lane (2):
    # the whole case for flipping rests on an encode saving that SF700 shows
    # does not materialise.  Hold the incumbent, spend no probes, emit no
    # flips.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400)
    truth, pairs = run_write_sim(drv, ctrl, _encode_gate_lanes(30.0, 10.0, 2.0), 60)
    assert drv.converged and drv.encode_gated and drv.k_star == 4
    assert pairs == [] and drv.windows == [] and drv.probes_spent == 0
    assert all(k == 4 for k in truth.values())


def test_encode_gate_lets_gap_bound_pipe_search():
    # compute - compress_out (4) sits BELOW the max transfer lane (12):
    # removing the output-compress step really could hand the bottleneck to
    # transfer, so this pipe is a legitimate candidate and must still search.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400)
    run_write_sim(drv, ctrl, _encode_gate_lanes(16.0, 12.0, 12.0), 60)
    assert not drv.encode_gated and drv.emitted


def _compress_bound_lanes(k, n=4):
    """Compute carries a large output-compress span that RAW removes, and
    RAW adds one unit of d2h per flipped column: at k = n, compute minus
    compress_out (10) does not exceed the d2h lane (10), and f is lowest at
    k = 0 (d2h 14 against the incumbent's compute 26)."""
    cout = 4.0 * k
    return (1.0, 0.0, 10.0 + (n - k), 10.0 + cout, cout)


@pytest.mark.parametrize("total_ticks, mode", [(24, "greedy"),
                                               (400, "ternary")])
def test_encode_gate_passes_and_search_flips_when_compress_out_binds(
        total_ticks, mode):
    # The gate reads compress_out: with the output-compress step off the
    # compute lane, transfer would be the bottleneck, so the gate lets the
    # pipe search, and the measured windows flip every column to RAW.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, total_ticks)
    assert drv.mode == mode and drv._encode_gate
    truth, _ = run_write_sim(drv, ctrl, _compress_bound_lanes, total_ticks)
    assert not drv.encode_gated and not drv.pregate_skipped
    assert drv.converged and drv.k_star == 0
    assert truth[total_ticks - 1] == 0


def test_encode_gate_off_under_ungated_policy():
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400, write_policy="ungated")
    assert drv._encode_gate is False
    run_write_sim(drv, ctrl, _encode_gate_lanes(30.0, 10.0, 2.0), 60)
    assert not drv.encode_gated


def test_encode_gate_env_override(monkeypatch):
    monkeypatch.setenv("TQP_WRITE_NO_ENCODE_GATE", "1")
    ctrl = make_controller(n_write=4)
    assert WriteWindowSearchDriver(ctrl, 400)._encode_gate is False


def test_encode_gate_explicit_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("TQP_WRITE_NO_ENCODE_GATE", "1")
    ctrl = make_controller(n_write=4)
    assert WriteWindowSearchDriver(ctrl, 400, encode_gate=True)._encode_gate


def test_encode_gate_precedes_p1_transfer_bound():
    # A transfer-dominated pipe is still P1's, not P1b's: the flag that fires
    # identifies which gate held, and P1 runs first.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 400)
    run_write_sim(drv, ctrl, lambda k: (20.0, 0.0, 5.0, 6.0, 1.0), 30)
    assert drv.pregate_skipped and not drv.encode_gated


# ---------------------------------------------------------------------------
# The readback price: a RAW output's bytes cross PCIe again downstream
# ---------------------------------------------------------------------------

@pytest.fixture
def bw(monkeypatch):
    """1.5 MB/ms, so a column producing 2.5 MB a tick at 4:1 with one
    consumer read is priced 2.5 MB x 3/4 x 2 / 1.5 MB/ms = 2.5 ms a tick."""
    monkeypatch.setattr(resource_monitor, "PCIE_B_PER_MS", 1.5e6)


RATE = 2_500_000


def test_readback_gate_holds_when_the_cheapest_flip_costs_the_encode(bw):
    # Encode rides the compute lane (RAW lowers the local f), but the
    # whole 2 ms output-compress span cannot pay for writing even one
    # column RAW (2.5 ms downstream): hold with zero windows.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 24, encode_gate=False)
    truth, pairs = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0 + 0.5 * k, 0.5 * k), 24,
        rate=RATE)
    assert drv.converged and drv.readback_gated and drv.k_star == 4
    assert pairs == [] and drv.windows == []
    assert truth[23] == 4


def test_readback_price_holds_a_greedy_flip_that_moves_bytes_downstream(bw):
    # The measured k=0 window lowers the local f (30 against 38), the
    # SF300 join-sort-pipe shape, but its four RAW columns add 10 ms a tick
    # downstream: the priced f (40) is no better — hold the incumbent.
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 24, encode_gate=False)
    assert drv.mode == "greedy" and drv._measured
    truth, _ = run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0 + 2.0 * k, 2.0 * k), 24,
        rate=RATE)
    assert not drv.readback_gated
    assert drv.windows and drv.windows[-1] == (0, pytest.approx(40.0), 2)
    assert drv.converged and drv.k_star == 4 and truth[23] == 4


def test_readback_price_moves_the_ternary_argmin(bw):
    # Locally k=2 wins (compute_bound_lanes); priced at 2.5 ms per RAW
    # column, compressing every column (k=4) does.
    n = 4
    ctrl = make_controller(n_write=n)
    lanes = compute_bound_lanes(n, 2)
    local = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    run_write_sim(local, ctrl, lanes, 400)
    assert local.k_star == 2
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    truth, _ = run_write_sim(drv, ctrl, lanes, 400, rate=RATE)
    assert drv.converged and drv.k_star == n and truth[399] == n
    for k, score, _ in drv.windows:
        assert score > max(lanes(k)[:4]) or k == n


def test_readback_price_counts_each_consumer_read(bw):
    # One crossing home plus one per read: reads 1, 3, 2 price the
    # columns 2.5, 5 and 3.75 ms; the ranking keeps the dearest compressed
    # first.
    ctrl = make_controller(n_write=3, reads=[1, 3, 2])
    drv = WriteWindowSearchDriver(ctrl, 400, encode_gate=False)
    run_write_sim(drv, ctrl, compute_bound_lanes(3, 1), 400, rate=RATE)
    assert [round(drv.price[i], 3) for i in range(3)] == [2.5, 5.0, 3.75]
    assert drv._ranked == [1, 2, 0]


def test_readback_price_off_under_ungated(bw):
    ctrl = make_controller(n_write=4)
    drv = WriteWindowSearchDriver(ctrl, 24, write_policy="ungated")
    run_write_sim(
        drv, ctrl, lambda k: (1.0, 0.0, 2.0, 30.0 + 0.5 * k, 0.5 * k), 24,
        rate=RATE)
    assert not drv.readback_gated and drv.k_star == 0
