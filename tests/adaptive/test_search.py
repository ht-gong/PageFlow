"""The search math and the streaming per-chunk driver.

CPU-only.  The simulator below plays the pipeline's role faithfully: a pick
vector returned at sample *t* becomes the active regime at tick *t+2* (the
t+2 rule), and each tick's ``f`` is generated from the regime that actually
built it — so the driver's attribution log can be checked against ground
truth.  Views are real :class:`ResourceMonitor` instances.
"""
import pytest

from adaptive.compression_controller import CompressionController, RAW
from adaptive.compression_search import (
    CompressionSearchDriver,
    argmin_unimodal,
    max_evaluations,
)
from adaptive.decision_menu import DecisionMenu, build_input_choices
from adaptive.resource_monitor import ComputeTimes, ResourceMonitor, TransferTimes


def mk_view(tick, f=0.0, prologue=False, transfer=0.0):
    """A monitor last-tick view: tick index + f carried on the compute leg
    (so the bottleneck pre-gate sees compute-dominated ticks and searches);
    the prologue tick is delivered with ``last_compute is None``."""
    mon = ResourceMonitor()
    mon.last_tick = tick
    mon.last_transfer = TransferTimes(transfer, 0.0, 0.0)
    mon.last_compute = None if prologue else ComputeTimes(f)
    return mon


def make_controller(n_read=6, n_singleton=2):
    """A menu with ``n_read`` decision columns of strictly descending
    benefit (deterministic ranking 0, 1, ...) plus operator-output-fed
    singleton columns (one compressed, rest RAW)."""
    cols = []
    for i in range(n_read):
        raw_b = 1000 - i
        cols.append(build_input_choices(
            {RAW: raw_b, "BITCOMP": raw_b // 2}, raw_b))
    for j in range(n_singleton):
        cols.append(build_input_choices(
            {"ANS": 10} if j == 0 else {RAW: 10}, 20))
    menu = DecisionMenu(pipe_name="synth pipe", inputs=tuple(cols), outputs=())
    return CompressionController(menu)


def run_sim(driver, controller, f_true, total_ticks, noise_fn=None):
    """Faithful pipeline: t+2 actuation, per-regime f generation.
    Returns {tick: k_that_built_it}."""
    decision = controller.decision_indices()
    active_k, pending = controller.decision_count(), None
    truth = {}
    for t in range(total_ticks):
        if pending is not None and t >= pending[0]:
            active_k, pending = pending[1], None
        truth[t] = active_k
        f = 0.0 if t == 0 else (
            f_true(active_k) + (noise_fn(t) if noise_fn else 0.0))
        ret = driver.on_view(mk_view(t, f=f, prologue=(t == 0)))
        if ret is not None:
            k_ret = sum(1 for i in decision if ret[i] != RAW)
            pending = (t + 2, k_ret)
    return truth


# --------------------------------------------------------------------------- #
# Search math: clean, noisy, few-chunk
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 6, 10, 17, 30])
def test_pure_ternary_exact_on_every_vshape(n):
    import math
    # Absolute O(log N) cap, independent of the module's own budget math
    # (max_evaluations mirrors the search's split, so it would drift WITH a
    # regression; this bound would not): the interval shrinks to ~2/3 per
    # iteration at 2 evals each, + the final <=3-point scan.
    absolute_cap = 2 * math.ceil(math.log(max(n, 2)) / math.log(1.5)) + 5
    for m in range(n + 1):
        k, evals = argmin_unimodal(lambda k: abs(k - m), n)
        assert k == m
        assert evals <= max_evaluations(n)
        assert evals <= absolute_cap, (n, m, evals)


def test_pure_ternary_plateau_returns_valid_k():
    for n in (0, 1, 4, 9):
        k, _ = argmin_unimodal(lambda _k: 0.0, n)
        assert 0 <= k <= n


def test_driver_clean_unimodal_converges_within_budget():
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    assert not drv.inert
    run_sim(drv, ctrl, lambda k: abs(k - 3) + 1.0, 200)
    assert drv.converged
    assert drv.k_star == 3
    assert drv.probes_spent <= drv._budget_ticks
    # Converged state holds: the last emission is the k* vector.
    assert drv.emitted[-1][1] == 3


def test_driver_noisy_unimodal_within_one():
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=400, probes_per_k=3)
    run_sim(drv, ctrl, lambda k: 4.0 * (k - 3) ** 2 + 10.0, 400,
            noise_fn=lambda t: 1.0 if t % 2 else -1.0)
    assert drv.converged
    assert abs(drv.k_star - 3) <= 1


def test_driver_noise_comparable_to_curvature_needs_averaging():
    # Noise amplitude (±0.9) is comparable to the adjacent-k separation
    # (|k-3|+1 has Δf = 1), so single-sample probes can genuinely flip
    # comparisons — this exercises the probe-averaging mechanism itself,
    # unlike the weak-noise case above.  Seeded => deterministic.
    import random
    rng = random.Random(7)
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=800, probes_per_k=5,
                                  max_probe_frac=0.25)
    assert not drv.inert
    run_sim(drv, ctrl, lambda k: abs(k - 3) + 1.0, 800,
            noise_fn=lambda t: rng.uniform(-0.9, 0.9))
    assert drv.converged
    assert abs(drv.k_star - 3) <= 1


def test_driver_few_chunk_pipeline_keeps_policy_default():
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=10, probes_per_k=2)
    assert drv.inert
    for t in range(10):
        assert drv.on_view(mk_view(t, f=5.0, prologue=(t == 0))) is None
    assert drv.emitted == []
    assert drv.k_star is None


def test_driver_probe_cap_fraction():
    # Enough ticks in absolute terms, but the budget exceeds the 25% cap.
    ctrl = make_controller(n_read=6)
    budget = CompressionSearchDriver(ctrl, total_ticks=10 ** 6)._budget_ticks
    drv = CompressionSearchDriver(ctrl, total_ticks=budget * 2,
                                  max_probe_frac=0.25)
    assert drv.inert


def test_pregate_transfer_bound_skips_search_entirely():
    # Bottleneck pre-gate (transfer is monotone in k): transfer >= compute
    # at the incumbent k=N regime proves k*=N — zero probes, zero emissions.
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    for t in range(30):
        ret = drv.on_view(mk_view(t, f=2.0, prologue=(t == 0), transfer=5.0))
        assert ret is None
    assert drv.converged and drv.pregate_skipped
    assert drv.k_star == ctrl.decision_count()
    assert drv.emitted == [] and drv.attribution == []


def test_pregate_tie_also_skips():
    # transfer == compute: f(N) = transfer(N) and transfer(k) >= transfer(N)
    # for all k < N, so no k improves — the tie skips too.
    ctrl = make_controller(n_read=4)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    for t in range(10):
        drv.on_view(mk_view(t, f=3.0, prologue=(t == 0), transfer=3.0))
    assert drv.converged and drv.pregate_skipped and drv.k_star == 4


def test_pregate_compute_bound_proceeds_to_search():
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    run_sim(drv, ctrl, lambda k: abs(k - 3) + 1.0, 200)
    assert drv.converged and not drv.pregate_skipped
    assert drv.k_star == 3


def test_unknown_geometry_menu_is_inert():
    # total_ticks=0 — a menu built outside a pipeline (DecisionMenu's
    # default geometry) — must keep the driver inert, never searching.
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=0)
    assert drv.inert


def test_budget_covers_actual_convergence_ticks():
    # The declared budget (the inert gate's input) must genuinely cover a
    # full clean search: prologue skip, per-probe actuation lag, samples,
    # and the final converged actuation all land within _budget_ticks.
    for n in (1, 2, 3, 6, 10, 17):
        for p in (1, 2, 3):
            ctrl = make_controller(n_read=n)
            drv = CompressionSearchDriver(ctrl, total_ticks=10 ** 6,
                                          probes_per_k=p)
            run_sim(drv, ctrl, lambda k: abs(k - n // 2) + 1.0,
                    4 * drv._budget_ticks)
            assert drv.converged, (n, p)
            last = drv.emitted[-1][0] + drv._lag
            assert last <= drv._budget_ticks, (n, p, last, drv._budget_ticks)


def test_epilogue_tail_never_sampled_nor_emitted():
    # The pipeline's two-tick epilogue produces drain-artifact samples; the
    # driver must neither collect nor emit there (ticks > total-3).
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    run_sim(drv, ctrl, lambda k: abs(k - 3) + 1.0, 200)
    assert drv.converged and drv.k_star == 3
    assert all(t <= 197 for t, _ in drv.attribution)
    assert all(t <= 197 for t, _, _ in drv.emitted)


def test_truncation_edge_mid_probe_holds_regime():
    # A driver still probing when the valid window closes: tail ticks are
    # ignored (no collect, no emission), it never converges, and the
    # pipeline simply keeps the last-applied vector (the truncation edge —
    # picks never change results, so correctness is not the search's job).
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    drv._last_valid_tick = 8            # force a tiny valid window
    run_sim(drv, ctrl, lambda k: abs(k - 3) + 1.0, 200)
    assert not drv.converged
    assert all(t <= 8 for t, _, _ in drv.emitted)
    assert all(t <= 8 for t, _ in drv.attribution)


# --------------------------------------------------------------------------- #
# t+2 attribution: every sample credited to the vector that built it
# --------------------------------------------------------------------------- #

def test_attribution_matches_ground_truth():
    ctrl = make_controller(n_read=6)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=2)
    truth = run_sim(drv, ctrl, lambda k: abs(k - 2) + 1.0, 200)
    assert drv.converged
    assert len(drv.attribution) > 0
    for tick, k in drv.attribution:
        assert k == truth[tick], f"tick {tick}: attributed {k}, built by {truth[tick]}"


def test_attribution_no_offbyone_around_actuation():
    ctrl = make_controller(n_read=4)
    drv = CompressionSearchDriver(ctrl, total_ticks=200, probes_per_k=1)
    run_sim(drv, ctrl, lambda k: abs(k - 1) + 1.0, 200)
    assert drv.converged
    attributed = dict(drv.attribution)
    att_ticks = sorted(attributed)
    for i, (emit_tick, k, _vec) in enumerate(drv.emitted):
        # The actuation-lag tick is NEVER collected — the driver must wait
        # out the t+2 window (a collect there would credit the old regime's
        # bytes to the new k).
        assert emit_tick + 1 not in attributed, (i, emit_tick)
        later = [t for t in att_ticks if t > emit_tick]
        if i < len(drv.emitted) - 1:
            # Non-final probe: collection resumes at >= emit_tick + 2 and
            # the first collected sample is credited to the emitted k.
            assert later, (i, emit_tick)
            assert later[0] >= emit_tick + 2, (i, emit_tick, later[0])
            assert attributed[later[0]] == k, (i, emit_tick, later[0])
        else:
            # Final (converged) emission: nothing is collected afterwards —
            # k* holds with no further probing.
            assert not later, (emit_tick, later[:3])


# --------------------------------------------------------------------------- #
# Input replies: emissions are full input vectors whose deltas touch
# decision columns only; singleton entries are fixed
# --------------------------------------------------------------------------- #

def test_emitted_vectors_touch_decision_columns_only():
    ctrl = make_controller(n_read=5, n_singleton=2)
    n_cols = len(ctrl._inputs)
    decision = set(ctrl.decision_indices())
    drv = CompressionSearchDriver(ctrl, total_ticks=300, probes_per_k=2)
    run_sim(drv, ctrl, lambda k: abs(k - 2) + 1.0, 300)
    assert drv.converged
    assert len(drv.emitted) >= 2
    for _tick, _k, vec in drv.emitted:
        assert len(vec) == n_cols          # full-length, input side only
        for i, pick in enumerate(vec):
            if i not in decision:
                # Singletons carry their only entry — the compressed
                # operator-output singleton stays at its codec, never RAW'd.
                assert pick == ctrl._inputs[i].choices[0]
    for (_, _, prev), (_, _, cur) in zip(drv.emitted, drv.emitted[1:]):
        delta = {i for i in range(n_cols) if cur[i] != prev[i]}
        assert delta <= decision


def test_min_searchable_ticks_matches_gate():
    from adaptive.compression_search import min_searchable_ticks
    # Defaults: N=1 budget = 2 + 2*3 + 2 = 10; sample cap 4/0.25 = 16.
    assert min_searchable_ticks() == 16
    # Any pipe at/above the floor with N>=1 must be judgeable; below it,
    # every driver is inert regardless of the menu.
    ctrl = make_controller(n_read=1)
    assert not CompressionSearchDriver(ctrl, total_ticks=16).inert
    assert CompressionSearchDriver(ctrl, total_ticks=15).inert
