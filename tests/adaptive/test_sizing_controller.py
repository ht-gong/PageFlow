"""Unit tier for the gather-window sizing session (CPU-only, torch-free).

Covers the SENSE→MARCH search in ``adaptive/sizing_controller.py``:

* sense phase — the stream starts at ``c_start − δ`` (downward
  first, zero OOM risk), returns to the base, then takes gated upward
  δ-probes; sensing is memory-only (throughput never judged at δ
  granularity); a fitted per-pipe envelope exists before any actuation
  above the base;
* additive march — growth is ``c += Δ``, accepted only on measured
  ≥ ε gain (knee-stop: the first sub-ε step reverts and freezes
  ``converged``); ``tail_jump`` is retired (economy gate freezes);
* increment bound — ``T_obs + r_worst·Δ ≤ budget`` gates every
  growth actuation on top of the predictive envelope-vs-grow-budget gate;
* shrink-to-safety in every state (inversion + Δ belt);
  ``line_blocked`` re-opens when the budget recovers by ≥ ``r_worst·Δ``,
  ``converged``/``breach`` do not;
* envelope breach freezes at the incumbent, the budget with the cache
  region's lendable bytes (a step over the operator space alone taken, the
  contiguity belt's run at the boundary extended by them), sample
  attribution, warm start (down-probe refresh), close() lifecycle, factory
  wiring.

The session is built directly with real :class:`TransientEnvelopeModel`
pipe + class models and a mutable fake arena-stats dict;
samples are hand-built :class:`GatherIterSample` streams.  Transients ride
the ``transient = eff_c`` line so the fitted slope is 1 (< the 2.0 seed)
and no test breaches the envelope unless it means to.
"""
import pytest

import constants
import sizing_resolver
from adaptive import adaptive_driver, sizing_controller
from adaptive.adaptive_driver import GatherIterSample
from adaptive.sizing_controller import (
    GB,
    MB,
    GatherSizingSession,
    SizingSearchConfig,
)
from adaptive.sizing_memory_model import TransientEnvelopeModel

ROW_BYTES = 8
ROW_OVERHEAD = 8
BPR = ROW_BYTES + ROW_OVERHEAD          # bytes per row (16)

SF, QID, INST, CODEC = 1, 99, "col0", "ANS"
KEY = (SF, QID, INST, CODEC)


# --------------------------------------------------------------------------- #
# Global-state hygiene (resolver layers, run store, factory)
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _logger():
    from utility.logger import set_message_logger
    set_message_logger(None, False)


@pytest.fixture(autouse=True)
def _clean_state():
    sizing_resolver.reset()
    sizing_controller.reset_run_store()
    yield
    sizing_resolver.reset()
    sizing_controller.reset_run_store()
    adaptive_driver.set_gather_sizing_factory(None)
    adaptive_driver.set_gather_sizing_enabled(True)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def huge_stats():
    return {"capacity_b": 1 << 40, "allocated_b": 0, "free_b": 1 << 40,
            "largest_free_run_b": 1 << 40, "pending_b": 0}


def budget_stats(budget_b, M_abs_b=64 * MB, lendable_b=0):
    """Stats whose operator space leaves exactly ``budget_b`` (headroom-
    limited; the largest run does not bind) and whose budget is that plus
    ``lendable_b``, what the cache region can lend."""
    return {"capacity_b": budget_b + M_abs_b, "allocated_b": 0,
            "free_b": budget_b + M_abs_b,
            "largest_free_run_b": 1 << 40, "pending_b": 0,
            "lendable_b": lendable_b}


def fragmented_stats(largest_run_b, capacity_b=64 * GB):
    """Plenty of TOTAL headroom, but scattered: the largest contiguous
    free run is the binding term (the SF700 Q7 OOM shape)."""
    return {"capacity_b": capacity_b, "allocated_b": 0,
            "free_b": capacity_b,
            "largest_free_run_b": largest_run_b, "pending_b": 0}


def mk_cfg(**kw):
    # grow_budget_frac pinned to 1.0: the rung-arithmetic tests below assert
    # exact clamp values; the hysteresis margin (default 0.9) has its own
    # dedicated test that opts in explicitly.  δ=64 MB, Δ=128 MB keep the
    # walk arithmetic small and exact.
    # march_escalation pinned to 1.0 and march_patience to 1: the march
    # tests below assert exact flat-Δ, first-rejection-knee arithmetic;
    # Δ-escalation and patience have their own dedicated tests
    # that opt in explicitly.
    base = dict(c0_b=256 * MB, c_min_b=16 * MB, c_max_b=1 * GB,
                sense_delta_b=64 * MB, march_delta_b=128 * MB,
                sense_up_probes=2, epsilon=0.02, samples_per_step=1,
                full_window_frac=0.9, M_abs_b=64 * MB, pad_min_b=16 * MB,
                seed_slope=2.0, grow_budget_frac=1.0, march_escalation=1.0,
                march_patience=1)
    base.update(kw)
    return SizingSearchConfig(**base).validate()


def mk_models(cfg):
    return (TransientEnvelopeModel(cfg.pad_min_b, cfg.march_delta_b,
                                   cfg.seed_slope),
            TransientEnvelopeModel(cfg.pad_min_b, cfg.march_delta_b,
                                   cfg.seed_slope))


def mk_session(cfg, stats, model=None, class_model=None,
               run_store=None, total_rows=10 ** 10,
               instance=INST):
    pipe, klass = mk_models(cfg)
    return GatherSizingSession(
        instance=instance, row_bytes=ROW_BYTES, row_overhead=ROW_OVERHEAD,
        total_rows=total_rows, codec_epoch=CODEC, sf=SF, query=QID,
        cfg=cfg, model=model if model is not None else pipe,
        class_model=class_model if class_model is not None else klass,
        arena_stats_fn=(stats if callable(stats) else (lambda: stats)),
        run_store=run_store if run_store is not None else {})


def mk_sample(window_b, tp, rows=None, transient_b=None, iter_tp=None):
    """One gather iteration at ``window_b``; ``tp`` encodes throughput in
    bytes/ms (gpu leg time = chunk_bytes / tp).  Default rows = the full
    window; default transient sits exactly on the transient = eff_c line."""
    if rows is None:
        rows = window_b // BPR
    chunk_bytes = rows * ROW_BYTES
    if transient_b is None:
        transient_b = rows * BPR
    return GatherIterSample(
        window_b=window_b, rows=rows, row_bytes=ROW_BYTES,
        chunk_bytes=chunk_bytes, h2d_ms=chunk_bytes / tp,
        compress_ms=0.0, d2h_ms=0.0,
        iter_peak_b=transient_b, live_at_entry_b=0,
        iter_ms=(chunk_bytes / iter_tp) if iter_tp else 0.0)


def run_sense(sess, tp=100.0, max_iters=8):
    """Feed full-window samples at the current size until the sense phase
    ends (the tp value is irrelevant: sensing is memory-only)."""
    for _ in range(max_iters):
        if sess.state != "sense":
            return
        sess.observe(mk_sample(sess.c_cur, tp))
    assert sess.state != "sense", "sense did not terminate"


def resolved(instance=INST):
    return sizing_resolver.resolve("rearrange", instance)


# --------------------------------------------------------------------------- #
# Downward-first start
# --------------------------------------------------------------------------- #

def test_stream_starts_at_base_minus_delta():
    sess = mk_session(mk_cfg(), huge_stats())
    assert sess.state == "sense"
    assert sess.c_cur == (256 - 64) * MB
    d = resolved()
    assert (d.chunk_b, d.source) == ((256 - 64) * MB, "override")


def test_start_falls_back_to_resolver_static_when_c0_none():
    sizing_resolver.set_config_static("rearrange", 128 * MB)
    sess = mk_session(mk_cfg(c0_b=None), huge_stats())
    assert sess.c_cur == (128 - 64) * MB
    assert resolved().chunk_b == (128 - 64) * MB


def test_start_clamped_into_bounds():
    # base − δ below c_min clamps to c_min.
    sess = mk_session(mk_cfg(c0_b=64 * MB, c_min_b=32 * MB), huge_stats())
    assert sess.c_cur == 32 * MB
    assert resolved().chunk_b == 32 * MB


# --------------------------------------------------------------------------- #
# The sense walk — down, base, gated upward δ-probes
# --------------------------------------------------------------------------- #

def test_sense_walk_visits_planned_sizes_then_marches():
    sess = mk_session(mk_cfg(), huge_stats())
    seen = [sess.c_cur]
    for _ in range(4):
        sess.observe(mk_sample(sess.c_cur, 100.0))
        seen.append(sess.c_cur)
    # 192 → 256 (base) → 320 → 384 (two upward δ-probes), then march.
    assert seen[:4] == [192 * MB, 256 * MB, 320 * MB, 384 * MB]
    assert sess.state == "growing"
    assert sess.incumbent_c == 384 * MB      # best sensed size
    assert sess.model.has_fit                # fitted before any march step


def test_sense_is_memory_only_ignores_throughput():
    # Wildly regressing throughput during sense must not revert/freeze:
    # no time comparison happens at δ granularity.
    sess = mk_session(mk_cfg(), huge_stats())
    for tp in (500.0, 5.0, 400.0, 3.0):
        if sess.state != "sense":
            break
        sess.observe(mk_sample(sess.c_cur, tp))
    assert sess.state == "growing"
    assert sess.incumbent_c == 384 * MB


def test_no_upward_actuation_without_pipe_fit():
    # Degenerate sense: c0 == c_min, so the down-probe coincides with the
    # base — one distinct size, no fit.  The upward δ-probes and the march
    # must both withhold growth (never a seeded-gate upward actuation).
    cfg = mk_cfg(c0_b=64 * MB, c_min_b=64 * MB)
    sess = mk_session(cfg, huge_stats())
    assert sess.c_cur == 64 * MB
    for tp in (100.0, 120.0, 150.0):
        sess.observe(mk_sample(sess.c_cur, tp))
    assert sess.c_cur == 64 * MB             # never grew
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert not sess.model.has_fit


def test_sense_upward_probe_gated_by_grow_budget():
    # Full budget 528 vouches for the seeded base return (cold gate
    # 2·256+16 = 528 ≤ 528), but the hysteresis-fractioned grow budget
    # (0.6·528 = 316) does not vouch for the fitted +δ probe
    # (gate(320) = 336 > 316) → the probe is withheld, sense ends at the
    # base, and the walk marches from there.
    cfg = mk_cfg(grow_budget_frac=0.6)
    sess = mk_session(cfg, budget_stats(528 * MB))
    sess.observe(mk_sample(sess.c_cur, 100.0))       # 192 → 256
    assert sess.c_cur == 256 * MB
    sess.observe(mk_sample(sess.c_cur, 100.0))       # probe 320 withheld
    assert sess.state == "growing"                    # sense ended early
    assert sess.incumbent_c == 256 * MB


def test_sense_base_return_gated_when_entry_shrank():
    # The entry pre-check shrinks below the base (seeded cold gate
    # 2·192+16 = 400 > budget 300 → inversion (300−16)/2 = 142): the line
    # does not vouch for the base, so the return probe is withheld too —
    # the stream never actuates a size the shrink-to-safety predicate rejects.
    cfg = mk_cfg()
    sess = mk_session(cfg, budget_stats(300 * MB))
    assert sess.c_cur == 142 * MB
    sess.observe(mk_sample(sess.c_cur, 100.0))
    assert sess.c_cur == 142 * MB                     # base return blocked
    assert sess.state != "sense"


# --------------------------------------------------------------------------- #
# Additive march — accept, knee-stop, cap freeze
# --------------------------------------------------------------------------- #

def test_march_is_additive_and_accepts_on_gain():
    sess = mk_session(mk_cfg(), huge_stats())
    run_sense(sess)                                   # incumbent 384
    sess.observe(mk_sample(384 * MB, 100.0))          # baseline → grow +Δ
    assert sess.c_cur == 512 * MB                     # 384 + 128, not ×γ
    sess.observe(mk_sample(512 * MB, 110.0))          # +10% ≥ ε → accept
    assert sess.c_cur == 640 * MB
    assert sess.incumbent_c == 512 * MB
    assert sess.state == "growing"


def test_knee_stop_reverts_and_freezes_converged():
    sess = mk_session(mk_cfg(), huge_stats())
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # baseline → 512
    sess.observe(mk_sample(512 * MB, 101.0))          # +1% < ε → knee
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.c_cur == 384 * MB == sess.incumbent_c
    assert resolved().chunk_b == 384 * MB
    epoch = sess.size_epoch
    # Converged: further samples cause no actuation (no auto re-open).
    sess.observe(mk_sample(384 * MB, 500.0))
    assert sess.state == "frozen" and sess.size_epoch == epoch


def test_march_cap_freeze():
    cfg = mk_cfg(c_max_b=512 * MB)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)                                   # incumbent 384
    sess.observe(mk_sample(384 * MB, 100.0))          # baseline → 512 (cap)
    assert sess.c_cur == 512 * MB
    sess.observe(mk_sample(512 * MB, 110.0))          # accept at cap
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.c_cur == 512 * MB


# --------------------------------------------------------------------------- #
# Δ-escalation on consecutive accepts (still increment-bound gated)
# --------------------------------------------------------------------------- #

def test_step_escalates_on_consecutive_accepts():
    cfg = mk_cfg(march_escalation=2.0, march_step_max_b=512 * MB,
                 c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)                                   # incumbent 384
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → +Δ (128)
    assert sess.c_cur == 512 * MB
    sess.observe(mk_sample(512 * MB, 110.0))          # accept → step 256
    assert sess.c_cur == 768 * MB
    sess.observe(mk_sample(768 * MB, 121.0))          # accept → step 512 (cap)
    assert sess.c_cur == 1280 * MB
    sess.observe(mk_sample(1280 * MB, 134.0))         # accept → step stays 512
    assert sess.c_cur == 1792 * MB
    assert sess._march_step_b == 512 * MB
    assert sess.state == "growing"


def test_knee_after_escalation_reverts_to_incumbent():
    cfg = mk_cfg(march_escalation=2.0, march_step_max_b=512 * MB,
                 c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 110.0))          # accept → 768 (step 256)
    sess.observe(mk_sample(768 * MB, 111.0))          # +0.9% < ε → knee
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.c_cur == 512 * MB == sess.incumbent_c


def test_economy_tail_de_escalates_to_base_delta():
    # Rows: sense consumes 192+256+320+384, anchor 384, accepted step 512
    # (all MB-of-rows).  With 700 MB-rows left the escalated step to 768
    # cannot be measured, but the base-Δ step to 640 can.
    total_rows = (192 + 256 + 320 + 384 + 384 + 512 + 700) * MB // BPR
    cfg = mk_cfg(march_escalation=2.0, march_step_max_b=512 * MB,
                 c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats(), total_rows=total_rows)
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 110.0))          # accept → step 256
    assert sess.c_cur == 640 * MB                     # fell back to +Δ
    assert sess._march_step_b == 128 * MB
    assert sess.state == "growing"


def test_clamped_growth_resets_step():
    # Budget 704 MB: the escalated proposal 768 gates seeded (beyond the
    # fit's extrapolation horizon) and fails the grow budget; the clamp
    # lands at the fit-validity cap 640 and the step de-escalates back to Δ.
    cfg = mk_cfg(march_escalation=2.0, march_step_max_b=512 * MB,
                 c_max_b=8 * GB)
    sess = mk_session(cfg, budget_stats(704 * MB))
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 110.0))          # accept → propose 768
    assert sess.c_cur == 640 * MB                     # clamped, not 768
    assert sess._march_step_b == 128 * MB
    assert sess.state == "growing"


def test_safety_shrink_resets_step():
    stats = {}
    stats.update(huge_stats())
    cfg = mk_cfg(march_escalation=2.0, march_step_max_b=512 * MB,
                 c_max_b=8 * GB)
    sess = mk_session(cfg, lambda: stats)
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 110.0))          # accept → step 256
    assert sess._march_step_b == 256 * MB
    stats.clear()
    stats.update(budget_stats(256 * MB))              # line collapses
    sess.observe(mk_sample(768 * MB, 121.0))          # shrink-to-safety fires
    assert sess.c_cur < 768 * MB
    assert sess._march_step_b == 128 * MB


def test_default_config_escalates():
    cfg = SizingSearchConfig()
    assert cfg.march_escalation == 2.0
    assert cfg.march_step_max_b == 2 * GB
    with pytest.raises(AssertionError):
        SizingSearchConfig(march_escalation=0.5).validate()


def test_load_config_march_escalation(tmp_path):
    p = tmp_path / "sizing.json"
    p.write_text('{"march": {"escalation": 1.5, "step_max_mb": 1024, '
                 '"patience": 3}}')
    cfg = sizing_controller.load_config(str(p))
    assert cfg.march_escalation == 1.5
    assert cfg.march_step_max_b == 1024 * MB
    assert cfg.march_patience == 3


def test_a_config_with_the_retired_governor_section_still_loads(tmp_path):
    p = tmp_path / "sizing.json"
    p.write_text('{"governor": {"slot_floor_slots": 2}, '
                 '"march": {"patience": 3}}')
    cfg = sizing_controller.load_config(str(p))
    assert cfg.march_patience == 3
    assert not hasattr(cfg, "slot_floor_slots")


# --------------------------------------------------------------------------- #
# Patience — tolerated sub-ε rungs, cumulative-ε acceptance
# --------------------------------------------------------------------------- #

def test_tolerated_rung_then_cumulative_accept():
    cfg = mk_cfg(march_patience=2, c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)                                   # incumbent 384
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 101.0))          # +1% < ε → tolerated
    assert sess.state == "growing"
    assert sess.c_cur == 640 * MB                     # kept marching
    assert sess.incumbent_c == 384 * MB               # anchor did not move
    sess.observe(mk_sample(640 * MB, 104.0))          # +4% vs 384 → accept
    assert sess.incumbent_c == 640 * MB
    assert sess.c_cur == 768 * MB
    assert sess.state == "growing"


def test_patience_exhausted_reverts_to_best():
    cfg = mk_cfg(march_patience=2, c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 101.0))          # tolerated (1/2)
    sess.observe(mk_sample(640 * MB, 101.5))          # sub-ε again → knee
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.c_cur == 384 * MB == sess.incumbent_c
    assert resolved().chunk_b == 384 * MB


def test_tolerated_rung_resets_escalation():
    cfg = mk_cfg(march_escalation=2.0, march_step_max_b=512 * MB,
                 march_patience=2, c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 110.0))          # accept → step 256
    sess.observe(mk_sample(768 * MB, 111.0))          # sub-ε → tolerated
    assert sess.c_cur == 896 * MB                     # 768 + base Δ, not +256
    assert sess._march_step_b == 128 * MB
    sess.observe(mk_sample(896 * MB, 113.0))          # +2.7% vs 512 → accept
    assert sess.incumbent_c == 896 * MB
    assert sess.c_cur == 1152 * MB                    # escalation resumed (256)


def test_economy_freeze_mid_sojourn_settles_to_incumbent():
    # Rows: sense 192+256+320+384, anchor 384, rung 512; then only 600
    # MB-rows remain — the next rung to 640 fails the economy gate while
    # the stream sits on the tolerated 512 → settle back to 384.
    total_rows = (192 + 256 + 320 + 384 + 384 + 512 + 600) * MB // BPR
    cfg = mk_cfg(march_patience=2, c_max_b=8 * GB)
    sess = mk_session(cfg, huge_stats(), total_rows=total_rows)
    run_sense(sess)
    sess.observe(mk_sample(384 * MB, 100.0))          # anchor → 512
    sess.observe(mk_sample(512 * MB, 101.0))          # tolerated → grow 640?
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.c_cur == 384 * MB == sess.incumbent_c


def test_default_patience_is_two():
    assert SizingSearchConfig().march_patience == 2
    with pytest.raises(AssertionError):
        SizingSearchConfig(march_patience=0).validate()


# --------------------------------------------------------------------------- #
# The increment bound + line_blocked re-open
# --------------------------------------------------------------------------- #

def _sensed_session(cfg, stats):
    """A session sensed under a loose budget, marching at incumbent 256;
    the caller then tightens ``stats`` in place to exercise the gates."""
    stats.clear()
    stats.update(huge_stats())
    sess = mk_session(cfg, lambda: stats)
    run_sense(sess)
    assert sess.state == "growing" and sess.incumbent_c == 256 * MB
    return sess


def test_increment_bound_line_blocked_and_reopens():
    cfg = mk_cfg(sense_up_probes=0, grow_budget_frac=0.9)
    stats = {}
    sess = _sensed_session(cfg, stats)
    sess.observe(mk_sample(256 * MB, 100.0))          # baseline → grow 384
    assert sess.c_cur == 384 * MB
    # Tighten into the hysteresis band: full budget 420 keeps c_cur safe
    # (gate(384)=400 ≤ 420) but the grow budget (378) vouches for nothing
    # above the incumbent → the accepted step's follow-up is line-blocked.
    stats.clear()
    stats.update(budget_stats(420 * MB))
    sess.observe(mk_sample(384 * MB, 110.0))          # accept, then block
    assert sess.state == "frozen"
    assert sess.frozen_reason == "line_blocked"
    assert sess.c_cur == 384 * MB
    # Recovery below r_worst·Δ (= 128) does NOT re-open...
    stats.clear()
    stats.update(budget_stats(500 * MB))              # +80 < 128
    sess.observe(mk_sample(384 * MB, 110.0))
    assert sess.state == "frozen"
    assert sess.frozen_reason == "line_blocked"
    # ...but a recovery ≥ r_worst·Δ over the blocked level does.
    stats.clear()
    stats.update(budget_stats(600 * MB))              # ≥ 420 + 128
    sess.observe(mk_sample(384 * MB, 110.0))          # re-open + re-anchor
    assert sess.state == "growing"
    sess.observe(mk_sample(384 * MB, 110.0))          # baseline → grow
    assert sess.c_cur == 512 * MB


# --------------------------------------------------------------------------- #
# Predictive safety — clamp path with the Δ belt, line invariant
# --------------------------------------------------------------------------- #

def test_clamp_via_inversion_and_line_invariant(monkeypatch):
    cfg = mk_cfg(sense_up_probes=0)
    stats = {}
    holder = {}
    seen = []                                        # (c, gate_at_set, budget)
    real_set = sizing_resolver.set_override

    def spy(pipe_class, instance, chunk_b):
        sess = holder.get("sess")
        if sess is not None:
            seen.append((chunk_b, sess.gate.gate_b(chunk_b)[0],
                         sess._budget(stats)))
        real_set(pipe_class, instance, chunk_b)

    monkeypatch.setattr(sizing_resolver, "set_override", spy)
    sess = _sensed_session(cfg, stats)
    holder["sess"] = sess
    stats.clear()
    stats.update(budget_stats(420 * MB))
    # Step 256→384 fits 420 outright (gate 400, increment 384) → grows.
    sess.observe(mk_sample(256 * MB, 100.0))
    assert sess.c_cur == 384 * MB
    # Next step 384→512 fails both gates (increment 512, gate 528 > 420) →
    # clamp by inversion: invert(420) = 420−16 = 404 — a sub-Δ step that
    # passes both gates (increment 384+20=404 ≤ 420).
    sess.observe(mk_sample(384 * MB, 110.0))
    assert sess.c_cur == 404 * MB
    # The invariant: no window was ever actuated with its gate above the
    # budget.
    assert seen, "override spy saw no actuations"
    for c, gate_b, budget in seen:
        assert gate_b <= budget, (
            f"actuated c={c} with gate {gate_b} > budget {budget}")


# --------------------------------------------------------------------------- #
# The cache region's lendable bytes count in the budget
# --------------------------------------------------------------------------- #

def _lending_stats(avail_b, lendable_b, M_abs_b=64 * MB):
    """``budget_stats`` with the run at the boundary its whole free space."""
    st = budget_stats(avail_b, M_abs_b, lendable_b)
    st["top_free_run_b"] = st["free_b"]
    return st


def test_the_lendable_bytes_fund_a_step():
    """512 MB is over what the operator space leaves (420) but within the
    budget with what the cache region can lend (560): the step is taken (its
    allocations borrow inside the allocator)."""
    cfg = mk_cfg(sense_up_probes=0)
    stats = {}
    sess = _sensed_session(cfg, stats)
    stats.clear()
    stats.update(_lending_stats(420 * MB, 140 * MB))
    sess.observe(mk_sample(256 * MB, 100.0))          # 384 fits outright
    assert sess.c_cur == 384 * MB
    sess.observe(mk_sample(384 * MB, 110.0))          # 512 on the lendable
    assert sess.c_cur == 512 * MB and sess.state == "growing"


def test_a_window_over_the_operator_space_is_kept_on_the_lendable_bytes():
    """The operator space shrinks under the running window (gate(384) = 400
    > 300) but the cache can lend the difference: no shrink, and the
    accept's next step is taken too."""
    cfg = mk_cfg(sense_up_probes=0)
    stats = {}
    sess = _sensed_session(cfg, stats)
    sess.observe(mk_sample(256 * MB, 100.0))          # baseline → 384
    assert sess.c_cur == 384 * MB
    stats.clear()
    stats.update(_lending_stats(300 * MB, 1 * GB))
    epoch = sess.size_epoch
    sess.observe(mk_sample(384 * MB, 120.0))          # keep, accept, step
    assert sess.c_cur == 512 * MB and sess.incumbent_c == 384 * MB
    assert sess.size_epoch == epoch + 1               # the step, no shrink


def test_the_lendable_bytes_extend_only_the_run_at_the_boundary():
    """With the contiguity belt, the lendable bytes extend the free run at
    the boundary, not the largest run elsewhere: 100 MB at the boundary plus
    4 GiB lendable vouch for the step to 1536 MB; with nothing at the
    boundary and 400 MB lendable, the 1100 MB run elsewhere binds and the
    step clamps."""
    cfg = mk_cfg(c0_b=1 * GB, c_max_b=4 * GB, march_delta_b=512 * MB,
                 sense_up_probes=0)
    stats = huge_stats()
    sess = mk_session(cfg, lambda: stats)
    run_sense(sess)
    assert sess.state == "growing" and sess.incumbent_c == 1 * GB
    stats.clear()
    stats.update({"capacity_b": 64 * GB, "allocated_b": 0,
                  "free_b": 64 * GB, "largest_free_run_b": 1100 * MB,
                  "top_free_run_b": 100 * MB, "pending_b": 0,
                  "lendable_b": 4 * GB})
    assert sess._budget(stats) == 4196 * MB
    sess.observe(mk_sample(1 * GB, 100.0))            # baseline → 1536
    assert sess.c_cur == 1536 * MB
    stats.update(top_free_run_b=0, lendable_b=400 * MB)
    assert sess._budget(stats) == 1100 * MB


def test_a_step_over_the_budget_clamps_to_it():
    """From 1 GiB, the 2 GiB step is over the budget (1244 MB the operator
    space leaves + 512 MB lendable): the step clamps to what it vouches for,
    1740 MB, and the march step falls back to Δ."""
    cfg = mk_cfg(c0_b=1 * GB, c_max_b=4 * GB, march_delta_b=1 * GB,
                 sense_up_probes=0)
    stats = huge_stats()
    sess = mk_session(cfg, lambda: stats)
    run_sense(sess)
    stats.clear()
    stats.update(_lending_stats(1244 * MB, 512 * MB))
    sess.observe(mk_sample(1 * GB, 100.0))            # baseline → step
    assert sess.c_cur == 1740 * MB
    assert sess._march_step_b == cfg.march_delta_b


# --------------------------------------------------------------------------- #
# Mid-stream shrink-to-safety under a collapsed line (Q7 shape)
# --------------------------------------------------------------------------- #

def test_shrink_from_frozen_on_budget_collapse():
    cfg = mk_cfg(sense_up_probes=0)
    stats = huge_stats()
    sess = mk_session(cfg, lambda: stats)
    run_sense(sess)
    sess.observe(mk_sample(256 * MB, 100.0))          # baseline → 384
    sess.observe(mk_sample(384 * MB, 101.0))          # knee → frozen @256
    assert sess.state == "frozen" and sess.c_cur == 256 * MB
    epoch = sess.size_epoch
    # The resident base grew: the budget collapses under the current size.
    # Fit slope 1 → gate(256) = 272 > 100; inversion: 100−16 = 84 MB.
    stats.clear()
    stats.update(budget_stats(100 * MB))
    sess.observe(mk_sample(256 * MB, 100.0))
    assert sess.c_cur == 84 * MB                      # inversion clamp
    assert resolved().chunk_b == 84 * MB
    assert sess.size_epoch == epoch + 1
    assert sess.state == "frozen"                     # frozen sessions shrink
    # A safety shrink is not a measured verdict: acceptance re-anchors.
    assert sess.incumbent_c == 84 * MB
    assert sess.incumbent_tp is None


# --------------------------------------------------------------------------- #
# Envelope breach freezes at the incumbent
# --------------------------------------------------------------------------- #

def test_breach_freezes_at_incumbent_and_blocks_store():
    cfg = mk_cfg(sense_up_probes=0)
    run_store = {}
    sess = mk_session(cfg, huge_stats(), run_store=run_store)
    run_sense(sess)
    sess.observe(mk_sample(256 * MB, 100.0))          # baseline → 384
    # Fit is slope-1: gate(384) = 400 MB.  A 500 MB transient breaches.
    sess.observe(mk_sample(384 * MB, 120.0, transient_b=500 * MB))
    assert sess.breaches == 1
    assert sess.state == "frozen"
    assert sess.frozen_reason == "breach"
    assert sess.c_cur == 256 * MB == sess.incumbent_c
    epoch = sess.size_epoch
    sess.observe(mk_sample(256 * MB, 500.0))          # no re-open, no growth
    assert sess.state == "frozen" and sess.size_epoch == epoch
    # A breached stream never persists its window.
    sess.close()
    assert run_store == {}


# --------------------------------------------------------------------------- #
# Sample attribution + the economy (remaining-stream) gate
# --------------------------------------------------------------------------- #

def test_partial_tail_samples_not_counted_for_acceptance():
    cfg = mk_cfg(sense_up_probes=0, samples_per_step=2)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)
    full_rows = (256 * MB) // BPR
    sess.observe(mk_sample(256 * MB, 100.0))                    # 1 of 2
    assert sess.c_cur == 256 * MB
    sess.observe(mk_sample(256 * MB, 100.0, rows=full_rows // 2))  # tail
    assert sess.c_cur == 256 * MB                     # not counted
    sess.observe(mk_sample(256 * MB, 100.0))                    # 2 of 2
    assert sess.c_cur == 384 * MB                     # baseline adopted, grew


def test_economy_gate_freezes_converged_no_tail_jump():
    # 24 windows of 256 MB: sense (192 + 256) spends ~1.75 windows-worth;
    # the first march step to 384 MB fits, the next to 512 MB does not.
    total_rows = 24 * ((256 * MB) // BPR)
    cfg = mk_cfg(sense_up_probes=0, samples_per_step=8)
    sess = mk_session(cfg, huge_stats(), total_rows=total_rows)
    run_sense(sess)
    for _ in range(8):
        sess.observe(mk_sample(256 * MB, 100.0))      # baseline x8 → grow
    assert sess.c_cur == 384 * MB
    for _ in range(8):
        sess.observe(mk_sample(384 * MB, 110.0))      # accept → next step?
    # Remaining rows cannot host 8 full 512 MB windows → economy freeze,
    # and NO terminal jump beyond +Δ ever happens (tail_jump retired).
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.c_cur == 384 * MB


# --------------------------------------------------------------------------- #
# Warm start (down-probe refresh) + close() lifecycle
# --------------------------------------------------------------------------- #

def test_warm_start_down_probe_then_frozen():
    run_store = {KEY: 384 * MB}
    sess = mk_session(mk_cfg(), huge_stats(), run_store=run_store)
    assert sess.state == "sense"
    assert sess.c_cur == (384 - 64) * MB              # down-probe first
    sess.observe(mk_sample(sess.c_cur, 100.0))
    assert sess.c_cur == 384 * MB                     # back at the store
    sess.observe(mk_sample(384 * MB, 100.0))
    assert sess.state == "frozen"
    assert sess.frozen_reason == "converged"
    assert sess.model.has_fit                         # fit refreshed
    sess.observe(mk_sample(384 * MB, 500.0))          # frozen: no search
    assert sess.c_cur == 384 * MB
    sess.close()
    assert run_store[KEY] == 384 * MB                 # clean stream persists


def test_close_clears_override():
    sizing_resolver.set_config_static("rearrange", 200 * MB)
    sess = mk_session(mk_cfg(), huge_stats())
    assert resolved().source == "override"
    sess.close()
    # The override is gone: resolution falls back to the static layer.
    d = resolved()
    assert (d.chunk_b, d.source) == (200 * MB, "config")


def test_close_persists_converged_window():
    run_store = {}
    cfg = mk_cfg(sense_up_probes=0)
    sess = mk_session(cfg, huge_stats(), run_store=run_store)
    run_sense(sess)
    sess.observe(mk_sample(256 * MB, 100.0))          # baseline → 384
    sess.observe(mk_sample(384 * MB, 101.0))          # knee → frozen @256
    sess.close()
    assert run_store == {KEY: 256 * MB}
    d = resolved()
    assert (d.chunk_b, d.source) == (constants.get_chunk_size_b(),
                                     "constants")


# --------------------------------------------------------------------------- #
# Factory wiring (install/uninstall, threshold, warmup exclusion)
# --------------------------------------------------------------------------- #

def test_factory_install_threshold_enable_uninstall():
    cfg = mk_cfg(c0_b=64 * MB)
    stats = huge_stats()
    thresh_rows = (64 * MB) // BPR                  # 4194304
    try:
        sizing_controller.install_gather_sizing(cfg, lambda: stats)
        assert adaptive_driver.has_gather_sizing_factory()
        # Single-iteration stream (total bytes <= c0): nothing to search.
        assert adaptive_driver.make_gather_sizing_session(
            "rearrange gather", "colA", ROW_BYTES, ROW_OVERHEAD,
            thresh_rows, CODEC) is None
        # One row over the threshold: a live session with its override.
        sess = adaptive_driver.make_gather_sizing_session(
            "rearrange gather", "colB", ROW_BYTES, ROW_OVERHEAD,
            thresh_rows + 1, CODEC)
        assert isinstance(sess, GatherSizingSession)
        assert resolved("colB").source == "override"
        # Warmup exclusion: disabled -> no sessions, predicate off.
        adaptive_driver.set_gather_sizing_enabled(False)
        assert not adaptive_driver.has_gather_sizing_factory()
        assert adaptive_driver.make_gather_sizing_session(
            "rearrange gather", "colC", ROW_BYTES, ROW_OVERHEAD,
            10 ** 8, CODEC) is None
        adaptive_driver.set_gather_sizing_enabled(True)
        assert adaptive_driver.has_gather_sizing_factory()
        # Uninstall clears the factory and any live overrides.
        sizing_controller.uninstall_gather_sizing()
        assert not adaptive_driver.has_gather_sizing_factory()
        assert adaptive_driver.make_gather_sizing_session(
            "rearrange gather", "colD", ROW_BYTES, ROW_OVERHEAD,
            10 ** 8, CODEC) is None
        assert resolved("colB").source != "override"
    finally:
        adaptive_driver.set_gather_sizing_factory(None)
        adaptive_driver.set_gather_sizing_enabled(True)
        sizing_controller.reset_run_store()
        sizing_resolver.reset()


# --------------------------------------------------------------------------- #
# Objective denominator ("legs" vs host-bracketed "iter") — march phase
# --------------------------------------------------------------------------- #

def test_objective_iter_uses_host_bracket():
    # Legs say +10% (would accept), the host bracket says +1% (must revert):
    # only the "iter" objective sees the flat true iteration cost.
    cfg = mk_cfg(sense_up_probes=0, objective="iter", epsilon=0.02)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)
    sess.observe(mk_sample(256 * MB, 100.0, iter_tp=80.0))
    assert sess.c_cur == 384 * MB                     # baseline adopted, grew
    sess.observe(mk_sample(384 * MB, 110.0, iter_tp=80.8))
    assert sess.state == "frozen"
    assert sess.c_cur == 256 * MB                     # reverted on iter tp


def test_objective_legs_ignores_bracket():
    cfg = mk_cfg(sense_up_probes=0, objective="legs", epsilon=0.02)
    sess = mk_session(cfg, huge_stats())
    run_sense(sess)
    sess.observe(mk_sample(256 * MB, 100.0, iter_tp=80.0))
    sess.observe(mk_sample(384 * MB, 110.0, iter_tp=80.8))
    assert sess.state == "growing"                    # legs +10% accepted
    assert sess.c_cur == 512 * MB


# --------------------------------------------------------------------------- #
# Every stream's first run starts at c0
# --------------------------------------------------------------------------- #

def test_sibling_stream_starts_at_c0_whatever_the_first_converged():
    store = {}
    cfg = mk_cfg(sense_up_probes=0)
    a = mk_session(cfg, huge_stats(), instance="colA", run_store=store)
    assert a.c_cur == (256 - 64) * MB                 # c0 − δ
    run_sense(a)
    a.observe(mk_sample(256 * MB, 100.0))             # baseline → 384
    a.observe(mk_sample(384 * MB, 120.0))             # accept → 512
    a.observe(mk_sample(512 * MB, 121.0))             # knee → 384
    a.close()
    assert store[(SF, QID, "colA", CODEC)] == 384 * MB
    # Same (sf, query, codec), another instance: its own first run, sensed
    # one δ below c0, not at what colA converged to.
    b = mk_session(cfg, huge_stats(), instance="colB", run_store=store)
    assert b.state == "sense"
    assert b._sense_base == 256 * MB
    assert b.c_cur == (256 - 64) * MB


# --------------------------------------------------------------------------- #
# Contiguity belt (largest_free_run gate) + grow hysteresis
# --------------------------------------------------------------------------- #

def test_contiguity_gate_bounds_growth_headroom_cannot_see():
    # After sense the fitted gate(384) = 400 MB.  Total headroom is huge;
    # the largest run (390 MB) is the binding budget → inversion clamp.
    cfg = mk_cfg(sense_up_probes=0)
    stats = {}
    sess = _sensed_session(cfg, stats)
    stats.clear()
    stats.update(fragmented_stats(390 * MB))
    sess.observe(mk_sample(256 * MB, 100.0))
    assert sess.c_cur == 374 * MB                     # invert(390) = 390−16
    assert sess.c_cur < 384 * MB
    # Same shape with the belt disarmed: headroom-only, the full Δ step.
    cfg2 = mk_cfg(sense_up_probes=0, contiguity_gate=False)
    stats2 = {}
    sess2 = _sensed_session(cfg2, stats2)
    stats2.clear()
    stats2.update(fragmented_stats(390 * MB))
    sess2.observe(mk_sample(256 * MB, 100.0))
    assert sess2.c_cur == 384 * MB


def test_contiguity_gate_missing_stat_falls_back_to_headroom():
    stats = {"capacity_b": 1 << 40, "allocated_b": 0, "free_b": 1 << 40,
             "pending_b": 0}                          # no largest_free_run_b
    cfg = mk_cfg(sense_up_probes=0)
    sess = mk_session(cfg, stats)
    run_sense(sess)
    sess.observe(mk_sample(256 * MB, 100.0))
    assert sess.c_cur == 384 * MB


def test_grow_hysteresis_no_shrink_regrow_pingpong():
    """A window clamped at the line must not oscillate: growth plans into
    grow_budget_frac of the budget, the shrink rung fires at the full
    budget, and the band between them absorbs envelope-pad breathing."""
    cfg = mk_cfg(sense_up_probes=0, grow_budget_frac=0.9)
    stats = {}
    sess = _sensed_session(cfg, stats)
    stats.clear()
    stats.update(fragmented_stats(440 * MB))
    sess.observe(mk_sample(256 * MB, 100.0))
    c1 = sess.c_cur
    # Growth planned into 0.9·440 = 396: invert(396) = 380 (fitted line).
    assert c1 == 380 * MB
    # Iterate at the clamped size with transients riding eff_c: the gate
    # stays inside the FULL budget, so no shrink fires — no ping-pong.
    for tp in (120.0, 121.0, 122.0):
        sess.observe(mk_sample(sess.c_cur, tp))
        assert sess.c_cur >= c1                       # never shrinks back


def test_run_store_is_kept_per_region_size():
    """Convergence memory is selected per key (the cache region's size): a
    key gets back what it stored, another key starts empty, and a reset
    drops every key's."""
    sc = sizing_controller
    sc.select_run_store(10)
    sc._run_store["a"] = 1
    sc._tick_store["t"] = {"x": 1}
    sc.select_run_store(0)
    assert not sc._run_store and not sc._tick_store
    sc._run_store["b"] = 2
    sc.select_run_store(10)
    assert sc._run_store == {"a": 1} and sc._tick_store == {"t": {"x": 1}}
    sc.select_run_store(0)
    assert sc._run_store == {"b": 2}
    sc.reset_run_store()
    sc.select_run_store(10)
    assert not sc._run_store and not sc._tick_store
