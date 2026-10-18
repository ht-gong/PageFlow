"""Sizing-plane memory model unit tier.

Covers the transient envelope model and its registry:

* seed phase — <2 distinct ``c`` values predicts on the analytic
  seed line, and inversion is the seed-line floor;
* fit phase — exact linear samples reproduce the line; predictions
  inside validity are fit-based and dominate every observed sample;
* envelope pad — pad = max(max positive residual, pad_min), so the
  envelope covers ALL samples, never the mean;
* slope clamp — decreasing samples clamp ``a_hat`` to 0 (constant
  prediction, never a negative slope);
* additive extrapolation limit — beyond ``max_observed_c + step_b``
  the seed is the only oracle even with a good fit (an additive horizon,
  not a γ-multiplicative one);
* inversion — largest ``c`` whose envelope prediction fits the budget;
  capped at fit validity unless the seed line vouches beyond it;
* running state — ``r_worst`` and ``max_transient_at`` are lifetime
  maxima that survive fit-window eviction;
* breach check — observed transient above its gating prediction;
* registry keying — arbitrary tuple keys (per-pipe and per-class);
  fits are never read across keys.

CPU-only, torch-free.  The model is instance-scoped (no module-level state),
so no resolver/governor/run-store resets are needed here.
"""
import pytest

from adaptive.sizing_memory_model import (
    ModelRegistry,
    Prediction,
    TransientEnvelopeModel,
    _FIT_WINDOW,
)

MB = 1024 ** 2


def mk_model(pad_min_b=5, step_b=200, seed_slope=2.0):
    return TransientEnvelopeModel(pad_min_b=pad_min_b, step_b=step_b,
                                  seed_slope=seed_slope)


# --------------------------------------------------------------------------- #
# Seed-only phase
# --------------------------------------------------------------------------- #

def test_seed_phase_predict_no_samples():
    m = mk_model(pad_min_b=1000, step_b=200, seed_slope=2.0)
    assert not m.has_fit
    p = m.predict(4096)
    assert isinstance(p, Prediction)
    assert p.seeded is True
    assert p.transient_b == int(2.0 * 4096) + 1000


def test_seed_phase_persists_below_two_distinct_c():
    # Two samples at the SAME c are still <2 distinct sizes: no fit.
    m = mk_model(pad_min_b=7, step_b=200, seed_slope=2.0)
    m.add_sample(100, 90)
    m.add_sample(100, 110)
    assert not m.has_fit
    p = m.predict(100)
    assert p.seeded is True
    assert p.transient_b == int(2.0 * 100) + 7


def test_seed_phase_invert_is_seed_line_floor():
    m = mk_model(pad_min_b=1000, step_b=200, seed_slope=2.0)
    # Exact division: (10000 - 1000) / 2 = 4500.
    assert m.invert(10000) == 4500
    # Floor, not round: (9001) / 2 -> 4500.
    assert m.invert(10001) == 4500
    # The inverted size's seed prediction fits the budget.
    assert m.predict(m.invert(10000)).transient_b <= 10000
    # Budget below pad_min: clamped to 0, never negative.
    assert m.invert(500) == 0


# --------------------------------------------------------------------------- #
# Fit phase on exact linear samples
# --------------------------------------------------------------------------- #

def _feed_exact_line(m):
    """t = 3*c + 50 at four distinct c (all arithmetic exact in float)."""
    samples = [(c, 3 * c + 50) for c in (100, 200, 300, 400)]
    for c, t in samples:
        m.add_sample(c, t)
    return samples


def test_fit_phase_predict_inside_validity():
    m = mk_model(pad_min_b=5, step_b=200)
    samples = _feed_exact_line(m)
    assert m.has_fit
    assert m.max_observed_c == 400
    # Fit-based inside validity (c <= max_observed_c + step_b = 600).
    p = m.predict(500)
    assert p.seeded is False
    # Exact line + zero residuals => pad = pad_min; prediction = line + pad.
    assert p.transient_b == 3 * 500 + 50 + 5
    # Envelope property: prediction at each observed c >= that sample's t.
    for c, t in samples:
        pc = m.predict(c)
        assert pc.seeded is False
        assert pc.transient_b >= t


def test_fit_lets_prediction_undercut_generous_seed():
    # A well-behaved pipe (slope < seed_slope) grows past the 2c seed line.
    m = mk_model(pad_min_b=5, step_b=200, seed_slope=2.0)
    for c in (100, 200, 300):
        m.add_sample(c, c // 2 + 50)     # t = 0.5*c + 50
    p = m.predict(300)
    assert p.seeded is False
    assert p.transient_b < int(2.0 * 300) + 5


# --------------------------------------------------------------------------- #
# Envelope pad over noisy samples
# --------------------------------------------------------------------------- #

def test_envelope_pad_covers_all_noisy_samples():
    # Least squares over these is exactly a=2, b=60 (symmetric c, exact
    # float arithmetic); residuals -10, +20, -10 => pad = 20 > pad_min = 5.
    m = mk_model(pad_min_b=5, step_b=300)
    noisy = [(100, 250), (200, 480), (300, 650)]
    for c, t in noisy:
        m.add_sample(c, t)
    a_fit, b_fit, max_pos_resid = 2.0, 60.0, 20
    for c in (100, 150, 200, 300, 450, 600):
        p = m.predict(c)
        assert p.seeded is False
        gap = p.transient_b - (a_fit * c + b_fit)
        assert gap >= max_pos_resid
        assert gap >= m.pad_min_b
    # The envelope covers every observed sample.
    for c, t in noisy:
        assert m.predict(c).transient_b >= t


def test_pad_never_below_pad_min():
    # Zero residuals: the pad floors at pad_min, not at 0.
    m = mk_model(pad_min_b=64, step_b=200)
    for c in (100, 200):
        m.add_sample(c, 3 * c)           # exact line through origin
    p = m.predict(200)
    assert p.seeded is False
    assert p.transient_b == 3 * 200 + 64


# --------------------------------------------------------------------------- #
# Slope clamp
# --------------------------------------------------------------------------- #

def test_slope_clamped_to_zero_on_decreasing_samples():
    m = mk_model(pad_min_b=5, step_b=200)
    decreasing = [(100, 900), (200, 600), (300, 300)]
    for c, t in decreasing:
        m.add_sample(c, t)
    # a_hat clamped to 0: prediction is constant in c, never a negative slope.
    p_lo = m.predict(100)
    p_hi = m.predict(300)
    assert p_lo.seeded is False and p_hi.seeded is False
    assert p_lo.transient_b == p_hi.transient_b
    # b = mean_y = 600; max positive residual = 300 => envelope = 900.
    assert p_hi.transient_b == 900
    # Still an envelope over every sample.
    for c, t in decreasing:
        assert m.predict(c).transient_b >= t


# --------------------------------------------------------------------------- #
# Additive one-step extrapolation limit
# --------------------------------------------------------------------------- #

def test_extrapolation_beyond_step_falls_back_to_seed():
    m = mk_model(pad_min_b=5, step_b=200, seed_slope=2.0)
    _feed_exact_line(m)                  # perfect fit, max_observed_c = 400
    assert m.has_fit
    # At the validity boundary (c == max_observed_c + step_b) fit rules.
    assert m.predict(600).seeded is False
    # One byte past it the seed is the only oracle, despite the good fit —
    # the horizon is a constant increment, not a multiple.
    p = m.predict(601)
    assert p.seeded is True
    assert p.transient_b == int(2.0 * 601) + 5


def test_validity_horizon_is_additive_not_multiplicative():
    # With samples at large c, a γ-style horizon (1.5x) would vouch far
    # beyond max_observed; the additive horizon stays one Δ out.
    m = mk_model(pad_min_b=5, step_b=100, seed_slope=2.0)
    for c in (1000, 2000):
        m.add_sample(c, 3 * c)
    assert m.predict(2100).seeded is False   # max + Δ
    assert m.predict(2101).seeded is True    # one past the additive horizon
    assert m.predict(3000).seeded is True    # γ-era 1.5x point: NOT valid


# --------------------------------------------------------------------------- #
# Inversion
# --------------------------------------------------------------------------- #

def test_invert_fit_region_is_largest_fitting_c():
    m = mk_model(pad_min_b=5, step_b=200, seed_slope=2.0)
    _feed_exact_line(m)                  # envelope = 3c + 55, validity <= 600
    budget = 1000
    inv = m.invert(budget)
    assert inv == 315                    # (1000 - 55) // 3
    assert m.predict(inv).transient_b <= budget
    # Anything meaningfully larger no longer fits.
    assert m.predict(int(inv * 1.05) + 1).transient_b > budget
    assert m.predict(inv + 1).transient_b > budget


def test_invert_capped_at_validity_when_seed_does_not_vouch():
    # Shallow fit (a = 0.5 < seed_slope): the raw fit inversion lands far
    # beyond validity, but the seed line does not vouch => cap at
    # max_observed_c + step_b (no fit extrapolation past one step).
    m = mk_model(pad_min_b=5, step_b=150, seed_slope=2.0)
    for c in (100, 200, 300):
        m.add_sample(c, c // 2 + 50)     # t = 0.5*c + 50, exact
    validity_cap = 300 + 150             # 450
    budget = 400
    # seed_c = (400 - 5) // 2 = 197 <= cap: seed does not vouch beyond it.
    inv = m.invert(budget)
    assert inv == validity_cap
    assert m.predict(inv).transient_b <= budget
    # And this really is the largest fitting c: one past the cap the seed
    # gates, and the seed prediction blows the budget.
    assert m.predict(inv + 1).transient_b > budget


def test_invert_follows_seed_beyond_validity_when_seed_vouches():
    m = mk_model(pad_min_b=5, step_b=150, seed_slope=2.0)
    for c in (100, 200, 300):
        m.add_sample(c, c // 2 + 50)
    validity_cap = 300 + 150             # 450
    budget = 2000
    # seed_c = (2000 - 5) // 2 = 997 > cap: the generous seed line vouches.
    inv = m.invert(budget)
    assert inv == 997
    assert inv > validity_cap
    # Beyond validity the prediction is the seed line, and it fits.
    p = m.predict(inv)
    assert p.seeded is True
    assert p.transient_b <= budget


# --------------------------------------------------------------------------- #
# r_worst + per-size max survive fit-window eviction
# --------------------------------------------------------------------------- #

def test_r_worst_tracks_lifetime_max_ratio():
    m = mk_model()
    assert m.r_worst == 0.0
    m.add_sample(100, 250)
    assert m.r_worst == pytest.approx(2.5)
    m.add_sample(200, 300)               # ratio 1.5 — does not lower it
    assert m.r_worst == pytest.approx(2.5)
    m.add_sample(100, 800)               # ratio 8.0 — raises it
    assert m.r_worst == pytest.approx(8.0)


def test_max_transient_at_is_per_size_running_max():
    m = mk_model()
    assert m.max_transient_at(100) == 0
    m.add_sample(100, 250)
    m.add_sample(100, 180)
    assert m.max_transient_at(100) == 250
    assert m.max_transient_at(200) == 0


def test_r_worst_survives_fit_window_eviction():
    m = mk_model()
    m.add_sample(100, 900)               # ratio 9.0
    # Flood the window with low-ratio samples until the first is evicted.
    for i in range(_FIT_WINDOW + 50):
        m.add_sample(1000 + i, 1000 + i)
    assert m.r_worst == pytest.approx(9.0)
    assert m.max_transient_at(100) == 900


# --------------------------------------------------------------------------- #
# Breach check
# --------------------------------------------------------------------------- #

def test_check_breach_true_iff_observed_exceeds_gated_prediction():
    m = mk_model()
    assert m.check_breach(100, 501, 500) is True
    assert m.check_breach(100, 500, 500) is False
    assert m.check_breach(100, 499, 500) is False


# --------------------------------------------------------------------------- #
# Registry keying (arbitrary tuples — per-pipe and per-class)
# --------------------------------------------------------------------------- #

def test_registry_keys_by_full_tuple():
    reg = ModelRegistry(pad_min_b=5, step_b=200)
    m = reg.model_for(1, "rearrange", "ANS")
    assert reg.model_for(1, "rearrange", "ANS") is m
    assert reg.model_for(1, "rearrange", "RAW") is not m
    assert reg.model_for(100, "rearrange", "ANS") is not m


def test_registry_pipe_key_distinct_from_class_key():
    # The per-pipe key and the class-prior key are different models.
    reg = ModelRegistry(pad_min_b=5, step_b=200)
    pipe = reg.model_for(1, 5, "sort", "q5 sort pipeline", "ANS")
    klass = reg.model_for(1, "sort", "ANS")
    assert pipe is not klass
    other = reg.model_for(1, 5, "sort", "q5 other pipeline", "ANS")
    assert other is not pipe


def test_registry_fits_never_read_across_keys():
    reg = ModelRegistry(pad_min_b=5, step_b=200)
    m = reg.model_for(1, "rearrange", "ANS")
    for c, t in ((100, 350), (200, 650)):
        m.add_sample(c, t)
    assert m.has_fit
    # Sibling keys stay in the seed phase: no fit leaks across.
    for key in ((1, "rearrange", "RAW"), (100, "rearrange", "ANS")):
        sibling = reg.model_for(*key)
        assert not sibling.has_fit
        assert sibling.predict(100).seeded is True
