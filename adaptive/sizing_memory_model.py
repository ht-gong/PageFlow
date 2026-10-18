"""The sizing plane's naive memory model.

Answers exactly one question: *what will the arena transient be if the next
gather iteration / tick section of this pipe runs at chunk budget ``c``?*
Deliberately the simplest thing that can gate a fatal step — a per-key linear
fit with a conservative upper envelope.  It is NOT a cost model of time: time
is measured, never predicted.

* **Transient** — only the transient is fitted; ``live_at_entry`` is read
  from the allocator at decision time by the caller.  Training target =
  ``iteration_peak − live_at_entry``.
* **Keying** — the registry keys are arbitrary tuples: the *pipe* model is
  keyed ``(sf, query, pipe_class, instance, codec_epoch)`` and the *class*
  model ``(sf, pipe_class, codec_epoch)``.  Every sample enters both; the
  class model is read only as the cold-pipe prior (the caller's gate
  logic).  Fits are never read across keys.
* **Envelope** — gating prediction = fit + pad, ``pad = max(max positive
  residual over the fit window, pad_min)`` — an upper envelope, never the
  mean.
* **Validity** — no fit-based extrapolation below 2 distinct ``c`` values,
  and never further than one additive march step (``step_b`` = Δ) beyond
  the largest observed ``c``; outside that validity the generous analytic
  seed is the only oracle.  Sense probes (δ) are inside the horizon by
  δ < Δ.
* **r_worst** — the largest observed ``transient / c`` ratio for this key,
  monotone over the model's lifetime (never decays when samples age out of
  the fit window).  Feeds the increment bound and the cold-pipe gate.
* **Seed envelope** — the per-row budget charges output + scratch and
  a compressing spill transiently holds the compressed copy beside the
  output; ``seed_slope`` defaults to 2.0 for rearrange, 3.0 for tick pipes
  (deliberately generous).
* **Breach** — an observed transient above its gating prediction is an
  envelope breach: the caller acts (freeze / tolerate per its rules); the
  model refits with the breach included (the sample is added like any other
  — the refit is automatic).

The module is torch-free and process-local (no cross-run persistence).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Samples kept per key for the fit window (the pad's residual window), and the
# per-distinct-size cap.  A converged stream emits hundreds of samples at ONE
# size; without the cap they evict every other size from the window, the fit
# collapses below its 2-distinct-sizes floor mid-stream, and predictions
# snap back to the pessimistic seed — spuriously shrinking a healthy frozen
# stream (review finding).  Capping per-size representation keeps size
# diversity indefinitely while the recent residuals of the converged size
# still refresh the envelope pad.
_FIT_WINDOW = 160
_PER_SIZE_CAP = 8


@dataclass(frozen=True)
class Prediction:
    transient_b: int
    seeded: bool          # True when fit validity forced the analytic seed


class TransientEnvelopeModel:
    """One key's envelope fit — keyed by the registry below."""

    def __init__(self, pad_min_b: int, step_b: int,
                 seed_slope: float = 2.0):
        if pad_min_b < 0 or step_b <= 0 or seed_slope <= 0:
            raise ValueError(
                f"TransientEnvelopeModel: bad params pad_min_b={pad_min_b} "
                f"step_b={step_b} seed_slope={seed_slope}")
        self.pad_min_b = int(pad_min_b)
        self.step_b = int(step_b)          # additive validity horizon
        self.seed_slope = float(seed_slope)
        self._samples: List[Tuple[int, int]] = []
        # Cached fit: (a_hat, b_hat, pad_b) or None when < 2 distinct c.
        self._fit: Optional[Tuple[float, float, int]] = None
        # Lifetime running state — survives fit-window eviction.
        self._r_worst: float = 0.0
        self._max_at: Dict[int, int] = {}   # size -> max observed transient

    # -- training ----------------------------------------------------------

    def add_sample(self, c_b: int, transient_b: int) -> None:
        c_b = int(c_b)
        transient_b = max(0, int(transient_b))
        self._samples.append((c_b, transient_b))
        if c_b > 0:
            self._r_worst = max(self._r_worst, transient_b / c_b)
        self._max_at[c_b] = max(self._max_at.get(c_b, 0), transient_b)
        same = [i for i, (cc, _) in enumerate(self._samples) if cc == c_b]
        if len(same) > _PER_SIZE_CAP:
            del self._samples[same[0]]
        if len(self._samples) > _FIT_WINDOW:
            # Trim the oldest sample of any over-represented size; never
            # drop the last sample of a distinct size.
            counts: Dict[int, int] = {}
            for cc, _ in self._samples:
                counts[cc] = counts.get(cc, 0) + 1
            for i, (cc, _) in enumerate(self._samples):
                if counts[cc] > 1:
                    del self._samples[i]
                    break
            else:
                del self._samples[0]
        self._refit()

    def _refit(self) -> None:
        xs = [c for c, _ in self._samples]
        ys = [t for _, t in self._samples]
        if len(set(xs)) < 2:
            self._fit = None
            return
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        var = sum((x - mean_x) ** 2 for x in xs)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        a = max(0.0, cov / var)                     # slope clamp
        b = mean_y - a * mean_x
        max_resid = max(y - (a * x + b) for x, y in zip(xs, ys))
        pad = max(int(max_resid), self.pad_min_b)   # envelope pad
        self._fit = (a, b, pad)

    # -- queries -----------------------------------------------------------

    @property
    def has_fit(self) -> bool:
        return self._fit is not None

    @property
    def max_observed_c(self) -> int:
        return max((c for c, _ in self._samples), default=0)

    @property
    def r_worst(self) -> float:
        """Largest ``transient / c`` this key ever exhibited (0.0 before
        any sample) — monotone, never decays."""
        return self._r_worst

    def max_transient_at(self, c_b: int) -> int:
        """Largest transient ever observed at exactly ``c_b`` (the
        increment bound's ``T_obs``); 0 when the size was never observed."""
        return self._max_at.get(int(c_b), 0)

    def _seed_predict(self, c_b: int) -> int:
        return int(self.seed_slope * c_b) + self.pad_min_b

    def predict(self, c_b: int) -> Prediction:
        """Envelope prediction of the transient at budget ``c_b``.

        Fit-based only while the fit is valid (≥ 2 distinct sizes and
        ``c ≤ max_observed_c + step_b``); the generous analytic seed
        otherwise.  Inside validity the fitted envelope (fit + max-residual
        pad) IS the gate — that is what lets a well-behaved pipe grow past
        the deliberately pessimistic seed line.
        """
        if self._fit is None or c_b > self.max_observed_c + self.step_b:
            return Prediction(self._seed_predict(c_b), seeded=True)
        a, b, pad = self._fit
        return Prediction(max(0, int(a * c_b + b + pad)), seeded=False)

    def invert(self, transient_budget_b: int) -> int:
        """Largest ``c`` whose envelope prediction fits ``transient_budget_b``
        (the clamp path).  The prediction is piecewise — the fitted
        envelope inside fit validity (``c ≤ max_observed_c + step_b``), the
        seed line beyond — so the inversion checks the seed region first and
        falls back to the (validity-capped) fit region.  Returns 0 when no
        budget fits.
        """
        budget = int(transient_budget_b)
        seed_c = max(0, int((budget - self.pad_min_b) / self.seed_slope))
        validity_cap = self.max_observed_c + self.step_b
        if self._fit is None:
            return seed_c
        if seed_c > validity_cap:
            # The generous seed line vouches for a size beyond fit validity.
            return seed_c
        a, b, pad = self._fit
        if a <= 0.0:
            fit_c = validity_cap if (b + pad) <= budget else 0
        else:
            fit_c = int((budget - b - pad) / a)
        return max(0, min(fit_c, validity_cap))

    def check_breach(self, c_b: int, observed_transient_b: int,
                     gated_prediction_b: int) -> bool:
        """Did the observed transient exceed the prediction that gated
        it?  The caller acts on True; the sample itself has already
        re-entered the fit via :meth:`add_sample` (envelope self-heals)."""
        return observed_transient_b > gated_prediction_b


class ModelRegistry:
    """One :class:`TransientEnvelopeModel` per arbitrary key tuple.  Pipe
    models are keyed ``(sf, query, pipe_class, instance, codec_epoch)``,
    class priors ``(sf, pipe_class, codec_epoch)``.  Fits are never read
    across keys."""

    def __init__(self, pad_min_b: int, step_b: int, seed_slope: float = 2.0):
        self._pad_min_b = pad_min_b
        self._step_b = step_b
        self._seed_slope = seed_slope
        self._models: Dict[Tuple, TransientEnvelopeModel] = {}

    def model_for(self, *key_parts) -> TransientEnvelopeModel:
        key = tuple(str(p) if isinstance(p, str) else p for p in key_parts)
        m = self._models.get(key)
        if m is None:
            m = TransientEnvelopeModel(self._pad_min_b, self._step_b,
                                       self._seed_slope)
            self._models[key] = m
        return m
