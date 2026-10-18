"""Benefit-ranked prefix search over k = how many columns to compress.

The search half of the compression plane: ``f(k)`` — the per-tick
slowest-leg sample when the top-k benefit-ranked columns are compressed — is
unimodal in ``k`` (each transfer leg non-increasing, compute non-decreasing,
``max`` of the two envelopes quasi-convex), so its minimum is found by
integer ternary search in ``O(log N)`` evaluations.

Two layers:

* :func:`ternary_argmin_gen` / :func:`argmin_unimodal` — the pure search
  math, memoized, endpoints included.
* :class:`CompressionSearchDriver` — the streaming per-chunk driver: a pure
  ``view -> picks-or-None`` function (the tick-observer shape) that probes
  successive ``k`` values across a pipe's ticks.
  It consumes the monitor's last-tick view (``last_tick``, ``f_last``) and
  emits full input-pick vectors from
  :meth:`~adaptive.compression_controller.CompressionController.pick_vector`.
  It owns the probe schedule and attribution; actuation sequencing (the t+2
  rule) is the pipeline's job — the driver only *assumes* it when
  attributing samples.

Contract points implemented here:

* **t+2 attribution** — a vector returned from sample *t* is assumed active
  from tick *t+2*; every measured ``f`` is attributed to the ``k`` that
  actually built its tick (``attribution`` log).
* **Input vectors only** — replies are input vectors only; successive
  emissions differ on decision (two-entry) columns only — singleton entries
  are fixed by construction.
* **Short pipes** — a pipe with fewer ticks than the probe budget keeps the
  policy default: the driver goes inert and never returns picks.
* **Probe economy** — ``O(log N)`` probes, a configurable
  ≤-fraction-of-ticks probe cap, and a CONVERGED state that holds ``k*``
  stable with no oscillation.

(There is no host-stall veto: the search assumes the host is never the
bottleneck — ``f`` is device-event-derived only.)

CPU-only, torch-free.
"""
from __future__ import annotations

import os
from statistics import median
from typing import Callable, Dict, Generator, List, Optional, Tuple

from adaptive import resource_monitor
from adaptive.decision_menu import PickPair
from adaptive.resource_monitor import ResourceMonitor


# ---------------------------------------------------------------------------
# Pure search math
# ---------------------------------------------------------------------------

def ternary_argmin_gen(n: int) -> Generator[int, float, int]:
    """Generator protocol for integer ternary search over ``k in [0, n]``.

    Yields each ``k`` to evaluate; the caller ``send()``s the measured
    ``f(k)`` back.  Returns (``StopIteration.value``) the argmin.  Evaluations
    are memoized, endpoints are included in the final scan, and ties break
    toward the smaller ``k`` (deterministic).
    """
    memo: Dict[int, float] = {}
    lo, hi = 0, n
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if m1 not in memo:
            memo[m1] = yield m1
        if m2 not in memo:
            memo[m2] = yield m2
        if memo[m1] < memo[m2]:
            hi = m2
        else:
            lo = m1
    for k in range(lo, hi + 1):
        if k not in memo:
            memo[k] = yield k
    return min(range(lo, hi + 1), key=lambda k: (memo[k], k))


def argmin_unimodal(eval_fn: Callable[[int], float], n: int
                    ) -> Tuple[int, int]:
    """Run the ternary search against a black-box ``eval_fn``.

    Returns ``(k_star, evaluations_used)``.
    """
    gen = ternary_argmin_gen(n)
    evals = 0
    try:
        k = next(gen)
        while True:
            evals += 1
            k = gen.send(eval_fn(k))
    except StopIteration as stop:
        return stop.value, evals


def min_searchable_ticks(probes_per_k: int = 2, actuation_lag: int = 2,
                         max_probe_frac: float = 0.25) -> int:
    """The smallest ``total_ticks`` at which ANY driver with these settings
    can be non-inert (the N=1 budget + the probe sample-fraction cap).  The
    pipeline uses this as a construction-time skip hint: shorter pipes get
    no menu and no observer at all (their driver would be inert anyway)."""
    p, lag = max(1, int(probes_per_k)), int(actuation_lag)
    budget1 = p + max_evaluations(1) * (p + lag - 1) + lag
    samples1 = max_evaluations(1) * p
    import math
    return max(budget1, math.ceil(samples1 / max_probe_frac))


def max_evaluations(n: int) -> int:
    """Worst-case evaluation count of :func:`ternary_argmin_gen` on
    ``[0, n]`` (memoization can only reduce it) — the probe-budget input."""
    lo, hi, evals = 0, n, 0
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        evals += 2
        # Take whichever branch keeps the interval larger (worst case).
        if m2 - lo >= hi - m1:
            hi = m2
        else:
            lo = m1
    return evals + (hi - lo + 1)


# ---------------------------------------------------------------------------
# The streaming per-chunk driver
# ---------------------------------------------------------------------------

class CompressionSearchDriver:
    """Streaming ternary search over the decision-column subset size ``k``.

    Its call shape IS the tick observer (``adaptive.adaptive_driver.
    TickObserver``), which the search factories in ``main.py`` wrap.  Per
    view it either collects the tick's ``f_last`` for the probe that built
    the tick, or — when a probe completes — returns the next probe's full
    input-pick vector for the pipeline to actuate at ``t+2``.

    States: INERT (few-chunk / no decision columns — never returns picks),
    SEARCHING, CONVERGED (``k*`` installed, held stable).
    """

    def __init__(self, controller, total_ticks: int, *,
                 probes_per_k: int = 2,
                 max_probe_frac: float = 0.25,
                 actuation_lag: int = 2):
        self._controller = controller
        self._n = controller.decision_count()
        self._probes_per_k = max(1, int(probes_per_k))
        self._lag = int(actuation_lag)

        # Probe budget (search feasibility): each evaluation costs its samples
        # PLUS the actuation lag it waits out before the first attributable
        # sample (emit at sample t -> collect at t+lag .. t+lag+p-1, so the
        # per-eval increment is p + lag - 1); +1 for the compute-less
        # prologue tick the driver skips, + the passive pre-gate samples
        # (p, of which the last coincides with the first emission), + a
        # final lag for the converged vector to land.  A pipe shorter than
        # this cannot finish a search.
        self._budget_ticks = (self._probes_per_k
                              + max_evaluations(self._n)
                              * (self._probes_per_k + self._lag - 1)
                              + self._lag)
        # Probe economy: only *measured probe samples* count as probing
        # spend — the interleaved lag ticks run the previous regime at zero
        # overhead — so the ≤-fraction cap applies to the worst-case sample
        # count, not the lag-inclusive budget.
        probe_samples = max_evaluations(self._n) * self._probes_per_k
        frac_cap = (total_ticks * max_probe_frac if total_ticks else 0)
        self.inert = (self._n == 0
                      or total_ticks < self._budget_ticks
                      or probe_samples > frac_cap)

        # Trailing ticks whose samples are not steady-state (the pipeline's
        # two-tick epilogue: tick T-2 transfers an empty chunk, tick T-1
        # computes one — their ``f`` is a drain artifact, not an ``f(k)``
        # sample).  With the budget gate above the search converges well
        # before this tail; the guard is for the truncation edge.
        self._last_valid_tick = (total_ticks - 3 if total_ticks else -1)

        # Search state.
        self._gen: Optional[Generator[int, float, int]] = None
        self._probe_k: Optional[int] = None       # k being measured
        self._effective_tick: Optional[int] = None  # first tick built by it
        self._samples: List[float] = []
        self.k_star: Optional[int] = None
        self.converged = False

        # Bottleneck pre-gate (probe economy; from the legs' monotonicity):
        # the incumbent static regime IS k = N.  If, measured there, the
        # transfer envelope already >= compute, then for every k < N
        # transfer(k) >= transfer(N) = f(N) — no smaller k can improve f, so
        # k* = N is PROVEN without probing a single tick.  The driver
        # observes the first ``probes_per_k`` valid static-regime samples
        # (passive — no actuation, no cost) and only starts the ternary when
        # compute dominates.
        self._pregate: List[Tuple[float, float]] = []   # (transfer, compute)
        self.pregate_skipped = False

        # Baseline regime before any actuation: the static default compresses
        # every decision column (typed non-gather resolves its candidate),
        # i.e. k = N over the ranked decision columns.
        self._active_k = self._n

        # Introspection / test surfaces.
        self.attribution: List[Tuple[int, int]] = []   # (tick, k) consumed
        self.emitted: List[Tuple[int, int, Tuple[str, ...]]] = []  # (tick, k, vector)
        self.probes_spent = 0

    # -- internals -----------------------------------------------------------

    def _advance(self, measured_f: Optional[float], tick: int
                 ) -> Optional[Tuple[str, ...]]:
        """Feed the completed probe's aggregate into the generator (or start
        it) and emit the next probe's pick vector, or converge."""
        try:
            if self._gen is None:
                self._gen = ternary_argmin_gen(self._n)
                next_k = next(self._gen)
            else:
                next_k = self._gen.send(measured_f)
        except StopIteration as stop:
            self.k_star = stop.value
            self.converged = True
            self._probe_k = None
            final_picks = self._controller.pick_vector(self.k_star)
            self.emitted.append((tick, self.k_star, final_picks))
            self._effective_tick = tick + self._lag
            return final_picks
        self._probe_k = next_k
        self._effective_tick = tick + self._lag
        self._samples = []
        probe_picks = self._controller.pick_vector(next_k)
        self.emitted.append((tick, next_k, probe_picks))
        return probe_picks

    # -- the tick observer (pure view -> picks) -------------------------------

    def on_view(self, view: ResourceMonitor) -> Optional[Tuple[str, ...]]:
        if self.inert:
            return None
        tick = view.last_tick

        # A previously returned vector becomes the active regime at its
        # actuation tick (the pipeline applied it at t+lag).
        if self._effective_tick is not None and tick >= self._effective_tick:
            self._active_k = (self.k_star if self.converged
                              else self._probe_k)
            self._effective_tick = None

        if self.converged:
            return None

        # Prologue tick: transfer-only, delivered with last_compute None —
        # not a valid f sample.
        if view.last_compute is None:
            return None

        # Epilogue tail: drain-artifact samples (and any emission would
        # actuate past the end of the pipe) — hold the current regime.
        if tick > self._last_valid_tick:
            return None

        if self._gen is None:
            # Bottleneck pre-gate: observe the incumbent (k = N) regime
            # first.  Transfer-dominated => k* = N proven, zero probes.
            self._pregate.append((view.last_transfer.get_max()
                                  if view.last_transfer is not None else 0.0,
                                  view.last_compute.compute))
            if len(self._pregate) < self._probes_per_k:
                return None
            if (median(t for t, _ in self._pregate)
                    >= median(c for _, c in self._pregate)):
                self.k_star = self._n
                self.converged = True
                self.pregate_skipped = True
                self.probes_spent = len(self._pregate)
                return None       # incumbent already runs k = N — no picks
            # Compute-dominated: start the search, emit the first probe.
            return self._advance(None, tick)

        if self._probe_k is None or self._effective_tick is not None:
            # Awaiting actuation — the tick was built by the previous regime.
            return None

        # Collecting for the active probe.
        self.attribution.append((tick, self._active_k))
        self._samples.append(view.f_last)
        self.probes_spent += 1
        if len(self._samples) < self._probes_per_k:
            return None
        return self._advance(median(self._samples), tick)

    __call__ = on_view

    def settled(self) -> bool:
        """No more picks: inert, or converged (its last vector already
        returned)."""
        return self.inert or self.converged


# ---------------------------------------------------------------------------
# The write-leg window search (read-and-write)
# ---------------------------------------------------------------------------

class WriteWindowSearchDriver:
    """Probe-window search over the write prefix ``k_w`` (the boundary
    mechanics live on the pipeline side).

    The write analog of the read-leg ternary: configs are benefit-ranked
    write prefixes ``k in [0, n]`` (ranked ONCE from the in-run measured
    per-slot output bytes — ``view.out_slot_raw_bytes`` — and frozen, which
    is what keeps window-averaged ``f(k)`` unimodal: transfer legs
    non-increasing, ``compress_out`` non-decreasing in ``k``).  Each
    evaluation is one hermetic probe window:

      emit ``PickPair(outputs=vector(k), seal=True)``   (flip boundary, t+1)
        -> skip ``lag + warm_skip`` settling ticks
        -> collect ``window_ticks`` 4-lane samples
        -> emit a NULL boundary (same vector, ``seal=True``), so the
           following ``drain_ticks`` samples carry the window's own bytes
           still in flight to their homes
        -> score = max over lanes of the window+drain per-tick mean.

    Every pipe writes one region per tick and column, cached or not as
    usual, so a boundary costs nothing and every tick ships its own bytes.

    Budget-gated (as on the read leg): ternary above budget; below it, the
    **one-shot greedy** fallback — flip everything to RAW once when compute
    dominates for ``greedy_dominance_ticks`` consecutive ticks and the
    predicted encode saving over the remaining ticks clears
    ``greedy_min_benefit_ms`` (at most one flip per column per run).  The
    bottleneck pre-gate applies unchanged: the incumbent IS ``k = n``, so a
    transfer-dominated regime proves ``k* = n`` with zero windows.

    Three pre-gates run on the incumbent window before any probe is spent:
    **P1** (transfer-dominated ⇒ ``k* = n``) in every policy; **P1b**, the
    encode-realization gate (``compute − compress_out > max transfer lane``
    ⇒ hold), everywhere except ``ungated`` — see ``__init__`` for the SF700
    measurement that motivates it; and **P3**, the readback gate
    (``compress_out`` no larger than the cheapest single column's readback
    price ⇒ hold), everywhere except ``ungated``.  ``compress_out`` is the
    tick's output-compress span on the compute stream, host gaps included.

    **The readback price.**  A RAW output's extra bytes cross PCIe again
    after this pipe: home, and once per consumer read (the dataflow's
    expected reads, on the menu).  None of it lands in this pipe's lanes —
    with the granule cache not even its own d2h lane, since an admitted
    output goes home only on eviction — so a producer-local ``f`` sees the
    encode leave the compute lane and nothing else (SF300 Q21: three join
    sort pipes flipped to RAW took the query from 6.4 to 9.0 s, spilling
    and restoring 47 GB each way).  Each write decision column carries a
    price, the ms per tick its RAW bytes add downstream:
    ``raw/tick × (1 − 1/ratio) × (reads + 1) ÷ PCIE_B_PER_MS``
    (``resource_monitor``), from the
    pipe's byte counters on the incumbent; every window score, the greedy
    probe's and P2's signal add the price of the columns they write RAW.
    """

    def __init__(self, controller, total_ticks: int, *,
                 start_tick: int = 0,
                 window_ticks: int = 2,
                 drain_ticks: int = 2,
                 actuation_lag: int = 1,
                 warm_skip: int = 1,
                 settle_skip: int = 0,
                 min_production_ticks: int = 4,
                 greedy_dominance_ticks: int = 3,
                 greedy_min_benefit_ms: float = 5.0,
                 ternary_min_benefit_ms: float = 25.0,
                 min_saving_ms_per_tick: float = 1.0,
                 write_policy: Optional[str] = None,
                 encode_gate: Optional[bool] = None):
        self._controller = controller
        self._n = controller.write_decision_count()
        self._W = max(1, int(window_ticks))
        self._D = max(0, int(drain_ticks))
        self._lag = int(actuation_lag)
        self._skip = self._lag + max(0, int(warm_skip))
        self._settle = max(0, int(settle_skip))
        self._min_prod = int(min_production_ticks)
        self._dom_ticks = max(1, int(greedy_dominance_ticks))
        self._min_benefit = float(greedy_min_benefit_ms)
        self._ternary_min_benefit = float(ternary_min_benefit_ms)
        self._min_per_tick = float(min_saving_ms_per_tick)
        self._last_valid_tick = (total_ticks - 3 if total_ticks else -1)
        self._total_ticks = total_ticks

        eval_cost = self._skip + self._W + self._D
        self._ternary_budget = (self._settle + self._W
                                + max_evaluations(self._n) * eval_cost
                                + self._lag)
        self._greedy_budget = (self._settle + self._W + self._dom_ticks
                               + self._lag)
        ternary_ok = (start_tick + self._ternary_budget + self._min_prod
                      <= self._last_valid_tick)
        greedy_ok = (start_tick + self._greedy_budget + self._min_prod
                     <= self._last_valid_tick)
        self.mode = ("ternary" if ternary_ok
                     else "greedy" if greedy_ok else "inert")
        self.inert = self._n == 0 or self.mode == "inert"

        # Search state.
        self._gen: Optional[Generator[int, float, int]] = None
        self._ranked: Optional[List[int]] = None
        self._probe_k: Optional[int] = None
        self._active_vec: Optional[Tuple[str, ...]] = None
        self._samples: List[Tuple[float, float, float, float]] = []
        self._skip_left = 0
        self._null_sealed = False
        self._pregate: List[Tuple[float, float]] = []
        self._pregate_sig: List[float] = []   # per-tick min(cout, comp-xfer)
        self._pregate_res: List[float] = []   # per-tick compute - compress_out
        self._pregate_cout: List[float] = []  # per-tick compress_out
        self._wdec = list(controller.write_decision_indices())
        # The readback price (ms per tick) of writing each decision column
        # RAW, by output slot, and its inputs: the consumer reads per slot
        # and the previous view's raw byte counters (``_observe_bytes``).
        self._reads = controller.write_reads()
        self.price: Dict[int, float] = {}
        self._raw_prev: Optional[List[int]] = None
        self._pregate_done = False
        self._dom_run: List[Tuple[float, float]] = []   # (benefit, compress_out)
        self.pregate_skipped = False
        self.benefit_gated = False
        self.encode_gated = False
        self.readback_gated = False
        # Write-leg gating policy.  The P1 transfer-dominated pre-gate applies
        # in every policy; the policy only governs the compute-dominated
        # regime:
        #   "measured"   (default) — drop the predictive P2 benefit floor and,
        #       in the greedy fallback, replace the blind one-shot flip with a
        #       MEASURED k=0-vs-incumbent probe: flip to RAW only when it
        #       measurably lowers the bottleneck f, else hold the incumbent
        #       (never blind-flip).
        #   "predictive" — the original write leg: predictive P2 benefit
        #       floor + greedy benefit threshold (kept as an A/B fallback).
        #   "ungated"    — drop both floors; greedy blind-flips on sustained
        #       compute-dominance (diagnostic only).
        # Promoted to "measured" default 2026-07-22: the predictive floor almost
        # never flips (SF700: 2/80 write pipes) and left a validated ~4 % on the
        # table; measured flips the 34 compute-dominated pipes whose output-
        # encode sat on the critical path (SF700 143.3 vs 148.8s; verify_mode
        # read-and-write 22/22).  An explicit kwarg wins; otherwise a TQP_WRITE_*
        # env switch picks the default so the A/B artifacts still reproduce.
        if write_policy is None:
            if os.environ.get("TQP_WRITE_UNGATED"):
                write_policy = "ungated"
            elif os.environ.get("TQP_WRITE_PREDICTIVE"):
                write_policy = "predictive"
            else:                     # TQP_WRITE_MEASURED is now the default
                write_policy = "measured"
        if write_policy not in ("measured", "predictive", "ungated"):
            raise ValueError(f"unknown write_policy: {write_policy!r}")
        self.write_policy = write_policy
        self._ungated = write_policy == "ungated"
        self._measured = write_policy == "measured"
        self._drop_p2 = self._ungated or self._measured
        # P1b, the encode-realization gate (SF700 2026-07-27).  The whole
        # write leg is justified by one premise: dropping compression removes
        # the output-compress step from the compute lane.  Measured on the
        # SF700 suite at chunk 512 that premise is FALSE — over the 35 pipes
        # the plane flipped, 3.007 s of encode really did leave
        # `compress_out`, and the compute lane went UP 0.127 s (saving
        # realized at −4 %): a RAW output is a bigger page, and the extra
        # page/launch work inside the same compute bracket eats the encode
        # saving whole.
        #
        # So refuse to spend probes where the output-compress step is the
        # ONLY thing arguing for the flip.  ``compute − compress_out > max
        # transfer lane`` is exactly the condition ``sig = min(compress_out,
        # compute − transfer)`` binds on the compress term rather than on the
        # measured compute/transfer gap; there the expected benefit is ~0
        # while the transfer inflation is real and mostly lands on OTHER
        # pipes.  Pipes whose gap binds — where removing the compress step
        # could genuinely hand the bottleneck to the transfer lane — still
        # search normally.  ``compress_out`` is the span the pipeline already
        # records around the tick's output step on the compute stream (host
        # gaps included), so the gate reads no timing of its own; it bounds
        # the codec kernels' own time from above, so the gate holds fewer
        # pipes than one on pure kernel time would.  Measured retrospectively
        # on the a3 arm, with pure encode kernel time as the signal: gates 33
        # of 35 flips.  Not applied under "ungated", whose contract is "no
        # floors".
        if encode_gate is None:
            encode_gate = not (self._ungated
                               or os.environ.get("TQP_WRITE_NO_ENCODE_GATE"))
        self._encode_gate = bool(encode_gate)
        # The readback price (class docstring): off only under "ungated".
        self._priced = not self._ungated
        # measured-greedy probe state
        self._f_incumbent = 0.0
        self._gprobing = False
        self._gskip = 0
        self._gsamples: List[Tuple[float, float, float, float]] = []
        self.k_star: Optional[int] = None
        self.converged = False
        # Introspection / test surfaces.
        self.windows: List[Tuple[int, float, int]] = []   # (k, score, samples)
        self.emitted: List[Tuple[int, int, bool]] = []    # (tick, k, seal_only)
        self.probes_spent = 0

    # -- internals -----------------------------------------------------------

    def _vector(self, k: int) -> Tuple[str, ...]:
        return self._controller.write_pick_vector(k, self._ranked)

    def _emit(self, tick: int, k: int, seal_only: bool) -> PickPair:
        vec = self._vector(k)
        self._active_vec = vec
        self.emitted.append((tick, k, seal_only))
        return PickPair(outputs=vec, seal=True)

    def _advance(self, measured: Optional[float], tick: int
                 ) -> Optional[PickPair]:
        try:
            if self._gen is None:
                self._gen = ternary_argmin_gen(self._n)
                next_k = next(self._gen)
            else:
                next_k = self._gen.send(measured)
        except StopIteration as stop:
            self.k_star = stop.value
            self.converged = True
            self._probe_k = None
            return self._emit(tick, self.k_star, seal_only=False)
        self._probe_k = next_k
        self._samples = []
        self._skip_left = self._skip
        self._null_sealed = False
        return self._emit(tick, next_k, seal_only=False)

    # -- the tick observer ----------------------------------------------------

    def on_view(self, view: ResourceMonitor) -> Optional[PickPair]:
        if self.inert or self.converged:
            return None
        tick = view.last_tick
        if view.last_compute is None:       # prologue: no f sample
            return None
        if tick > self._last_valid_tick:
            # Epilogue truncation: hold whatever regime is active.
            self.k_star = self._probe_k if self._probe_k is not None else self._n
            self.converged = True
            return None

        transfer = (view.last_transfer.get_max()
                    if view.last_transfer is not None else 0.0)
        compute = view.last_compute.compute

        # Bottleneck pre-gate on the incumbent (k = n) regime.
        if not self._pregate_done:
            self._observe_bytes(view)
            if self._settle > 0:
                self._settle -= 1
                return None
            self._pregate.append((transfer, compute))
            self._pregate_sig.append(
                min(view.last_compute.compress_out,
                    max(0.0, compute - transfer)))
            self._pregate_res.append(
                max(0.0, compute - view.last_compute.compress_out))
            self._pregate_cout.append(view.last_compute.compress_out)
            if len(self._pregate) < self._W:
                return None
            self._pregate_done = True
            if (median(t for t, _ in self._pregate)
                    >= median(c for _, c in self._pregate)):
                self.k_star = self._n
                self.converged = True
                self.pregate_skipped = True
                return None
            # P1b — the encode-realization gate.  Removing the output
            # compress step is only worth probing when it could hand the
            # bottleneck to the transfer lane; where the compute lane stays
            # on top even with that step free, the flip's whole case rests on
            # an encode saving that does not materialise (see __init__).
            # Hold the incumbent.
            if (self._encode_gate
                    and median(self._pregate_res)
                    > median(t for t, _ in self._pregate)):
                self.k_star = self._n
                self.converged = True
                self.encode_gated = True
                return None
            # P3 — the readback gate.  Writing any subset RAW saves at most
            # the whole output-compress span and costs at least the
            # cheapest column's readback price: when that price already
            # covers the span, no k < n can win (class docstring).  Hold.
            if (self._priced and self.price
                    and median(self._pregate_cout)
                    <= min(self.price.get(i, 0.0) for i in self._wdec)):
                self.k_star = self._n
                self.converged = True
                self.readback_gated = True
                return None
            # Benefit floors (the remaining production must be worth probing —
            # both are measured SF100 Q13 lessons).  (1) Total ceiling: the
            # write leg's prize is the encode share it could shave off the
            # compute lane over the remaining ticks; below the floor,
            # probing costs more than it can win (boundary seals, drains,
            # off-optimum windows).  (2) Per-tick materiality: the in-run
            # objective is producer-local by construction (the consumer's
            # h2d for the now-RAW bytes lands in a LATER pipe, invisible
            # here), so a flip whose per-tick encode saving is marginal
            # loses globally even when the long remainder makes its total
            # clear the ceiling (measured: Q13's join-core flip, sub-ms/tick
            # saving, large joined outputs re-read downstream, +2.9 % on the
            # query).  Flip only when encode is a MATERIAL per-tick share,
            # net of the readback price of writing every column RAW.
            if not self._drop_p2:
                remaining = max(0, self._last_valid_tick - tick)
                sig = max(0.0, median(self._pregate_sig)
                          - self._price_of(0))
                if (sig * remaining < self._ternary_min_benefit
                        or sig < self._min_per_tick):
                    self.k_star = self._n
                    self.converged = True
                    self.benefit_gated = True
                    return None
            # Incumbent (k=N) bottleneck f from the pre-gate window — the
            # baseline the measured-greedy probe compares RAW against.
            self._f_incumbent = median(max(t, c) for t, c in self._pregate)
            # Compute-dominated and worth probing: rank once, the columns
            # dearest to write RAW first (compressed at the smallest k): by
            # readback price, else by the measured raw bytes; then enter
            # the mode the budget allows.
            self._ranked = self._controller.ranked_write_indices(
                [self.price.get(i, 0.0)
                 for i in range(len(view.out_slot_raw_bytes or ()))]
                if self.price else view.out_slot_raw_bytes)
            if self.mode == "ternary":
                return self._advance(None, tick)
            self._dom_run.append(self._greedy_signal(view, transfer, compute))
            return None

        if self.mode == "greedy":
            return self._greedy_step(view, tick, transfer, compute)

        # Ternary window collection.
        if self._skip_left > 0:
            self._skip_left -= 1
            return None
        self._samples.append((view.last_transfer.h2d_ms,
                              view.last_transfer.d2d_ms,
                              view.last_transfer.d2h_ms,
                              compute))
        self.probes_spent += 1
        n_s = len(self._samples)
        if n_s == self._W and not self._null_sealed and self._D > 0:
            # Null boundary (A-A): the drain ticks below carry the
            # window's own bytes.
            self._null_sealed = True
            return self._emit(tick, self._probe_k, seal_only=True)
        if n_s < self._W + self._D:
            return None
        lanes = [sum(s[i] for s in self._samples) / n_s for i in range(4)]
        score = max(lanes) + self._price_of(self._probe_k)
        self.windows.append((self._probe_k, score, n_s))
        return self._advance(score, tick)

    def _observe_bytes(self, view) -> None:
        """Refresh each write decision column's readback price from the
        pipe's cumulative per-slot byte counters on the incumbent (every
        column compressed).  A compressed output's stored bytes are counted
        when its region is packed, a tick after its raw bytes, so the ratio
        pairs this view's stored bytes with the previous view's raw ones.
        No price without the counters (a pipe whose outputs are not
        tracked) or before two views."""
        raw = getattr(view, "out_slot_raw_bytes", None)
        stored = getattr(view, "out_slot_stored_bytes", None)
        if not raw or not stored:
            return
        prev, self._raw_prev = self._raw_prev, list(raw)
        if prev is None:
            return
        ticks = max(1, view.last_tick)
        for i in self._wdec:
            if i >= len(raw) or i >= len(stored):
                continue
            if prev[i] <= 0 or stored[i] <= 0:
                continue
            saved = max(0.0, 1.0 - stored[i] / prev[i])   # 1 − 1/ratio
            self.price[i] = (raw[i] / ticks * saved
                             * (self._reads[i] + 1)
                             / resource_monitor.PCIE_B_PER_MS)

    def _price_of(self, k: int) -> float:
        """The readback price (ms per tick) of the write vector at prefix
        ``k``: the columns past the first ``k`` of the ranking are written
        RAW.  0 without prices or under ``ungated``."""
        if not self._priced or not self.price:
            return 0.0
        ranked = (self._ranked if self._ranked is not None
                  else self._controller.ranked_write_indices())
        return sum(self.price.get(i, 0.0) for i in ranked[k:])

    def _greedy_signal(self, view, transfer, compute):
        cout = view.last_compute.compress_out
        return (min(cout, max(0.0, compute - transfer)), cout)

    def _greedy_step(self, view, tick, transfer, compute
                     ) -> Optional[PickPair]:
        # --- TQP_WRITE_MEASURED: a k=0 probe window is in flight ---
        if self._gprobing:
            if self._gskip > 0:
                self._gskip -= 1
                return None
            self._gsamples.append((view.last_transfer.h2d_ms,
                                   view.last_transfer.d2d_ms,
                                   view.last_transfer.d2h_ms, compute))
            self.probes_spent += 1
            if len(self._gsamples) < self._W:
                return None
            lanes = [sum(s[i] for s in self._gsamples) / len(self._gsamples)
                     for i in range(4)]
            f0 = max(lanes) + self._price_of(0)
            self.windows.append((0, f0, len(self._gsamples)))
            self.converged = True
            if f0 < self._f_incumbent:            # RAW measurably lowers f
                self.k_star = 0
                return self._emit(tick, 0, seal_only=False)
            self.k_star = self._n                 # measured worse -> hold
            return self._emit(tick, self._n, seal_only=False)

        # --- sustained compute-dominance detection ---
        if compute > transfer:
            self._dom_run.append(self._greedy_signal(view, transfer, compute))
        else:
            self._dom_run = []
            return None
        if len(self._dom_run) < self._dom_ticks:
            return None

        if self._measured:
            # Measure f(k=0) before flipping; hold the incumbent if there is
            # no room to finish the probe (never blind-flip).
            remaining = max(0, self._last_valid_tick - tick)
            if remaining < self._skip + self._W + 1:
                self.k_star = self._n
                self.converged = True
                return None
            self._gprobing = True
            self._gskip = self._skip
            self._gsamples = []
            return self._emit(tick, 0, seal_only=False)   # emit RAW to measure

        if not self._ungated:
            remaining = max(0, self._last_valid_tick - tick)
            benefit = median(b for b, _ in self._dom_run) * remaining
            if benefit < self._min_benefit:
                return None
        # One-shot flip: everything RAW, held for the rest of the run.
        self.k_star = 0
        self.converged = True
        return self._emit(tick, 0, seal_only=False)

    __call__ = on_view

    def settled(self) -> bool:
        """No more picks: inert, or converged (its last pick pair already
        returned)."""
        return self.inert or self.converged


# ---------------------------------------------------------------------------
# read-and-write composition
# ---------------------------------------------------------------------------

def min_write_searchable_ticks(window_ticks: int = 2, drain_ticks: int = 2,
                               actuation_lag: int = 1,
                               greedy_dominance_ticks: int = 3,
                               min_production_ticks: int = 4) -> int:
    """Smallest ``total_ticks`` at which a write-only driver can be
    non-inert (the greedy budget from tick 0 + the epilogue tail)."""
    greedy_budget = window_ticks + greedy_dominance_ticks + actuation_lag
    return greedy_budget + min_production_ticks + 3 + 1


class ReadWriteSearchDriver:
    """Sequential composition of the two planes for ``read-and-write``:
    the per-chunk read search first (t+2 rule), then the write window
    search over the ticks that remain, observing the settled post-read
    regime.  Sequencing keeps each plane's ``f`` samples free of the other's
    actuation transients; the write phase's budget is computed from its
    ACTUAL start tick, so a long read search gracefully degrades the write
    phase toward greedy / inert."""

    def __init__(self, controller, total_ticks: int, *,
                 write_settle_skip: int = 2, **write_kwargs):
        self._controller = controller
        self._total_ticks = total_ticks
        self._write_kwargs = dict(write_kwargs)
        self._write_settle = int(write_settle_skip)
        self.read = CompressionSearchDriver(controller, total_ticks)
        self.write: Optional[WriteWindowSearchDriver] = None
        self._phase = "read" if not self.read.inert else "write-pending"
        # Factory feasibility: could EITHER plane act on this pipe?  The
        # write estimate is conservative (greedy from the read budget).
        write_probe = WriteWindowSearchDriver(
            controller, total_ticks,
            start_tick=(self.read._budget_ticks if not self.read.inert else 0),
            **write_kwargs)
        self.inert = self.read.inert and write_probe.inert

    def on_view(self, view: ResourceMonitor):
        if self._phase == "read":
            ret = self.read(view)
            if self.read.converged:
                self._phase = "write-pending"
            if ret is not None:
                return ret
            if self._phase == "read":
                return None
        if self._phase == "write-pending":
            self.write = WriteWindowSearchDriver(
                self._controller, self._total_ticks,
                start_tick=max(0, view.last_tick),
                settle_skip=(self._write_settle if not self.read.inert else 0),
                **self._write_kwargs)
            self._phase = "done" if self.write.inert else "write"
        if self._phase == "write":
            return self.write(view)
        return None

    __call__ = on_view

    def settled(self) -> bool:
        """No more picks from either leg: the read leg has settled and
        the write leg, built at the view after it, is inert or
        converged."""
        if self._phase == "write":
            return self.write.settled()
        return self._phase == "done"
