"""The sizing plane's controllers: the chunk budgets of the searches.

Two legs:

* **Gather-window sizing** (``mid_stream``) — the per-gather-iteration
  window search inside one rearrange stream (``_stream_gather``):
  :class:`GatherSizingSession`, a SENSE→MARCH walk.
  - **Per-pipe keying** — its memory models are keyed ``(sf, query,
    pipe_class, instance, codec_epoch)``; the class-keyed model trains on
    every sample but is read only as the cold-stream prior (while the
    stream's own fit is invalid the gate is ``max(class envelope,
    r_worst·c, seed)``).
  - **Sense** — small-δ (64 MB) perturbations, downward first, calibrate
    the memory equation before any upward step; memory only.
  - **March** — steps of Δ (256 MB), escalated ×2 per accepted step up to
    2 GB, each gated by the increment bound ``T_obs(c) + r_worst·step ≤
    budget`` and the envelope against the grow budget, kept only on a
    measured per-byte gain ≥ ε, with a patience of two sub-ε rungs before
    the knee freezes.
  - **Safety re-check** — the shrink-to-safety predicate runs at every
    iteration, and frozen splits into ``converged`` vs ``line_blocked``
    (re-opened when the budget recovers by ``r_worst·Δ``).
* **Tick-pipe sizing** (``per_pipe``, ``mid_stream``) — per-pipe budgets for
  the tick-loop classes (sort/core/agg/filter/project):
  :class:`TickSizingController` (each pipe's start size, the store) and
  :class:`TickJumpSession` (the learned jump: a re-cuttable pipe measures
  its tick time and transient per byte on its event-ordered ticks, doubles
  once, then moves to the size its fitted tick time and its memory bound
  point at, keeping each step only on a measured gain), further below.

The gather sessions actuate through ``sizing_resolver`` overrides, the tick
leg through its decision layer and the pipeline's ``_resection`` (a re-cut
of the ticks not yet issued).  Safety is predictive against the budget
with ``grow_budget_frac`` hysteresis; there is no OOM recovery anywhere
below.

* **The budget** (``_budget_from_stats``) is the whole arena's room: the
  operator space's headroom plus what the cache region could lend right
  now (``lendable_b``, in the arena's own stats), less ``M_abs``, capped
  by the contiguity belt.  The search takes any size the budget vouches
  for: an allocation that does not fit the operator space borrows the
  region's lowest blocks inside the allocator (``memory/README.md``,
  "Lending"), and a lend moves bytes from the lendable to the operator
  space without changing the budget, so it is every level recorded or
  compared across time.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import sizing_resolver
from adaptive.adaptive_driver import GatherIterSample
from adaptive.run_stats import record_sizing_sample
from IO.telemetry import gates
from utility.logger import message_logger

PIPE_CLASS = "rearrange"

MB = 1024 ** 2
GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SizingSearchConfig:
    # rearrange class (gather-window search)
    # The gather leg's base window (None: the static layer's); each gather
    # iteration is synchronized and the shrink-to-safety check runs from the
    # stream's entry.
    c0_b: Optional[int] = 1 * GB      # start window (None → static layer)
    static_b: Optional[int] = None    # optional static-layer override (config)
    c_min_b: int = 64 * MB
    c_max_b: int = 8 * GB             # the validated cap (old dedicated window)
    epsilon: float = 0.02             # min per-step gain
    samples_per_step: int = 1         # full-window samples per size decision
    full_window_frac: float = 0.9     # a sample counts only at ≥ this fill
    # Acceptance objective denominator: "legs" = the three CUDA-event legs,
    # "iter" = the host-bracketed synchronized iteration (sees the fixed
    # per-iteration cost a larger window amortizes; falls back to legs when
    # the sample carries no bracket).
    objective: str = "iter"
    # ── the sense/march step set (the gather leg's) ──
    sense_delta_b: int = 64 * MB      # δ — sense probe step
    march_delta_b: int = 256 * MB     # Δ — march step + fit validity horizon
    # Step escalation: each accepted march step
    # multiplies the NEXT step (Δ, 2Δ, 4Δ, …) up to ``march_step_max_b``.
    # Every escalated step is still gated by both the increment bound and
    # the envelope gate — ``_increment_ok`` takes the actual step as its
    # argument, and beyond the fit validity horizon the envelope gate
    # falls back to the seeded line — so escalation buys walk speed, not
    # risk.  The step resets to Δ whenever the line talks back (clamp,
    # safety shrink, line-blocked reopen, economy tail).  1.0 = a flat
    # march.
    march_escalation: float = 2.0
    march_step_max_b: int = 2 * GB
    # Patience: consecutive sub-ε rungs tolerated before the knee
    # freezes.  A tolerated rung does NOT move the incumbent — the accept
    # test stays anchored at the best measured size, so a later rung must
    # clear the CUMULATIVE ε over the tolerated stretch.  This lets the
    # walk see past one noisy single-sample verdict and past a shallow
    # monotone slope whose per-Δ gain sits under ε (which would freeze the
    # walk one rejected step above its sense window).
    # 1 = the first-rejection knee.  Exploration cost is bounded:
    # at most (patience − 1) windows run above the incumbent before the
    # revert, and a rejection de-escalates the escalated step to Δ.
    march_patience: int = 2
    sense_up_probes: int = 2          # optional upward δ-probes (0..2)
    # safety.  The contiguity belt gates every growth against
    # min(headroom, largest_free_run): a multi-GB window is one contiguous
    # alloc that scattered headroom cannot serve, however much is free in
    # total.
    contiguity_gate: bool = True
    # Hysteresis: growth may only plan its predicted transient into this
    # fraction of the safety budget; the shrink-to-safety step keeps firing
    # at the FULL budget.  The band absorbs envelope-pad breathing and the
    # bounded O(Δ) damage of a regime change inside one increment.
    grow_budget_frac: float = 0.9
    M_abs_b: int = 512 * MB           # absolute arena headroom kept free
    pad_min_b: int = 256 * MB         # envelope floor
    seed_slope: float = 2.0           # analytic seed for rearrange
    # tick-pipe classes (sort/core/agg/filter/project): the learned jump
    # (``TickJumpSession``).  Every pipe starts at c0; one that may be re-cut
    # mid-run measures ``tick_samples`` chunks per size it runs, doubles
    # once, then moves where its fitted tick time and its transient per
    # byte point, keeping each step only on a measured gain of ``epsilon``.
    tick_c0_b: Optional[int] = None   # start budget (None → --chunk_size_mb)
    tick_c_min_b: int = 64 * MB
    tick_c_max_b: int = 4 * GB
    tick_samples: int = 6             # measured chunks per size

    def validate(self) -> "SizingSearchConfig":
        assert 0 < self.c_min_b <= self.c_max_b, "bad [c_min, c_max]"
        if self.c0_b is not None:
            assert self.c_min_b <= self.c0_b <= self.c_max_b, (
                f"c0 {self.c0_b} outside [{self.c_min_b}, {self.c_max_b}]")
        # epsilon < 0 = always-accept (probe/forced-walk knob; the real
        # acceptance uses the positive default).
        assert self.epsilon > -1.0
        assert self.samples_per_step >= 1
        assert self.objective in ("legs", "iter"), (
            f"objective {self.objective!r} not in ('legs', 'iter')")
        assert 0.0 < self.grow_budget_frac <= 1.0, (
            f"grow_budget_frac {self.grow_budget_frac} outside (0, 1]")
        assert 0 < self.sense_delta_b <= self.march_delta_b, (
            f"sense δ {self.sense_delta_b} outside (0, Δ={self.march_delta_b}]")
        assert self.march_escalation >= 1.0, (
            f"march_escalation {self.march_escalation} < 1.0")
        assert self.march_step_max_b > 0
        assert self.march_patience >= 1, (
            f"march_patience {self.march_patience} < 1")
        assert 0 <= self.sense_up_probes <= 2
        assert 0 < self.tick_c_min_b <= self.tick_c_max_b, "bad tick bounds"
        if self.tick_c0_b is not None:
            assert self.tick_c_min_b <= self.tick_c0_b <= self.tick_c_max_b, (
                f"tick_c0 {self.tick_c0_b} outside bounds")
        assert self.tick_samples >= 2, (
            f"tick_samples {self.tick_samples} < 2")
        return self


def load_config(path: Optional[str]) -> SizingSearchConfig:
    """Build a config from the ``--sizing_config`` JSON (all keys optional).

    Schema::

        {"rearrange": {"c0_mb": 1024, "static_mb": null, "c_min_mb": 64,
                       "c_max_mb": 8192, "epsilon": 0.02,
                       "samples_per_step": 1, "objective": "iter"},
         "march":     {"delta_mb": 256, "escalation": 2.0,
                       "step_max_mb": 2048, "patience": 2,
                       "sense_delta_mb": 64, "up_probes": 2},
         "safety":    {"M_abs_mb": 512, "pad_min_mb": 256, "seed_slope": 2.0},
         "tick":      {"c0_mb": null, "c_min_mb": 64, "c_max_mb": 4096,
                       "samples": 6}}

    ``epsilon`` and ``safety`` serve both legs; ``march`` is the gather
    leg's.  Unknown keys are ignored, so older configs (the retired tick
    walk's ``k_seg``/``k_sense``/``seed_slope``/``jump_mb`` …, the retired
    slot governor's ``governor`` section) still load.
    """
    cfg = SizingSearchConfig()
    if path is None:
        return cfg.validate()
    with open(path, "r") as f:
        raw = json.load(f)
    r = raw.get("rearrange", {})
    if r.get("c0_mb") is not None:
        cfg.c0_b = int(r["c0_mb"] * MB)
    if r.get("static_mb") is not None:
        cfg.static_b = int(r["static_mb"] * MB)
    cfg.c_min_b = int(r.get("c_min_mb", cfg.c_min_b / MB) * MB)
    cfg.c_max_b = int(r.get("c_max_mb", cfg.c_max_b / MB) * MB)
    cfg.epsilon = float(r.get("epsilon", cfg.epsilon))
    cfg.samples_per_step = int(r.get("samples_per_step", cfg.samples_per_step))
    cfg.objective = str(r.get("objective", cfg.objective))
    m = raw.get("march", {})
    cfg.march_delta_b = int(m.get("delta_mb", cfg.march_delta_b / MB) * MB)
    cfg.march_escalation = float(m.get("escalation", cfg.march_escalation))
    cfg.march_step_max_b = int(m.get("step_max_mb",
                                     cfg.march_step_max_b / MB) * MB)
    cfg.march_patience = int(m.get("patience", cfg.march_patience))
    cfg.sense_delta_b = int(m.get("sense_delta_mb",
                                  cfg.sense_delta_b / MB) * MB)
    cfg.sense_up_probes = int(m.get("up_probes", cfg.sense_up_probes))
    s = raw.get("safety", {})
    cfg.contiguity_gate = bool(s.get("contiguity_gate", cfg.contiguity_gate))
    cfg.grow_budget_frac = float(s.get("grow_budget_frac",
                                       cfg.grow_budget_frac))
    cfg.M_abs_b = int(s.get("M_abs_mb", cfg.M_abs_b / MB) * MB)
    cfg.pad_min_b = int(s.get("pad_min_mb", cfg.pad_min_b / MB) * MB)
    cfg.seed_slope = float(s.get("seed_slope", cfg.seed_slope))
    t = raw.get("tick", {})
    if t.get("c0_mb") is not None:
        cfg.tick_c0_b = int(t["c0_mb"] * MB)
    cfg.tick_c_min_b = int(t.get("c_min_mb", cfg.tick_c_min_b / MB) * MB)
    cfg.tick_c_max_b = int(t.get("c_max_mb", cfg.tick_c_max_b / MB) * MB)
    cfg.tick_samples = int(t.get("samples", cfg.tick_samples))
    return cfg.validate()


# ---------------------------------------------------------------------------
# Shared gate arithmetic (cold-pipe prior + increment bound)
# ---------------------------------------------------------------------------

class _PipeGate:
    """The per-pipe envelope gate: the pipe fit when valid, else
    ``max(class envelope, r_worst·c)``.  ``r_worst`` prefers the pipe's
    own ratio, falls back to the class's."""

    def __init__(self, pipe_model, class_model):
        self.pipe = pipe_model
        self.klass = class_model

    def r_worst(self) -> float:
        return self.pipe.r_worst if self.pipe.r_worst > 0.0 \
            else self.klass.r_worst

    def gate_b(self, c_b: int) -> Tuple[int, bool]:
        """(envelope prediction, seeded?) that gates actuating ``c_b``."""
        p = self.pipe.predict(c_b)
        if not p.seeded:
            return p.transient_b, False
        cold = self.klass.predict(c_b).transient_b
        r = self.r_worst()
        return max(cold, int(r * c_b)), True

    def invert(self, budget_b: int) -> int:
        m = self.pipe if self.pipe.has_fit else self.klass
        return m.invert(budget_b)

    def t_obs(self, c_b: int) -> int:
        """``T_obs``: the largest transient this pipe actually
        observed at ``c_b`` — falls back to the gate prediction when the
        size was never observed (conservative stand-in)."""
        t = self.pipe.max_transient_at(c_b)
        return t if t > 0 else self.gate_b(c_b)[0]

    def add_sample(self, c_b: int, transient_b: int) -> None:
        """Every sample enters both the pipe fit and the class prior."""
        self.pipe.add_sample(c_b, transient_b)
        self.klass.add_sample(c_b, transient_b)


def _budget_from_stats(stats: dict, cfg: SizingSearchConfig) -> int:
    """The budget, the whole arena's room: the operator space's headroom
    plus what the cache region can lend right now (``lendable_b``, 0 when
    the stats carry none), less ``M_abs``; with the contiguity belt armed,
    capped at the larger of the largest free run and the run that ends at
    the boundary (``top_free_run_b``) plus the lendable bytes, since lent
    blocks are the region's lowest and join that run.  A lend moves bytes
    from the lendable to the operator space without changing it, so every
    decision, and every level recorded and compared across time, reads
    it."""
    lend = int(stats.get("lendable_b") or 0)
    h = stats["capacity_b"] + lend - stats["allocated_b"] - cfg.M_abs_b
    if cfg.contiguity_gate:
        run = stats.get("largest_free_run_b")
        if run is not None:
            top = int(stats.get("top_free_run_b") or 0)
            h = min(h, max(run, top + lend))
    return h


# ---------------------------------------------------------------------------
# Gather-window sizing: the per-stream session (sense → additive march)
# ---------------------------------------------------------------------------

def whole_quanta(c_b: int, quantum_b: int, floor_b: int = 0,
                 cap_b: Optional[int] = None) -> int:
    """The budget a pipe or a gather window cut in quanta of ``quantum_b``
    bytes realizes at ``c_b``: rounded down to whole quanta, or, when that
    falls under ``floor_b`` (a sizing search's class floor; 0 without one),
    up to the next whole quantum, unless that passes ``cap_b``.  Unchanged
    without a quantum (``quantum_b`` 0) or under one (cut by rows).  The
    one rounding the sessions, the tick cutter and ``window_rows`` share."""
    if not quantum_b or c_b < quantum_b:
        return c_b
    whole = c_b // quantum_b * quantum_b
    if whole < floor_b and whole < c_b:
        up = whole + quantum_b
        if cap_b is None or up <= cap_b:
            return up
    return whole


def _quantum_step(step_b: int, quantum_b: int) -> int:
    """A walk step of ``step_b`` in whole quanta, at least one."""
    if quantum_b:
        return max(quantum_b, step_b // quantum_b * quantum_b)
    return step_b


def window_rows(c_b: int, row_b: int, quantum_rows: int = 0,
                floor_b: int = 0) -> int:
    """Rows of a window of ``c_b`` bytes at ``row_b`` bytes per row, in
    whole quanta of ``quantum_rows`` (``whole_quanta``: rounded down, or up
    when down falls under ``floor_b``; none when 0, or when one quantum
    does not fit); at least one row."""
    if quantum_rows:
        c_b = whole_quanta(c_b, quantum_rows * row_b, floor_b)
    return max(1, c_b // row_b)


class GatherSizingSession:
    """One rearrange column stream's window walk.

    The pipeline resolves its window per iteration and calls
    :meth:`observe` after each; the session writes resolver overrides
    *between* iterations only.

    State machine::

        SENSE  (probes: c₀−δ, c₀ [, +δ, +2δ] — memory fit only)
          → MARCH   (c += step; step escalates Δ, 2Δ, 4Δ, … on consecutive
                     accepts up to ``march_step_max_b`` and de-escalates
                     whenever the line binds; increment bound on the
                     actual step; ε-acceptance on the iter objective
                     anchored at the incumbent; knee-stop after
                     ``march_patience`` consecutive sub-ε rungs)
          → FROZEN  (converged | line_blocked | breach)

    Shrink-to-safety armed in every state, at every iteration.
    """

    def __init__(self, *, instance: str, row_bytes: int, row_overhead: int,
                 total_rows: int, codec_epoch: str, sf, query,
                 cfg: SizingSearchConfig, model, class_model,
                 arena_stats_fn: Callable[[], dict],
                 run_store: Dict[Tuple, int],
                 quantum_rows: int = 0):
        self.instance = str(instance)
        self.row_bytes = int(row_bytes)
        self.row_overhead = int(row_overhead)
        # The output's rows per record (0: none): windows are whole
        # quanta of them, and the walk steps in quanta (``_clamp``,
        # ``_step``).
        self.quantum_rows = int(quantum_rows)
        self.quantum_b = self.quantum_rows * (self.row_bytes
                                              + self.row_overhead)
        self.total_rows = int(total_rows)
        self.codec_epoch = str(codec_epoch)
        self.cfg = cfg
        self.model = model                    # per-pipe fit
        self.gate = _PipeGate(model, class_model)
        self.arena_stats = arena_stats_fn
        self.run_store = run_store
        self.store_key = (sf, query, self.instance, self.codec_epoch)
        self.sf, self.query = sf, query

        self.rows_done = 0
        self.size_epoch = 0
        self.breaches = 0
        self.iterations = 0
        self.state = "sense"
        self.frozen_reason: Optional[str] = None
        self._blocked_budget_b: Optional[int] = None
        self._cur_tp_samples: List[float] = []
        self._t_obs_cur = 0              # max transient seen at c_cur
        self._march_step_b = cfg.march_delta_b   # current march step
        self._reject_streak = 0                  # consecutive sub-ε rungs

        static_b = sizing_resolver.resolve(PIPE_CLASS, self.instance).chunk_b
        delta = cfg.sense_delta_b
        warm = self.run_store.get(self.store_key)
        if warm is not None:
            # A same-key warm-frozen stream re-runs the down-probe only,
            # refreshing the process-local fit, then freezes converged at
            # the stored window.
            self._warm = True
            self._sense_base = self._clamp(warm)
            self._sense_plan = [self._sense_base]
            c_start = self._clamp(max(self.cfg.c_min_b,
                                      self._sense_base - self._step(delta)))
        else:
            self._warm = False
            base = cfg.c0_b if cfg.c0_b is not None else static_b
            self._sense_base = self._clamp(base)
            # Downward first, then upward δ-probes (gated).
            self._sense_plan = [self._sense_base]
            for k in range(1, cfg.sense_up_probes + 1):
                self._sense_plan.append(self._sense_base
                                        + k * self._step(delta))
            c_start = self._clamp(max(self.cfg.c_min_b,
                                      self._sense_base - self._step(delta)))
        self.c_cur = c_start
        self.incumbent_c = c_start
        self.incumbent_tp: Optional[float] = None
        # Entry safety pre-check: if the line already sits under c_start,
        # walk down before the first gather.
        self._shrink_to_safety(reason="entry")
        sizing_resolver.set_override(PIPE_CLASS, self.instance, self.c_cur)
        message_logger().info(
            "[sizing] Q%s %s: stream start c=%d MB (%s; base %d MB, static "
            "%d MB, codec=%s, rows=%d)", self.query, self.instance,
            self.c_cur // MB,
            "warm-refresh" if self._warm else "sense",
            self._sense_base // MB, static_b // MB, self.codec_epoch,
            self.total_rows)

    # -- geometry helpers --------------------------------------------------

    def _clamp(self, c: int) -> int:
        """``c`` within the bounds, in whole quanta (``whole_quanta``)."""
        return whole_quanta(max(self.cfg.c_min_b, min(self.cfg.c_max_b,
                                                      int(c))),
                            self.quantum_b, self.cfg.c_min_b,
                            self.cfg.c_max_b)

    def _step(self, step_b: int) -> int:
        """A step of ``step_b``, in whole quanta (at least one)."""
        return _quantum_step(step_b, self.quantum_b)

    def _window_rows(self, c_b: int) -> int:
        return window_rows(c_b, self.row_bytes + self.row_overhead,
                           self.quantum_rows, self.cfg.c_min_b)

    # -- safety ------------------------------------------------------------
    # The budget (``_budget_from_stats``, the arena's room with the cache
    # region's lendable bytes) decides every size and is every level
    # recorded or compared.

    def _budget(self, stats: dict) -> int:
        """The budget."""
        return _budget_from_stats(stats, self.cfg)

    def _is_safe(self, c_b: int, budget: int) -> bool:
        """The shrink-to-safety predicate — the FULL ``budget``."""
        return self.gate.gate_b(c_b)[0] <= budget

    def _grow_budget(self, budget: int) -> int:
        return int(budget * self.cfg.grow_budget_frac)

    def _is_growable(self, c_b: int, budget: int) -> bool:
        return self.gate.gate_b(c_b)[0] <= self._grow_budget(budget)

    def _increment_ok(self, step_b: int, budget: int) -> bool:
        """The increment bound: even if every byte of the increment
        expands at the worst ratio this pipe ever exhibited, the step
        cannot cross the fatal line.  Seed-slope stand-in when no ratio was
        ever observed."""
        r = self.gate.r_worst()
        if r <= 0.0:
            r = self.cfg.seed_slope
        t_obs = self._t_obs_cur if self._t_obs_cur > 0 \
            else self.gate.t_obs(self.c_cur)
        return t_obs + int(r * step_b) <= budget

    def _grow_gates_ok(self, c_next: int, budget: int) -> bool:
        """Both growth gates (increment bound + envelope) for ``c_next``."""
        return (self._increment_ok(c_next - self.c_cur, budget)
                and self._is_growable(c_next, budget))

    def _freeze(self, reason: str, stats: Optional[dict] = None) -> None:
        self.state = "frozen"
        self.frozen_reason = reason
        if reason == "line_blocked":
            if stats is None:
                stats = self.arena_stats()
            self._blocked_budget_b = self._budget(stats)

    def _safe_size(self, budget: int) -> int:
        """The largest window up to the current one whose envelope fits
        ``budget``: model inversion, then Δ steps down (to c_min)."""
        c = self._clamp(min(self.gate.invert(budget), self.c_cur))
        while c > self.cfg.c_min_b and not self._is_safe(c, budget):
            c_next = self._clamp(c - self._step(self.cfg.march_delta_b))
            if c_next == c:
                break
            c = c_next
        return c

    def _shrink_to_safety(self, reason: str) -> bool:
        """Shrink to safety: if the line moved under the current window,
        the largest window the budget vouches for.  Returns True when a
        shrink actuated."""
        stats = self.arena_stats()
        budget = self._budget(stats)
        if self._is_safe(self.c_cur, budget):
            return False
        c = self._safe_size(budget)
        if c >= self.c_cur:
            return False
        message_logger().warning(
            "[sizing] Q%s %s: safety budget moved under c=%d MB (%s) — "
            "shrink to %d MB (live=%d MB, budget=%d MB)",
            self.query, self.instance, self.c_cur // MB, reason, c // MB,
            stats["allocated_b"] // MB, budget // MB)
        self._actuate(c)
        # Re-anchor acceptance at the shrunk size; a safety shrink is not a
        # measured verdict, so the incumbent throughput resets (and the
        # escalated march step resets to Δ with it).
        self.incumbent_c = c
        self.incumbent_tp = None
        self._march_step_b = self.cfg.march_delta_b
        self._reject_streak = 0
        return True

    # -- the SENSE → MARCH state machine -----------------------------------

    def observe(self, s: GatherIterSample) -> None:
        self.iterations += 1
        self.rows_done += s.rows
        transient = max(0, s.iter_peak_b - s.live_at_entry_b)
        # Sample attribution: an exact full window is attributed to
        # the BUDGET (window_b); a partial tail chunk to what actually ran.
        window_rows_full = self._window_rows(s.window_b)
        exact_full = s.rows == window_rows_full
        c_attr = (s.window_b if exact_full
                  else s.rows * (self.row_bytes + self.row_overhead))
        gate_b, seeded = self.gate.gate_b(c_attr)
        self.gate.add_sample(c_attr, transient)
        if s.window_b == self.c_cur:
            self._t_obs_cur = max(self._t_obs_cur, transient)
        # The plane value marks a gather-window sample in the perf report
        # (TQP_EXPERIMENT_TELEMETRY's sizing_model section).
        if gates.EXPERIMENT:
            record_sizing_sample(
                "SZ-B", self.instance, c_attr, gate_b, transient,
                seeded, transient > gate_b)
        if transient > gate_b:
            # Envelope breach — one step from fatal.  Freeze at the
            # last-good size (the incumbent); the breach sample is already
            # in the fit (envelope self-heals for the next stream).
            self.breaches += 1
            message_logger().error(
                "[sizing] Q%s %s: ENVELOPE BREACH at c=%d MB (%s gate) — "
                "transient %d MB > predicted %d MB; freezing at %d MB",
                self.query, self.instance, c_attr // MB,
                "seeded" if seeded else "fitted", transient // MB,
                gate_b // MB, self.incumbent_c // MB)
            if self.c_cur != self.incumbent_c:
                self._actuate(self.incumbent_c)
            self._freeze("breach")
            return

        # Shrink-to-safety watch every iteration, every state.
        if self._shrink_to_safety(reason="mid-stream"):
            return

        # A line-blocked walk re-opens when the budget recovers by
        # ≥ r_worst·Δ over the level that blocked it.
        if (self.state == "frozen" and self.frozen_reason == "line_blocked"
                and self._blocked_budget_b is not None):
            stats = self.arena_stats()
            r = self.gate.r_worst() or self.cfg.seed_slope
            if self._budget(stats) >= (self._blocked_budget_b
                                       + int(r * self.cfg.march_delta_b)):
                message_logger().info(
                    "[sizing] Q%s %s: budget recovered (%d MB) — line-"
                    "blocked walk re-opens at c=%d MB", self.query,
                    self.instance, self._budget(stats) // MB,
                    self.c_cur // MB)
                self.state = "growing"
                self.frozen_reason = None
                self._blocked_budget_b = None
                self.incumbent_tp = None
                self._march_step_b = self.cfg.march_delta_b
                self._reject_streak = 0

        if self.state == "sense":
            # Sense is memory-only — the sample entered the fit above; no
            # time comparison is made at δ granularity.
            self.incumbent_c = self.c_cur     # last verified sensed size
            self._advance_sense()
            return

        # MARCH: throughput sample — like-for-like only: full windows of
        # the currently actuated size.
        denom_ms = s.h2d_ms + s.compress_ms + s.d2h_ms
        if self.cfg.objective == "iter" and s.iter_ms > 0:
            denom_ms = s.iter_ms
        is_full = (s.rows >= self.cfg.full_window_frac * window_rows_full)
        if (self.state == "growing" and is_full and denom_ms > 0
                and s.window_b == self.c_cur):
            self._cur_tp_samples.append(s.chunk_bytes / denom_ms)
        if (self.state == "growing"
                and len(self._cur_tp_samples) >= self.cfg.samples_per_step):
            self._evaluate(statistics.median(self._cur_tp_samples))

    def _advance_sense(self) -> None:
        """Walk the sense plan: sizes ≤ the sense base are unconditional
        (the base is the pre-checked static-posture size); sizes above it
        are upward δ-probes gated by the just-fitted envelope + the
        increment bound at δ."""
        while self._sense_plan:
            c_next = self._clamp(self._sense_plan[0])
            if c_next == self.c_cur:
                self._sense_plan.pop(0)
                continue
            budget = self._budget(self.arena_stats())
            if c_next <= self._sense_base:
                # Static-posture probe — but if the entry pre-check already
                # shrank below the base, the line does not vouch for it:
                # the shrink-to-safety predicate gates the return too
                # (conservative beyond off-mode).
                if not self._is_safe(c_next, budget):
                    self._sense_plan.clear()
                    break
                self._sense_plan.pop(0)
                self._actuate(c_next)
                return
            # Upward δ-probe: requires a fitted pipe envelope (the
            # fitted-before-upward invariant) and both growth gates.
            if (self.model.has_fit
                    and self._increment_ok(c_next - self.c_cur, budget)
                    and self._is_growable(c_next, budget)):
                self._sense_plan.pop(0)
                self._actuate(c_next)
                return
            self._sense_plan.clear()      # gate failed — end sensing
        # Plan exhausted → incumbent = best (largest verified) sensed
        # size; enter MARCH (or freeze for warm-refresh streams).
        if self._warm:
            self._freeze("converged")
            return
        self.state = "growing"
        self.incumbent_tp = None

    def _evaluate(self, tp_cur: float) -> None:
        eps = self.cfg.epsilon
        if self.incumbent_tp is None or self.c_cur == self.incumbent_c:
            # First measurement at the anchor — adopt and propose growth.
            self.incumbent_c = self.c_cur
            self.incumbent_tp = tp_cur
            self._reject_streak = 0
            self._try_grow()
        elif tp_cur >= self.incumbent_tp * (1.0 + eps):
            # Accept: the grown window measurably wins.  The accept
            # escalates the NEXT step (Δ, 2Δ, 4Δ, … up to the cap).
            self._march_step_b = min(
                int(self._march_step_b * self.cfg.march_escalation),
                max(self.cfg.march_step_max_b, self.cfg.march_delta_b))
            message_logger().info(
                "[sizing] Q%s %s: accept c=%d MB (tp %.0f vs %.0f MB/s, "
                "+%.1f%%; next step %d MB)", self.query, self.instance,
                self.c_cur // MB, tp_cur / 1e3, self.incumbent_tp / 1e3,
                100.0 * (tp_cur / self.incumbent_tp - 1.0),
                self._march_step_b // MB)
            self.incumbent_c = self.c_cur
            self.incumbent_tp = tp_cur
            self._reject_streak = 0
            self._try_grow()
        else:
            # Sub-ε rung.  Tolerate up to (patience − 1) of them — the
            # incumbent anchor does not move, so the next rung is judged on
            # the cumulative gain — then revert + freeze at the knee.
            self._reject_streak += 1
            self._march_step_b = self.cfg.march_delta_b   # streak broken
            if self._reject_streak < self.cfg.march_patience:
                message_logger().info(
                    "[sizing] Q%s %s: tolerate c=%d MB (tp %.0f vs %.0f "
                    "MB/s < +%.1f%%; patience %d/%d)", self.query,
                    self.instance, self.c_cur // MB, tp_cur / 1e3,
                    self.incumbent_tp / 1e3, 100.0 * eps,
                    self._reject_streak, self.cfg.march_patience)
                self._try_grow()
                return
            # Revert + freeze: the knee — growth stopped paying.
            message_logger().info(
                "[sizing] Q%s %s: revert c=%d->%d MB (tp %.0f vs %.0f MB/s "
                "< +%.1f%%) — converged", self.query, self.instance,
                self.c_cur // MB, self.incumbent_c // MB, tp_cur / 1e3,
                self.incumbent_tp / 1e3, 100.0 * eps)
            self._actuate(self.incumbent_c)
            self._freeze("converged")

    def _settle_to_incumbent(self) -> None:
        """A freeze that ends the walk must not strand the stream on a
        tolerated sub-ε rung — return to the best measured window."""
        if self.c_cur != self.incumbent_c:
            self._actuate(self.incumbent_c)

    def _try_grow(self) -> None:
        cfg = self.cfg
        c_next = self._clamp(self.c_cur + self._step(self._march_step_b))
        if c_next <= self.c_cur:
            self._settle_to_incumbent()
            self._freeze("converged")     # at cap
            return
        remaining = self.total_rows - self.rows_done
        if remaining < self._window_rows(c_next) * cfg.samples_per_step:
            # Economy gate: not enough stream left to measure the grown
            # size, so the stream finishes at the incumbent — but an
            # escalated step first falls back to the base Δ: the tail may
            # still host one plain step.
            c_small = self._clamp(self.c_cur + self._step(cfg.march_delta_b))
            if (c_small < c_next and remaining
                    >= self._window_rows(c_small) * cfg.samples_per_step):
                self._march_step_b = cfg.march_delta_b
                c_next = c_small
            else:
                self._settle_to_incumbent()
                self._freeze("converged")
                return
        if not self.model.has_fit:
            # Invariant: no upward actuation on a seed-only pipe gate.
            # Reached only when sensing was degenerate (one distinct size).
            message_logger().info(
                "[sizing] Q%s %s: no per-pipe fit after sense — growth "
                "withheld at %d MB", self.query, self.instance,
                self.c_cur // MB)
            self._settle_to_incumbent()
            self._freeze("converged")
            return
        stats = self.arena_stats()
        budget = self._budget(stats)
        if self._grow_gates_ok(c_next, budget):
            self._actuate(c_next)
            return
        # Pressure ladder: the largest window below the step that passes
        # both gates (a clamp).  A clamp de-escalates the march step back to
        # Δ — escalation resumes only from fresh accepts.
        c_got = self._clamp_growth(c_next, budget)
        if c_got > self.c_cur:
            self._march_step_b = cfg.march_delta_b
            message_logger().info(
                "[sizing] Q%s %s: clamp growth %d->%d MB (line-limited)",
                self.query, self.instance, c_next // MB, c_got // MB)
            self._actuate(c_got)
        else:
            # Cannot grow safely — hold the incumbent (the line-blocked
            # level is the total budget).
            message_logger().info(
                "[sizing] Q%s %s: growth to %d MB blocked by the fatal "
                "line — line-blocked at %d MB", self.query, self.instance,
                c_next // MB, self.c_cur // MB)
            self._settle_to_incumbent()
            self._freeze("line_blocked", stats)

    def _clamp_growth(self, c_hi: int, budget: int) -> int:
        """The largest window up to ``c_hi`` that ``budget`` vouches for:
        model inversion of its growth budget, then Δ steps down while either
        growth gate fails (the piecewise regimes), not below the current
        window."""
        c = self._clamp(min(self.gate.invert(self._grow_budget(budget)), c_hi))
        while c > self.c_cur and not self._grow_gates_ok(c, budget):
            c_step = self._clamp(c - self._step(self.cfg.march_delta_b))
            if c_step == c:
                break
            c = c_step
        return c

    def _actuate(self, c_new: int) -> None:
        self.c_cur = c_new
        self.size_epoch += 1
        self._cur_tp_samples = []
        self._t_obs_cur = self.model.max_transient_at(c_new)
        sizing_resolver.set_override(PIPE_CLASS, self.instance, c_new)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Stream end: persist the converged window for same-key streams,
        drop the override."""
        # The last VERIFIED size: after any freeze incumbent_c == c_cur; a
        # stream that ended mid-candidate keeps its incumbent.
        converged = self.incumbent_c
        if self.breaches == 0 and self.iterations > 0:
            self.run_store[self.store_key] = converged
        sizing_resolver.clear_override(PIPE_CLASS, self.instance)
        message_logger().info(
            "[sizing] Q%s %s: stream end — %d iterations, c=%d MB (%s%s), "
            "size_epoch=%d, breaches=%d", self.query,
            self.instance, self.iterations, converged // MB, self.state,
            f"/{self.frozen_reason}" if self.frozen_reason else "",
            self.size_epoch, self.breaches)


# ---------------------------------------------------------------------------
# Tick-pipe sizing: the learned jump
# ---------------------------------------------------------------------------

# Re-cuttable CLASS DEFAULTS: only pipes whose resolved budget has NO
# consumer outside the tick geometry may be re-cut mid-run.  ``agg``
# defaults to excluded — the two-pass PASS-2 and the single-pass family
# read the construction-time budget into out-of-pipeline contracts (the
# ``stages − 2`` final-chunk countdowns captured in the operator closures:
# aggregate.py ``global_stage_countdown`` / ``ticks_remaining``), which a
# mid-run re-split breaks (it can empty the result).  ``core`` is
# structurally excluded (``div`` geometry).  A pipe overrides its class
# with the ``sectionable`` flag its constructor passes down: aggregation
# pass 1 opts in (its chunk has no consumer outside the pipe; B and each
# group's bucket are fixed before it), and the contract-bearing pipes pass
# False next to their contracts.
SECTIONABLE_CLASSES = ("sort", "filter", "project")


def _fit_line(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Least squares ``y = F + t·x`` over ``points`` (``(x, y)``, two or
    more distinct ``x``): ``(F, t)``."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    t = (sum((x - mx) * (y - my) for x, y in points) / sxx) if sxx else 0.0
    return my - t * mx, t


class TickSizingController:
    """Per-pipe chunk budgets for the tick-loop classes (sort/core/agg/
    filter/project), the ``--sizing_search per_pipe`` leg.

    ``decide`` (the resolver's decision layer, once per pipe construction)
    gives a pipe its start size: ``c0`` the first time, else the size its
    last occurrence settled on, lowered (never under ``c0``) until the
    transient per byte it measured there fits the budget now.  ``session``
    gives a pipe that may be re-cut mid-run (uniform geometry; a
    re-cuttable class, or ``sectionable``) its :class:`TickJumpSession`,
    unless an earlier occurrence fixed its size.  Every other pipe (join
    cores, aggregation pass 2, the scalar aggregation) runs its start size,
    ``c0``, as with the search off.

    The store keeps per pipe ``(sf, query, class, instance, codec_epoch)``:
    ``c``, the size it settled on; ``fixed``; and ``rho``, the transient per
    input byte measured at the largest size it ran.
    """

    def __init__(self, cfg: SizingSearchConfig, arena_stats_fn,
                 codec_epoch: str = "static",
                 store: Optional[Dict[Tuple, Dict]] = None):
        self.cfg = cfg
        self.arena_stats = arena_stats_fn
        self.codec_epoch = str(codec_epoch)
        self.store: Dict[Tuple, Dict] = store if store is not None else {}
        self.decisions = 0
        self.clamps = 0
        self.jumps = 0
        self.accepts = 0
        self.reverts = 0
        self.breaches = 0
        # The query's sessions, one entry each (``take_report``).
        self._report: List[str] = []

    def _key(self, sf, query, pipe_class: str, instance: str) -> Tuple:
        return (sf, query, pipe_class, str(instance), self.codec_epoch)

    def _c0(self) -> int:
        import constants
        return (self.cfg.tick_c0_b if self.cfg.tick_c0_b is not None
                else constants.get_chunk_size_b())

    def _clamp(self, c: int) -> int:
        return max(self.cfg.tick_c_min_b, min(self.cfg.tick_c_max_b, int(c)))

    def room(self, stats: Dict) -> int:
        """What a pipe starting now may plan to hold above what is live: the
        budget (``_budget_from_stats``) without the cache region's lendable
        bytes.  A lend is all or nothing and gives nothing while the
        granules it needs are pinned, so what the region could lend at a
        pipe's start may be gone when its larger ticks allocate, and a pipe
        that planned its transients on it runs out of memory; the region
        stays the fallback for what no plan foresaw."""
        return _budget_from_stats({**stats, "lendable_b": 0}, self.cfg)

    def decide(self, pipe_class: str, instance: str) -> Optional[int]:
        if pipe_class not in sizing_resolver.TICK_CLASSES:
            return None            # rearrange stays with the gather sessions
        from adaptive.adaptive_driver import sizing_enabled
        if not sizing_enabled():
            return None            # warmup runs static budgets
        import constants
        key = self._key(constants.get_query_sf(), constants.get_query_id(),
                        pipe_class, instance)
        c0 = self._clamp(self._c0())
        st = self.store.get(key)
        c = c0 if st is None else self._clamp(st["c"])
        if c > c0 and st.get("rho"):
            limit = self.cfg.grow_budget_frac * self.room(self.arena_stats())
            fit = self._clamp(limit / st["rho"])
            if fit < c:
                self.clamps += 1
                message_logger().info(
                    "[sizing] Q%s %s:%s start %d -> %d MB (%.2f B/B over "
                    "%d MB of room)", key[1], pipe_class, instance, c // MB,
                    max(c0, fit) // MB, st["rho"], int(limit) // MB)
                c = max(c0, fit)
        self.decisions += 1
        return c

    def session(self, pipe_name: str, pipe_class: str, instance: str,
                chunk_b: int, n_ticks: int, input_b: int,
                sectionable: Optional[bool] = None,
                quantum_b: int = 0) -> Optional["TickJumpSession"]:
        """The factory ``VortexPipeline._exchange_ticks`` consults for a
        uniform-geometry sized pipe on event-ordered ticks (it feeds the
        session only ticks that run on events)."""
        from adaptive.adaptive_driver import sizing_enabled
        if not sizing_enabled() or sectionable is False:
            return None
        if sectionable is None and pipe_class not in SECTIONABLE_CLASSES:
            return None
        if n_ticks < self.cfg.tick_samples + 3:
            return None            # too short to measure its start size
        import constants
        key = self._key(constants.get_query_sf(), constants.get_query_id(),
                        pipe_class, instance)
        st = self.store.get(key)
        if st is None:
            st = self.store[key] = {"c": int(chunk_b), "fixed": False,
                                    "rho": None}
        if st["fixed"]:
            return None
        stats = self.arena_stats()
        return TickJumpSession(self, key, st, pipe_name, int(chunk_b),
                               int(input_b), self.room(stats),
                               int(stats["allocated_b"]), int(quantum_b))

    def note(self, line: str) -> None:
        self._report.append(line)

    def take_report(self) -> List[str]:
        """The query's sessions since the last call, one entry each."""
        report, self._report = self._report, []
        return report


class TickJumpSession:
    """One pipe's learned jump, on its event-ordered ticks.

    At each size the pipe runs, the session skips the first chunk (the
    switch) and measures ``k`` (``tick_samples``) chunks:

    * ``q``, the tick time: the gap between the chunk's compute end and the
      previous chunk's on the GPU timeline (the compute events every
      event-ordered tick records), so whatever bounds the pipe, host or
      GPU, plus the chunk's readback: its outputs' stored bytes times their
      later crossings (home and each consumer read) at the probed PCIe rate
      (``resource_monitor.PCIE_B_PER_MS``), the price the write leg puts on
      a RAW output;
    * ``b``, the chunk's input bytes;
    * ``T``, the transient: the arena's peak while the chunk computes, less
      what was live when the pipe started.

    From those it picks the next size:

    * memory: ``rho = T/b`` at the largest size measured bounds the
      transient of any larger size (a transient linear in the chunk with a
      non-negative intercept falls in ``T/b``), so a size ``c`` fits when
      ``rho·c ≤ grow_budget_frac·room``, ``room`` the budget when the pipe
      started, without the cache region's lendable bytes (``room``);
    * time: the pipe's time at size ``x`` is ``pt(x) = q(x)·(D/x + 1)``,
      ``D`` the pipe's input bytes and the 1 the drain tick, which a bigger
      tick makes longer: the time the pipe would take run whole at ``x``,
      which is what its store keeps for the next occurrence (the bytes left
      would judge a step measured near the pipe's end on one tick);
    * the first step (the walk) doubles the size.  With two sizes measured,
      the tick time is fitted as ``q(x) = F + t·x`` (least squares over the
      sizes measured) and the next size is the ``pt`` minimizer ``x* =
      sqrt(D·F/t)`` (the memory bound when ``t ≤ 0``), taken only when the
      fit predicts ``pt(x*)`` at least ``epsilon`` under the incumbent's;
    * every size is capped by the memory bound, by ``tick_c_max`` and by
      ``R/(k + 2)``, ``R`` the input bytes not yet issued (so it can still
      be measured), and a step that
      cannot gain ``epsilon`` even with no per-byte cost (``x ≤
      c/(1 − epsilon)``) is not taken.

    A measured size whose ``pt`` is ``epsilon`` under the incumbent's
    becomes the incumbent, and the next size is planned from the refitted
    model.  One that is not, or whose transient passes
    ``grow_budget_frac·room``, is reverted: the pipe is re-cut at the
    incumbent and fixed there.  The store gets the outcome (``c``,
    ``fixed``, ``rho``), so a repeated qid starts where this one stopped.
    """

    def __init__(self, ctl: TickSizingController, key: Tuple, st: Dict,
                 pipe_name: str, start_b: int, input_b: int, room_b: int,
                 live_b: int, quantum_b: int):
        self.ctl, self.key, self.st, self.cfg = ctl, key, st, ctl.cfg
        self.pipe_name = pipe_name
        self.k = self.cfg.tick_samples
        self.input_b = max(1, int(input_b))
        self.room_b, self.live_b, self.quantum_b = room_b, live_b, quantum_b
        self.c0 = ctl._clamp(ctl._c0())
        self.inc = self.cur = int(start_b)
        # size -> (q ms per tick, b bytes per chunk, T transient bytes)
        self.points: Dict[int, Tuple[float, float, int]] = {}
        self._q: List[float] = []
        self._b: List[int] = []
        self._t = 0
        self._skip = True
        self.fixed = False
        self.path: List[str] = []

    def settled(self) -> bool:
        return self.fixed

    def observe(self, built_b: int, nbytes: int, q_ms: float, peak_b: int,
                remaining_b: int) -> Optional[int]:
        """Chunk built at ``built_b`` with ``nbytes`` of input, its tick
        time ``q_ms``, the arena's peak while it computed ``peak_b``, and
        ``remaining_b`` input bytes not yet issued.  Returns the budget to
        re-cut the rest of the pipe at, or None."""
        if self.fixed or built_b != self.cur:
            return None
        if self._skip:
            self._skip = False
            return None
        if nbytes <= 0 or q_ms <= 0:
            return None
        self._q.append(q_ms)
        self._b.append(int(nbytes))
        self._t = max(self._t, int(peak_b) - self.live_b)
        if len(self._q) < self.k:
            return None
        self.points[self.cur] = (statistics.median(self._q),
                                 statistics.fmean(self._b), self._t)
        self._q, self._b, self._t = [], [], 0
        return self._step(int(remaining_b))

    def _pipe_ms(self, c: int) -> float:
        """The pipe's time were it run whole at size ``c``: its input's
        chunks at ``q`` each, and one more for the drain."""
        q, b, _ = self.points[c]
        return q * (self.input_b / b + 1.0)

    def _step(self, R: int) -> Optional[int]:
        eps, c = self.cfg.epsilon, self.cur
        q, b, T = self.points[c]
        limit = self.cfg.grow_budget_frac * self.room_b
        self._record(c, b, T, limit)
        self.path.append(f"{c // MB} MB: {q:.2f} ms/{b / MB:.0f} MB, "
                         f"T {T / MB:.0f} MB")
        if c != self.inc:
            # A step was measured: keep it on a gain, else revert and fix.
            if T > limit:
                self.ctl.breaches += 1
                return self._fix(f"breach (T {T // MB} > {int(limit) // MB}"
                                 f" MB)", revert=True)
            gain = 1.0 - self._pipe_ms(c) / self._pipe_ms(self.inc)
            if gain < eps:
                self.ctl.reverts += 1
                return self._fix(f"lost ({100 * gain:+.1f}%)", revert=True)
            self.ctl.accepts += 1
            self.path.append(f"won ({100 * gain:+.1f}%)")
            self.inc = c
        elif c > self.c0 and T > limit:
            # A start size from an earlier occurrence no longer fits.
            self.ctl.breaches += 1
            self.inc = self.c0
            return self._fix(f"breach at start (T {T // MB} > "
                             f"{int(limit) // MB} MB)", revert=True)
        nxt, why = self._next(R)
        if nxt is None:
            return self._fix(why, revert=False)
        self.ctl.jumps += 1
        self.path.append(f"-> {nxt // MB} MB ({why})")
        self.cur, self._skip = nxt, True
        self._persist()
        return nxt

    def _next(self, R: int) -> Tuple[Optional[int], str]:
        """The next size to measure from the incumbent, or ``(None, why)``."""
        cfg, eps, inc = self.cfg, self.cfg.epsilon, self.inc
        _, b_big, T_big = self.points[max(self.points)]
        caps = {"c_max": cfg.tick_c_max_b, "ticks": R / (self.k + 2)}
        if T_big > 0:
            caps["memory"] = cfg.grow_budget_frac * self.room_b * b_big / T_big
        bound = min(caps, key=caps.get)
        cap = caps[bound]
        if len(self.points) < 2:
            x, why = 2.0 * inc, "walk"
            fit = None
        else:
            F, t = _fit_line([(b, q) for q, b, _ in self.points.values()])
            if F <= 0:
                return None, f"no fixed cost to spread (F {F:.2f} ms)"
            D = self.input_b
            x = math.sqrt(D * F / t) if t > 0 else cap
            why = f"fit F {F:.2f} ms, t {t * MB:.4f} ms/MB"
            fit = (F, t)
        if x >= cap:
            x, why = cap, f"{why}, {bound} cap"
        x = self.ctl._clamp(whole_quanta(int(x), self.quantum_b,
                                         cfg.tick_c_min_b, cfg.tick_c_max_b))
        if x <= inc / (1.0 - eps):
            return None, f"no step past {inc // MB} MB ({why}: {x // MB} MB)"
        if fit is not None:
            F, t = fit
            gain = 1.0 - ((F + t * x) * (self.input_b / x + 1.0)
                          / self._pipe_ms(inc))
            if gain < eps:
                return None, f"no gain predicted ({why}: {x // MB} MB " \
                             f"{100 * gain:+.1f}%)"
            why = f"{why}, {100 * gain:+.1f}% predicted"
        return int(x), why

    def _fix(self, why: str, revert: bool) -> Optional[int]:
        self.fixed = True
        self.path.append(f"fixed at {self.inc // MB} MB: {why}")
        self._persist()
        if revert and self.cur != self.inc:
            self.cur = self.inc
            return self.inc
        return None

    def _persist(self) -> None:
        st = self.st
        st["c"], st["fixed"] = self.inc, self.fixed
        _, b_big, T_big = self.points[max(self.points)]
        if b_big > 0 and T_big > 0:
            st["rho"] = T_big / b_big

    def _record(self, c: int, b: float, T: int, limit: float) -> None:
        if not gates.EXPERIMENT:
            return
        smaller = [s for s in self.points if s < c]
        prior = self.points[max(smaller)] if smaller else None
        predicted = (int(prior[2] / prior[1] * b)
                     if prior is not None and prior[1] > 0 else 0)
        record_sizing_sample(
            "SZ-A", f"{self.key[2]}:{self.key[3]}", c, predicted, T,
            prior is None, T > limit, budget_b=int(self.room_b))

    def recut_refused(self) -> None:
        """The pipeline could not re-cut (only the epilogue was left): the
        pipe ends at what it runs."""
        if not self.fixed:
            self._fix("pipe ended", revert=False)

    def close(self) -> None:
        """Pipe end: persist and file the session's path for the query's
        ``[sizing]`` line."""
        if self.points:
            self._persist()
        self.ctl.note(f"{self.key[2]}:{self.key[3]} "
                      + (" | ".join(self.path) if self.path
                         else f"{self.cur // MB} MB: not measured"))


# ---------------------------------------------------------------------------
# Install / factory (main.py wiring)
# ---------------------------------------------------------------------------

_run_store: Dict[Tuple, int] = {}
_tick_store: Dict[Tuple, Dict] = {}


def install_gather_sizing(cfg: SizingSearchConfig,
                          arena_stats_fn: Callable[[], dict]) -> None:
    """Arm the gather-window search: configure the resolver bounds and
    static layer, the model registry, and install the per-stream session
    factory on the driver.  The sessions read ``arena_stats_fn``'s stats
    (the arena's ``get_arena_stats``, with ``lendable_b``)."""
    from adaptive import adaptive_driver
    from adaptive.sizing_memory_model import ModelRegistry

    cfg.validate()
    sizing_resolver.set_bounds(PIPE_CLASS, cfg.c_min_b, cfg.c_max_b)
    if cfg.static_b is not None:
        sizing_resolver.set_config_static(PIPE_CLASS, cfg.static_b)
    registry = ModelRegistry(cfg.pad_min_b, cfg.march_delta_b, cfg.seed_slope)

    def _factory(pipe_name, instance, row_bytes, row_overhead, total_rows,
                 codec_epoch, quantum_rows=0):
        import constants
        sf, query = constants.get_query_sf(), constants.get_query_id()
        start_b = (cfg.c0_b if cfg.c0_b is not None
                   else sizing_resolver.resolve(PIPE_CLASS, instance).chunk_b)
        if total_rows * (row_bytes + row_overhead) <= start_b:
            return None   # single-iteration stream — nothing to search
        # The per-pipe fit + the class prior (every sample enters
        # both; the class model is read only while the pipe fit is cold).
        model = registry.model_for(sf, query, PIPE_CLASS, str(instance),
                                   codec_epoch)
        class_model = registry.model_for(sf, PIPE_CLASS, codec_epoch)
        return GatherSizingSession(
            instance=instance, row_bytes=row_bytes,
            row_overhead=row_overhead, total_rows=total_rows,
            codec_epoch=codec_epoch, sf=sf, query=query, cfg=cfg,
            model=model, class_model=class_model,
            arena_stats_fn=arena_stats_fn,
            run_store=_run_store,
            quantum_rows=quantum_rows)

    adaptive_driver.set_gather_sizing_factory(_factory)


def uninstall_gather_sizing() -> None:
    from adaptive import adaptive_driver
    adaptive_driver.set_gather_sizing_factory(None)
    sizing_resolver.clear_all_overrides()


def install_per_pipe_sizing(cfg: SizingSearchConfig,
                            arena_stats_fn: Callable[[], dict],
                            codec_epoch: str = "static"
                            ) -> TickSizingController:
    """Arm the per-pipe leg (``--sizing_search per_pipe``; also armed by
    ``mid_stream``): configure the tick-class resolver bounds, build the
    controller and install it as the resolver decision layer and the
    pipeline's tick sizing session factory.  The controller reads
    ``arena_stats_fn``'s stats (the arena's ``get_arena_stats``, with
    ``lendable_b``).

    ``codec_epoch`` is the run-level compression-policy fingerprint, part
    of the store's key."""
    from adaptive import adaptive_driver

    cfg.validate()
    for c in sizing_resolver.TICK_CLASSES:
        sizing_resolver.set_bounds(c, cfg.tick_c_min_b, cfg.tick_c_max_b)
    ctl = TickSizingController(cfg, arena_stats_fn, codec_epoch=codec_epoch,
                               store=_tick_store)
    sizing_resolver.set_decision_fn(ctl.decide)
    adaptive_driver.set_tick_sizing_factory(ctl.session)
    return ctl


def uninstall_per_pipe_sizing() -> None:
    from adaptive import adaptive_driver
    sizing_resolver.set_decision_fn(None)
    adaptive_driver.set_tick_sizing_factory(None)


_store_banks: Dict[object, Tuple[Dict, ...]] = {}
_store_key: object = None


def reset_run_store() -> None:
    """--sizing_reset analog (tests): drop in-run convergence memory, every
    region size's."""
    global _store_key
    _run_store.clear()
    _tick_store.clear()
    _store_banks.clear()
    _store_key = None


def select_run_store(key) -> None:
    """Keep the in-run convergence memory per ``key`` (main.py: the cache
    region's size, which fixes the operator space the budgets converged
    in): set the current memory aside under the current key and bring back
    ``key``'s, empty the first time.  The stores change in place, so the
    installed sessions and controller see the switch."""
    global _store_key
    if key == _store_key:
        return
    stores = (_run_store, _tick_store)
    _store_banks[_store_key] = tuple(dict(s) for s in stores)
    for s, saved in zip(stores, _store_banks.pop(key, ({},) * len(stores))):
        s.clear()
        s.update(saved)
    _store_key = key
