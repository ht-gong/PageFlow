"""Adaptive-plane registration point: the feedback records and the
process-wide factories the pipelines consult.

The one registration point both adaptive planes share:

* The per-tick feedback is the **monitor's last-tick view**:
  ``VortexPipeline`` passes its own per-pipe :class:`~adaptive.
  resource_monitor.ResourceMonitor` to the observer once per tick.  The view
  is ephemeral — valid only inside the callback — and carries no epoch
  stamps (stamps belong on *stored* samples).
* :class:`GatherIterSample` — one rearrange gather iteration's record
  (the gather-window sizing session's sample).  The tick pipes' sizing
  session is fed by the pipeline directly (``make_tick_sizing_session``).

A non-``None`` tick return is a **full input-pick vector** — positional over
the pipe's published menu (``adaptive/decision_menu.py``), applied by the
*pipeline* (sequencing knowledge — the t+2 rule — lives entirely on the
pipeline side).

The module is deliberately torch-free so plane logic and its unit tests
run CPU-only.

Pipeline attachment: ``VortexPipeline`` accepts a ``tick_observer``
constructor argument.  Because pipelines are constructed inside the
operators, a process-wide *factory* is also provided —
:func:`set_tick_observer_factory` — which the pipeline consults when no
explicit observer is passed.  The factory receives the pipe name **and the
pipe's decision menu** (the menu publication — once, at construction,
one-directional) and returns a per-pipe observer (or ``None``).  One pipe ==
one tick sequence, so observers are per-pipe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from adaptive.decision_menu import DecisionMenu
from adaptive.resource_monitor import ResourceMonitor


# ---------------------------------------------------------------------------
# Stored boundary records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GatherIterSample:
    """One rearrange gather iteration's feedback (the gather-window sizing
    probe sample).

    Built by ``_stream_gather`` after each internally-synchronized
    gather→compress→spill iteration and handed to the stream's sizing
    session.  ``window_b`` is the resolved budget that built the iteration
    (sample attribution — exact, not the controller's belief); the GPU legs
    are CUDA-event times (the iteration is serialized around syncs, so
    their sum is the iteration's measured device time — the timing rule's
    device-derived analog for a pipe with no tick loop).  ``iter_peak_b`` /
    ``live_at_entry_b`` bracket the arena over the iteration
    (transient = peak − entry).
    """
    window_b: int
    rows: int
    row_bytes: int
    chunk_bytes: int
    h2d_ms: float
    compress_ms: float
    d2h_ms: float
    iter_peak_b: int
    live_at_entry_b: int
    # Host-bracketed time of the WHOLE synchronized iteration (resolve →
    # gather → compress → spill → syncs), perf_counter-based.  Superset of
    # the three CUDA legs: it additionally sees the per-iteration fixed host
    # cost (sync latency, producer bookkeeping) that CUDA
    # events cannot, which is exactly the cost a larger window amortizes.
    # 0.0 when the caller did not bracket (older callers / unit fixtures).
    iter_ms: float = 0.0


# ---------------------------------------------------------------------------
# Callback shapes
# ---------------------------------------------------------------------------

# The observer consumes the monitor's last-tick view and may return a full
# input-pick vector (positional over the pipe's decision menu), else None.
# An observer may carry ``settled()``: True once it returns nothing more,
# from which tick the pipe leaves the tick barrier for event-ordered ticks
# (``VortexPipeline._searches_settled``); one without it keeps the barrier.
TickObserver = Callable[[ResourceMonitor], Optional[Sequence[str]]]


# ---------------------------------------------------------------------------
# Process-wide observer factory (pipeline attachment + menu publication)
# ---------------------------------------------------------------------------

# fn(pipe_name, menu) -> tick observer for that pipe, or None.  The menu is
# published once per pipe, at construction, with no reply.
ObserverFactory = Callable[[str, DecisionMenu], Optional[TickObserver]]

_tick_observer_factory: Optional[ObserverFactory] = None
_factory_min_ticks: int = 0


def set_tick_observer_factory(factory: Optional[ObserverFactory],
                              min_total_ticks: int = 0) -> None:
    """Install (or clear, with ``None``) the process-wide per-pipe observer
    factory.  Consulted by ``VortexPipeline.__init__`` when no explicit
    ``tick_observer`` was passed; default ``None`` keeps every pipeline
    byte-identical to the unobserved build — the pipeline then skips menu
    construction entirely.

    ``min_total_ticks`` is the factory's declared floor: a pipe with fewer
    ticks would get ``None`` from the factory regardless of its menu (too
    short to finish a search, it keeps the static default), so the pipeline
    skips menu construction — and its per-page byte accounting — for such
    pipes wholesale.  Measured motivation: menu builds on the many short
    pipes of join/rearrange-heavy SF100 queries cost a consistent +3-35 %%
    wall with zero searches run."""
    global _tick_observer_factory, _factory_min_ticks
    _tick_observer_factory = factory
    _factory_min_ticks = int(min_total_ticks) if factory is not None else 0


def has_tick_observer_factory() -> bool:
    """Cheap predicate so the pipeline can skip menu construction (and every
    other observer cost) when no factory is installed."""
    return _tick_observer_factory is not None


def tick_observer_min_ticks() -> int:
    """The installed factory's declared minimum pipe length (0 = none)."""
    return _factory_min_ticks


def make_tick_observer_for(pipe_name: str,
                           menu: DecisionMenu) -> Optional[TickObserver]:
    """The pipeline-side hook: publish the pipe's decision menu and resolve
    the observer for the new pipe."""
    if _tick_observer_factory is None:
        return None
    return _tick_observer_factory(pipe_name, menu)


# ---------------------------------------------------------------------------
# Process-wide gather-sizing session factory
# ---------------------------------------------------------------------------
# The sizing-plane analog of the tick-observer factory: rearrange gather
# streams have no tick loop, so their probe granularity is the gather
# iteration.  ``_stream_gather`` asks for a per-stream session at stream
# start; per iteration it re-resolves its window from the sizing resolver
# and feeds the session one :class:`GatherIterSample`.  The session
# (adaptive/sizing_controller.py) actuates ONLY through resolver overrides +
# governor leases — sequencing stays pipeline-side.
#
# fn(pipe_name, instance, row_bytes, row_overhead, total_rows, codec_epoch,
#    quantum_rows=0)
#   -> session with .observe(GatherIterSample) / .close(), or None.

GatherSizingFactory = Callable[..., Optional[object]]

_gather_sizing_factory: Optional[GatherSizingFactory] = None
_gather_sizing_enabled: bool = True


def set_gather_sizing_factory(factory: Optional[GatherSizingFactory]) -> None:
    """Install (or clear, with ``None``) the process-wide gather-sizing
    session factory.  Default ``None`` keeps every rearrange stream
    byte-identical to the unobserved build."""
    global _gather_sizing_factory
    _gather_sizing_factory = factory


def has_gather_sizing_factory() -> bool:
    """Cheap predicate so ``_stream_gather`` can take the exact legacy path
    (single up-front window resolve, no per-iteration work) when sizing is
    off or excluded."""
    return _gather_sizing_factory is not None and _gather_sizing_enabled


def set_gather_sizing_enabled(flag: bool) -> None:
    """Warmup exclusion: ``main.warmup`` disables sessions so
    warmup's differently-resident pipes contribute no fit samples and run
    the static window."""
    global _gather_sizing_enabled
    _gather_sizing_enabled = bool(flag)


def make_gather_sizing_session(pipe_name: str, instance: str, row_bytes: int,
                               row_overhead: int, total_rows: int,
                               codec_epoch: str, quantum_rows: int = 0
                               ) -> Optional[object]:
    """``quantum_rows``: the output's rows per record, the unit of the
    window (0: none)."""
    if _gather_sizing_factory is None or not _gather_sizing_enabled:
        return None
    return _gather_sizing_factory(pipe_name, instance, row_bytes,
                                  row_overhead, total_rows, codec_epoch,
                                  quantum_rows=quantum_rows)


def sizing_enabled() -> bool:
    """The sizing plane's warmup switch.  One flag for both legs:
    ``set_gather_sizing_enabled(False)`` during ``main.warmup`` silences the
    gather-window session factory above AND the tick-pipe sizing surface
    below (decision layer + sample delivery), so warmup's differently-resident
    pipes run static budgets and contribute no fit samples."""
    return _gather_sizing_enabled


# ---------------------------------------------------------------------------
# Process-wide tick sizing session factory (the tick leg)
# ---------------------------------------------------------------------------
# Tick-pipe budgets are decided at construction (the resolver's decision
# layer); a pipe that may be re-cut mid-run also gets a per-pipe session at
# its start (``VortexPipeline._exchange_ticks``), which its event-ordered
# ticks feed one sample per chunk and which may ask for a re-cut of the
# ticks not yet issued (``sizing_controller.TickJumpSession``).  Default
# ``None`` keeps every tick pipe byte-identical to the unsized build.
#
# fn(pipe_name, pipe_class, instance, chunk_b, n_ticks, input_b,
#    sectionable=None, quantum_b=0)
#   -> session with ``k``, ``.observe(built_b, nbytes, q_ms, peak_b,
#      remaining_b)``, ``.recut_refused()`` and ``.close()``, or None.

TickSizingFactory = Callable[..., Optional[object]]

_tick_sizing_factory: Optional[TickSizingFactory] = None


def set_tick_sizing_factory(factory: Optional[TickSizingFactory]) -> None:
    """Install (or clear, with ``None``) the process-wide tick sizing
    session factory."""
    global _tick_sizing_factory
    _tick_sizing_factory = factory


def has_tick_sizing() -> bool:
    """Whether a tick sizing factory is installed and the warmup has not
    switched sizing off."""
    return _tick_sizing_factory is not None and _gather_sizing_enabled


def make_tick_sizing_session(pipe_name: str, pipe_class: str, instance: str,
                             chunk_b: int, n_ticks: int, input_b: int,
                             sectionable: Optional[bool] = None,
                             quantum_b: int = 0) -> Optional[object]:
    """``input_b``: the pipe's raw input bytes; ``quantum_b``: the raw bytes
    of one of the pipe's quanta (0: the pipe is cut by rows)."""
    if not has_tick_sizing():
        return None
    return _tick_sizing_factory(pipe_name, pipe_class, instance, chunk_b,
                                n_ticks, input_b, sectionable=sectionable,
                                quantum_b=quantum_b)
