"""The sizing plane's single actuation channel for chunk budgets.

``resolve(pipe_class, instance)`` is the ONLY sanctioned way a sized
construction/consumption site obtains its chunk byte budget.  Layering,
most-specific wins:

1. **controller override** — present only while a sizing search is live,
   written exclusively by ``adaptive/sizing_controller.py`` (the
   single-writer rule), keyed per ``(pipe_class, instance)``;
2. **static config** — an optional ``--sizing_config`` JSON ``static_mb``
   per class (the ``sizing_tuning.json`` layer);
3. **constants default** — the uniform chunk budget
   (``constants.get_chunk_size_b()``, CLI ``--chunk_size_mb``).

**Scope:** every pipe class in ``CLASSES`` is sized.  The ``rearrange``
class (gather-window sizing) actuates through per-instance **overrides**
written by ``GatherSizingSession``; the tick-pipe classes (``sort``/
``core``/``agg``/``filter``/``project``, tick-pipe sizing) actuate through
the **decision function** — a pull-model controller layer installed by
``TickSizingController`` and consulted exactly once per pipe construction.
With no override, no decision function and no config, resolution is
byte-identical to ``constants.get_chunk_size_b()``.

Bounds: every override must sit inside the class's ``[c_min, c_max]``
(both sides police — ``set_override`` raises, and ``resolve`` asserts).
Self-documentation: one ``[sizing]`` log line per resolved-value *change*
per key (per-iteration re-resolution in the gather stream would otherwise
log thousands of identical lines).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import constants
from utility.logger import message_logger

# The sized pipe classes: the rearrange gather window plus the five
# tick-pipe classes.
TICK_CLASSES = ("sort", "core", "agg", "filter", "project")
CLASSES = ("rearrange",) + TICK_CLASSES

# Hard per-class bounds on any resolved budget.  Overridable via the
# sizing config; these are the resolver's own last-line clamps.
_TICK_BOUNDS = (16 * 1024 ** 2, 16 * 1024 ** 3)
DEFAULT_BOUNDS_B: Dict[str, Tuple[int, int]] = {
    "rearrange": (16 * 1024 ** 2, 64 * 1024 ** 3),
    **{c: _TICK_BOUNDS for c in TICK_CLASSES},
}


@dataclass(frozen=True)
class SizeDecision:
    chunk_b: int
    source: str  # 'override' | 'config' | 'constants'


_overrides: Dict[Tuple[str, str], int] = {}
_config_static_b: Dict[str, int] = {}
_bounds_b: Dict[str, Tuple[int, int]] = dict(DEFAULT_BOUNDS_B)
_last_logged: Dict[Tuple[str, str], Tuple[int, str]] = {}
# The tick-pipe sizing controller layer: fn(pipe_class, instance) ->
# chunk_b | None.  Pull-model because the tick-pipe decision is made AT the
# single construction-time resolve from decision-time arena stats — there
# is no earlier moment at which a push-style override could be computed.
# Written exclusively by adaptive/sizing_controller.py (single writer).
_decision_fn: Optional[Callable[[str, str], Optional[int]]] = None


def set_decision_fn(fn: Optional[Callable[[str, str], Optional[int]]]) -> None:
    """Install (or clear, with ``None``) the per-pipe decision layer."""
    global _decision_fn
    _decision_fn = fn


def has_decision_fn() -> bool:
    return _decision_fn is not None


def _check_class(pipe_class: str) -> None:
    if pipe_class not in CLASSES:
        raise ValueError(
            f"sizing_resolver: unknown pipe class {pipe_class!r} "
            f"(sized classes this build: {CLASSES})")


def bounds(pipe_class: str) -> Tuple[int, int]:
    _check_class(pipe_class)
    return _bounds_b[pipe_class]


def set_bounds(pipe_class: str, c_min_b: int, c_max_b: int) -> None:
    """Install the class's [c_min, c_max] (from the sizing config)."""
    _check_class(pipe_class)
    if not (0 < c_min_b <= c_max_b):
        raise ValueError(
            f"sizing_resolver: invalid bounds [{c_min_b}, {c_max_b}] "
            f"for {pipe_class!r}")
    _bounds_b[pipe_class] = (int(c_min_b), int(c_max_b))


def set_config_static(pipe_class: str, chunk_b: int) -> None:
    """Install the static-config layer value for ``pipe_class``."""
    _check_class(pipe_class)
    lo, hi = _bounds_b[pipe_class]
    if not (lo <= chunk_b <= hi):
        raise ValueError(
            f"sizing_resolver: config static {chunk_b} for {pipe_class!r} "
            f"outside bounds [{lo}, {hi}]")
    _config_static_b[pipe_class] = int(chunk_b)


def set_override(pipe_class: str, instance: str, chunk_b: int) -> None:
    """Controller-plane override for one ``(class, instance)`` stream.

    Raises on an unknown class or a value outside the class bounds — the
    boundary polices, the controller never writes blind.
    """
    _check_class(pipe_class)
    lo, hi = _bounds_b[pipe_class]
    if not (lo <= chunk_b <= hi):
        raise ValueError(
            f"sizing_resolver: override {chunk_b} for "
            f"{pipe_class!r}:{instance!r} outside bounds [{lo}, {hi}]")
    _overrides[(pipe_class, str(instance))] = int(chunk_b)


def clear_override(pipe_class: str, instance: str) -> None:
    _overrides.pop((pipe_class, str(instance)), None)


def clear_all_overrides() -> None:
    _overrides.clear()


def resolve(pipe_class: str, instance: str) -> SizeDecision:
    """Resolve the chunk budget for one pipe instance.

    Consulted once per pipe at construction (tick-pipe classes) or once
    per gather iteration (the rearrange gather window).  With no override
    and no config this returns the constants default — bit-identical to
    the pre-resolver read.
    """
    _check_class(pipe_class)
    key = (pipe_class, str(instance))
    decided_b = _decision_fn(pipe_class, key[1]) if _decision_fn else None
    if decided_b is not None:
        lo, hi = _bounds_b[pipe_class]
        if not (lo <= decided_b <= hi):
            raise ValueError(
                f"sizing_resolver: decision {decided_b} for "
                f"{pipe_class!r}:{instance!r} outside bounds [{lo}, {hi}]")
        decision = SizeDecision(int(decided_b), "controller")
    elif key in _overrides:
        decision = SizeDecision(_overrides[key], "override")
    elif pipe_class in _config_static_b:
        decision = SizeDecision(_config_static_b[pipe_class], "config")
    else:
        decision = SizeDecision(constants.get_chunk_size_b(), "constants")
    lo, hi = _bounds_b[pipe_class]
    assert lo <= decision.chunk_b <= hi or decision.source == "constants", (
        f"sizing_resolver: resolved {decision} outside [{lo}, {hi}]")
    logged = _last_logged.get(key)
    if logged != (decision.chunk_b, decision.source):
        _last_logged[key] = (decision.chunk_b, decision.source)
        _q = constants.get_query_id()
        message_logger().info(
            "[sizing] Q%s %s:%s -> %d MB (%s)",
            _q, pipe_class, instance,
            decision.chunk_b // (1024 ** 2), decision.source)
    return decision


def reset() -> None:
    """Drop every layer above constants (tests / --sizing_reset analog)."""
    global _decision_fn
    _decision_fn = None
    _overrides.clear()
    _config_static_b.clear()
    _bounds_b.clear()
    _bounds_b.update(DEFAULT_BOUNDS_B)
    _last_logged.clear()
