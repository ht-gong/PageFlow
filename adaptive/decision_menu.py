"""Boundary objects: the per-pipe decision menu, picks, and validation.

The codec plane's single actuation interface: per pipe, the pipeline
publishes each column's legal codec **range** and the adaptive side returns
**picks** within those ranges, positionally.  Nothing plan-shaped crosses the
boundary — no tags, no column names, no operator structure; a two-entry range
is the *only* signal that a column is searchable (pick legality holds by
construction: an illegal choice is inexpressible).

* Input ranges are enumerated from the reps a column's CCR slice **actually
  holds** (ground truth — a choice is offered iff taking it is pure rep
  selection).  A consumer of an operator output naturally sees a singleton
  range (the producer wrote one rep); a scan consumer sees two.
* Output ranges come from the structural rules + typed policy: a RAW-resolved
  output (untyped, or RAW-policy class) is a singleton; a
  codec-resolved output offers ``(RAW, candidate)``.
* The menu is **publish-only** — published once at pipe construction, no
  reply.  All actuation flows through the per-tick observer return (input
  picks under the t+2 rule; output picks only at probe-window boundaries
  under ``read-and-write``).

Torch-free: builders take plain byte counts, so the whole surface unit-tests
CPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

RAW = "RAW"


# ---------------------------------------------------------------------------
# The observer return (adaptive -> pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PickPair:
    """The ``read-and-write`` observer return.

    ``inputs`` is a full input-pick vector (or ``None`` = no read-leg
    change); ``outputs`` is a full output-pick vector (or ``None`` = no
    write-leg change).  ``seal`` requests a *null* probe-window boundary
    (seal-and-evict with no codec change — an A-A boundary): the pipeline
    runs the boundary protocol on the write-decision columns even when
    ``outputs`` matches the current state.  A plain sequence return remains
    the read-only shape (input picks under t+2); this typed pair is the only
    way output picks cross the boundary."""
    inputs: Optional[Sequence[str]] = None
    outputs: Optional[Sequence[str]] = None
    seal: bool = False


# ---------------------------------------------------------------------------
# The menu (pipeline -> adaptive; publish-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnChoices:
    """One column's legal codec range plus its benefit-ranking byte counts.

    ``choices`` carries at most one non-RAW entry: *which* codec is still the
    policy's contract — the menu carries only *whether* (and, for singleton
    ranges, *that there is nothing to decide*).
    """
    choices: Tuple[str, ...]           # ("RAW",) / ("RAW", cand) / (cand,)
    raw_bytes: int                     # uncompressed bytes of this pipe's span
    stored_bytes: Mapping[str, int] = field(default_factory=dict)  # rep -> bytes
    # An output's consumer reads (the dataflow's expected reads): the
    # write leg's readback price counts one crossing per read, plus home.
    reads: int = 1

    def __post_init__(self):
        if not (1 <= len(self.choices) <= 2):
            raise ValueError(
                f"ColumnChoices: range must have 1 or 2 entries, got "
                f"{self.choices!r}")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(
                f"ColumnChoices: duplicate entries in {self.choices!r}")
        non_raw = [a for a in self.choices if a != RAW]
        if len(non_raw) > 1:
            raise ValueError(
                f"ColumnChoices: at most one non-RAW entry allowed, got "
                f"{self.choices!r}")

    @property
    def is_decision(self) -> bool:
        """A two-entry range — the search variables."""
        return len(self.choices) == 2

    @property
    def candidate(self) -> Optional[str]:
        """The range's single non-RAW entry, or None."""
        for a in self.choices:
            if a != RAW:
                return a
        return None


@dataclass(frozen=True)
class DecisionMenu:
    """The per-pipe decision universe.

    ``pipe_name`` is opaque identity — autotune keying only; menus and picks
    are positional (``inputs`` follows the pipeline's input column order,
    ``outputs`` the output slot order).

    ``total_ticks`` is the pipe's tick count (``VortexPipeline.stages`` —
    prologue and epilogue included): pure pipe *geometry*, not plan shape,
    published so the search can budget-gate itself without a second channel
    (a pipe too short to finish a search keeps the static default).  ``0``
    (the default for menus built outside a pipeline) means "unknown" and
    keeps any budget-gated consumer inert.
    """
    pipe_name: str
    inputs: Tuple[ColumnChoices, ...]
    outputs: Tuple[ColumnChoices, ...]
    total_ticks: int = 0


# ---------------------------------------------------------------------------
# Builders (pipeline-side; pure functions over plain byte counts)
# ---------------------------------------------------------------------------

def build_input_choices(rep_bytes: Mapping[str, int],
                        raw_bytes: int) -> ColumnChoices:
    """One input column's range from the reps it actually holds.

    ``rep_bytes`` maps each materialized rep's codec name (``"RAW"`` for the
    uncompressed rep) to its stored bytes.  RAW is listed first when present;
    more than one non-RAW rep violates single-candidate materialization and
    raises.
    """
    if not rep_bytes:
        raise ValueError("build_input_choices: column holds no reps")
    non_raw = sorted(a for a in rep_bytes if a != RAW)
    if len(non_raw) > 1:
        raise ValueError(
            f"build_input_choices: more than one non-RAW rep materialized "
            f"({non_raw}) — violates the single-candidate rule")
    choices = ((RAW,) if RAW in rep_bytes else ()) + tuple(non_raw)
    return ColumnChoices(choices=choices, raw_bytes=int(raw_bytes),
                         stored_bytes=dict(rep_bytes))


def build_output_choices(candidate: Optional[str],
                         raw_bytes: int = 0,
                         locked: bool = False,
                         reads: int = 1) -> ColumnChoices:
    """One output column's range from the structural rules + typed policy.

    ``candidate`` is the column's statically resolved non-RAW codec, or
    ``None`` when structure/typing forces RAW (untyped, RAW-policy class,
    ``TQP_RAW_INTERMEDIATES``) — those publish a singleton, so the adaptive side can
    never widen a structural decision.  Stored bytes are unknown at
    construction (nothing is produced yet); the write leg's ranking bytes are
    in-run measured counters.

    ``locked`` publishes the resolved codec as a singleton even when a
    candidate exists: the pipeline uses it for the non-base slots of a
    slot group (a B-partitioned logical column spans B consecutive output
    slots that must flip together).  Only the group's base slot publishes a
    decision range; a vector that desyncs the group is inexpressible, and
    the pipeline propagates the base slot's pick to the whole group on
    actuation.

    ``reads`` is the output's consumer reads (the dataflow's expected
    reads), the write leg's readback-price input."""
    if candidate is None or candidate == RAW:
        choices = (RAW,)
    elif locked:
        choices = (candidate,)
    else:
        choices = (RAW, candidate)
    return ColumnChoices(choices=choices, raw_bytes=int(raw_bytes),
                         stored_bytes={}, reads=max(1, int(reads)))


# ---------------------------------------------------------------------------
# Pick validation and per-pipe resolution state (both sides police)
# ---------------------------------------------------------------------------

def validate_pick_vector(columns: Sequence[ColumnChoices],
                         picks: Sequence[str], *,
                         what: str = "pick vector") -> Tuple[str, ...]:
    """Boundary validation: a non-None vector MUST be full-length and every
    element within its column's range.  Raises ``ValueError`` with no state
    touched; returns the validated tuple."""
    picks = tuple(picks)
    if len(picks) != len(columns):
        raise ValueError(
            f"{what}: wrong length {len(picks)}, expected {len(columns)} "
            f"(any non-None vector MUST be full-length)")
    for i, (col, pick) in enumerate(zip(columns, picks)):
        if pick not in col.choices:
            raise ValueError(
                f"{what}: element {i} = {pick!r} outside its range "
                f"{col.choices!r}")
    return picks


class PickState:
    """One pipe's current input-pick resolution (pipeline-side).

    Holds the effective per-column codec vector: the static defaults at
    construction, replaced whole by each applied observer return (replace-
    not-merge, like the menu contract).  A rejected vector leaves the state
    unchanged — validation runs before any mutation.
    """

    def __init__(self, columns: Sequence[ColumnChoices],
                 defaults: Sequence[str]):
        self._columns = tuple(columns)
        self._current = validate_pick_vector(
            self._columns, defaults, what="static defaults")

    @property
    def current(self) -> Tuple[str, ...]:
        return self._current

    def apply(self, picks: Sequence[str]) -> Tuple[str, ...]:
        """Validate and install a full input-pick vector (observer return)."""
        self._current = validate_pick_vector(
            self._columns, picks, what="input picks")
        return self._current

    def resolve(self, idx: int) -> str:
        """The codec the tick builder must select for input column ``idx``."""
        return self._current[idx]
