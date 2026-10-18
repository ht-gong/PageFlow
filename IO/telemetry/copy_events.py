"""One CUDA event pair per copy batch: the per-tick transfer times the
compression searches read.

A *copy batch* is the copies one site issues on one stream in one launch:
the tick's H2D pack and D2D pack (``DirectIn``) and the zero-copy stage's
mask copies.  The site
records a start event on its stream right before its first copy and an end
event right after its last (:func:`begin` / :func:`end`), so a batch's time
is its copies plus any host gap between them, never the host work before
the first copy or after the last.  (The d2h lane is the home writer's,
``IO/codec/home_writer.py``.)

Batches are collected in a :class:`CopyLog`.  ``NaiveExchange.launch`` opens
one per tick and ``sync`` closes it.  With no
log open, :func:`begin` records nothing.  Once a log's streams are
synchronized:

* ``lane_ms()`` are the tick's input lanes (``TransferTimes``): per
  stream, the span from its first batch's start to its last batch's end, 0
  when the stream copied nothing.

These are engine inputs, so they are recorded whatever the telemetry switch
says.  Event pairs are pooled and reused once a log is released.
"""
from __future__ import annotations

import torch

H2D, D2D = "h2d", "d2d"
LANES = (H2D, D2D)

_free_pairs: list = []      # (start, end) timing events ready for reuse
_logs: list = []            # open logs, innermost last


class CopyBatch:
    """One site's copies on one stream: its lane and an event pair."""

    __slots__ = ("lane", "start", "end", "ended")

    def __init__(self, lane, start, end):
        self.lane = lane
        self.start = start
        self.end = end
        self.ended = False


class CopyLog:
    """The copy batches issued while this log was the innermost open one, in
    issue order (which is stream order within each lane)."""

    __slots__ = ("batches",)

    def __init__(self):
        self.batches = []

    def _ended(self, lane):
        return [b for b in self.batches if b.ended and b.lane == lane]

    def lane_ms(self) -> dict:
        """Per stream, the span from its first batch's start to its last
        batch's end (ms); 0 for a stream with no batch."""
        out = {}
        for lane in LANES:
            bs = self._ended(lane)
            out[lane] = bs[0].start.elapsed_time(bs[-1].end) if bs else 0.0
        return out

    def release(self) -> None:
        """Return the batches' events to the pool.  Call once nothing reads
        this log any more."""
        for b in self.batches:
            _free_pairs.append((b.start, b.end))
        self.batches = []


def open_log() -> CopyLog:
    """Push a new log; batches begun from now on are filed in it."""
    log = CopyLog()
    _logs.append(log)
    return log


def close_log(log: CopyLog) -> None:
    """Pop ``log``, and any log opened after it and left open by a
    failure."""
    for i in range(len(_logs) - 1, -1, -1):
        if _logs[i] is log:
            del _logs[i:]
            return


def discard_log(log: CopyLog) -> None:
    """Pop ``log`` unread and release its events (a launch a failure left
    unsynced)."""
    close_log(log)
    log.release()


def begin(lane):
    """Record a batch's start event on the current stream, right before its
    first copy; ``None`` when no log is open."""
    if not _logs:
        return None
    if _free_pairs:
        start, end = _free_pairs.pop()
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    batch = CopyBatch(lane, start, end)
    start.record()
    _logs[-1].batches.append(batch)
    return batch


def end(batch) -> None:
    """Record ``batch``'s end event on the current stream, right after its
    last copy."""
    if batch is None:
        return
    batch.end.record()
    batch.ended = True


def reset() -> None:
    """Drop any log a failed query left open (top of each query, when no
    launch is in flight)."""
    _logs.clear()
