"""The copy-batch event pairs (``IO/telemetry/copy_events.py``) and what is
read off them: the input lanes the compression searches take
(``TransferTimes``).

A :class:`CopyLog` over fake events: lanes are spans per stream, an unended
batch does not count, and the log stack drops what a failure left open.
"""
from __future__ import annotations

import pytest

from IO.telemetry import copy_events


@pytest.fixture(autouse=True)
def _clean():
    copy_events.reset()
    yield
    copy_events.reset()


# ── the log, over fake events ────────────────────────────────────────────

class FakeEvent:
    """A recorded event at a fixed device time (ms)."""

    def __init__(self, t):
        self.t = t

    def elapsed_time(self, other):
        return other.t - self.t


def _batch(lane, start, end, ended=True):
    b = copy_events.CopyBatch(lane, FakeEvent(start), FakeEvent(end))
    b.ended = ended
    return b


def test_lanes_are_spans_per_stream():
    log = copy_events.CopyLog()
    log.batches += [
        _batch("h2d", 0.0, 2.0),
        _batch("h2d", 3.0, 4.0),                      # h2d: 0 -> 4
        _batch("d2d", 0.5, 1.0),
        _batch("d2d", 1.5, 4.5),                      # d2d: 0.5 -> 4.5
        _batch("h2d", 9.0, 10.0, ended=False),        # begun, never ended
    ]
    assert log.lane_ms() == {"h2d": 4.0, "d2d": 4.0}
    assert copy_events.CopyLog().lane_ms() == {"h2d": 0.0, "d2d": 0.0}


def test_begin_records_nothing_without_an_open_log():
    assert copy_events.begin(copy_events.H2D) is None
    copy_events.end(None)                  # a no-op


def test_close_log_also_drops_logs_a_failure_left_open():
    outer = copy_events.open_log()
    copy_events.open_log()                 # never closed
    copy_events.close_log(outer)
    assert copy_events._logs == []
    log = copy_events.open_log()
    copy_events.discard_log(log)
    assert copy_events._logs == [] and log.batches == []
