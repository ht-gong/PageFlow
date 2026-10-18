"""The home writer: copies a pipe's output regions from GPU buffers into their
pinned homes.

A region producer (``IO/codec/region_producer.py``) packs each tick's output
of a column instance into a dense GPU buffer (or, for a RAW output, keeps the
operator's output tensor) and gives the writer one *shipment* per tick: the
copies from those buffers into the regions' homes, the buffers to hold, and
an event recorded after the work that filled them.  The writer is the only
code that copies into homes.

With the granule cache the writer also copies granules out of their slots
into their homes: the cache's cleaning copies (shipped by the producer
behind the tick's slot fills) and a reclaim's dirty victims (``copy_home``).
And it runs the producer's slot fills (``ship``'s ``fill``): a fill reuses
slots whose dirty victims this stream is still copying home, so it runs
behind those copies in stream order and the compute stream never waits on
them; the shipment holds the slabs it reads until the fill has run
(``fill_keep``, dropped at an event recorded after it).

The copies run right away on one background stream (``writer_stream``),
behind the shipment's event.  The tick barrier does not wait for that
stream, so a copy can run under this tick's compute and later ones.

Lifetime: a buffer the copies read was allocated on the compute stream, and
the arena orders a free only on the allocating stream (``record_stream`` does
nothing under the engine's allocator), so the writer holds every source buffer
until the event recorded after its copies has fired, and only then drops it
(``reap``).  A pipe keeps at most ``cap`` ticks' shipments in flight (two
ticks' buffers; a tick's RAW shipments and its pack count as one): when a
third tick's would start, the producer waits on the oldest tick's
(``wait_for_room``).

Timing.  A shipment forms one event pair (its start before its fill and first
copy, its end after the last): the pipe's d2h lane, the writer stream's work
for the pipe.  ``take_lane`` hands the
searches each finished shipment's span and bytes once its end has fired,
normally one tick later.
"""
from __future__ import annotations

from collections import deque

import torch

from utility import cuda_streams

_stream = None
# Timing event pairs of finished shipments, and their fill events, for reuse.
_free_pairs: list = []
_free_events: list = []


def _pair():
    if _free_pairs:
        return _free_pairs.pop()
    return (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))


def _event():
    if _free_events:
        return _free_events.pop()
    return torch.cuda.Event()


_native = None


def _tqp():
    """The native compression module, imported once (the module stays
    importable without it)."""
    global _native
    if _native is None:
        import TQPlib.tqpcompress as native
        _native = native
    return _native


def writer_stream() -> torch.cuda.Stream:
    """The background stream every pipe's shipments run on."""
    global _stream
    if _stream is None:
        _stream = cuda_streams.dedicated('home writer')
    return _stream


def copy_home(srcs, dsts, sizes):
    """Copy ``sizes[i]`` bytes from GPU address ``srcs[i]`` into home
    ``dsts[i]`` on the writer's stream, behind the work queued so far on the
    current stream; returns an event recorded after the copies.  The granule
    cache's reclaim writes dirty granules home through it: their slots were
    filled in earlier ticks, so no buffer needs holding."""
    after = torch.cuda.Event()
    after.record(torch.cuda.current_stream())
    end = torch.cuda.Event()
    stream = writer_stream()
    with torch.cuda.stream(stream):
        stream.wait_event(after)
        import TQPlib.tqpcompress as tqpcompress
        tqpcompress.batched_d2h_to_ptrs([int(x) for x in srcs],
                                        [int(x) for x in dsts],
                                        [int(x) for x in sizes])
        end.record(stream)
    return end


class Shipment:
    """One tick's copies into homes: ``(src, dst, bytes)`` per region, the
    buffers they read, the event they wait for, and their event pair."""

    __slots__ = ("srcs", "dsts", "sizes", "keep", "fill_keep", "filled",
                 "after", "start", "end", "wire_b", "done", "tick")

    def __init__(self, srcs, dsts, sizes, keep, after, tick, fill_keep=None):
        self.srcs = srcs
        self.dsts = dsts
        self.sizes = sizes
        self.keep = keep
        # The buffers only the fill reads, dropped once ``filled`` (recorded
        # after the fill) has fired, ahead of the copies' end.
        self.fill_keep = fill_keep
        self.filled = None
        self.after = after
        self.start, self.end = _pair()
        self.wire_b = int(sum(sizes))
        self.done = False
        self.tick = tick

    def ms(self) -> float:
        """The copies' span (the end event must have fired)."""
        return self.start.elapsed_time(self.end)


class HomeWriter:
    """One pipe's shipments, in issue order (module docstring)."""

    def __init__(self, cap: int = 2):
        assert cap >= 1
        self.cap = cap
        self._ships = deque()        # not yet reaped, issue order
        self._unreported = []        # ended, lane not yet taken
        self.cap_waits = 0           # waits for room under the cap

    # -- issuing ----------------------------------------------------------

    def ship(self, srcs, dsts, sizes, keep, after, tick=None,
             fill=None, fill_keep=()) -> Shipment:
        """Copy ``sizes[i]`` bytes from GPU address ``srcs[i]`` into home
        ``dsts[i]`` once ``after`` has fired, holding ``keep`` until the
        copies end.  ``tick`` labels the output tick the buffers hold, for
        the cap: shipments with equal labels count as one tick (``None``: a
        tick of its own).  ``fill``, if given, is called with the writer's
        stream current before the copies: the pack's slot fills
        (``region_producer``); ``fill_keep`` holds the buffers only it reads
        until it has run (``reap``), ``keep`` those the copies read too.
        The shipment's span, its lane time, covers the fill and the copies.
        The copies are issued by switching the current stream directly,
        without the stream context; the end event is recorded even when the
        fill or the copies fail, so ``abort`` waits for what was issued."""
        s = Shipment(list(srcs), list(dsts), list(sizes), list(keep), after,
                     tick if tick is not None else object(),
                     list(fill_keep) or None)
        self._ships.append(s)
        stream = writer_stream()
        stream.wait_event(s.after)
        s.start.record(stream)
        prev = torch.cuda.current_stream(stream.device_index)
        torch.cuda.set_stream(stream)
        try:
            if fill is not None:
                fill()
                if s.fill_keep:
                    s.filled = _event()
                    s.filled.record(stream)
            if s.srcs:
                _tqp().batched_d2h_to_ptrs(s.srcs, s.dsts, s.sizes)
        finally:
            s.end.record(stream)
            torch.cuda.set_stream(prev)
        return s

    # -- the cap and frees -------------------------------------------------

    def in_flight(self) -> int:
        """Ticks with shipments whose copies have not been seen to end."""
        self.reap()
        return len({s.tick for s in self._ships})

    def wait_for_room(self, tick=None) -> bool:
        """Before shipping tick ``tick``'s buffers: while ``cap`` other
        ticks have shipments in flight, wait for the oldest one's to end.
        Returns True when it waited."""
        waited = False
        while True:
            self.reap()
            ticks = []
            for s in self._ships:
                if s.tick != tick and s.tick not in ticks:
                    ticks.append(s.tick)
            if len(ticks) < self.cap:
                break
            for s in self._ships:
                if s.tick == ticks[0]:
                    s.end.synchronize()
            waited = True
        if waited:
            self.cap_waits += 1
        return waited

    def wait_all(self) -> None:
        """Wait for every shipment's copies to end and drop their buffers."""
        for s in self._ships:
            s.end.synchronize()
        self.reap()

    def reap(self) -> None:
        """Drop the buffers of every shipment, oldest first, whose copies
        have ended."""
        for s in self._ships:
            if s.fill_keep and s.filled is not None and s.filled.query():
                s.fill_keep = None
        while self._ships:
            s = self._ships[0]
            if not s.end.query():
                break
            s.done = True
            s.keep = s.fill_keep = None
            self._unreported.append(s)
            self._ships.popleft()

    def abort(self) -> None:
        """A failure ended the pipe: wait for every copy, so no buffer it
        reads or home it writes is freed under it, and drop the buffers."""
        for s in self._ships:
            s.end.synchronize()
        for s in self._ships:
            s.done = True
            s.keep = s.fill_keep = None
        self._ships.clear()
        self._unreported.clear()

    @staticmethod
    def _recycle(s: Shipment) -> None:
        """Give a finished shipment's events back for reuse."""
        if isinstance(s.end, torch.cuda.Event):
            _free_pairs.append((s.start, s.end))
        if s.filled is not None:
            _free_events.append(s.filled)
        s.start = s.end = s.filled = None

    # -- timing ------------------------------------------------------------

    def take_lane(self) -> tuple:
        """The span (ms) and bytes of the shipments that ended since the
        last call, each reported once."""
        self.reap()
        done = self._unreported
        if not done:
            return 0.0, 0
        ms = 0.0
        wire = 0
        for s in done:
            ms += s.ms()
            wire += s.wire_b
            self._recycle(s)
        done.clear()
        return ms, wire

    def drain(self) -> None:
        """End of pipe: wait for every shipment and drop their buffers.
        Their spans go unreported: no tick reads the lane after the pipe."""
        self.wait_all()
        assert not self._ships, "a shipment outlived the drain"
        for s in self._unreported:
            self._recycle(s)
        self._unreported.clear()
