import torch
from typing import Dict, List
from IO.transfer.in_stage import InStage
from IO.telemetry import copy_events
from utility import cuda_streams

# The naive exchange is the only transfer backend.
_exchange = None


class NaiveExchange:
    def __init__(self):
        # Kept for the process (utility/cuda_streams.py): no pipe's
        # compute stream is one of them.
        self.h2d_stream = cuda_streams.dedicated('h2d')
        self.d2d_stream = cuda_streams.dedicated('d2d')
        # The tick's input lane times (h2d, d2d ms), set by sync().
        self._last_transfer = (0.0, 0.0)
        # The copy batches of the launch in flight (open until sync).  See
        # IO/telemetry/copy_events.py.
        self._log = None

    def bind(self, input_info: List[InStage]):
        """The host half of a launch: the in-stages grouped by exact class,
        in first-seen order, each group with what its class binds before
        any allocation (``InStage.bind_input``).  Hand the result to
        ``launch``."""
        from collections import defaultdict
        groups: Dict[type, List[InStage]] = defaultdict(list)
        for s in input_info:
            groups[type(s)].append(s)
        return [(cls, stages, cls.bind_input(stages))
                for cls, stages in groups.items()]

    def launch(self, input_info: List[InStage], bound=None, holds=None):
        """Issue the tick's input: ``bound`` (``bind``'s result; bound here
        when ``None``) is allocated and copied, and the stages finalized.

        Without ``holds`` (the tick barrier) the launch's copy log stays
        open until ``sync``.  With ``holds`` (event-ordered ticks), the
        buffers the compute stream reads are appended to it, and the
        launch closes its own copy log and returns it: the caller reads it
        once the tick's events have fired and releases it; the exchange
        keeps no log, so a nested launch and sync (``col_to_gpu`` inside an
        operator) does not touch it."""
        if holds is None:
            if self._log is not None:
                # A launch a failure left unsynced: its batches are never
                # read.
                copy_events.discard_log(self._log)
            # Every copy batch from here to sync() is this tick's.
            self._log = copy_events.open_log()
            self._issue(input_info, bound, None)
            return None
        log = copy_events.open_log()
        try:
            self._issue(input_info, bound, holds)
        finally:
            copy_events.close_log(log)
        return log

    def _issue(self, input_info, bound, holds):
        # Class-agnostic per-tick orchestration.  Every InStage subclass
        # exposes ``bind_input`` / ``alloc_input`` / ``launch_h2d`` /
        # ``launch_d2d`` / ``build_decompress_metadata`` (defaults are
        # no-ops); stages are grouped by class (``bind``) and each class's
        # batched sequence runs here.
        # launch_h2d runs on h2d_stream (PCIe), launch_d2d on
        # d2d_stream (HBM) — different copy engines so they overlap.
        # h2d_stream.wait_stream(d2d_stream) after launch_d2d orders
        # all subsequent readers (build_decompress_metadata,
        # phase_finalize) behind the D2D fills.
        if bound is None:
            bound = self.bind(input_info)

        plans: List[tuple] = []  # (cls, state, plan)
        with torch.cuda.stream(self.h2d_stream):
            for cls, stages, token in bound:
                state, plan = cls.alloc_input(stages, token, holds)
                if plan is None:
                    continue
                plans.append((cls, state, plan))
                cls.launch_h2d(state, plan)

        if holds is not None and plans:
            # The allocator may serve the tick's buffers with blocks whose
            # frees on another stream (the compute stream's, while the host
            # runs ahead) have not fired yet; it orders only the allocating
            # stream (h2d) behind them.  The d2d stream writes into the
            # same buffers, so it waits for h2d first.
            self.d2d_stream.wait_stream(self.h2d_stream)
        for cls, state, plan in plans:
            if isinstance(state, torch.Tensor):
                state.record_stream(self.d2d_stream)
            with torch.cuda.stream(self.d2d_stream):
                cls.launch_d2d(state, plan)
        # Every copy reading a bound GPU region is queued: the free-list
        # baseline may evict them from here on (BoundTickPlan.unpin).
        for _, _, plan in plans:
            unpin = getattr(plan, "unpin", None)
            if unpin is not None:
                unpin()

        self.h2d_stream.wait_stream(self.d2d_stream)
        with torch.cuda.stream(self.h2d_stream):
            for cls, state, plan in plans:
                cls.build_decompress_metadata(state, plan)
            for stage in input_info:
                stage.phase_finalize(self.d2d_stream)
        if holds is not None:
            for stage in input_info:
                holds.extend(stage.issue_holds())

    def sync(self):
        self.h2d_stream.synchronize()
        self.d2d_stream.synchronize()
        self._calculate_times()

    def _calculate_times(self):
        """Close the launch's copy log and take its lanes: per stream, the
        span from the first copy batch's start to the last one's end.  A
        sync with no launch since the last one changes nothing."""
        log = self._log
        if log is None:
            return
        self._log = None
        copy_events.close_log(log)
        lanes = log.lane_ms()
        self._last_transfer = (lanes['h2d'], lanes['d2d'])
        log.release()

    def get_last_times(self) -> tuple:
        """The last synced tick's ``(h2d, d2d)`` lane times (ms)."""
        return self._last_transfer


def set_exchange_to_naive():
    global _exchange
    _exchange = NaiveExchange()


def exchange():
    global _exchange
    if _exchange is None:
        _exchange = NaiveExchange()
    return _exchange


def h2d_b_per_ms(nbytes: int = 256 << 20, reps: int = 3) -> float:
    """The host-to-device copy rate (bytes per ms) of ``nbytes`` from pinned
    memory, the best of ``reps`` copies on the exchange's h2d stream: what
    one more crossing of a byte costs (the write leg's readback price,
    ``adaptive.resource_monitor.PCIE_B_PER_MS``)."""
    stream = exchange().h2d_stream
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device='cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = 0.0
    with torch.cuda.stream(stream):
        for _ in range(reps):
            start.record(stream)
            dev.copy_(host, non_blocking=True)
            end.record(stream)
            end.synchronize()
            best = max(best, nbytes / start.elapsed_time(end))
    del host, dev
    # Give the pinned buffer back: torch's host allocator would otherwise
    # keep it cached for the life of the process.
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is not None:
        empty()
    return best
