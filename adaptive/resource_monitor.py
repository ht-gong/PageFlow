from dataclasses import dataclass
from typing import Optional

# The host-to-device copy rate (bytes per ms) the searches price a byte's
# later crossing of PCIe at: the write leg's readback price
# (``compression_search.WriteWindowSearchDriver``) and the sizing probe's
# (``sizing_controller.TickJumpSession``).  ``main.py`` sets it from a pinned H2D
# copy probe (``data_exchange.h2d_b_per_ms``) when it installs a search;
# 25 GB/s otherwise.
PCIE_B_PER_MS: float = 25e6

@dataclass
class TransferTimes:
    """One tick's transfer lanes (ms), one per exchange stream: the span from
    the stream's first copy batch's start to its last one's end
    (``IO/telemetry/copy_events.py``), host gaps between batches included;
    0 when the stream copied nothing.  The d2d lane holds the eviction
    write-back, which runs on ``d2d_stream``."""
    h2d_ms: float
    d2d_ms: float
    d2h_ms: float
    
    def get_max(self) -> float:
        return max(self.h2d_ms, self.d2d_ms, self.d2h_ms)

@dataclass
class ComputeTimes:
    """One tick's compute-stream spans (ms), host gaps included:
    ``compute`` from the start of the tick's input decompress to the end of
    its output step, and inside it ``compress_out``, the output step (compress
    and pageify).  The searches read ``compute``; the write search's encode
    gate also reads ``compress_out``."""
    compute: float
    compress_out: float = 0.0

class ResourceMonitor:
    def __init__(self):
        # Last-tick retention: references to the very objects the collect_*
        # calls receive, plus the tick index.  An EPHEMERAL view — valid only
        # inside the tick-observer callback (overwritten at the next tick; a
        # consumer that needs history copies the scalars it wants).
        self.last_tick: int = -1
        self.last_transfer: Optional[TransferTimes] = None
        self.last_compute: Optional[ComputeTimes] = None

        # Write-leg view extensions: LIVE references to the pipeline's
        # cumulative per-output-slot byte counters (accumulated at each slot
        # group's BASE slot; a compressed output's stored bytes when its
        # region is packed, a tick after its raw bytes): raw and stored
        # (wire) bytes, the readback-price inputs (a RAW flip grows a
        # slot's bytes by raw − stored on every later crossing).  None
        # until the pipeline wires them (read-and-write mode only).
        self.out_slot_raw_bytes: Optional[list] = None
        self.out_slot_stored_bytes: Optional[list] = None

    def collect_transfer(self, info: TransferTimes):
        """The tick's transfer lanes; the first collect of every tick."""
        # One transfer collect per tick, in tick order — so the tick index is
        # the collect count.  last_compute clears here and is re-set by
        # collect_compute on compute ticks, leaving None on the transfer-only
        # prologue tick.
        self.last_tick += 1
        self.last_transfer = info
        self.last_compute = None

    def collect_compute(self, info: ComputeTimes):
        """The tick's compute spans; absent on the transfer-only prologue."""
        self.last_compute = info

    @property
    def f_last(self) -> float:
        """The per-tick search objective: the slowest of the four overlapped
        legs — all three ``TransferTimes`` lanes plus the FULL compute bracket
        (``ComputeTimes.compute`` = decompress + operator + encode).  Not the
        operator alone: taking the codec work out would erase the term the
        search exists to balance.
        """
        c = self.last_compute.compute if self.last_compute is not None else 0.0
        t = (self.last_transfer.get_max()
             if self.last_transfer is not None else 0.0)
        return max(t, c)
