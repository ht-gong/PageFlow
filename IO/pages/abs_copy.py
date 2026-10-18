"""Copies by absolute device address (``batched_memcpy_abs``) and the pinned
buffers their address arrays are staged through.

A launch's arrays (sources, destinations and sizes, int64) sit in a pinned
buffer, which the kernel reads over UVA: no copy ahead of it, so nothing
queues behind other host-to-device copies on the copy engine.  The buffer
can be written again only once the kernel has run.  ``staging`` hands out
kept buffers by their event: one whose last use has ended, else a new one
(the oldest is waited for only when every kept buffer is in flight).  The
caller records the buffer's event after its last kernel reading it and
before it stages again; ``launch`` records it itself.

Users: the region producer's pack (slot parts and dense buffers, its own
kernels over one staged buffer) and the slot copies of its RAW outputs
(``IO/codec/region_producer.py``, ``launch``), and the reader's slot pieces
(``IO/transfer/tick_layout.py``, ``launch_slot_pieces``, ``launch``).
"""
from __future__ import annotations

import torch

# Kept buffers: (buffer, its numpy view, the event after its last copy to the
# GPU).  At most ``_STAGES_MAX``: a buffer too small for a request is
# replaced by a larger one, not kept beside it.
_STAGES: list = []
_STAGES_MAX = 8


def staging(n: int):
    """A pinned int64 buffer of at least ``n`` entries whose last copy out
    has ended, its numpy view and its event: a kept one when one fits, else
    a new one, kept in place of a free one that is too small (with the list
    full and every buffer in flight, of the oldest, once its copy ended)."""
    free = None
    for k, (host, view, event) in enumerate(_STAGES):
        if event.query():
            if host.numel() >= n:
                return host, view, event
            free = k
    if free is None and len(_STAGES) >= _STAGES_MAX:
        free = 0
        _STAGES[0][2].synchronize()
        if _STAGES[0][0].numel() >= n:
            return _STAGES[0]
    host = torch.empty(max(n, 1024), dtype=torch.int64, pin_memory=True)
    event = torch.cuda.Event()
    # Created now (CUDA creates an event at its first record), so ``launch``
    # can record it by its handle.
    event.record()
    stage = (host, host.numpy(), event)
    if free is not None:
        _STAGES[free] = stage
    else:
        _STAGES.append(stage)
    return stage


def launch(src, dst, size, dst_base: int = 0) -> None:
    """One ``batched_memcpy_abs`` on the current stream: ``size[i]`` bytes
    from device address ``src[i]`` to ``dst_base + dst[i]``.  Its arrays are
    staged (``staging``) and the kernel reads them over UVA; the launch and
    the staging buffer's event record are one native call
    (``batched_memcpy_abs_staged``), which spares the host the tensor views,
    the device-pointer lookups and a Python event record per launch."""
    import TQPlib.tqpcompress as tqpcompress
    m = len(size)
    if not m:
        return
    host, view, event = staging(3 * m)
    view[:m] = src
    view[m:2 * m] = dst
    view[2 * m:3 * m] = size
    tqpcompress.batched_memcpy_abs_staged(view.ctypes.data, m, dst_base,
                                          event.cuda_event)
