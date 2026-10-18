"""The CUDA streams the engine draws from torch's stream pool.

``torch.cuda.Stream()`` hands out the streams of a fixed pool round-robin,
32 per device and priority, so a draw 32 draws after another returns the same
stream.  The engine keeps a few streams for the whole process (the exchange's
H2D and D2D streams, the home writer's) and draws a new compute stream for every pipe.  A
pipe's compute stream that came round to a kept stream ran its kernels in
order with that stream's copies: on the exchange's H2D stream a scan pipe's
compute queued behind the next tick's prefetch instead of overlapping it, and
which pipes it hit moved with the number of streams drawn before them.

So the kept streams are taken together, every role at once, the first time
any of them is asked for on a device (``dedicated``), and every other draw
(``compute_stream``) skips them.  The pipe running at the first take drew
its compute stream before it, and the take draws only ``len(ROLES)`` < 32
more, so none of them is that stream; no later draw returns a kept one."""
from __future__ import annotations

from typing import Dict, Set, Tuple

import torch

# torch's pool size per device and priority (c10/cuda/CUDAStream.cpp).
POOL_SIZE = 32

# Every role a kept stream serves, taken together on first use.
ROLES = ('h2d', 'd2d', 'home writer')

_kept: Dict[Tuple[str, int], torch.cuda.Stream] = {}
_kept_handles: Set[Tuple[int, int]] = set()


def _device_index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    device = torch.device(device)
    return device.index if device.index is not None \
        else torch.cuda.current_device()


def _draw(index: int) -> torch.cuda.Stream:
    """The pool's next stream on device ``index`` that is not kept."""
    for _ in range(POOL_SIZE + 1):
        stream = torch.cuda.Stream(device=index)
        if (index, stream.cuda_stream) not in _kept_handles:
            return stream
    raise RuntimeError("every stream of the pool is kept")


def dedicated(role: str, device=None) -> torch.cuda.Stream:
    """The stream kept for ``role`` (one of ``ROLES``) on ``device``
    (default: the current one).  The first call on a device takes every
    role's stream from the pool; later calls return the same streams."""
    assert role in ROLES, f"unknown kept-stream role {role!r}"
    index = _device_index(device)
    if (role, index) not in _kept:
        for r in ROLES:
            stream = _draw(index)
            _kept[(r, index)] = stream
            _kept_handles.add((index, stream.cuda_stream))
    return _kept[(role, index)]


def compute_stream(device=None) -> torch.cuda.Stream:
    """A stream from the pool that is none of the kept streams."""
    return _draw(_device_index(device))
