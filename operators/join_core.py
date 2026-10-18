"""The join core pipe's kernels (``operators/kernels/join_core.cu``, the
``TQPlib.tqpkernels`` module), called by ``operators/hashjoin.py`` once per
core tick:

- ``unique_hash_pairs``: an inner join on a build tuple (1-2 int64 keys)
  proven unique, one partner per probe row;
- ``semi_anti``: a semi or anti join on a hash set of the build keys, with an
  optional per-key min/max summary that answers one residual comparison
  between a build column and a probe column;
- ``expand``: the pairs of the sort/search probe (each probe row's run of
  equal keys in the per-bucket sorted build), for every join the hash paths
  do not take.

Each call waits for the GPU once.  The hash paths queue every kernel, then
wait for the output row count: their outputs are allocated for every probe
row and narrowed here, so the count is the only thing the host reads.
``expand`` waits for the pair count, which sizes its outputs.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

import TQPlib.tqpkernels as tqpkernels

# The probe rows a call takes, and a unique-hash build's (join_core.cu's
# MAX_ROWS): CUB counts them in int, a tile below INT_MAX.  Over it a call
# raises.
MAX_ROWS = (1 << 31) - (1 << 16)

# The residual operators a summary answers, oriented build.v OP probe.v
# (join_core.cu's Op): <> reads the key's min and max, < and <= its min,
# > and >= its max.
SUMMARY_OPS = {'<>': 1, '<': 2, '<=': 3, '>': 4, '>=': 5}

# Where a call's kernels leave its output row count and, for semi_anti, the
# flag of a build key that equals the empty-slot value: one buffer for the
# process, since a call has read it back before it returns.
_host: Optional[torch.Tensor] = None


def _host_buffer() -> torch.Tensor:
    global _host
    if _host is None:
        _host = torch.empty(2, dtype=torch.int64, pin_memory=True)
    return _host


def _empty(device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64, device=device)


def expand(sorted_ids: torch.Tensor, start: torch.Tensor,
           length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The sort/search probe's pairs.  ``sorted_ids``: the build rows in key
    order; ``start[i]``, ``length[i]``: probe row i's run of equal keys in it
    (0: no match).  Returns (build rows, probe rows), in probe-row order and,
    within a probe row, run order."""
    return tuple(tqpkernels.join_expand(sorted_ids, start, length,
                                        _host_buffer()))


def unique_hash_peak_bytes(n_build: int, n_probe: int) -> int:
    """The scratch of a ``unique_hash_pairs`` call: the table (int32, two
    slots a build row), the partner and the two outputs, sized for every
    probe row."""
    return 2 * n_build * 4 + n_probe * (4 + 8 + 8)


def _validate_keys(build_keys: Sequence[torch.Tensor],
                   probe_keys: Sequence[torch.Tensor]) -> None:
    if len(build_keys) not in (1, 2) or len(build_keys) != len(probe_keys):
        raise ValueError(
            "unique_hash_pairs expects one or two aligned build/probe keys")
    device = build_keys[0].device
    for side, keys in (("build", build_keys), ("probe", probe_keys)):
        rows = keys[0].numel()
        for key in keys:
            if not key.is_cuda or key.dtype != torch.int64 or key.dim() != 1:
                raise ValueError(
                    f"{side} keys must be 1-D CUDA int64 tensors; got "
                    f"device={key.device}, dtype={key.dtype}, "
                    f"shape={tuple(key.shape)}")
            if not key.is_contiguous():
                raise ValueError(f"{side} keys must be contiguous")
            if key.numel() != rows:
                raise ValueError(f"{side} key lengths disagree")
            if key.device != device:
                raise ValueError(
                    "all build/probe keys must be on one CUDA device")
        if rows > MAX_ROWS:
            raise ValueError(
                f"unique hash core takes at most {MAX_ROWS} rows a side")


def unique_hash_pairs(
    build_keys: Sequence[torch.Tensor],
    probe_keys: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(build rows, probe rows)`` of an inner join whose build tuple is
    unique, in probe-row order.  Correctness rests on the uniqueness proof of
    the static parsing pipeline: a probe row takes the first build row with
    its tuple."""
    _validate_keys(build_keys, probe_keys)
    device = build_keys[0].device
    if build_keys[0].numel() == 0 or probe_keys[0].numel() == 0:
        return _empty(device), _empty(device)
    host = _host_buffer()
    build_idx, probe_idx = tqpkernels.join_unique_hash(
        list(build_keys), list(probe_keys), host)
    n = int(host[0])
    return build_idx[:n], probe_idx[:n]


def semi_anti(build_key: torch.Tensor, probe_key: torch.Tensor, anti: bool,
              summary: Optional[Tuple[torch.Tensor, torch.Tensor, str]] = None
              ) -> Optional[torch.Tensor]:
    """The probe rows, ascending, that have a matching build key (semi) or
    none (anti); with ``summary = (build column, probe column, op)``, a match
    must also pass ``build column op probe column`` (``SUMMARY_OPS``).  Keys
    and columns are 1-D contiguous int64 CUDA tensors, the probe side at most
    ``MAX_ROWS`` rows.  None when a build key equals the table's empty-slot
    value (int64 min): the caller takes the sort/search path."""
    device = build_key.device
    n_build, n_probe = build_key.numel(), probe_key.numel()
    if n_probe == 0:
        return _empty(device)
    if n_build == 0:
        return (torch.arange(n_probe, dtype=torch.int64, device=device)
                if anti else _empty(device))
    build_val = probe_val = None
    op = 0
    if summary is not None:
        build_val, probe_val, op_name = summary
        op = SUMMARY_OPS[op_name]
    host = _host_buffer()
    out = tqpkernels.join_semi_anti(build_key, probe_key, build_val, probe_val,
                                    op, anti, host)
    n, bad = host.tolist()
    if bad:
        return None
    return out[:n]
