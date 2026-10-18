"""mix64, the Murmur3 64-bit finalizer the join hashes its keys with (a key's
radix bucket is the top log2(B) bits), for the tests' references: computed
in numpy uint64 so the wrapping multiplies and logical shifts are exact."""
import numpy as np
import torch

_M1 = np.uint64(0xff51afd7ed558ccd)
_M2 = np.uint64(0xc4ceb9fe1a85ec53)
_S = np.uint64(33)


def mix64_np(keys: np.ndarray) -> np.ndarray:
    """mix64 of int64 keys, as uint64."""
    k = np.ascontiguousarray(keys, dtype=np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        k = k ^ (k >> _S)
        k = k * _M1
        k = k ^ (k >> _S)
        k = k * _M2
        k = k ^ (k >> _S)
    return k


def mix64(keys: torch.Tensor) -> torch.Tensor:
    """mix64 of an int64 tensor, as int64 on the tensor's device."""
    out = mix64_np(keys.detach().cpu().numpy()).view(np.int64)
    return torch.from_numpy(out.copy()).to(keys.device)


def bucket_of(keys: torch.Tensor, B: int) -> torch.Tensor:
    """Each key's radix bucket among B (a power of two): the top log2(B)
    bits of mix64."""
    log_b = B.bit_length() - 1
    h = mix64_np(keys.detach().cpu().numpy())
    b = (h >> np.uint64(64 - log_b)).astype(np.int64) if log_b else np.zeros_like(h, np.int64)
    return torch.from_numpy(b).to(keys.device)
