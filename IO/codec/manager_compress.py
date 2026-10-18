"""nvCOMP manager codec facts the region producer compresses with.

The engine compresses through ``IO.codec.region_producer`` only: records of
whole rows, one batched nvCOMP manager (``NVCOMP_NATIVE``) call per piece of
worst-case output.  This module holds what that path asks nvCOMP about a
codec, cached per process: its codec id and manager chunk
(:func:`_codec_info`) and a record's worst-case compressed bound
(:func:`_worst_case_aligned`).  It also tears the manager caches down at
exit.
"""
from __future__ import annotations

from typing import Optional

# Tear the manager caches down BEFORE interpreter/CUDA/allocator shutdown:
# managers retain scratch allocated through torch's (pluggable) allocator, and
# their static-destructor-time release would land after the allocator is gone.
import atexit as _atexit


def _mgr_atexit():
    try:
        from TQPlib.tqpcompress import mgr_shutdown
        mgr_shutdown()
    except Exception:
        pass


_atexit.register(_mgr_atexit)


def _codec_info(algo_name: str, chunk_b: Optional[int] = None):
    """``(codec id, manager chunk bytes)`` for *algo_name*.

    ``chunk_b`` overrides the policy's chunk (``mgr_chunk_size``); the sweep
    uses it to measure every chunk of its grid through the producer.
    """
    from compression.compression_algorithms import (
        name_to_int_map, mgr_chunk_size)
    if chunk_b is None:
        chunk_b = mgr_chunk_size(algo_name)
    assert int(chunk_b) > 0, f"manager chunk must be positive, got {chunk_b}"
    return name_to_int_map()[algo_name], int(chunk_b)


_worst_case_cache: dict = {}
# ``TQPlib.tqpcompress.mgr_max_compressed_size``, imported on the first cache
# miss and then kept: ``_worst_case_aligned`` runs once per record, and the
# module stays importable without the native extension.
_mgr_max_compressed_size = None


def _worst_case_aligned(codec_idx, elem_size, chunk_b, nbytes, alignment):
    global _mgr_max_compressed_size
    key = (codec_idx, elem_size, chunk_b, nbytes)
    m = _worst_case_cache.get(key)
    if m is None:
        if _mgr_max_compressed_size is None:
            from TQPlib.tqpcompress import mgr_max_compressed_size
            _mgr_max_compressed_size = mgr_max_compressed_size
        m = _mgr_max_compressed_size(codec_idx, elem_size, chunk_b, nbytes)
        _worst_case_cache[key] = m
    return (m + alignment - 1) // alignment * alignment
