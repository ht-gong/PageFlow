"""Compression codec registry.

Every compressed record is an nvCOMP manager (``NVCOMP_NATIVE``) object
produced and consumed through ``TQPlib.tqpcompress``'s ``mgr_*`` bindings
(``compression/nvcomp_manager.cu``).  This module holds the codec names and
indices those bindings key on; a record's index lives in its record table
(``IO/pages/record_table.py``).  The per-codec manager chunk size is not
here: it comes from the codec map (``compression/format_policy.py``, the
sweep's ``profiling/results/codec_map.json``) through :func:`mgr_chunk_size`.
The module imports no CUDA, so plan-time and solver code can use it freely.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Static registry (no CUDA)
# ---------------------------------------------------------------------------

# Codec names in index order; the index is the codec id the manager bindings
# (nvcomp_manager.cu's kCodec* constants) and the dataflow code key on
# (0 = RAW, no compression).
_ALGO_NAMES = ("RAW", "ANS", "CASCADED", "BITCOMP",
               "LZ4", "SNAPPY", "GDEFLATE", "DEFLATE")


def mgr_chunk_size(algo_name: str) -> int:
    """nvCOMP manager internal chunk bytes for *algo_name*.

    A codec parallelism/ratio knob, NOT a storage unit.  The value is the
    codec map's ``settings.<codec>.chunk_b`` (``profiling/results/
    codec_map.json``, written by ``python3 -m profiling.sweep``), or the
    built-in default when no map exists.
    """
    from compression.format_policy import chunk_bytes
    return chunk_bytes(algo_name)


def algo_names() -> list[str]:
    """Ordered codec names, e.g. ["RAW", "ANS", ...]."""
    return list(_ALGO_NAMES)


def name_to_int_map() -> dict[str, int]:
    """{"RAW": 0, "ANS": 1, ...}."""
    return {name: i for i, name in enumerate(_ALGO_NAMES)}
