"""Per-(type, sorted) compression format policy and the codec map.

Single source of truth for "which compressed format to materialize on the scan
path and select at runtime per column", keyed on ``(mapped_type, is_sorted)``,
and for the nvCOMP manager chunk size of every codec.

There are two compression *modes*:

* **fixed** — force one codec for every column (``--compression <codec>``),
  *except* tags the plan marks RAW-required, which the applier keeps RAW.
* **adaptive** — pick the swept best codec for the column's
  ``(type, is_sorted)`` class; the specific pick may vary at runtime as a
  column's sortedness becomes known.

Both the class table and the chunk sizes come from **the codec map**,
``profiling/results/codec_map.json``, written by the offline sweep
(``python3 -m profiling.sweep``, ``make profiling``).  This module is its
only reader.  The file is loaded once per process, on first use
(:func:`codec_map`), never at import:

* a **missing** file means the built-in defaults (:data:`_DEFAULT_BEST`,
  :data:`_DEFAULT_SETTINGS`) plus one warning on stderr;
* anything **wrong** in a present file (bad JSON, a missing or extra key, an
  unknown codec, a non-positive chunk) raises and stops the run.

The module is intentionally **dependency-free** (stdlib + ``dataclasses`` only)
so it is trivially unit-testable and safe to import from plan-time code with
no CUDA.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

RAW = "RAW"

# Mapped data types (see IO/data_scanner.py::TYPE_MAPPING) and the compressed
# codecs the engine knows about (compression/compression_algorithms.py,
# compression/nvcomp_manager.cu).  Kept as local constants to honour the
# dependency-free design goal.
MAPPED_TYPES = ("int", "float", "date", "string")
CODECS = ("ANS", "CASCADED", "BITCOMP", "LZ4", "SNAPPY", "GDEFLATE", "DEFLATE")
_VALID_CODECS = (RAW,) + CODECS

# Built-in class table: the last shipped verdict (byte-weighted over the 22
# TPC-H queries).  key: (mapped_type, is_sorted) -> codec (None => RAW-only).
_DEFAULT_BEST: Dict[Tuple[str, bool], Optional[str]] = {
    ("int",    False): "BITCOMP",  ("int",    True): "CASCADED",
    ("float",  False): "ANS",      ("float",  True): "CASCADED",
    ("date",   False): "BITCOMP",  ("date",   True): "CASCADED",
    ("string", False): "ANS",      ("string", True): "ANS",
}

# Built-in nvCOMP manager internal chunk sizes (uncompressed bytes per
# manager chunk): the tuned values for ANS / CASCADED / BITCOMP, 64 KiB for
# the codecs the default table never picks (they only matter under
# ``--compression <codec>`` without a map).
_DEFAULT_SETTINGS: Dict[str, int] = {
    "ANS": 131_072,       # 128 KiB
    "CASCADED": 32_768,   # 32 KiB
    "BITCOMP": 131_072,   # 128 KiB
    "LZ4": 65_536,
    "SNAPPY": 65_536,
    "GDEFLATE": 65_536,
    "DEFLATE": 65_536,
}

# Default location: <repo>/profiling/results/codec_map.json.  The profiling
# service writes it there and every engine-image service mounts the directory.
DEFAULT_CODEC_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "profiling", "results", "codec_map.json")


def _norm_codec(codec: Optional[str]) -> Optional[str]:
    """Normalize a codec string: ``None``/``""``/``"RAW"``/``"NONE"`` -> ``None``."""
    if codec in (None, "", RAW, "NONE", "None", "none"):
        return None
    if codec not in _VALID_CODECS:
        raise ValueError(f"unknown codec {codec!r}; valid: {_VALID_CODECS}")
    return codec


# ---------------------------------------------------------------------------
# The codec map: one load per process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodecMap:
    """The sweep's verdict: the class table plus one chunk size per codec."""
    best: Dict[Tuple[str, bool], Optional[str]]
    chunk_b: Dict[str, int]
    source: str                 # file path, or "defaults"
    mtime: Optional[float] = None

    def describe(self) -> str:
        """One-line summary for startup / preflight lines."""
        if self.mtime is None:
            src = "defaults"
        else:
            src = (f"{self.source} (mtime "
                   f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.mtime))} UTC)")
        cells = " ".join(
            f"{t}:{self.best[(t, False)] or RAW}/{self.best[(t, True)] or RAW}"
            for t in MAPPED_TYPES)
        chunks = " ".join(f"{c}={self.chunk_b[c]}" for c in CODECS)
        return f"source={src} | codec(unsorted/sorted) {cells} | chunk_b {chunks}"


def default_codec_map() -> CodecMap:
    return CodecMap(best=dict(_DEFAULT_BEST), chunk_b=dict(_DEFAULT_SETTINGS),
                    source="defaults", mtime=None)


def parse_codec_map(doc, path: str = "<codec map>") -> CodecMap:
    """Strictly validate a decoded ``codec_map.json`` document.

    The shape is exactly::

        {"codec":    {<type>: {"unsorted": <codec|RAW>, "sorted": <codec|RAW>}
                      for every mapped type},
         "settings": {<codec>: {"chunk_b": <positive int>} for all 7 codecs}}

    Any deviation raises ``ValueError``.
    """
    def bad(msg):
        raise ValueError(f"[format_policy] {path}: {msg}")

    if not isinstance(doc, dict) or set(doc) != {"codec", "settings"}:
        bad("top level must have exactly the keys 'codec' and 'settings'"
            f" (got {sorted(doc) if isinstance(doc, dict) else type(doc).__name__})")
    codec, settings = doc["codec"], doc["settings"]
    if not isinstance(codec, dict) or set(codec) != set(MAPPED_TYPES):
        bad(f"'codec' must have exactly the types {list(MAPPED_TYPES)}")
    best: Dict[Tuple[str, bool], Optional[str]] = {}
    for t in MAPPED_TYPES:
        cell = codec[t]
        if not isinstance(cell, dict) or set(cell) != {"unsorted", "sorted"}:
            bad(f"'codec.{t}' must have exactly the keys 'unsorted' and 'sorted'")
        for k, srt in (("unsorted", False), ("sorted", True)):
            v = cell[k]
            if not isinstance(v, str) or v not in _VALID_CODECS:
                bad(f"'codec.{t}.{k}' = {v!r} is not one of {_VALID_CODECS}")
            best[(t, srt)] = None if v == RAW else v
    if not isinstance(settings, dict) or set(settings) != set(CODECS):
        bad(f"'settings' must have exactly the codecs {list(CODECS)}")
    chunk_b: Dict[str, int] = {}
    for c in CODECS:
        s = settings[c]
        if not isinstance(s, dict) or set(s) != {"chunk_b"}:
            bad(f"'settings.{c}' must have exactly the key 'chunk_b'")
        v = s["chunk_b"]
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            bad(f"'settings.{c}.chunk_b' = {v!r} is not a positive integer")
        chunk_b[c] = v
    return CodecMap(best=best, chunk_b=chunk_b, source=path)


def load_codec_map(path: str = DEFAULT_CODEC_MAP_PATH) -> CodecMap:
    """Load and validate the map at ``path``.

    Missing file: the defaults, with one warning on stderr.  Present but
    broken (unreadable, bad JSON, wrong shape): ``ValueError``/``OSError``,
    which is meant to stop the run.
    """
    if not os.path.exists(path):
        rel = os.path.relpath(path, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        print(f"[format_policy] WARNING: no codec map at {rel}; using built-in "
              "defaults. Run: make profiling TQP_GPU=<idle>",
              file=sys.stderr, flush=True)
        return default_codec_map()
    with open(path, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except ValueError as e:
            raise ValueError(f"[format_policy] {path}: invalid JSON ({e})") from None
    m = parse_codec_map(doc, path)
    return CodecMap(best=m.best, chunk_b=m.chunk_b, source=path,
                    mtime=os.path.getmtime(path))


_MAP: Optional[CodecMap] = None


def codec_map() -> CodecMap:
    """The process-wide codec map, loaded from :data:`DEFAULT_CODEC_MAP_PATH`
    on first use."""
    global _MAP
    if _MAP is None:
        _MAP = load_codec_map(DEFAULT_CODEC_MAP_PATH)
    return _MAP


def set_codec_map(m: Optional[CodecMap]) -> None:
    """Replace the singleton (tests); ``None`` re-arms the lazy file load."""
    global _MAP
    _MAP = m


def chunk_bytes(codec: str) -> int:
    """nvCOMP manager internal chunk bytes for ``codec`` from the codec map."""
    try:
        return codec_map().chunk_b[codec]
    except KeyError:
        raise KeyError(f"no chunk size for codec {codec!r}; known: {CODECS}") from None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class CompressionPolicy:
    """Resolve the compressed format for a column under the active mode."""

    # "fixed" (force fixed_codec, except RAW-required) | "adaptive" (per (type,sorted))
    mode: str = "adaptive"
    fixed_codec: Optional[str] = None          # used iff mode == "fixed"; None/"RAW" => RAW
    best: Dict[Tuple[str, bool], Optional[str]] = field(
        default_factory=lambda: dict(_DEFAULT_BEST))

    def __post_init__(self):
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError(f"mode must be 'fixed' or 'adaptive', got {self.mode!r}")
        if self.fixed_codec is not None and self.fixed_codec not in _VALID_CODECS:
            raise ValueError(
                f"fixed_codec {self.fixed_codec!r} not in {_VALID_CODECS}")

    # -- queries ----------------------------------------------------------

    def compressed_format(self, mapped_type: str, is_sorted: bool) -> Optional[str]:
        """The single compressed codec for this column, or ``None`` for RAW-only."""
        if self.mode == "fixed":
            return _norm_codec(self.fixed_codec)
        return self.best.get((mapped_type, bool(is_sorted)))

    def default_for(self, mapped_type: str) -> Optional[str]:
        """Safe within-range pick when ``(type, sorted)`` is unresolved.

        The unsorted-class best for the type (or the fixed codec in fixed mode);
        never a sorted-only codec a column might not actually be monotone for.
        """
        if self.mode == "fixed":
            return _norm_codec(self.fixed_codec)
        return self.best.get((mapped_type, False))

    def formats_for(self, mapped_type: str, is_sorted: bool) -> List[str]:
        """The reps the scan path materializes for this column under the mode.

        ``["RAW"]`` when the policy resolves to RAW-only, else
        ``["RAW", <codec>]`` — RAW is always present (correctness + gather), at
        most one compressed rep beside it.
        """
        algo = self.compressed_format(mapped_type, is_sorted)
        return [RAW] if algo is None else [RAW, algo]

    # -- constructors -----------------------------------------------------

    @classmethod
    def fixed(cls, codec: Optional[str]) -> "CompressionPolicy":
        """Fixed mode: force ``codec`` everywhere (``None``/``"RAW"`` => RAW)."""
        return cls(mode="fixed", fixed_codec=codec)

    @classmethod
    def adaptive(cls, best: Optional[Dict[Tuple[str, bool], Optional[str]]] = None
                 ) -> "CompressionPolicy":
        """Adaptive mode seeded from the codec map (or an explicit table)."""
        table = dict(best) if best is not None else dict(codec_map().best)
        return cls(mode="adaptive", best=table)
