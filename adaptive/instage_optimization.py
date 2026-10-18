from dataclasses import dataclass
from typing import Dict, List, Optional
from compression.compression_algorithms import algo_names as _algo_names_fn
from utility.logger import message_logger


# ---------------------------------------------------------------------------
# Fixed-compression mode: a single algorithm for all reads and writes.
# Set via --fixed_compression CLI argument.
# ---------------------------------------------------------------------------

_global_fixed_compression: str = "RAW"

def set_fixed_compression(alg: str):
    global _global_fixed_compression
    assert alg in _algo_names_fn(), (
        f"fixed_compression={alg!r} not in registry: {_algo_names_fn()}"
    )
    _global_fixed_compression = alg

def get_fixed_compression() -> str:
    return _global_fixed_compression


# ---------------------------------------------------------------------------
# Adaptive compression mode (per-(type, sorted) profiled policy).
# Mutually exclusive with non-RAW fixed compression; off by default so the
# fixed/RAW paths are unchanged.
# ---------------------------------------------------------------------------

_global_adaptive_compression: bool = False

def set_adaptive_compression(on: bool = True):
    global _global_adaptive_compression
    _global_adaptive_compression = bool(on)

def get_adaptive_compression() -> bool:
    return _global_adaptive_compression


# ---------------------------------------------------------------------------
# --compression_search fixed: path of the per-tag codec JSON whose pins the
# applier applies instead of searching.  None in every other search mode.
# ---------------------------------------------------------------------------

_global_fixed_search_json = None

def set_fixed_search_json(path):
    global _global_fixed_search_json
    _global_fixed_search_json = path

def get_fixed_search_json():
    return _global_fixed_search_json


def active_compression_policy():
    """The CompressionPolicy implied by the current global mode.

    The single source of truth shared by the scan path (which reps to
    *materialize*) and the applier (which codec to *select*): adaptive ->
    profiled per-(type, sorted) policy; otherwise fixed at the global codec
    (RAW => RAW-only).  Because scan and applier both read this, a column's
    materialized compressed rep always matches the codec the selector requests.
    """
    from compression.format_policy import CompressionPolicy
    if _global_adaptive_compression:
        return CompressionPolicy.adaptive()
    return CompressionPolicy.fixed(_global_fixed_compression)


# ---------------------------------------------------------------------------
# Zero-copy selectivity thresholds.
# Zero-copy is attempted when selectivity < threshold.
# Set threshold to 0.0 to disable the corresponding zero-copy path.
# ---------------------------------------------------------------------------

_global_mask_selectivity: float = 1.0

def set_zero_copy_selectivity(mask: float = 0.1):
    global _global_mask_selectivity
    _global_mask_selectivity = mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_all_cpu_rep(ccr):
    """Find a RAW rep in the CombinedColumnRep whose regions are all in
    pinned host memory (``TableVariable.is_host_raw``)."""
    for rep in ccr.reps:
        if rep.table_backed and rep.is_host_raw():
            return rep
    return None


_mixed_rep_acceptance: bool = False

def set_mixed_rep_acceptance(on: bool = True):
    """Mixed reps (``read-and-write``): a producer that flipped its write
    picks mid-run leaves its single materialized rep with mixed RAW /
    candidate records (or all-RAW despite a non-RAW static resolution).  With
    this on, :func:`find_rep_by_algorithm` accepts a single-rep column's
    only rep when the target codec isn't materialized — exactly the
    singleton-menu columns whose rep selection has no alternative.  Scan
    columns (multi-rep) keep the strict-selector semantics untouched."""
    global _mixed_rep_acceptance
    _mixed_rep_acceptance = bool(on)


def find_rep_by_algorithm(data_col, target_alg: str, strict: bool = False):
    """Find the Variable in a CombinedColumnRep matching *target_alg*.

    When *strict* is True (oracle/applier paths), raises if the algorithm
    is not found.  When False (fixed-compression default), falls back to
    the first rep since operators like Sort legitimately produce RAW-only
    output.
    """
    for rep in data_col.reps:
        if target_alg == "RAW":
            if not rep.is_compressed:
                return rep
        else:
            if rep.is_compressed:
                # The first compressed record speaks for the rep's codec;
                # a mixed rep (write-flip straddle) may lead with RAW
                # records, so don't insist on the first record.
                if rep.first_codec_name() == target_alg:
                    return rep
    if _mixed_rep_acceptance and len(data_col.reps) == 1:
        # Mixed-rep acceptance: singleton-menu column (one materialized
        # rep) whose producer's write picks diverged from the consumer's
        # static resolution — the rep IS the column; records
        # self-describe and the mixed decode paths handle them per codec.
        return data_col.reps[0]
    available = []
    for rep in data_col.reps:
        if not rep.is_compressed:
            available.append("RAW")
        else:
            available.append(rep.leading_codec_name() or "?")
    tag = getattr(data_col, 'tag', None)
    n_reps = len(data_col.reps)
    message_logger().debug(
        "find_rep_by_algorithm: target=%s available=%s tag=%s n_reps=%d strict=%s",
        target_alg, available, tag, n_reps, strict,
    )
    if strict:
        raise RuntimeError(
            f"oracle requires algorithm {target_alg!r} but CombinedColumnRep "
            f"(tag={tag}, n_reps={n_reps}) only has: {available}"
        )
    return data_col.reps[0]


# ---------------------------------------------------------------------------
# Zero-copy detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Init-time decision helper: decide transfer type once per column.
# ---------------------------------------------------------------------------

@dataclass
class ColumnDecision:
    stage_type: str        # "direct" | "zero_copy_mask"
    algorithm: str | None  # compression alg for direct, None for zero-copy

def decide_column_type(data_col, mask_col, fallback_alg: str) -> ColumnDecision:
    """Decide transfer type once per column (not per chunk).

    A zero-copy mask read when the column has a RAW rep in pinned host
    memory and its mask keeps fewer rows than the threshold (its
    pre-computed ``selectivity``); else a direct transfer with the
    pre-resolved *fallback_alg* compression algorithm.  The caller
    (VortexPipeline) is responsible for resolving *fallback_alg* —
    typically via the applier or the global fixed-compression setting.

    Args:
        data_col: CombinedColumnRep for this column (first chunk as representative).
        mask_col: CombinedColumnRep for the mask column, or None.
        fallback_alg: Compression algorithm name to use when zero-copy is
            not selected.

    Returns:
        ColumnDecision with stage_type and algorithm.
    """
    if (mask_col is not None and _global_mask_selectivity > 0
            and mask_col.reps and _find_all_cpu_rep(data_col) is not None):
        selectivity = mask_col.reps[0].selectivity
        assert selectivity is not None, "mask Variable missing pre-computed selectivity"
        if selectivity < _global_mask_selectivity:
            return ColumnDecision("zero_copy_mask", None)
    return ColumnDecision("direct", fallback_alg)
