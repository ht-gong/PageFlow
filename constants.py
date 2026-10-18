import torch

string_dtype = torch.int8
int_dtype = torch.int64
float_dtype = torch.float64
date_dtype = torch.int32

eps = 1e-5
null = -64

COMPRESSION_ALIGNMENT = 512

# ── Active query context ────────────────────────────────────────────────
# Bound once per query by ``main.exec`` before the plan is built.  Read by
# the adaptive sizing plane (``sizing_resolver``, ``adaptive.sizing_controller``)
# to key memory models and log lines by (scale factor, query id).  Both stay
# ``None`` outside a query (unit tests that call operators directly), so
# readers must tolerate that.
_query_sf = None
_query_id = None

def set_query_context(scale_factor, qid) -> None:
    global _query_sf, _query_id
    _query_sf = int(scale_factor)
    try:
        _query_id = int(qid)
    except (TypeError, ValueError):
        _query_id = qid

def get_query_sf():
    return _query_sf

def get_query_id():
    return _query_id

# ── Tunable pipeline sizes (bytes) ──────────────────────────────────────
# THE chunk budget — ONE uniform value for every chunked pipeline: scan /
# filter / project, aggregate, the join core, the join's sort pipe, and the
# rearrange gather window.  Set once from --chunk_size_mb.  There is no
# per-pipeline chunk knob: the old --sort_chunk_size_mb (1024 MB sort-pipe
# override) and --rearrange_chunk_gb (8 GB gather window) were removed when
# the fixed-chunk baseline was established, so that a single swept number
# describes the whole configuration.
#
# The budget is *charged* differently per pipe class.  That is inherent, not
# an inconsistency — it is one byte ceiling, not one row count:
#   * tick pipes (filter / aggregate / join core / sort) charge INPUT bytes
#     per tick;
#   * rearrange charges OUTPUT row bytes PLUS argsort scratch
#     (_ARGSORT_OVERHEAD_PER_ROW = 24 B/row, IO/pipeline/rearrange_pipeline.py), so for
#     an 8-byte column only ~25% of the budget is actual transfer.
#
# Raising it cuts per-chunk barrier count and deepens in-flight PCIe, at a
# ~linear cost in peak GPU.  The rearrange leg dominates that peak (the full
# output index stays resident regardless of the window), and the peak is what
# OOMs first at large SF — so the slot pool is sized against whatever value
# is chosen here, not the other way round.
_chunk_size_b: int = 256 * 1024 ** 2      # default 256 MiB

def get_chunk_size_b() -> int:
    return _chunk_size_b

def set_chunk_size_b(val: int):
    global _chunk_size_b
    _chunk_size_b = val

# The partition budget (--partition_budget_mb): the bytes of the largest unit
# an operator holds whole.  A join's unit is a bucket pair (its left and right
# rows, the gidx included); a keyed aggregation's is a pass-2 round of
# adjacent buckets (its partial rows times the run's bytes per row).  Each
# join and keyed aggregation derives its hash-partition count B from it
# (operators/partitions.py): the smallest power of two whose largest unit is
# predicted to fit; it then packs whole measured buckets.  A bucket is
# un-splittable — the join core packs whole buckets into ticks — so this is
# the per-tick floor the join core cannot go below.
_partition_budget_b: int = 512 * 1024 ** 2   # default 512 MiB

def get_partition_budget_b() -> int:
    return _partition_budget_b

def set_partition_budget_b(val: int):
    global _partition_budget_b
    assert val > 0, f"partition budget must be positive, got {val}"
    _partition_budget_b = int(val)


