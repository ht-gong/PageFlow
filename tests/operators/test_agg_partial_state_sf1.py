"""Q8 and Q14 over several aggregation ticks at SF1 (``operators/aggregate.py``
partial states).

Their results are ratios of two sums: Q14's ``promo_revenue`` (a global
aggregation) and Q8's ``mkt_share`` per year (a keyed one).  Evaluated per
chunk and merged by SUM they came out about k-fold over k chunks (Q14 at
SF100 with the default 256 MiB chunk: 33.27 against DuckDB's 16.64).  At
SF1 the default chunk holds each aggregation's input in one tick, so these
runs give the aggregation pipes alone (the ``agg`` class, through the
sizing resolver's static layer) a 64 KiB chunk: Q14's aggregation then
takes about fifty ticks and Q8's pass 1 two.  Every result must match
``tests/verification/SF1_reference.out`` at
:data:`~tests.sigfig_compare.SIGFIGS_GEOMETRY`, as in
``tests/operators/test_agg_partitioned_sf1.py``, whose in-process runner
these runs share.
"""
from __future__ import annotations

import sizing_resolver
from operators import aggregate
from tests.operators.test_agg_partitioned_sf1 import _run

MiB = 1 << 20
AGG_CHUNK_B = 64 << 10


def test_q8_and_q14_over_several_aggregation_ticks(sf1_parquet, monkeypatch,
                                                   capsys):
    monkeypatch.setattr(aggregate, "AGGREGATION_PARTITIONS", 0)
    _, hi = sizing_resolver.bounds("agg")
    monkeypatch.setitem(sizing_resolver._bounds_b, "agg", (AGG_CHUNK_B, hi))
    monkeypatch.setitem(sizing_resolver._config_static_b, "agg", AGG_CHUNK_B)
    lines = _run(monkeypatch, capsys, [8, 14], 256 * MiB, 512 * MiB)
    # Q8's keyed aggregation spans pass-1 ticks, so its groups have
    # partials from several of them to merge.
    (q8,) = [a for a in lines.agg_b if a.qid == 8]
    assert q8.ticks >= 2, q8
