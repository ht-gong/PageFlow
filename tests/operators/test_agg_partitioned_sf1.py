"""Partitioned aggregation end to end at SF1 (``operators/aggregate.py``).

Every run here is in-process, as ``tests/smoke`` does, and every result
matches ``tests/verification/SF1_reference.out`` at :data:`SIGFIGS_GEOMETRY`
(6, a chunk geometry reorders float partial sums), every row in any order.
The aggregations' own log lines show what ran: each keyed aggregation's
``[agg_b]`` line has its B and the facts it was chosen from, and each
partitioned one's ``partitioned pass 2`` line the route, the number of
rounds and the run's bytes per row.

**B from the budget.**  All 22 queries with B chosen as it ships, at the
default geometry (256 MiB chunks, the 512 MiB budget: no SF1 aggregation
partitions) and at the e2e ladder's (4 MiB chunks, a 2 MiB budget: twelve
do).
Each keyed aggregation's ``[agg_b]`` line must show its cell in
``tests/operators/test_partitions.py`` (``AGG_CELLS``): P, pass 1's ticks,
G, w, the route, the key route's range bound, B and its reason.  Those cells
are where the rule is checked at SF100 and SF1000, so this ties the rule's
inputs there to what the engine computes.

**B forced** (``AGGREGATION_PARTITIONS``) on Q1, Q3, Q13, Q16, Q17, Q18 and
Q20, whose keyed aggregations cover every pass-2 route: hash (Q1, Q13's
first, Q16's two, Q18's last), packed (Q3, Q20), key (Q13's second, Q18's
first two) and dense (Q17).  The cells:

* B = 8 and 64 at a 1 MiB budget: the larger runs (Q13, Q16, Q18) take
  several rounds, Q18's at B = 8 a bucket each;
* B = 8 and 64 at the default 512 MiB: one round each, every row of
  pass 1's bucket-major row space;
* B = 8 at 64 KiB: the packed runs take several rounds too;
* B = 8 at 1 MiB with ``SORTED_RUN_MERGE`` off: the dynamic accumulator
  (Q17 stays dense), in several rounds.

Every forced cell runs at the e2e ladder's 4 MiB chunk: at the default
256 MiB each SF1 pass 1 is one tick and each pass 2 one feed, so no group
would have partials from several ticks to find in its one bucket, and no
feed would cross a round's end.  The budget is the joins' too, so the small
ones split the joins into more buckets as well.

In every run, each partitioned pass 2 runs at the bytes per row its B was
chosen from: the ``[agg_b]`` line's w on the key and hash routes, no more
than it on the packed route and the dynamic accumulator's (the hash route's
row over pass 1's stored dtypes).
"""
from __future__ import annotations

import logging
import pathlib
import re
from collections import namedtuple

import pytest

import constants
from operators import aggregate
from tests.operators.test_partitions import (AGG_CELLS, GEOMETRIES,
                                             agg_cell_facts)

_REPO = pathlib.Path(__file__).resolve().parents[2]
QUERIES = [1, 3, 13, 16, 17, 18, 20]
ALL_QUERIES = list(range(1, 23))
MiB = 1 << 20
CHUNK_MB = 4
FIXED_ROUTES = {"hash", "packed", "key", "dense"}

# (B, budget, sorted-run merge, the routes some aggregation runs in
# several rounds); one round is the only alternative.
CELLS = [
    pytest.param(8, 1 * MiB, True, {"hash", "key"}, id="B8-1MiB"),
    pytest.param(64, 1 * MiB, True, {"hash", "key"}, id="B64-1MiB"),
    pytest.param(8, 512 * MiB, True, set(), id="B8-512MiB"),
    pytest.param(64, 512 * MiB, True, set(), id="B64-512MiB"),
    pytest.param(8, 64 << 10, True, {"hash", "key", "packed"},
                 id="B8-64KiB"),
    pytest.param(8, 1 * MiB, False, {"dynamic"}, id="B8-1MiB-dynamic"),
]

AggB = namedtuple(
    "AggB", "qid name b reason rows ticks groups row_b route key_range")
Pass2 = namedtuple("Pass2", "qid name b rounds route run_row")

_AGG_B = re.compile(
    r"\[agg_b\] Q(\d+) (.+?): B=(\d+) \(([a-z ]+)\)(?: for [\d.]+ MB)? "
    r"\((\d+) rows, (\d+) ticks, <= (\S+) groups \[.*?\], (\d+) B/row, "
    r"route (\w+)(?:, key range <= (\S+) \[.*?\])?\)")
_PASS2 = re.compile(
    r"\[(.+?)\] partitioned pass 2: B=(\d+) rounds=(\d+) route=(\w+) "
    r"run_row=(\d+) B")


def _int(value):
    return None if value in (None, "?") else int(value)


class _AggLines(logging.Handler):
    """Each keyed aggregation's ``[agg_b]`` line (``AggB``) and partitioned
    pass-2 line (``Pass2``), in the order they are logged."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.agg_b, self.pass2 = [], []

    def emit(self, record):
        if not isinstance(record.msg, str):
            return
        if record.msg.startswith("[agg_b]"):
            (qid, name, b, reason, rows, ticks, groups, row_b, route,
             key_range) = _AGG_B.match(record.getMessage()).groups()
            self.agg_b.append(AggB(
                int(qid), name, int(b), reason, int(rows), int(ticks),
                _int(groups), int(row_b), route, _int(key_range)))
        elif "partitioned pass 2:" in record.msg:
            name, b, rounds, route, run_row = _PASS2.match(
                record.getMessage()).groups()
            self.pass2.append(Pass2(
                int(constants.get_query_id()), name, int(b), int(rounds),
                route, int(run_row)))


def _run(monkeypatch, capsys, qids, chunk_b, budget_b):
    """Run ``qids`` at SF1 in-process at ``chunk_b`` and ``budget_b``,
    check every result against the reference, and return the aggregations'
    log lines."""
    import main
    from IO.cache.config import BufferStrategy, set_global_buffer_strategy
    from IO.transfer.data_exchange import set_exchange_to_naive
    from adaptive.instage_optimization import set_adaptive_compression
    from tests.sigfig_compare import (SIGFIGS_GEOMETRY, compare_results,
                                      results_of)
    from utility.logger import message_logger, set_message_logger

    # main.exec silences the message log under no_debug; let the
    # aggregations' info lines through to the handler alone.
    lines = _AggLines()

    def set_logger(filename=None, enable=True):
        set_message_logger(filename, enable)
        message_logger().setLevel(logging.INFO)
        message_logger().addHandler(lines)
    monkeypatch.setattr(main, "set_message_logger", set_logger)

    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.CPU)
    set_exchange_to_naive()
    set_adaptive_compression(True)
    default_chunk = constants.get_chunk_size_b()
    default_budget = constants.get_partition_budget_b()
    constants.set_chunk_size_b(chunk_b)
    constants.set_partition_budget_b(budget_b)
    try:
        main.exec(1, qids, None, None, True, None)
    finally:
        constants.set_chunk_size_b(default_chunk)
        constants.set_partition_budget_b(default_budget)
        set_adaptive_compression(False)
        message_logger().removeHandler(lines)
        set_message_logger(None, False)

    out = capsys.readouterr().out
    failed = [ln for ln in out.splitlines() if "[QFAIL]" in ln]
    assert not failed, "query failures:\n" + "\n".join(failed)
    res = (_REPO / "results.out").read_text(encoding="utf-8")
    ref = (_REPO / "tests" / "verification" / "SF1_reference.out").read_text(
        encoding="utf-8")
    compare_results(res, results_of(ref, qids), SIGFIGS_GEOMETRY)
    return lines


def _check_run_rows(lines):
    """Each partitioned pass 2 belongs to one [agg_b] line, has its B and
    runs at the bytes per row B was chosen from: w on the key and hash
    routes, no more than w on the packed route and the dynamic
    accumulator's.  A dense state's row is not what B is chosen from."""
    by_agg = {(a.qid, a.name): a for a in lines.agg_b}
    assert len(by_agg) == len(lines.agg_b), lines.agg_b
    for p2 in lines.pass2:
        agg = by_agg[(p2.qid, p2.name)]
        assert p2.b == agg.b, (agg, p2)
        if p2.route in ("key", "hash"):
            assert p2.run_row == agg.row_b, (agg, p2)
        elif p2.route in ("packed", "dynamic"):
            assert p2.run_row <= agg.row_b, (agg, p2)


@pytest.mark.parametrize("geometry", ["SF1", "e2e"])
def test_b_from_the_budget_is_the_rules(sf1_parquet, monkeypatch, capsys,
                                        geometry):
    monkeypatch.setattr(aggregate, "AGGREGATION_PARTITIONS", 0)
    _, chunk_mb, budget_mb = GEOMETRIES[geometry]
    lines = _run(monkeypatch, capsys, ALL_QUERIES, chunk_mb * MiB,
                 budget_mb * MiB)

    # Every keyed aggregation of the 22 queries is a cell, and logs the
    # cell's facts and B.
    column = list(GEOMETRIES).index(geometry)
    want = {}
    for cell in AGG_CELLS:
        rows, ticks, groups, row_b, key_range = agg_cell_facts(cell, geometry)
        b, route = cell[-1][column], cell[7]
        reason = ("dense" if route == "dense" else "fits" if b == 1
                  else "budget")
        for name in cell[2]:
            want[(cell[1], name)] = AggB(
                cell[1], name, b, reason, rows, ticks, groups, row_b, route,
                key_range)
    got = {(a.qid, a.name): a for a in lines.agg_b}
    assert len(got) == len(lines.agg_b), lines.agg_b
    assert got == want

    partitioned = {(a.qid, a.name) for a in lines.agg_b if a.b > 1}
    assert {(p.qid, p.name) for p in lines.pass2} == partitioned
    assert len(partitioned) == (0 if geometry == "SF1" else 12), partitioned
    _check_run_rows(lines)


@pytest.mark.parametrize(
    "forced_b, budget_b, sorted_run_merge, several", CELLS)
def test_partitioned_aggregations_match_the_reference(
        sf1_parquet, monkeypatch, capsys, forced_b, budget_b,
        sorted_run_merge, several):
    monkeypatch.setattr(aggregate, "AGGREGATION_PARTITIONS", forced_b)
    monkeypatch.setattr(aggregate, "SORTED_RUN_MERGE", sorted_run_merge)
    lines = _run(monkeypatch, capsys, QUERIES, CHUNK_MB * MiB, budget_b)

    # Every query's keyed aggregations took the forced B, and each logged
    # its pass 2 right after its [agg_b] line.
    assert {a.qid for a in lines.agg_b} == set(QUERIES), lines.agg_b
    assert len(lines.pass2) == len(lines.agg_b), (lines.agg_b, lines.pass2)
    for agg, p2 in zip(lines.agg_b, lines.pass2):
        assert (agg.b, agg.reason) == (forced_b, "forced"), agg
        assert (p2.qid, p2.name) == (agg.qid, agg.name), (agg, p2)
    _check_run_rows(lines)
    assert max(a.ticks for a in lines.agg_b) > 1, lines.agg_b
    routes = {p.route for p in lines.pass2}
    assert routes == (FIXED_ROUTES if sorted_run_merge
                      else {"dynamic", "dense"}), lines.pass2
    in_several = {p.route for p in lines.pass2 if p.rounds > 1}
    assert in_several == several, lines.pass2

    with capsys.disabled():
        most = {}
        for p in lines.pass2:
            most[p.route] = max(most.get(p.route, 0), p.rounds)
        print(f"\n[agg partitions] SF1 B={forced_b} at {budget_b >> 10} KiB"
              f"{'' if sorted_run_merge else ', sorted-run merge off'}: "
              f"{len(lines.pass2)} aggregations; most rounds by route "
              + ", ".join(f"{r} {n}" for r, n in sorted(most.items())))

