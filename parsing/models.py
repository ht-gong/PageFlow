"""Shared data models for parsed physical plans."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryPlanRep:
    ops_idx: list = field(default_factory=list)
    ops_dict: dict = field(default_factory=dict)
    query_parent_map: dict = field(default_factory=dict)
    query_child_map: dict = field(default_factory=dict)
    disk_read_cols: dict = field(default_factory=dict)
    # "1/SF" on a subquery plan whose scalar result the runtime rescales to the
    # scale factor's fraction (Q11: tpch-q11.sql hard-codes the SF1 fraction;
    # pipeline.subquery_value).  A class attribute default, so pickles that
    # predate the field read None.
    subquery_scale: str | None = None


@dataclass
class ScannedTuple:
    name: str
    identifier: int
    table: str
