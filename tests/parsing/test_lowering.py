"""The checked-in plan text is the artifact of record for the plan pickles.

``queries/plan-q<N>.txt`` is captured by ``parsing/regenerate.py`` under the
pyspark version ``requirements.txt`` pins; each pickle must be exactly the
lowering of its text.  CPU-only: lowering needs neither Spark nor a GPU.
"""

from __future__ import annotations

import pickle

import pytest

from parsing.regenerate import (
    ALL_QUERIES, captured_version, lower_text, pickle_path,
    pinned_pyspark_version, text_path)


def test_pyspark_is_pinned_exactly():
    version = pinned_pyspark_version()
    assert version and version[0].isdigit(), version


@pytest.mark.parametrize("qid", ALL_QUERIES)
def test_plan_text_was_captured_under_the_pinned_pyspark(qid):
    text = text_path(qid).read_text(encoding="utf-8")
    assert captured_version(text) == pinned_pyspark_version(), (
        f"{text_path(qid).name} was not captured by `make plans` under the pinned "
        "pyspark; its column ids would not reproduce")


@pytest.mark.parametrize("qid", ALL_QUERIES)
def test_pickle_is_the_lowering_of_the_plan_text(qid):
    with pickle_path(qid).open("rb") as f:
        pickled = pickle.load(f)
    assert lower_text(qid) == pickled, (
        f"{pickle_path(qid).name} is not the lowering of {text_path(qid).name}; "
        f"run `make plans Q={qid}` (or `--lower-only` after a parsing change)")


@pytest.mark.parametrize("qid", ALL_QUERIES)
def test_only_q11_subquery_plan_is_scaled(qid):
    plans = lower_text(qid)
    scales = [plan.subquery_scale for plan in plans]
    if qid == 11:
        assert scales == ["1/SF", None]
    else:
        assert scales == [None] * len(plans)
