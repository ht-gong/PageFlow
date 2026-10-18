"""Post-load static plan pipeline shared by cached and freshly parsed plans."""

from .materialization import compute_deferred_materialization
from .uniqueness import annotate_query_plans


def prepare_query_plans(query_plans, qid: int):
    """Run every deterministic post-load plan analysis in dependency order.

    Cached plan pickles hold the stable lowered plan, query fixes included.
    Relational proof metadata and payload deferral are reconstructed on
    every load so fresh and cached plans share one static parsing path.
    """
    annotate_query_plans(query_plans)
    for plan in query_plans:
        compute_deferred_materialization(plan)
    return query_plans


def q11_fraction(scale_factor):
    """Q11's FRACTION at ``scale_factor`` as qgen prints it: 0.0001 / SF to 10
    decimals (SF700 gives 0.0000001429).

    The same as ``fraction`` in ``baselines/results/queries/stream.py``, which
    fills it into the baselines' query stream; the engine image cannot import
    it (``.dockerignore`` leaves ``baselines/`` out).
    """
    return "%.10f" % (0.0001 / scale_factor)


def subquery_value(raw, producing_plan, scale_factor):
    """The scalar a subquery plan produced, scaled as the plan asks.

    Applied once, where the value is captured: ``producing_plan`` is the
    plan that ran, and only a plan lowered with ``subquery_scale == "1/SF"``
    (Q11's subquery, whose text has the SF1 FRACTION, 0.0001) is rescaled to
    the scale factor's: multiplied by ``q11_fraction(scale_factor) / 0.0001``.
    """
    if producing_plan.subquery_scale == "1/SF":
        return raw * (float(q11_fraction(scale_factor)) / 0.0001)
    return raw
