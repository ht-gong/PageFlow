"""Shared helpers for the adaptive-plane unit tier (CPU-only).

``build_plan_input_menus`` builds a decision menu for every input of a real
query plan's dataflow graph, with synthetic rep presence.
"""


# --------------------------------------------------------------------------- #
# Menu construction over a real plan with synthetic rep presence
# --------------------------------------------------------------------------- #

# Synthetic byte counts for simulated reps (the decision-universe and
# pick-vector tests need ranges, not real sizes; distinct values keep raw
# vs stored distinguishable in assertions).
RAW_B, STORED_B = 1000, 400


def build_plan_input_menus(graph, policy):
    """Simulate rep presence for every input instance of every node
    in ``graph`` and build its choice range.

    * scan-produced instance: the RAW rep always exists; the single
      compressed rep exists beside it iff the policy resolves a candidate;
    * operator-output instance: the producer wrote exactly one rep, at the
      static default resolution (read-leg discovery is emergent).

    Returns ``(applier, entries)`` with ``entries`` a list of
    ``(node, instance, ColumnChoices)``.
    """
    from adaptive.decision_menu import RAW, build_input_choices
    from dataflow.compression_applier import CompressionApplier
    ap = CompressionApplier(graph, policy=policy)
    entries = []
    for node in graph.nodes:
        for inst in node.inputs:
            if inst.producer.kind == "scan":
                rep_bytes = {RAW: RAW_B}
                cand = ap.candidate_for(inst)
                if cand is not None:
                    rep_bytes[cand] = STORED_B
            else:
                rep_bytes = {ap.codec(inst): STORED_B}
            entries.append((node, inst, build_input_choices(rep_bytes, RAW_B)))
    return ap, entries
