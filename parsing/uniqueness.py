"""Relational unique-key proof propagation in the static parsing pipeline.

The physical-plan pickles do not carry trustworthy uniqueness metadata and the
legacy ``queries/schema.json::PRIMARY_KEY`` field cannot represent composite
keys.  This module deliberately keeps a separate, correct TPC-H key schema and
annotates each Join with the unique-key sets proven on its physical left/right
inputs.

The annotations are proof metadata only.  They never select an algorithm by
themselves; ``operators.hashjoin`` still checks join type, key dtype/layout,
scratch bounds, and the explicit A/B gate before dispatching to the hash core.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set


UniqueKey = FrozenSet[int]


# TPC-H relational keys.  Keep this separate from schema.json until that file
# can represent a list of (possibly composite) unique-key sets without changing
# its existing scan/compression meaning.
TPC_H_UNIQUE_KEYS = {
    "PART": (("P_PARTKEY",),),
    "SUPPLIER": (("S_SUPPKEY",),),
    "PARTSUPP": (("PS_PARTKEY", "PS_SUPPKEY"),),
    "CUSTOMER": (("C_CUSTKEY",),),
    "ORDERS": (("O_ORDERKEY",),),
    "LINEITEM": (("L_ORDERKEY", "L_LINENUMBER"),),
    "NATION": (("N_NATIONKEY",),),
    "REGION": (("R_REGIONKEY",),),
}


def _sorted_key_sets(keys: Iterable[UniqueKey]) -> List[List[int]]:
    """Return a deterministic, JSON/pickle-friendly representation."""
    return sorted((sorted(key) for key in keys), key=lambda key: (len(key), key))


def _scan_unique_keys(plan, args) -> Set[UniqueKey]:
    scanned = {
        int(item.identifier): (str(item.table).upper(), str(item.name).upper())
        for item in plan.disk_read_cols
    }
    output_ids = set(args.get("output ids", []))
    table = str(args["table name"]).upper()
    name_to_id = {
        scanned[col_id][1]: col_id
        for col_id in output_ids
        if col_id in scanned and scanned[col_id][0] == table
    }

    result: Set[UniqueKey] = set()
    for key_names in TPC_H_UNIQUE_KEYS.get(table, ()):
        if all(name in name_to_id for name in key_names):
            result.add(frozenset(name_to_id[name] for name in key_names))
    return result


def _project_unique_keys(input_keys: Iterable[UniqueKey], args) -> Set[UniqueKey]:
    output_ids = set(args.get("output ids", []))
    rename_map = {
        int(src): int(dst) for src, dst in args.get("rename map", {}).items()
    }
    result: Set[UniqueKey] = set()
    for key in input_keys:
        mapped = frozenset(rename_map.get(col_id, col_id) for col_id in key)
        # Spark assigns new ids to computed expressions.  A surviving id or an
        # explicit rename-map edge is therefore a direct projection/alias.
        if mapped <= output_ids:
            result.add(mapped)
    return result


def _parent_for_group(plan, parents: Sequence[int], group: Sequence[int], *,
                      exclude: int | None = None) -> int:
    """Resolve the physical parent whose output is exactly ``group``.

    ``ParsingLayer.rewrite_ops`` may swap physical join roles without changing
    ``query_parent_map`` order, so list position is not a safe side identity.
    The stored left/right groups are copied from their parent's output ids and
    are the authoritative mapping.
    """
    wanted = set(group)
    candidates = [
        parent for parent in parents
        if parent != exclude
        and set(plan.ops_dict[parent].get("output ids", [])) == wanted
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Repeated subplans can expose identical ids.  Either candidate carries
        # the same relational proof; choose deterministically.
        return candidates[0]
    raise AssertionError(
        f"cannot map join group {sorted(wanted)} to parents {list(parents)}"
    )


def _covers_unique(join_keys: Sequence[int], unique_keys: Iterable[UniqueKey]) -> bool:
    key_set = set(join_keys)
    return any(unique_key <= key_set for unique_key in unique_keys)


def _equality_substitutions(keys: Iterable[UniqueKey], left_keys: Sequence[int],
                            right_keys: Sequence[int]) -> Set[UniqueKey]:
    """Substitute any joined-equal column into a preserved unique key.

    If ``a = x`` and ``b = y``, uniqueness of ``(a, b)`` also proves
    ``(x, b)``, ``(a, y)``, and ``(x, y)`` in the inner-join result.  Carrying
    those equivalents is important when the next Project drops one side's key.
    """
    equivalents: Dict[int, Set[int]] = {}
    for left, right in zip(left_keys, right_keys):
        equivalents.setdefault(left, {left}).add(right)
        equivalents.setdefault(right, {right}).add(left)

    result: Set[UniqueKey] = set()
    for key in keys:
        columns = [sorted(equivalents.get(col_id, {col_id})) for col_id in key]
        for replacement in product(*columns):
            result.add(frozenset(replacement))
    return result


def annotate_plan_unique_keys(plan) -> None:
    """Attach proven input unique-key sets to every Join in ``plan``.

    The pass follows the already-rewritten physical operator order.  It is
    intentionally conservative: an unknown operator drops proof rather than
    guessing, and a join side is marked unique only when its complete equality
    key set covers a propagated relational key.
    """
    proven_by_op: Dict[int, Set[UniqueKey]] = {}

    for op_id in plan.ops_idx:
        args = plan.ops_dict[op_id]
        name = args["name"]
        parents = plan.query_parent_map.get(op_id, [])

        if name == "Scan":
            proven = _scan_unique_keys(plan, args)
        elif name in {"Filter", "Sort", "take ordered"}:
            proven = set(proven_by_op.get(parents[0], set())) if parents else set()
            if args.get("distinct") and args.get("distinct ids"):
                proven.add(frozenset(args["distinct ids"]))
        elif name == "Project":
            parent_keys = proven_by_op.get(parents[0], set()) if parents else set()
            proven = _project_unique_keys(parent_keys, args)
        elif name == "Aggregate":
            group_keys = args.get("out key ids") or []
            proven = {frozenset(group_keys)} if group_keys else set()
        elif name == "Join":
            if len(parents) != 2:
                raise AssertionError(
                    f"join {op_id} expected two parents, got {parents}"
                )
            left_parent = _parent_for_group(plan, parents, args["left group"])
            right_parent = _parent_for_group(
                plan, parents, args["right group"], exclude=left_parent
            )
            left_unique = set(proven_by_op.get(left_parent, set()))
            right_unique = set(proven_by_op.get(right_parent, set()))

            args["left unique keys"] = _sorted_key_sets(left_unique)
            args["right unique keys"] = _sorted_key_sets(right_unique)

            left_join_unique = _covers_unique(args["left keys"], left_unique)
            right_join_unique = _covers_unique(args["right keys"], right_unique)

            join_type = args["join type"]
            if join_type == "Inner":
                preserved: Set[UniqueKey] = set()
                # A side's rows cannot be duplicated when the other side's
                # equality key is unique.
                if right_join_unique:
                    preserved.update(left_unique)
                if left_join_unique:
                    preserved.update(right_unique)
                proven = _equality_substitutions(
                    preserved, args["left keys"], args["right keys"]
                )
            elif join_type in {"LeftSemi", "LeftAnti"}:
                # rewrite_ops makes the retained logical-left relation the
                # physical right side consumed by join_kernel_new.
                proven = right_unique
            elif join_type == "LeftOuter":
                # Physical right rows are retained; they remain unique only if
                # at most one physical-left row can match each of them.
                proven = right_unique if left_join_unique else set()
            else:
                proven = set()
        else:
            # Output/adaptor nodes do not feed a later join in current plans.
            # Preserve a unary input when possible, otherwise fail closed.
            proven = (
                set(proven_by_op.get(parents[0], set()))
                if len(parents) == 1 else set()
            )

        proven_by_op[op_id] = proven


def annotate_query_plans(query_plans) -> None:
    """Annotate every runtime fragment belonging to one TPC-H query."""
    for plan in query_plans:
        annotate_plan_unique_keys(plan)
