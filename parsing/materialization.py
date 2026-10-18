"""Static late-materialization analysis for parsed query plans.

The three passes, in the order they run:

1. :func:`seed_materialization` — at lowering.  Normalizes every join so the
   build side is the physical left (swapping keys and inputs under
   ``BuildRight``) and seeds ``materialize_in`` (what the operator reads to
   produce its output; the mask walk uses it), ``materialize_now`` (what it
   produces) and an empty ``materialize_deferred`` on every Join, Filter,
   Aggregate and Project.  A Sort or ``take ordered`` gets only
   ``materialize_in``.
2. :func:`apply_simple_materialization` — at lowering.  Simple late
   materialization: a simple Project materializes nothing, and a Filter
   emits only a mask column, threaded through ``mask map`` down to the
   first Join, Aggregate, Sort or ``take ordered``, which applies it.
3. :func:`compute_deferred_materialization` — on every load, through
   ``pipeline.prepare_query_plans``.  Payload
   columns that can ride a row lineage across a join chain are gathered once,
   at their first consumer after the chain.

The pickles hold the result of the first two; the third is recomputed on
every load.
"""

from .expressions import get_literals, get_project_used_cols, is_simple_projection
from .models import QueryPlanRep


def seed_materialization(plan: QueryPlanRep) -> None:
    """Normalize joins to build-left and seed every operator's materialization."""
    for i in plan.ops_idx:
        if plan.ops_dict[i]['name'] == 'Join':
            assert len(plan.query_parent_map[i]) == 2, "Plan should accept 2 sets of inputs"
            ls, rs = plan.query_parent_map[i]
            left_keys, right_keys = plan.ops_dict[i]['left keys'], plan.ops_dict[i]['right keys']
            if plan.ops_dict[i]['build'] == 'right':
                plan.ops_dict[i]['right keys'], plan.ops_dict[i]['left keys'] = left_keys, right_keys
                ls, rs = rs, ls
            else:
                assert 'Left' not in plan.ops_dict[i]['join type'], "Anti/Semi/Outer only support BuildRight"

            plan.ops_dict[i]['left group'] = plan.ops_dict[ls]['output ids']
            plan.ops_dict[i]['right group'] = plan.ops_dict[rs]['output ids']
            plan.ops_dict[i]['input ids'] = plan.ops_dict[i]['left group'] + plan.ops_dict[i]['right group']
            plan.ops_dict[i]['output ids'] = plan.ops_dict[i]['input ids'].copy()

            used_set = set()
            for key in plan.ops_dict[i]['left keys'] + plan.ops_dict[i]['right keys']:
                used_set.add(key)
            if 'join condition' in plan.ops_dict[i]:
                get_literals(plan.ops_dict[i]['join condition'], used_set)

            plan.ops_dict[i]['join in'] = list(used_set)
            plan.ops_dict[i]['materialize_now'] = plan.ops_dict[i]['output ids'].copy()
            plan.ops_dict[i]['materialize_deferred'] = []
            plan.ops_dict[i]['materialize_in'] = plan.ops_dict[i]['materialize_now'].copy()

        elif plan.ops_dict[i]['name'] == 'Filter':
            used_set = set()
            get_literals(plan.ops_dict[i]['filter tree'], used_set)
            used_set = used_set & set(plan.ops_dict[i]['input ids'])
            plan.ops_dict[i]['materialize_in'] = list(used_set)
            plan.ops_dict[i]['materialize_now'] = list(used_set)
            plan.ops_dict[i]['materialize_deferred'] = []

        elif plan.ops_dict[i]['name'] == 'Aggregate':
            plan.ops_dict[i]['materialize_in'] = plan.ops_dict[i]['input ids'].copy()
            plan.ops_dict[i]['materialize_now'] = plan.ops_dict[i]['output ids'].copy()
            plan.ops_dict[i]['materialize_deferred'] = []

        elif plan.ops_dict[i]['name'] == 'Project':
            rename_map, material_in, material_out, used_expr = get_project_used_cols(
                plan.ops_dict[i]['output'], set(plan.ops_dict[i]['input ids']))
            plan.ops_dict[i]['rename map'] = rename_map
            plan.ops_dict[i]['materialize_in'] = list(material_in)
            plan.ops_dict[i]['materialize_now'] = material_out
            plan.ops_dict[i]['materialize_deferred'] = []
            plan.ops_dict[i]['used expr'] = used_expr

        elif plan.ops_dict[i]['name'] in ('Sort', 'take ordered'):
            # Read by the mask walk only: a filter feeding a sort (a HAVING)
            # hands it a mask, and the sort keeps the rows it selects.
            plan.ops_dict[i]['materialize_in'] = plan.ops_dict[i]['input ids'].copy()


# The operators at which a filter's mask is materialized: the mask walk stops
# on the first one it meets below the filter.
_MASK_MATERIALIZERS = ('Aggregate', 'Join', 'Sort', 'take ordered')


def apply_simple_materialization(plan: QueryPlanRep, next_mask_id) -> None:
    """Simple late materialization for simple projects and filters.

    ``next_mask_id(used_ids)`` allocates a fresh mask column id not in
    ``used_ids`` (the lowering keeps one counter per query, across its plans).
    """
    used_ids = set()
    for i in plan.ops_idx:
        for used_id in plan.ops_dict[i].get('output ids', []) + plan.ops_dict[i].get('input ids', []):
            used_ids.add(used_id)

    for i in plan.ops_idx:
        if plan.ops_dict[i]['name'] == 'Project':
            is_simple, material_set = is_simple_projection(plan.ops_dict[i]['output'])
            if is_simple:
                assert len(plan.query_parent_map[i]) == 1, "Multiple inputs into Project not anticipated"
                parent = plan.query_parent_map[i][0]
                if plan.ops_dict[parent]['name'] == 'Join':
                    plan.ops_dict[parent]['materialize_in'] = list(material_set | set(plan.ops_dict[parent]['join in']))
                plan.ops_dict[parent]['materialize_now'] = list(material_set)
                plan.ops_dict[i]['materialize_now'] = []

    for i in plan.ops_idx:
        if plan.ops_dict[i]['name'] == 'Filter' and len(plan.ops_dict[i]['materialize_now']) > 0:
            mask_id = next_mask_id(used_ids)
            plan.ops_dict[i]['materialize_now'] = [mask_id]  # the filter outputs only its mask
            child = plan.query_child_map[i]

            while True:
                if 'mask map' not in plan.ops_dict[child]:
                    plan.ops_dict[child]['mask map'] = {}
                for j in plan.ops_dict[child].get('materialize_in', []):
                    if j in plan.ops_dict[i]['output ids']:
                        plan.ops_dict[child]['mask map'][j] = mask_id
                if plan.ops_dict[child]['name'] in _MASK_MATERIALIZERS:  # materialized here
                    break
                child = plan.query_child_map[child]

            plan.ops_dict[i]['out mask id'] = mask_id


def consumed_columns(op_args: dict) -> set:
    """The columns an operator reads.

    ======================  =====================================
    operator                fields
    ======================  =====================================
    Filter                  ``materialize_in``
    Project                 ``materialize_in``, ``rename map`` keys
    Aggregate               ``input ids``, ``in key ids``
    Sort, ``take ordered``  ``input ids``, ``key ids``
    Join                    ``join in``
    AdaptiveSparkPlan       ``output ids``
    ======================  =====================================

    Masks (``mask map``) are not listed: the deferral pass, the one caller,
    looks up only join outputs, and a mask is never one.
    """
    name = op_args['name']
    if name == 'Filter':
        return set(op_args.get('materialize_in', []))
    if name == 'Project':
        return set(op_args.get('materialize_in', [])) | set(op_args.get('rename map', {}))
    if name == 'Aggregate':
        return set(op_args.get('input ids', [])) | set(op_args.get('in key ids', []))
    if name in ('Sort', 'take ordered'):
        return set(op_args.get('input ids', [])) | set(op_args.get('key ids', []))
    if name == 'Join':
        return set(op_args.get('join in', []))
    if name == 'AdaptiveSparkPlan':
        return set(op_args.get('output ids', []))
    return set()


def _full_output(op_args: dict) -> list:
    """Everything an operator produces: gathered now plus deferred."""
    return list(op_args.get('materialize_now', [])) + list(op_args.get('materialize_deferred', []))


def compute_deferred_materialization(plan: QueryPlanRep) -> dict:
    """Annotate payload columns that can ride lineage across join chains.

    Moves a join's deferred columns from ``materialize_now`` to
    ``materialize_deferred`` (only joins defer; every other operator's
    ``materialize_deferred`` stays empty), writes ``deferred_groups`` on the
    join and ``gather_deferred`` on the operator that gathers a group.  The dataflow builder turns these into the gather nodes (a
    join's rearrange, the lineage composes, the deferred gathers).
    Returns ``{column id: {'chain', 'gather_at', 'group'}}``.
    """
    operators, order = plan.ops_dict, plan.ops_idx
    position = {op_id: index for index, op_id in enumerate(order)}

    key_columns: set = set()
    for op_id in order:
        name = operators[op_id]['name']
        if name == 'Join':
            key_columns |= set(operators[op_id].get('join in', []))
        elif name in ('Sort', 'take ordered'):
            key_columns |= set(operators[op_id].get('key ids', []))
        elif name == 'Aggregate':
            key_columns |= set(operators[op_id].get('in key ids', []))

    scan_of_column: dict = {}
    for op_id in order:
        if operators[op_id]['name'] == 'Scan':
            for column_id in operators[op_id].get('output ids', []):
                scan_of_column.setdefault(column_id, op_id)

    chain_of: dict = {}
    for op_id in order:
        if operators[op_id]['name'] != 'Join':
            continue
        join_inputs = set(operators[op_id].get('join in', []))
        for column_id in _full_output(operators[op_id]):
            if column_id not in join_inputs and column_id not in key_columns:
                chain_of.setdefault(column_id, []).append(op_id)

    def first_consumer_after(column_id, after_position):
        """The first operator after ``after_position`` that reads ``column_id``:
        where a deferred column is gathered."""
        best = None
        for op_id in order:
            if position[op_id] <= after_position:
                continue
            if column_id in consumed_columns(operators[op_id]) and (
                    best is None or position[op_id] < position[best]):
                best = op_id
        return best

    deferred: dict = {}
    for column_id, chain in chain_of.items():
        gather_at = first_consumer_after(column_id, position[chain[-1]])
        if gather_at is not None:
            deferred[column_id] = {'chain': chain, 'gather_at': gather_at}

    group_id_of: dict = {}
    next_group_id = 0
    for column_id, info in deferred.items():
        group_key = (
            scan_of_column.get(column_id, ('col', column_id)),
            tuple(info['chain']),
        )
        if group_key not in group_id_of:
            group_id_of[group_key] = next_group_id
            next_group_id += 1
        info['group'] = group_id_of[group_key]

    for op_id in order:
        if operators[op_id]['name'] != 'Join':
            continue
        materialized_out = _full_output(operators[op_id])
        deferred_here = [
            column_id for column_id in materialized_out
            if column_id in deferred and op_id in deferred[column_id]['chain']
        ]
        operators[op_id]['materialize_now'] = [
            column_id for column_id in materialized_out
            if column_id not in deferred_here
        ]
        operators[op_id]['materialize_deferred'] = deferred_here
        deferred_groups = {
            column_id: deferred[column_id]['group'] for column_id in deferred_here
        }
        if deferred_groups:
            operators[op_id]['deferred_groups'] = deferred_groups

    group_size: dict = {}
    for info in deferred.values():
        group = info['group']
        group_size[group] = group_size.get(group, 0) + 1
    for column_id, info in deferred.items():
        operators[info['gather_at']].setdefault('gather_deferred', []).append({
            'col': column_id,
            'group': info['group'],
            'n': group_size[info['group']],
        })

    return deferred
