"""Expression-tree helpers used while lowering Spark plans."""

import re


def get_id(name):  # name#id --> id
    return int(name.split('#')[-1].replace('L', ''))


def get_name(name):  # name#id --> name
    return name.split('#')[0]


_COLUMN_REF = re.compile(r'#(\d+)L?$')


def column_ref(token):
    """The column id a plan expression token references, or ``None``.

    Expression trees hold column references as ``name#id`` tokens (``#idL``
    for a long-marker suffix; a cast reads ``name#id as double``) and string
    literals as bare tokens.  One literal family looks like a reference:
    ``Brand#NN`` (Q16, Q17, Q19), whose number can collide with a real column
    id.  This is the single place that rule lives; every resolver that turns
    a token into a column id goes through it.
    """
    if not isinstance(token, str) or token.startswith('Brand#'):
        return None
    head = token.split(' as', 1)[0] if ' as' in token else token
    match = _COLUMN_REF.search(head)
    return int(match.group(1)) if match else None


def get_literals(tree, literal_set):
    if isinstance(tree, str):
        column = column_ref(tree)
        if column is not None:
            literal_set.add(column)

    if isinstance(tree, list) and tree[0] == 'isnotnull':
        return

    if isinstance(tree, (list, tuple)):
        if len(tree) == 3 and tree[1] == '#':
            literal_set.add(get_id(tree[-1]))
            get_literals(tree[0], literal_set)
        else:
            for element in tree:
                get_literals(element, literal_set)


def right_most(tree):
    if isinstance(tree, list):
        return right_most(tree[-1])
    return tree


def extract_number_after_hash(value):
    match = re.search(r'#(\d+)', value)
    return int(match.group(1)) if match else None


def get_project_output_ids(expr_tree):
    output_ids = []
    for tree in expr_tree:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]
        output_ids.append(get_id(right_most(tree)))
    return output_ids


def get_project_used_cols(expr_tree, available_cols_set):
    rename_map = {}
    literal_set = set()
    material_out, used_expr = [], []
    for tree in expr_tree:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]
        renamed_id = get_id(right_most(tree))
        if not isinstance(tree, str):
            if tree[1] == 'AS' and isinstance(tree[0], str):
                rename_map[get_id(tree[0])] = renamed_id
            else:
                get_literals(tree, literal_set)
                material_out.append(renamed_id)
                used_expr.append(tree)

    return rename_map, literal_set & available_cols_set, material_out, used_expr


def get_aggregation_ops(tree):
    if isinstance(tree, str):
        if tree in {'sum', 'min', 'max', 'count', 'avg'}:
            return tree
        return ''
    if isinstance(tree, list):
        for element in tree:
            result = get_aggregation_ops(element)
            if result != '':
                return result
    return ''


def get_aggregation_key_and_values(expr_tree):
    input_key_ids, output_key_ids, output_agg_ids, agg_ops = [], [], [], []
    for tree in expr_tree:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]
        if isinstance(tree, str) or (
            len(tree) == 3 and tree[1] == 'AS' and isinstance(tree[0], str)
        ):
            input_key_ids.append(get_id(tree if isinstance(tree, str) else tree[0]))
            output_key_ids.append(get_id(tree if isinstance(tree, str) else tree[-1]))
        else:
            output_agg_ids.append(get_id(right_most(tree)))
            agg_ops.append(get_aggregation_ops(tree))
    return input_key_ids, output_key_ids, output_agg_ids, agg_ops


def is_simple_projection(expr_trees):
    used_set = set()
    for tree in expr_trees:
        while isinstance(tree, list) and len(tree) == 1:
            tree = tree[0]

        agg_id = get_id(right_most(tree))
        if isinstance(tree, str):
            used_set.add(agg_id)
        elif tree[1] == 'AS' and isinstance(tree[0], str):
            used_set.add(get_id(tree[0]))
        else:
            return False, set()
    return True, used_set
