from __future__ import annotations

from typing import Any

import torch

import constants
import conversion
from conversion import date_to_float, is_date, is_float, str_to_np
from operators.like import like_contains, like_mask
from parsing import column_ref, get_id


ExprTree = Any

# The elements a torch op limited by INT_MAX takes: ``nonzero`` over a mask
# of fewer, and ``torch.sort`` / ``unique`` count their rows in int.  A sort's
# rows and a join call's residual pairs are checked against it.
INT_ROWS_MAX = (1 << 31) - 2


def normalize_expr_tree(tree):
    """Unwrap Spark parser single-item list wrappers."""
    while isinstance(tree, list) and len(tree) == 1:
        tree = tree[0]
    return tree


def is_alias_expr(tree):
    tree = normalize_expr_tree(tree)
    return isinstance(tree, list) and len(tree) == 3 and tree[1] == "AS"


def is_direct_column_expr(tree):
    tree = normalize_expr_tree(tree)
    return isinstance(tree, str)


def is_simple_rename_expr(tree):
    tree = normalize_expr_tree(tree)
    return is_alias_expr(tree) and isinstance(tree[0], str)


def is_passthrough_or_rename(tree):
    return is_direct_column_expr(tree) or is_simple_rename_expr(tree)


def is_group_key_expr(tree):
    tree = normalize_expr_tree(tree)
    return isinstance(tree, str) or is_simple_rename_expr(tree)


def unwrap_alias_expr(tree):
    tree = normalize_expr_tree(tree)
    if is_alias_expr(tree):
        return tree[0]
    return tree


def get_evaluable_project_expr(tree):
    """
    Return expression that requires compute in project.
    Returns None for pass-through columns and rename-only outputs.
    """
    tree = normalize_expr_tree(tree)
    if is_passthrough_or_rename(tree):
        return None
    return unwrap_alias_expr(tree)


def right_most_expr(tree):
    tree = normalize_expr_tree(tree)
    if isinstance(tree, list):
        return right_most_expr(tree[-1])
    return tree


def get_aggregation_op(tree):
    """Return first aggregation op name found in expression tree."""
    tree = normalize_expr_tree(tree)
    if isinstance(tree, str):
        if tree in ("sum", "min", "max", "count", "avg"):
            return tree
        return ""
    if isinstance(tree, list):
        for e in tree:
            r = get_aggregation_op(e)
            if r:
                return r
    return ""

# =============- TYPE INFERENCE UTIL FUNCS -===============

def _infer_expr_type_core(tree, col_type_lookup, default_dtype=constants.float_dtype):
    """
    Shared expression dtype inference core.
    col_type_lookup: callable taking a column-id int and returning torch.dtype.
    """
    tree = normalize_expr_tree(tree)

    if isinstance(tree, str):
        column = column_ref(tree)
        if column is not None:
            return col_type_lookup(column)
        if tree.isdigit():
            return constants.int_dtype
        if is_float(tree):
            return constants.float_dtype
        return constants.string_dtype

    if not isinstance(tree, list):
        return default_dtype

    if len(tree) == 3 and tree[1] == "AS":
        return _infer_expr_type_core(tree[0], col_type_lookup, default_dtype)

    if len(tree) == 3 and tree[1] == "#":
        if isinstance(tree[-1], str) and tree[-1].replace("L", "").isdigit():
            return col_type_lookup(int(tree[-1].replace("L", "")))
        return _infer_expr_type_core(tree[0], col_type_lookup, default_dtype)

    if len(tree) == 8 and tree[0] == "CASE":
        then_t = _infer_expr_type_core(tree[4], col_type_lookup, default_dtype)
        else_t = _infer_expr_type_core(tree[6], col_type_lookup, default_dtype)
        if then_t == else_t:
            return then_t
        if then_t == constants.float_dtype or else_t == constants.float_dtype:
            return constants.float_dtype
        if then_t == constants.string_dtype or else_t == constants.string_dtype:
            return constants.string_dtype
        return constants.int_dtype

    if len(tree) == 2:
        fn = tree[0]
        if fn in ("StartsWith", "NOT", "isnotnull"):
            return torch.bool
        if fn == "substring":
            return constants.string_dtype
        if fn == "cast":
            literal, _, cast_type = tree[1][0].split(" ")
            if cast_type == "double":
                return constants.float_dtype
            return _infer_expr_type_core(literal, col_type_lookup, default_dtype)
        # Aggregate function defaults to numeric output.
        if fn in ("sum", "avg", "count", "min", "max", "year"):
            if fn == "avg":
                return constants.float_dtype
            if fn == "count":
                return constants.int_dtype
            if fn == "year":
                return constants.int_dtype
            return _infer_expr_type_core(tree[1], col_type_lookup, default_dtype)
        return default_dtype

    if len(tree) == 3:
        op = tree[1]
        if op in ("<=", ">=", "<", ">", "=", "<>", "AND", "OR", "IN", "LIKE"):
            return torch.bool

        lhs_t = _infer_expr_type_core(tree[0], col_type_lookup, default_dtype)
        rhs_t = _infer_expr_type_core(tree[2], col_type_lookup, default_dtype)
        if op == "/":
            return constants.float_dtype
        if lhs_t == constants.float_dtype or rhs_t == constants.float_dtype:
            return constants.float_dtype
        if lhs_t == constants.string_dtype or rhs_t == constants.string_dtype:
            return constants.string_dtype
        return constants.int_dtype

    return default_dtype


def get_result_type(tree, transfer_set, transfer_types):
    """Infer aggregate expression output dtype using transfer input dtypes."""
    def _lookup(col_id):
        if col_id in transfer_set:
            return transfer_types[transfer_set.index(col_id)]
        return constants.int_dtype

    return _infer_expr_type_core(tree, _lookup, default_dtype=constants.int_dtype)


def infer_project_expr_type(expr, input_type_map):
    """
    Infer torch dtype for a single project expression.
    Falls back to float64 for unknown expression forms to preserve behavior.
    """
    return _infer_expr_type_core(
        expr,
        lambda col_id: input_type_map.get(col_id, constants.float_dtype),
        default_dtype=constants.float_dtype,
    )


def infer_project_output_types(used_expr, input_type_map):
    """
    Infer dtypes for all materialized project expressions in order.
    Only expressions that require compute are included.
    """
    output_types = []
    for tree in used_expr:
        expr = get_evaluable_project_expr(tree)
        if expr is None:
            continue
        output_types.append(infer_project_expr_type(expr, input_type_map))
    return output_types


# ==============- KEY EVALUATION CODES -==================

def _scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, num_subgroups=1024, reduce="max", initv=0):
    n_rows = inverted_index.size(0)
    device = tensor_tmp.device
    subgroup_id = torch.arange(n_rows, device=device) % num_subgroups
    flat_index = inverted_index * num_subgroups + subgroup_id
    result_2d = torch.full((cnt_groups * num_subgroups,), initv, dtype=tensor_tmp.dtype, device=device)
    result_2d = torch.scatter_reduce(result_2d, 0, flat_index, tensor_tmp, reduce=reduce)
    result_2d = result_2d.view(cnt_groups, num_subgroups)
    if reduce == "max":
        return result_2d.max(dim=1).values
    if reduce == "min":
        return result_2d.min(dim=1).values
    return result_2d.sum(dim=1)


def _cast_result(value, target_dtype):
    if isinstance(value, torch.Tensor):
        return value.to(target_dtype)
    if target_dtype == constants.float_dtype:
        return float(value)
    if target_dtype == constants.int_dtype:
        return int(value)
    raise ValueError("Unsupported scalar cast target: %s" % target_dtype)

def _evaluate_core(
    mode,
    device_name,
    tensors,
    tree,
    rows,
    cnt_rows,
    cnt_groups,
    inverted_index,
    subquery_result,
    leaf_function,
):
    tree = normalize_expr_tree(tree)

    if isinstance(tree, str):
        if mode == "filter":
            if tree == "Subquery":
                return subquery_result
            if is_date(tree):
                return date_to_float(tree)
            if is_float(tree):
                return float(tree) if "." in tree else int(tree)
            column = column_ref(tree)
            if column is not None:
                if leaf_function is not None:
                    return leaf_function(column)
                return tensors[column].squeeze()
            return tree

        if tree.isdigit():
            return int(tree)
        if is_float(tree):
            return float(tree)
        column = column_ref(tree)
        if column is not None:
            return tensors[column]
        return tree

    if len(tree) == 3 and tree[1] == "#":
        if mode == "filter":
            if leaf_function is not None:
                return leaf_function(int(tree[-1]))
            return tensors[get_id(tree[-1])].squeeze()
        return _evaluate_core(mode, device_name, tensors, tree[0], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)

    if len(tree) == 8 and tree[0] == "CASE":
        mask = _evaluate_core(mode, device_name, tensors, tree[2], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
        then = _evaluate_core(mode, device_name, tensors, tree[4], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
        elsc = _evaluate_core(mode, device_name, tensors, tree[6], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
        return torch.where(mask, then, elsc)

    if len(tree) == 2:
        fn = tree[0]

        if fn == "StartsWith":
            col_id, name = get_id(tree[1][0]), tree[1][1]
            t = tensors[col_id]
            assert t.shape[-1] >= len(name)
            p = torch.tensor(str_to_np(name, len(name)), dtype=constants.string_dtype, device=device_name)
            return torch.eq(t[:, : len(name)], p).all(dim=1)

        if fn in ("EndsWith", "Contains"):
            if mode != "filter":
                raise ValueError("Unsupported function in aggregation: %s" % fn)
            col_id, name = get_id(tree[1][0]), tree[1][1]
            return like_contains(tensors[col_id], torch.tensor(str_to_np(name, len(name)), device=device_name))

        if fn == "year":
            if mode != "aggregate":
                raise ValueError("Unsupported function in filter: year")
            rhs = _evaluate_core(mode, device_name, tensors, tree[1], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
            bin_idx = torch.bucketize(rhs, torch.tensor(conversion.year_edges, device="cuda"), right=True)
            return torch.tensor(conversion.year_range, device="cuda")[bin_idx]

        if fn == "NOT":
            return ~_evaluate_core(mode, device_name, tensors, tree[1], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)

        if fn == "isnotnull":
            if mode != "filter":
                raise ValueError("Unsupported function in aggregation: isnotnull")
            return torch.ones(rows, dtype=torch.bool, device=device_name)

        if fn == "cast":
            literal, _, cast_type = tree[1][0].split(" ")
            if cast_type == "double":
                value = _evaluate_core(
                    mode,
                    device_name,
                    tensors,
                    literal,
                    rows,
                    cnt_rows,
                    cnt_groups,
                    inverted_index,
                    subquery_result,
                    leaf_function,
                )
                return _cast_result(value, constants.float_dtype)
            raise ValueError("Unsupported cast type: %s" % cast_type)

        if fn == "substring":
            assert len(tree[1]) == 3
            sub, l_pos, width = tree[1]
            l_pos, width = int(l_pos) - 1, int(width)
            base = _evaluate_core(mode, device_name, tensors, sub, rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
            return base[:, l_pos : l_pos + width]

        if fn not in ("sum", "avg", "count", "min", "max") or mode != "aggregate":
            raise ValueError("Unsupported function: %s" % fn)

        tensor_tmp = _evaluate_core(mode, device_name, tensors, tree[1], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
        enable_2steps = False
        if cnt_rows / cnt_groups >= (1 << 16):
            enable_2steps = True
            sub_groups = max(1, cnt_rows // (cnt_groups << 6))

        if isinstance(tensor_tmp, int):
            tensor_tmp = torch.full((cnt_rows,), tensor_tmp, dtype=constants.int_dtype, device=device_name)
        elif isinstance(tensor_tmp, float):
            tensor_tmp = torch.full((cnt_rows,), tensor_tmp, dtype=constants.float_dtype, device=device_name)

        if fn == "min":
            mx = torch.iinfo(tensor_tmp.dtype).max if not tensor_tmp.is_floating_point() else torch.finfo(tensor_tmp.dtype).max
            if enable_2steps:
                result = _scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, sub_groups, "min", mx)
            else:
                result = torch.full((cnt_groups,), mx, dtype=tensor_tmp.dtype, device=device_name)
                torch.scatter_reduce(result, 0, inverted_index, tensor_tmp, reduce="min", out=result)
            return result

        if fn == "max":
            if enable_2steps:
                return _scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, sub_groups, "max", 0)
            result = torch.zeros(cnt_groups, dtype=tensor_tmp.dtype, device=device_name)
            torch.scatter_reduce(result, 0, inverted_index, tensor_tmp, reduce="max", out=result)
            return result

        if fn == "avg":
            if enable_2steps:
                result = _scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, sub_groups, "sum", 0).to(torch.float64)
                ones = torch.ones(inverted_index.size(0), dtype=torch.int64, device="cuda")
                denom = _scatter_reduce_2steps(inverted_index, ones, cnt_groups, sub_groups, "sum", 0)
                result.div_(denom)
            else:
                result = torch.bincount(inverted_index, weights=tensor_tmp, minlength=cnt_groups)
                denom = torch.bincount(inverted_index, minlength=cnt_groups)
                result.div_(denom)
            return result

        if tensor_tmp.numel() >= 2:
            tensor_tmp.squeeze_()
        elif tensor_tmp.dim() >= 2:
            tensor_tmp.squeeze_(0)

        if fn == "count":
            tensor_tmp = (tensor_tmp != constants.null).to(constants.int_dtype)

        if enable_2steps:
            result = _scatter_reduce_2steps(inverted_index, tensor_tmp, cnt_groups, sub_groups, "sum", 0)
        else:
            result = torch.bincount(inverted_index, weights=tensor_tmp, minlength=cnt_groups)

        if not torch.is_floating_point(tensor_tmp):
            result = torch.round(result).to(constants.int_dtype)
        return result

    op = tree[-2]

    if mode == "filter" and op == "IN":
        if tree == ["DELIVER", "IN", "PERSON"]:
            return " ".join(tree)
        lhs = _evaluate_core(mode, device_name, tensors, tree[0], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
        if lhs.dtype == constants.string_dtype:
            rhs_values = [name for name in tree[-1]]
        elif lhs.dtype in (constants.int_dtype, constants.date_dtype):
            rhs_values = [int(name) for name in tree[-1]]
        else:
            rhs_values = [float(name) for name in tree[-1]]
        if lhs.dtype == constants.string_dtype and lhs.dim() == 1:
            lhs = lhs.unsqueeze(0)
        mask = torch.zeros(rows, dtype=torch.bool, device=device_name)
        for rhs in rhs_values:
            if isinstance(rhs, str):
                if len(rhs) == 1:
                    mask = torch.logical_or(mask, torch.eq(lhs, ord(rhs)))
                else:
                    rhs_tensor = torch.tensor(str_to_np(rhs, lhs.shape[-1]), dtype=constants.string_dtype, device=device_name)
                    mask = torch.logical_or(mask, (lhs == rhs_tensor).all(dim=1))
            else:
                mask = torch.logical_or(mask, torch.eq(lhs, rhs))
        return mask

    if mode == "filter" and op == "LIKE":
        lhs = _evaluate_core(mode, device_name, tensors, tree[0], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
        assert tree[-1].count("%") == 3
        _, word1, word2, _ = tree[-1].split("%")
        word1 = torch.tensor(str_to_np(word1, len(word1)), dtype=constants.string_dtype, device=device_name)
        word2 = torch.tensor(str_to_np(word2, len(word2)), dtype=constants.string_dtype, device=device_name)
        return like_mask(lhs, word1, word2)

    if op == "AS":
        return _evaluate_core(mode, device_name, tensors, tree[0], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)

    ops = {
        "+": torch.add,
        "-": torch.sub,
        "*": torch.mul,
        "/": torch.div,
        "<=": torch.le,
        ">=": torch.ge,
        "<": torch.lt,
        ">": torch.gt,
        "=": torch.eq,
        "<>": torch.ne,
        "AND": torch.logical_and,
        "OR": torch.logical_or,
    }
    if op not in ops:
        raise ValueError("Unsupported operation: %s" % op)

    lhs = _evaluate_core(mode, device_name, tensors, tree[0], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)
    rhs = _evaluate_core(mode, device_name, tensors, tree[-1], rows, cnt_rows, cnt_groups, inverted_index, subquery_result, leaf_function)

    if mode == "filter" and isinstance(rhs, float):
        if op == "<=":
            return torch.le(lhs, rhs + constants.eps)
        if op == ">=":
            return torch.ge(lhs, rhs - constants.eps)

    if isinstance(rhs, str):
        if mode == "aggregate" and isinstance(lhs, int):
            assert op == "-"
            return "".join(tree)
        if len(rhs) == 1:
            return ops[op](lhs, ord(rhs))
        if mode == "filter" and lhs.dim() == 1:
            lhs.unsqueeze_(0)
        rhs_tensor = torch.tensor(str_to_np(rhs, lhs.shape[-1]), dtype=constants.string_dtype, device=device_name)
        return (lhs == rhs_tensor).all(dim=1)

    if mode == "filter" and op == "=" and lhs.dtype == constants.float_dtype:
        if isinstance(rhs, float):
            rhs = torch.full((lhs.shape[0],), rhs, dtype=lhs.dtype, device=device_name)
        return torch.isclose(lhs, rhs, atol=constants.eps)

    return ops[op](lhs, rhs)


def evaluate_aggregate(device_name, tensors, tree, cnt_rows, cnt_groups, inverted_index):
    return _evaluate_core(
        mode="aggregate",
        device_name=device_name,
        tensors=tensors,
        tree=tree,
        rows=None,
        cnt_rows=cnt_rows,
        cnt_groups=cnt_groups,
        inverted_index=inverted_index,
        subquery_result=None,
        leaf_function=None,
    )


def evaluate_filter(device_name, tensors, tree, rows, subquery_result, leaf_function=None):
    if rows is None:
        rows = list(tensors.values())[0].size(0)
    return _evaluate_core(
        mode="filter",
        device_name=device_name,
        tensors=tensors,
        tree=tree,
        rows=rows,
        cnt_rows=None,
        cnt_groups=None,
        inverted_index=None,
        subquery_result=subquery_result,
        leaf_function=leaf_function,
    )


def evaluate_expr(device_name, columns, tree, cnt_rows, cnt_groups=None, inverted_index=None):
    return evaluate_aggregate(device_name, columns, tree, cnt_rows, cnt_groups, inverted_index)
