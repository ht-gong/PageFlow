import torch
from typing import Dict
import constants
from variable import gpu_region_column
from IO.pipeline.common import col_to_gpu
from IO.pages.combined_column_rep import CombinedColumnRep
from utility.logger import perf_logger
from operators.operator_utils import INT_ROWS_MAX


def sort_inputs(args, name="sort"):
    """The column ids a sort reads (its input, key and distinct columns) and
    the id of the mask in its ``mask map``, or None."""
    input_ids = args["input ids"]
    distinct_ids = (list(args.get("distinct ids", input_ids))
                    if args.get("distinct", False) else [])
    mask_ids = set((args.get("mask map") or {}).values())
    assert len(mask_ids) <= 1, f"{name}: one mask expected, got {sorted(mask_ids)}"
    ids = set(input_ids + args["key ids"] + distinct_ids)
    return ids, (mask_ids.pop() if mask_ids else None)


def sort_tensors(cuda_tensors: Dict[int, torch.Tensor], mask, args, with_limit,
                 name="sort"):
    """The sort over GPU tensors: ``cuda_tensors`` holds every column
    ``sort_inputs`` names, contiguous, and ``mask`` (1-D bool, or None) the
    rows to sort.  Stable lexicographic sort on the key columns, then
    DISTINCT and LIMIT.  Returns ``({input id: sorted tensor}, count)``,
    count the distinct rows under DISTINCT, else the rows before the limit.
    """
    perm, count = sort_perm(cuda_tensors, mask, args, with_limit, name)
    outs = {col_id: cuda_tensors[col_id][perm] for col_id in args["input ids"]}
    return outs, count


def order_ids(args):
    """The columns the sort's order is computed from: its keys and, under
    DISTINCT, the distinct columns.  The other input columns only follow
    the order."""
    distinct_ids = (list(args.get("distinct ids", args["input ids"]))
                    if args.get("distinct", False) else [])
    return set(args["key ids"]) | set(distinct_ids)


def sort_perm(cuda_tensors: Dict[int, torch.Tensor], mask, args, with_limit,
              name="sort"):
    """The sort's order: ``cuda_tensors`` holds the ``order_ids`` columns,
    contiguous, and ``mask`` (1-D bool, or None) the rows to sort.  Returns
    ``(perm, count)``: the input rows of the output, in order (int64 on the
    GPU), and the distinct rows under DISTINCT, else the rows before the
    limit."""
    if with_limit:
        limit = args["limit"]

    input_ids = args["input ids"]
    key_ids, key_descs = args["key ids"], args["key orders"]

    # Opt-in DISTINCT, for a plan whose semi-join or DISTINCT became plain
    # inner joins (no lowered plan sets it).  ``distinct ids`` is the full
    # output tuple to dedup on; it defaults to the input columns.
    distinct = args.get("distinct", False)
    distinct_ids = list(args.get("distinct ids", input_ids)) if distinct else []

    first_key = cuda_tensors[key_ids[0]]
    in_rows = first_key.shape[0]
    if in_rows > INT_ROWS_MAX:
        raise RuntimeError(
            f"{name}: {in_rows} input rows are over the {INT_ROWS_MAX} rows "
            f"one sort takes")

    if first_key.numel() * first_key.element_size() >= 8_000_000_000:
        raise RuntimeError("ooc sort not integrated yet")

    # When deduping, every distinct column must take part in the sort order as a
    # lowest-priority tiebreak so identical full-tuples land adjacent; otherwise
    # adjacency-dedup is unsound whenever a real sort key does not functionally
    # determine the other distinct columns.  This never changes the user-visible
    # ORDER BY (real keys keep priority) — it only orders rows that are equal on
    # every real key, which dedup then collapses.
    sort_key_ids, sort_key_descs = list(key_ids), list(key_descs)
    for cid in distinct_ids:
        if cid not in sort_key_ids:
            sort_key_ids.append(cid)
            sort_key_descs.append(False)

    # A filter right below the sort emits only its mask, so the input columns
    # arrive unfiltered: the sort starts from the rows the mask keeps.
    if mask is not None:
        assert mask.shape[0] == in_rows, f"{name}: mask has {mask.shape[0]} rows, inputs {in_rows}"
        perm = mask.nonzero().squeeze(1).contiguous()
    else:
        perm = torch.arange(in_rows, device="cuda", dtype=torch.int64).contiguous()
    rows = perm.shape[0]
    torch.cuda.synchronize()

    for key_id, desc in zip(reversed(sort_key_ids), reversed(sort_key_descs)):
        key_col = cuda_tensors[key_id]

        if key_col.dim() == 2:
            tmp = torch.gather(key_col, 0, perm.unsqueeze(1).expand(-1, key_col.shape[1]))
            _, inv = torch.unique(tmp, sorted=True, return_inverse=True, dim=0)
            _, inverse_index = torch.sort(inv, stable=True, descending=bool(desc))
            perm = torch.gather(perm, 0, inverse_index)
        else:
            tmp = torch.gather(key_col, 0, perm)
            _, inverse_index = torch.sort(tmp, stable=True, descending=bool(desc))
            perm = torch.gather(perm, 0, inverse_index)

    if distinct:
        # Drop adjacent duplicates: keep a row iff it differs from its
        # predecessor in ANY distinct column (the rows are now contiguous in
        # full-tuple order thanks to the tiebreak keys appended above).
        row_changed = None
        for cid in distinct_ids:
            col = cuda_tensors[cid]
            if col.dim() == 2:
                g = torch.gather(col, 0, perm.unsqueeze(1).expand(-1, col.shape[1]))
                d = (g[1:] != g[:-1]).any(dim=1)
            else:
                g = torch.gather(col, 0, perm)
                d = g[1:] != g[:-1]
            row_changed = d if row_changed is None else (row_changed | d)
        keep = torch.ones(perm.shape[0], dtype=torch.bool, device="cuda")
        if row_changed is not None:
            keep[1:] = row_changed
        perm = perm[keep].contiguous()

    if with_limit:
        perm = perm[:limit].contiguous()

    return perm, (perm.shape[0] if distinct else rows)


def _take_rows(ccr: CombinedColumnRep, perm: torch.Tensor) -> torch.Tensor:
    """Rows ``perm`` of the column ``ccr``, in ``perm``'s order, decoded a
    window of whole records at a time (``get_sliced_rep``, then
    ``col_to_gpu``): only the windows that hold one of the rows are read,
    and the sort holds one window of the column beside its output."""
    from adaptive.sizing_controller import window_rows
    from IO.codec.region_producer import record_rows
    rep = ccr.reps[0]
    n = ccr.col_length()
    row_b = max(1, rep.col_width() * rep.orig_elem_size())
    w = window_rows(constants.get_chunk_size_b(), row_b, record_rows(row_b))
    if perm.numel() == 0:
        return col_to_gpu(ccr.get_sliced_rep(slice(0, 1)))[:0]
    rows, order = torch.sort(perm)
    starts = list(range(0, n, w))
    cuts = torch.searchsorted(
        rows, torch.tensor(starts + [n], device=rows.device)).tolist()
    out = None
    for i, lo in enumerate(starts):
        a, b = cuts[i], cuts[i + 1]
        if a == b:
            continue
        part = col_to_gpu(ccr.get_sliced_rep(slice(lo, min(lo + w, n))))
        if out is None:
            out = torch.empty((perm.numel(),) + tuple(part.shape[1:]),
                              dtype=part.dtype, device=part.device)
        out[order[a:b]] = part[rows[a:b] - lo]
        del part
    return out


def _windowed(ccr: CombinedColumnRep) -> bool:
    """Whether the column is larger than the chunk budget."""
    rep = ccr.reps[0]
    return (ccr.col_length() * rep.col_width() * rep.orig_elem_size()
            > constants.get_chunk_size_b())


def tqp_sort(
    tensor_groups: Dict[int, CombinedColumnRep],
    args,
    with_limit,
    name="sort",
):
    """
    Decompress and materialize the columns the order is computed from
    (``order_ids``) to GPU, perform stable lexicographic sort
    (``sort_perm``), then take the output rows of every input column: from
    the GPU copy, or for a column that only follows the order and is larger
    than the chunk budget, decoded window by window (``_take_rows``).  A
    mask in ``mask map`` (from a filter right below, e.g. a HAVING) selects
    the rows that are sorted.
    """
    perf_logger().start(name)

    ids, mask_id = sort_inputs(args, name)
    whole = order_ids(args)
    windowed = {cid for cid in ids - whole if _windowed(tensor_groups[cid])}
    cuda_tensors = {cid: col_to_gpu(tensor_groups[cid]).contiguous()
                    for cid in ids - windowed}
    mask = None if mask_id is None else col_to_gpu(tensor_groups[mask_id])
    perm, count = sort_perm({cid: cuda_tensors[cid] for cid in whole}, mask,
                            args, with_limit, name)
    del mask
    outs = {cid: (_take_rows(tensor_groups[cid], perm) if cid in windowed
                  else cuda_tensors[cid][perm])
            for cid in args["input ids"]}
    del cuda_tensors

    # Each output column is one GPU region (``gpu_region_column``), read by
    # the output step through a DirectIn.
    for col_id, out in outs.items():
        if out.dim() == 1:
            out = out.unsqueeze(1)
        tensor_type = tensor_groups[col_id].reps[0].tensor_type
        tensor_groups[col_id] = CombinedColumnRep(
            [gpu_region_column(out.contiguous(), tensor_type)])

    torch.cuda.synchronize()
    perf_logger().stop(name)
    return count
