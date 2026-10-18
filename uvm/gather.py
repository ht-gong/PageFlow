"""The UVM executor's gathers: the dataflow graph's gather nodes on plain
tensors.

A gather node (``dataflow/dataflow_gen.py``) reads its *sources* (every
input but the last) and its *index* (the last input) and writes each
source gathered by the index to the output instance at the same position:

- ``rearrange``: a join's payload, gathered by the side's ``out_index``
  right after the join's core;
- ``compose``: a deferred group's lineage, gathered by a later join's
  ``out_index`` (the lineage seed is the seed join's ``out_index``
  instance itself: no node);
- ``deferred``: a deferred group's columns, gathered by the group's
  lineage just before the op that first reads them.

Each output is ``src[idx]``, one source at a time and one morsel of the
index at a time, and an index entry of ``constants.null`` (an outer
join's unmatched row) gives a null row: every element of the row is
``constants.null``, a string's bytes included.  The index holds pre-mask
row positions of the source, as the streaming engine's gidx does.  The
null mask is built only when the index holds a null (one check per
gather).

The lifetime rule is the streaming engine's (``rearrange_pipeline``): a
source with no row is not gathered, its output is the source itself; a
source is dropped from the payloads as soon as it is gathered (the gather
is its last reader), the index where the gather is its last consumer.  A
rearrange's and a deferred gather's outputs are columns of the tensor
group; a compose's is the group's next lineage, on its instance only.
"""
from __future__ import annotations

import torch

import constants
from utility.tensor_utils import row_bytes
from uvm.column import DEVICE, UColumn
from uvm.morsel import morsels


def has_null(idx: torch.Tensor) -> bool:
    """Whether ``idx`` holds ``constants.null``, one morsel at a time, with
    one host sync."""
    found = torch.zeros((), dtype=torch.bool, device=DEVICE)
    for a, b in morsels(idx.shape[0], 8 + 1):
        found |= (idx[a:b] == constants.null).any()
    return bool(found.item())


def gather_rows(src: torch.Tensor, idx: torch.Tensor,
                nulls: bool) -> torch.Tensor:
    """``src`` at rows ``idx`` (``src[idx]``), one morsel of ``idx`` at a
    time, with a null row wherever ``idx`` is ``constants.null`` (as
    ``conversion.index_with_null``; ``src`` has a row).  The null mask and
    the safe index are taken only when ``nulls`` (``has_null(idx)``)."""
    n = idx.shape[0]
    out = torch.empty((n,) + tuple(src.shape[1:]), dtype=src.dtype,
                      device=src.device)
    # The output rows, the index and, with a null, its null mask and the
    # safe index.
    for a, b in morsels(n, row_bytes(src) + 8 + (1 + 8 if nulls else 0)):
        i = idx[a:b]
        if nulls:
            null = i == constants.null
            i = i.masked_fill(null, 0)
        dst = out[a:b]
        torch.index_select(src, 0, i, out=dst)
        if nulls:
            dst.masked_fill_(null.view((-1,) + (1,) * (dst.dim() - 1)),
                             constants.null)
    return out


def run_gather(node, tensor_group, payloads) -> None:
    """Run gather ``node`` (a rearrange, compose or deferred gather): each
    source gathered by the index onto the output at its position, in
    ``payloads``; a rearrange's or deferred gather's outputs also into
    ``tensor_group``.  One source at a time: a source's payload goes as
    soon as it is gathered, so only one source and its output are held
    beyond what the gather has already handed over."""
    *sources, index = node.inputs
    assert len(sources) == len(node.outputs), (
        node.name, len(sources), len(node.outputs))
    index_col = payloads.get(index)
    assert index_col is not None, (
        f"{node.name}: index {index.name} has no payload")
    idx = nulls = None
    for src, out in zip(sources, node.outputs):
        col = payloads.get(src)
        assert col is not None, f"{node.name}: source {src.name} has no payload"
        if col.col_length() > 0:
            if idx is None:
                idx = index_col.tensor.reshape(-1).long()
                nulls = has_null(idx)
            col = UColumn(gather_rows(col.tensor, idx, nulls), col.type_name)
        payloads.publish(out, col)
        if node.kind != "compose":
            tensor_group[out.local] = col
        # The gather is a source's last reader.
        payloads.drop(src)
        del col
    del idx, index_col
    # The index goes where this is its last reader.
    payloads.consumed(node)
