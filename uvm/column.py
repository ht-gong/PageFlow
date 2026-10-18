"""A UVM column: a tensor in the managed arena and its type name; a query
occurrence's payloads, the column each dataflow instance carries; and how
an operator reads its node's input columns, publishes its outputs and
drops its inputs."""
from __future__ import annotations

import torch

from utility.tensor_utils import row_bytes
from variable import type_to_torch

DEVICE = 'cuda'


class UColumn:
    """One column of the UVM executor: ``tensor``, shaped ``[rows]`` or
    ``[rows, width]`` (a width-1 column is kept 1-D, as ``Variable.data``
    gives it to the streaming operators), its type name (``'int'``,
    ``'float'``, ``'date'``, ``'string'``, ``'bool'``) and, for a mask, its
    selectivity.  It answers what the plan code asks of a column."""

    def __init__(self, tensor: torch.Tensor, type_name: str,
                 selectivity: float = None):
        if tensor.dim() == 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        self.tensor = tensor
        self.type_name = type_name
        self.selectivity = selectivity

    def __repr__(self):
        return (f"UColumn(type={self.type_name!r}, "
                f"shape={tuple(self.tensor.shape)})")

    def col_length(self) -> int:
        return self.tensor.shape[0]

    def col_width(self) -> int:
        return 1 if self.tensor.dim() == 1 else self.tensor.shape[1]

    def row_bytes(self) -> int:
        return row_bytes(self.tensor)

    def get_type_torch(self) -> torch.dtype:
        return type_to_torch(self.type_name)

    def tensor_type(self) -> str:
        return self.type_name

    def is_floating_point(self) -> bool:
        return self.type_name == 'float'


class Payloads:
    """One query occurrence's payloads: the ``UColumn`` each dataflow
    instance (``dataflow/dataflow_gen.ColumnInstance``) carries, set by its
    producer.  The executor owns it for the occurrence and ``main.exec``
    drops it where the occurrence ends, so the graph itself never holds a
    tensor.  An entry goes when the node that reads the instance last has
    run (``consumed``) or when its column leaves the tensor group
    (``drop_columns``)."""

    def __init__(self):
        self._cols = {}

    def __len__(self):
        return len(self._cols)

    def get(self, inst):
        """Instance ``inst``'s column, or None (not produced, or dropped)."""
        return self._cols.get(inst)

    def publish(self, inst, col: UColumn) -> UColumn:
        """Set ``col`` as instance ``inst``'s payload."""
        self._cols[inst] = col
        return col

    def inputs(self, node):
        """A pipeline node's input columns, by input key."""
        return {key: self._cols.get(inst) for key, inst in node.input_pairs()}

    def drop(self, inst) -> None:
        """Drop instance ``inst``'s payload."""
        self._cols.pop(inst, None)

    def consumed(self, node) -> None:
        """Drop each input of ``node`` that ``node`` reads last (the graph's
        ``consumers``)."""
        for inst in node.inputs:
            if inst.consumers and inst.consumers[-1] is node:
                self._cols.pop(inst, None)

    def drop_columns(self, cols, tensor_group) -> None:
        """Drop every entry that carries one of ``cols``, unless the tensor
        group still holds that column under some id (a rename's alias keeps
        it), so its tensor goes back to the arena with its last
        reference."""
        held = {id(col) for col in tensor_group.values()}
        gone = {id(col) for col in cols if id(col) not in held}
        if gone:
            self._cols = {inst: col for inst, col in self._cols.items()
                          if id(col) not in gone}

    def clear(self) -> None:
        self._cols.clear()


def rows_of(columns) -> int:
    """The rows of ``columns``, all of one row space."""
    lengths = {col.col_length() for col in columns.values()}
    assert len(lengths) == 1, f"input columns of different lengths: {lengths}"
    return lengths.pop()


def row_bytes_of(columns) -> int:
    return sum(col.row_bytes() for col in columns.values())


def drop_inputs(tensor_group, payloads, input_ids, keep) -> None:
    """Drop ``input_ids`` that are not in ``keep`` from the tensor group, and
    their columns from ``payloads`` (``Payloads.drop_columns``)."""
    dropped = []
    for cid in input_ids:
        if cid in tensor_group and cid not in keep:
            dropped.append(tensor_group[cid])
            del tensor_group[cid]
    payloads.drop_columns(dropped, tensor_group)


def concat_or_empty(parts, dtype, trailing_shape=()) -> torch.Tensor:
    """``parts`` concatenated along rows: the one part itself, or an empty
    column of ``dtype`` and ``trailing_shape`` when there is none."""
    if len(parts) > 1:
        return torch.cat(parts)
    if parts:
        return parts[0]
    return torch.empty((0, *trailing_shape), dtype=dtype, device=DEVICE)
