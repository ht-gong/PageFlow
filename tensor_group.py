from __future__ import annotations

from typing import Dict

from IO.pages.combined_column_rep import CombinedColumnRep


class TensorGroup:
    """The cross-operator column map of a running query.

    ``tensor_group[col_id]`` is the current generation of a column: its base
    rep after the scan, then whatever the last producer set.  Within-operator
    intermediates never live here: the runtime holds them on the dataflow
    graph's ``ColumnInstance.ccr`` (``dataflow/dataflow_gen.py``).
    """

    def __init__(self):
        self._cross: Dict[int, CombinedColumnRep] = {}

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        return self._cross[key]

    def __setitem__(self, key, value):
        self._cross[key] = value

    def __delitem__(self, key):
        del self._cross[key]

    def __contains__(self, key):
        return key in self._cross

    def __iter__(self):
        return iter(self._cross)

    def __len__(self):
        return len(self._cross)

    def keys(self):
        return self._cross.keys()

    def items(self):
        return self._cross.items()

    def values(self):
        return self._cross.values()

    def alias(self, orig_id: int, new_id: int):
        """Project rename: make new_id point to same CCR as orig_id."""
        self._cross[new_id] = self._cross[orig_id]
