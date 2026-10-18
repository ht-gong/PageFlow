"""A column scanned as the scan builds it (``IO/data_scanner.py``), from a
tensor instead of the parquet: its rows fed whole to the scan's
``_ParquetColumn``, cut into the scan batches, each batch one tick of a
region producer whose output is the representation."""
from __future__ import annotations

import torch

from IO import data_scanner


def scan_table(x: torch.Tensor, tensor_type: str, pool, algo_name: str):
    """``x`` (2-D, on the host or the GPU, at least one row) as one
    representation of a scanned column, ``algo_name`` (``RAW`` or a codec):
    a ``TableVariable`` over a record table whose regions are in ``pool``."""
    rows, width = x.shape
    column = data_scanner._ParquetColumn(
        tensor_type, [algo_name], 1, rows, width * x.element_size(), pool,
        torch.cuda.current_stream())
    try:
        column.feed(x.to("cuda:0"))
        column.flush()
        (ccr,) = column.finish()
    except BaseException:
        column.abort()
        raise
    return ccr.reps[0]


def pinned_column(values: torch.Tensor, tensor_type: str, pool,
                  selectivity: float = None):
    """``values`` (1-D or 2-D) as the streaming engine's scanned column:
    one RAW representation in ``pool`` with ``selectivity`` (a mask's), as
    a ``CombinedColumnRep``."""
    from IO.pages.combined_column_rep import CombinedColumnRep
    rep = scan_table(data_scanner.ensure_2d(values).contiguous(), tensor_type,
                     pool, "RAW")
    rep.selectivity = selectivity
    return CombinedColumnRep([rep])
