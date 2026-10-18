"""The UVM scan: each column a query reads, decoded from the datagen parquet
into one managed tensor and left in host memory on the GPU's NUMA node,
before the query's timed region.

Per table, ``ParquetTable.batches`` (``IO/parquet_reader.py``) decodes the
columns the query reads on the GPU, chunk by chunk, with cuDF's small
buffers sharing arena blocks (``utility/rmm_arena.small_buffers``).  Per
column, one managed tensor holds the table's rows in the engine's dtype
(``data_scanner.engine_dtype``; a string int8 ``[rows, width]``).  Each
chunk is copied into its slice, cast by the copy, and the slice prefetched
to the GPU's host NUMA node right after, so the scan never holds more than
a chunk of the table on the GPU, and the query's first GPU access to each
page faults it in.  This is
the only prefetch of the UVM mode: nothing hints the driver inside the
timed region.  Every occurrence of a column in the query gets the same
``UColumn``: operators never write their inputs.
"""
from __future__ import annotations

from collections import defaultdict

import torch

import TQPlib.tqpmemory as cmp
from IO.data_scanner import (column_type, engine_dtype, parquet_table,
                             scan_occurrences)
from utility import numa, rmm_arena
from uvm.column import UColumn


def scan_to_managed(config_json, schema_table, SF, to_scan):
    """Load ``to_scan`` into managed tensors on the GPU's host NUMA node
    (module docstring).  Returns ``{table: {COLUMN: [UColumn, ...]}}``, one
    entry per occurrence, the structure the plan's Scan operators pop.  On
    an arena of device memory (the tests') each column stays where it was
    written: there is nothing to prefetch."""
    root = config_json.get('tables path')
    host_node = numa.gpu_numa_node(0) if cmp.is_managed() else None
    scan_group = defaultdict(lambda: defaultdict(list))
    stream = torch.cuda.current_stream()
    with rmm_arena.small_buffers():
        for table, counts in scan_occurrences(to_scan).items():
            pt = parquet_table(root, SF, table)
            types = {col_name: column_type(schema_table, table, col_name)
                     for col_name in counts}
            columns = {}
            for first, chunk in pt.batches(list(counts)):
                for name, tensor in chunk.items():
                    dst = columns.get(name)
                    if dst is None:
                        dst = columns[name] = torch.empty(
                            (pt.rows, *tensor.shape[1:]),
                            dtype=engine_dtype(tensor.dtype, types[name]),
                            device='cuda')
                    # The engine's dtype, cast by the copy.
                    piece = dst[first:first + tensor.shape[0]]
                    piece.copy_(tensor)
                    if host_node is not None:
                        cmp.prefetch(piece.data_ptr(),
                                     piece.numel() * piece.element_size(),
                                     host_node, stream=stream.cuda_stream)
                del chunk, tensor, piece
            for col_name, n in counts.items():
                scan_group[table][col_name] += (
                    [UColumn(columns[col_name], types[col_name])] * n)
    torch.cuda.synchronize()
    return dict(scan_group)
