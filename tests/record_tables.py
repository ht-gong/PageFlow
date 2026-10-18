"""Record-table reads the tests check against (``IO/pages/record_table.py``
keeps only what the engine reads)."""
import numpy as np


def region_records(table, region):
    """Region ``region``'s records, in byte order."""
    recs = np.flatnonzero(table.region == region)
    return recs[np.argsort(table.offset[recs], kind="stable")]


def stream_row_offsets(table):
    """Each stream's first row in the table's row space, plus the end."""
    return np.concatenate(([0], np.cumsum(table.stream_rows()))).astype(np.int64)
