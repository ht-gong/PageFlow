"""Bucketed morsel outputs regrouped into the kernel calls that read them.

A keyed aggregation's pass 1 and a join's partition write, per morsel, its
rows bucket by bucket; pass 2's rounds and the join's core calls each read
a range of whole buckets from every morsel.  ``split_buckets`` copies each
morsel's bucket ranges into per-group columns once and drops the morsel's
outputs as they are copied, so the rows are held once.
"""
from __future__ import annotations

import numpy as np
import torch


def _runs(src, dst, n):
    """``(source row, destination row, rows)`` of each copy for buckets
    starting at ``src`` and ``dst`` with ``n`` rows each (all non-empty):
    buckets adjacent in both are one copy."""
    brk = np.flatnonzero((src[:-1] + n[:-1] != src[1:])
                         | (dst[:-1] + n[:-1] != dst[1:])) + 1
    first = np.concatenate(([0], brk))
    last = np.concatenate((brk, [n.size]))
    cs = np.concatenate(([0], np.cumsum(n)))
    return zip(src[first].tolist(), dst[first].tolist(),
               (cs[last] - cs[first]).tolist())


def split_buckets(parts, bounds, bucket_major: bool):
    """Every group's rows, one tensor per output.

    ``parts`` holds, per morsel, ``(tensors, counts, starts)``: its output
    tensors, its rows per bucket and, per tensor, the row each bucket
    starts at; it is emptied as it goes.  ``bounds`` are each group's first
    bucket, then B.  A group's rows are bucket by bucket, each bucket's
    morsel by morsel (``bucket_major``), or morsel by morsel, each morsel's
    in bucket order.  Every group's tensors are allocated once, at its
    rows, then each morsel's bucket ranges are copied into them and its own
    tensors dropped.  A group with no row is None."""
    n_groups = len(bounds) - 1
    if not parts:
        return [None] * n_groups
    counts = np.stack([np.asarray(c, dtype=np.int64) for _, c, _ in parts])
    bounds = np.asarray(bounds, dtype=np.int64)
    # Each bucket's group, and that group's first bucket.
    group = np.repeat(np.arange(n_groups), np.diff(bounds))
    lo = bounds[:-1][group]
    total = np.concatenate(([0], np.cumsum(counts.sum(axis=0))))
    group_rows = (total[bounds[1:]] - total[bounds[:-1]]).tolist()
    if bucket_major:
        # The bucket's start in its group, then the earlier morsels' rows
        # of the bucket.
        dst = (total[:-1] - total[lo])[None, :] + \
            (np.cumsum(counts, axis=0) - counts)
    else:
        # The earlier morsels' rows of the group, then the morsel's rows of
        # the group's earlier buckets.
        cum = np.concatenate((np.zeros((counts.shape[0], 1), dtype=np.int64),
                              np.cumsum(counts, axis=1)), axis=1)
        per_group = cum[:, bounds[1:]] - cum[:, bounds[:-1]]
        before = np.cumsum(per_group, axis=0) - per_group
        dst = before[:, group] + (cum[:, :-1] - cum[:, lo])

    template = parts[0][0]
    groups = [[torch.empty((n,) + tuple(t.shape[1:]), dtype=t.dtype,
                           device=t.device) for t in template]
              if n else None for n in group_rows]
    del template
    m = 0
    while parts:
        tensors, _, starts = parts.pop(0)
        for g in range(n_groups):
            b0, b1 = int(bounds[g]), int(bounds[g + 1])
            n = counts[m, b0:b1]
            nz = np.flatnonzero(n)
            if nz.size == 0:
                continue
            n, d = n[nz], dst[m, b0:b1][nz]
            for out, src, st in zip(groups[g], tensors, starts):
                s = np.asarray(st, dtype=np.int64)[b0:b1][nz]
                for s0, d0, k in _runs(s, d, n):
                    out[d0:d0 + k] = src[s0:s0 + k]
        del tensors, starts
        m += 1
    return groups
