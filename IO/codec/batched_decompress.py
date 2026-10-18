"""Tick-level decompress orchestration.

``BatchedDecompress.decompress_all``: per-tick fanout for DirectIn stages.
The unified per-tick plan (``IO/transfer/in_stage.py:_alloc_agg_buf_and_bind_views``)
binds every compressed stage to a Level-B group ctx, so calling
``stage.decompress()`` on each member runs one nvCOMP call per codec group
(the first member pays the launch; later members narrow into the shared
output).  The group decodes the records the stages selected
(``IO/transfer/tick_layout.py``); the zero-copy mask stage reads its mask the
same way.
"""
from __future__ import annotations

from typing import Iterable


class BatchedDecompress:
    """Per-tick fanout to each DirectIn stage's group-batched decompress."""

    def decompress_all(self, stages: Iterable) -> None:
        for s in stages:
            if s.gpu_col is None or getattr(s, '_data_buf', None) is None:
                continue
            s.decompress()
