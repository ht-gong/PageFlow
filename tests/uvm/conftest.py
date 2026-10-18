"""Fixtures every UVM executor test shares: the engine's logging and
exchange setup, the CUDA skip mark and the budgets a test sets."""
from __future__ import annotations

import pytest
import torch

import constants

MIB = 1024 ** 2

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA + custom allocator required")


@pytest.fixture(autouse=True)
def _engine_setup():
    from IO.transfer.data_exchange import set_exchange_to_naive
    from utility.logger import (set_message_logger, set_perf_logger,
                                set_torch_profiler)
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_torch_profiler(None, False)
    set_exchange_to_naive()


@pytest.fixture
def budgets():
    """``budgets(chunk_b, partition_b=None)`` sets the chunk budget and,
    when given, the partition budget (bytes) for one test; both are
    restored afterwards."""
    chunk, partition = (constants.get_chunk_size_b(),
                        constants.get_partition_budget_b())

    def set_b(chunk_b, partition_b=None):
        constants.set_chunk_size_b(int(chunk_b))
        if partition_b is not None:
            constants.set_partition_budget_b(int(partition_b))

    yield set_b
    constants.set_chunk_size_b(chunk)
    constants.set_partition_budget_b(partition)
