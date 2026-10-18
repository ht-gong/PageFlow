"""The engine's streams (utility/cuda_streams.py): torch's pool hands its
32 streams out round-robin, so a pipe's compute stream drawn from it would
come round to the exchange's copy streams; compute draws skip every kept
stream."""
import pytest
import torch

from utility import cuda_streams

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")


def test_a_kept_stream_is_one_stream_per_role():
    a = cuda_streams.dedicated('h2d')
    assert cuda_streams.dedicated('h2d') is a
    assert cuda_streams.dedicated('h2d', 'cuda') is a
    assert cuda_streams.dedicated('d2d').cuda_stream != a.cuda_stream


def test_compute_streams_never_come_round_to_a_kept_stream():
    kept = {cuda_streams.dedicated(r).cuda_stream
            for r in ('h2d', 'd2d', 'home writer')}
    assert len(kept) == 3
    drawn = [cuda_streams.compute_stream().cuda_stream
             for _ in range(3 * cuda_streams.POOL_SIZE)]
    assert not kept & set(drawn)
    # They still rotate over the rest of the pool.
    assert len(set(drawn)) == cuda_streams.POOL_SIZE - len(
        cuda_streams._kept_handles)


def test_the_exchange_and_the_home_writer_use_kept_streams():
    from IO.codec.home_writer import writer_stream
    from IO.transfer.data_exchange import exchange
    ex = exchange()
    kept = {s.cuda_stream for s in (ex.h2d_stream, ex.d2d_stream,
                                    writer_stream())}
    assert len(kept) == 3
    for _ in range(2 * cuda_streams.POOL_SIZE):
        assert cuda_streams.compute_stream('cuda:0').cuda_stream not in kept


@pytest.fixture
def no_kept_streams(monkeypatch):
    """The module's kept streams emptied for one test (and restored after
    it), as before the process's first take."""
    monkeypatch.setattr(cuda_streams, "_kept", {})
    monkeypatch.setattr(cuda_streams, "_kept_handles", set())


def test_every_role_is_taken_at_the_first_take(no_kept_streams):
    cuda_streams.dedicated('d2d')
    assert {r for r, _ in cuda_streams._kept} == set(cuda_streams.ROLES)
