"""The two reads of a gather's source (``IO/pipeline/rearrange_pipeline.
py``): over UVA from a RAW rep, or streamed onto the GPU through a tick pipe
and gathered there.  Both give the same rows, null indices included, for
every type and for a compressed source; the gather takes the read that
moves fewer bytes (``uva_bytes`` against the streamed rep's stored bytes);
a producer's compressed column, with no RAW rep, is streamed; a RAW source
on the GPU is read in place; a source too large for the GPU is read over
UVA, from a RAW home decoded for it when it has no RAW rep."""
from __future__ import annotations

import pytest
import torch

import constants
from IO.pages.combined_column_rep import CombinedColumnRep

from tests.IO.test_direct_regions import (  # noqa: F401  (fixtures)
    _FixedCodec, _gather_node, _host, engine, pool, requires_cuda)
from tests.scan_table import scan_table


def _column(pool, x, tensor_type, codecs):
    """``x`` as one column with a rep per codec in ``codecs`` (a scan's
    column is ``["RAW", codec]``, a producer's ``[codec]``)."""
    return CombinedColumnRep([scan_table(x.contiguous(), tensor_type, pool, c)
                              for c in codecs])


def _index(pool, n_src, n_out, nulls, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_src, (n_out,), dtype=torch.int64, generator=g)
    if nulls:
        idx[torch.randperm(n_out, generator=g)[:n_out // 7]] = constants.null
    return idx, _column(pool, idx.unsqueeze(1), "int", ["RAW"])


def _expected(x, idx):
    out = x[idx.clamp(min=0)].clone()
    out[idx < 0] = constants.null
    return out


class _Reads:
    """Counts the UVA gathers and the tick pipes a gather runs."""

    def __init__(self, monkeypatch):
        from IO.pipeline import rearrange_pipeline as rp_mod
        from IO.pipeline.vortex_pipeline import VortexPipeline
        self.uva, self.pipes = 0, []
        real_gather, real_init = rp_mod._gather, VortexPipeline.__init__

        def gather(*a):
            self.uva += 1
            return real_gather(*a)

        def init(pipe, *a, **k):
            real_init(pipe, *a, **k)
            self.pipes.append(pipe)
        monkeypatch.setattr(rp_mod, "_gather", gather)
        monkeypatch.setattr(VortexPipeline, "__init__", init)


CASES = [
    # (tensor type, column, the source's codec)
    ("int", lambda g: torch.randint(0, 1 << 40, (1 << 20, 1),
                                    dtype=torch.int64, generator=g), "BITCOMP"),
    ("float", lambda g: torch.rand((1 << 20, 1), dtype=torch.float64,
                                   generator=g), "ANS"),
    ("date", lambda g: torch.randint(0, 3000, (1 << 20, 1),
                                     dtype=torch.int32, generator=g), "BITCOMP"),
    ("string", lambda g: torch.randint(-100, 100, (1 << 17, 25),
                                       dtype=torch.int8, generator=g), "ANS"),
]


@requires_cuda
@pytest.mark.parametrize("case", range(len(CASES)))
def test_streamed_and_uva_reads_give_the_rows(cuda_allocator, pool, engine,
                                              monkeypatch, case):
    """Each type: a scan's column (RAW and compressed reps) gathered by a
    right-outer index over UVA and streamed (its compressed rep, over
    several ticks) gives the same rows, nulls included."""
    from IO.pipeline import rearrange_pipeline as rp_mod
    tensor_type, make, codec = CASES[case]
    monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 1 << 20)
    x = make(torch.Generator().manual_seed(case))
    n_out = 3 * x.shape[0] // 2
    for read, uva_b in (("uva", 0), ("streamed", 1 << 62)):
        monkeypatch.setattr(rp_mod, "uva_bytes", lambda *a: uva_b)
        reads = _Reads(monkeypatch)
        idx, idx_ccr = _index(pool, x.shape[0], n_out, True, case + 1)
        node = _gather_node(_column(pool, x, tensor_type, ["RAW", codec]),
                            idx_ccr)
        rp_mod.rearrange_pipeline(node, pool, "right-outer",
                                  applier=_FixedCodec(codec))
        got = _host(node.outputs[0].ccr.reps[0]).reshape(n_out, -1)
        assert torch.equal(got, _expected(x, idx).reshape(n_out, -1)), read
        if read == "uva":
            assert reads.uva > 0 and not reads.pipes
        else:
            (load,) = reads.pipes
            assert reads.uva == 0 and load.stages - 2 >= 3
            assert load._VortexPipeline__input_compression_algs == [codec]


@requires_cuda
@pytest.mark.parametrize("n_out,read", [(2_000, "uva"), (3 << 20, "streamed")])
def test_the_cheaper_read_is_taken(cuda_allocator, pool, engine, monkeypatch,
                                   n_out, read):
    """A sparse gather of a scan's column reads its rows over UVA (fewer
    sectors than the source's stored bytes); a dense one streams the
    source."""
    from IO.pipeline import rearrange_pipeline as rp_mod
    monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 64 << 20)
    x = torch.randint(0, 1 << 20, (1 << 20, 1), dtype=torch.int64,
                      generator=torch.Generator().manual_seed(9))
    idx, idx_ccr = _index(pool, x.shape[0], n_out, False, 10)
    node = _gather_node(_column(pool, x, "int", ["RAW", "BITCOMP"]), idx_ccr)
    reads = _Reads(monkeypatch)
    rp_mod.rearrange_pipeline(node, pool, "inner",
                              applier=_FixedCodec("BITCOMP"))
    assert (reads.uva > 0) == (read == "uva")
    assert len(reads.pipes) == (read == "streamed")
    got = _host(node.outputs[0].ccr.reps[0]).reshape(-1, 1)
    assert torch.equal(got, x[idx])


@requires_cuda
def test_a_compressed_producer_column_is_streamed(cuda_allocator, pool,
                                                  engine, monkeypatch):
    """A producer's column has one rep, in its codec: the gather streams it,
    however sparse the gather."""
    from IO.pipeline import rearrange_pipeline as rp_mod
    monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 1 << 20)
    x = torch.randint(0, 1 << 30, (1 << 18, 1), dtype=torch.int64,
                      generator=torch.Generator().manual_seed(5))
    idx, idx_ccr = _index(pool, x.shape[0], 1_000, False, 6)
    node = _gather_node(_column(pool, x, "int", ["BITCOMP"]), idx_ccr)
    reads = _Reads(monkeypatch)
    rp_mod.rearrange_pipeline(node, pool, "inner",
                              applier=_FixedCodec("BITCOMP"))
    assert reads.uva == 0 and len(reads.pipes) == 1
    got = _host(node.outputs[0].ccr.reps[0]).reshape(-1, 1)
    assert torch.equal(got, x[idx])


@requires_cuda
@pytest.mark.parametrize("codecs", [["RAW", "BITCOMP"], ["BITCOMP"]])
def test_a_source_too_large_for_the_gpu_is_read_over_uva(
        cuda_allocator, pool, engine, monkeypatch, codecs):
    """A dense gather whose source does not fit on the GPU beside it: a
    scan's column is read over UVA from its RAW rep; a producer's, with
    none, is first decoded home RAW by a tick pipe, read over UVA, and its
    RAW home freed."""
    from IO.pipeline import rearrange_pipeline as rp_mod
    monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 1 << 20)
    monkeypatch.setattr(rp_mod, "_fits_on_gpu", lambda nbytes: False)
    x = torch.randint(0, 1 << 30, (1 << 18, 1), dtype=torch.int64,
                      generator=torch.Generator().manual_seed(7))
    idx, idx_ccr = _index(pool, x.shape[0], 1 << 19, True, 8)
    node = _gather_node(_column(pool, x, "int", codecs), idx_ccr)
    reads = _Reads(monkeypatch)
    rp_mod.rearrange_pipeline(node, pool, "right-outer",
                              applier=_FixedCodec("BITCOMP"))
    assert reads.uva > 0
    assert len(reads.pipes) == (0 if "RAW" in codecs else 1)
    if reads.pipes:
        (home,) = reads.pipes
        assert home._VortexPipeline__input_compression_algs == ["BITCOMP"]
        assert home.result_tables()[0].regions[0].freed
    got = _host(node.outputs[0].ccr.reps[0]).reshape(-1, 1)
    assert torch.equal(got, _expected(x, idx))


@requires_cuda
def test_a_raw_source_on_the_gpu_is_read_in_place(cuda_allocator, pool,
                                                  engine):
    """A RAW column on the GPU (the free-list baseline's intermediates) is
    read over UVA, by address, whatever the density."""
    from types import SimpleNamespace
    from IO.pipeline import rearrange_pipeline as rp_mod
    from variable import gpu_region_column
    x = torch.arange(1 << 16, dtype=torch.int64, device="cuda").unsqueeze(1)
    src = SimpleNamespace(
        ccr=CombinedColumnRep([gpu_region_column(x, "int")]), name="src")
    how, rep = rp_mod._read_plan(src, 1 << 20, "BITCOMP",
                                 _FixedCodec("BITCOMP"))
    assert how == "uva" and rep is src.ccr.reps[0]


def test_uva_bytes_bounds():
    """Sparse windows read a sector per row, a dense window at most every
    sector of the source, a wide row its own sectors."""
    from IO.pipeline.rearrange_pipeline import uva_bytes
    n = 1 << 26
    assert uva_bytes(n, 1000, 1 << 20, 8) == pytest.approx(32 * 1000,
                                                           rel=1e-3)
    assert uva_bytes(n, 1 << 30, 1 << 30, 8) <= n * 8 + 32
    assert uva_bytes(n, 10, 1 << 20, 117) == 10 * (117 // 32 + 2) * 32
    # Windows count apart: the same rows in two windows read twice as many
    # sectors when each window is dense.
    assert uva_bytes(n, 2 * n, n, 8) == pytest.approx(
        2 * uva_bytes(n, n, n, 8), rel=1e-6)
