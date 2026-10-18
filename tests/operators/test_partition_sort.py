"""The fused sort-pipe partition (``operators/partition_sort.py``) against a
reference in torch ops (``_reference``).

Every output (each column, then the gidx) must hold, in the padded layout the
region producer reads, the bucket counts and in every bucket the kept rows
whose key hashes there (the top log2(B) bits of mix64), in source order, the
gidx being the tick's first global row plus the pre-mask row.  Padding rows
are unwritten, so they are not compared.  The cases cover the bucket counts,
row widths, tick sizes around the kernel's 2048-row tile, masks from
all-False to all-True, compacted, raw and mixed columns under a mask, both
mask prologues and both raw-column modes, a global offset near SF1000's,
offset mask views, a mask count that the mask contradicts, ticks past 1024
tiles, the ticks the kernel refuses and its bucket limit, the join's B cap
(``operators/partitions.py``).  A test that must take one mode
forces the selectivity threshold to 0.0 (never select-first) or 2.0
(always).
"""

from __future__ import annotations

import re
from math import gcd
from pathlib import Path

import pytest
import torch

import operators.partition_sort as partition_sort
from operators.partition_sort import partition
from operators.partitions import JOIN_B_MAX
from tests.mix64 import mix64


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _keys(n, seed, skew=False):
    if skew:
        return torch.full((n,), 7, dtype=torch.int64, device="cuda")
    g = _gen(seed)
    # Negative keys and repeats included.
    return torch.randint(-(1 << 40), 1 << 40, (n,), dtype=torch.int64,
                         generator=g).cuda()


def _column(n, kind, seed):
    g = _gen(seed)
    if kind == "i64":
        return torch.randint(-(1 << 62), 1 << 62, (n,), dtype=torch.int64,
                             generator=g).cuda()
    if kind == "i32":
        return torch.randint(-(1 << 30), 1 << 30, (n,), dtype=torch.int32,
                             generator=g).cuda()
    if kind == "i16":
        return torch.randint(-(1 << 14), 1 << 14, (n,), dtype=torch.int16,
                             generator=g).cuda()
    if kind == "i8":
        return torch.randint(-128, 128, (n,), dtype=torch.int8,
                             generator=g).cuda()
    if kind == "s25":
        return torch.randint(0, 256, (n, 25), dtype=torch.uint8,
                             generator=g).cuda()
    if kind.startswith("u8x"):
        # 2-D byte rows: 2, 4 and 8 B take the typed copy, others the byte loop
        return torch.randint(0, 256, (n, int(kind[3:])), dtype=torch.uint8,
                             generator=g).cuda()
    if kind == "i64x2":
        return torch.randint(-(1 << 62), 1 << 62, (n, 2), dtype=torch.int64,
                             generator=g).cuda()
    raise ValueError(kind)


def _table(n, kinds, seed, skew=False, key_pos=0):
    cols = [_column(n, k, seed + 1 + c) for c, k in enumerate(kinds)]
    cols.insert(key_pos, _keys(n, seed, skew))
    return cols


def _mask(n, keep, seed):
    if keep <= 0:
        return torch.zeros(n, dtype=torch.bool, device="cuda")
    if keep >= 1:
        return torch.ones(n, dtype=torch.bool, device="cuda")
    return (torch.rand(n, generator=_gen(seed)) < keep).cuda()


def _bucket_rows(pt):
    """Each bucket's rows of a PartitionedTensor, in bucket order."""
    t = pt.tensor
    rb = t.element_size() * (t.shape[1] if t.dim() == 2 else 1)
    unit = 512 // gcd(rb, 512)
    rows, at = [], 0
    for c in pt.partition_row_counts:
        rows.append(t[at:at + c])
        at += -(-c // unit) * unit
    assert t.shape[0] == at, (t.shape[0], at)
    return rows


def _reference(cols, key_pos, mask, goff, B):
    """The bucket counts and, per output (each column, then the gidx), each
    bucket's rows: the kept rows stably sorted by bucket."""
    n_pre = mask.shape[0] if mask is not None else cols[key_pos].shape[0]
    gidx = torch.arange(goff, goff + n_pre, dtype=torch.int64, device="cuda")
    if mask is not None:
        cols = [c[mask] for c in cols]
        gidx = gidx[mask]
    shift = 64 - (B.bit_length() - 1)
    bucket = (mix64(cols[key_pos].contiguous()) >> shift) & (B - 1)
    order = torch.sort(bucket, stable=True).indices
    counts = torch.bincount(bucket, minlength=B).tolist()
    return counts, [list(torch.split(t[order], counts)) for t in cols + [gidx]]


def _assert_same(got, want):
    counts, outs = want
    assert len(got) == len(outs)
    for g, w in zip(got, outs):
        assert g.partition_row_counts == counts
        assert g.tensor.dtype == w[0].dtype
        for b, (gb, wb) in enumerate(zip(_bucket_rows(g), w)):
            assert torch.equal(gb, wb), f"bucket {b} differs"


def _check(cols, key_pos, mask, goff, B, compact=None, count=False):
    """The kernel on ``cols`` (raw, or compacted by ``mask`` where
    ``compact`` says so; every column when it is None), with the mask's kept
    rows when ``count``, against the reference."""
    if mask is None:
        given = list(cols)
    else:
        compact = compact if compact is not None else [True] * len(cols)
        given = [c[mask] if k else c for c, k in zip(cols, compact)]
    n_valid = int(mask.sum()) if count else None
    got = partition(given, key_pos, mask, goff, B, n_valid=n_valid)
    _assert_same(got, _reference(cols, key_pos, mask, goff, B))


@pytest.mark.parametrize("n", [0, 1, 25, 2047, 2048, 2049, 300_000])
def test_unmasked_tick_sizes(n):
    _check(_table(n, ["i64"], seed=n), 0, None, 0, 32)


@pytest.mark.parametrize("B", [1, 2, 8, 16, 32, 64, 128, 256, 512])
def test_every_bucket_count(B):
    _check(_table(100_000, ["i64"], seed=B), 0, None, 12345, B)


@pytest.mark.parametrize("B", [64, 512])
def test_six_columns_of_four_widths_with_the_key_inside(B):
    cols = _table(70_001, ["i64", "i32", "i8", "s25", "i64"], seed=3,
                  key_pos=2)
    _check(cols, 2, None, 0, B)


@pytest.mark.parametrize("kind", ["i32", "i16", "i8", "s25", "u8x2", "u8x3",
                                  "u8x4", "u8x8", "i64x2"])
def test_each_row_width(kind):
    _check(_table(50_000, [kind], seed=11), 0, None, 0, 16)


def test_one_bucket_holds_every_row():
    _check(_table(40_000, ["i64", "s25"], seed=5, skew=True), 0, None, 0, 64)


def test_global_offset_near_sf1000s_last_row():
    n = 1 << 18
    _check(_table(n, ["i64"], seed=6), 0, None, 6_000_000_000 - n, 64)


@pytest.mark.parametrize("keep", [0.0, 0.01, 0.095, 0.3, 0.63, 1.0])
@pytest.mark.parametrize("select_first_below", [0.0, 2.0])
def test_compacted_columns_under_a_mask(keep, select_first_below, monkeypatch):
    """Every column compacted, as zero-copy reads arrive, with the mask's
    count the pipe passes (under an all-True mask the compacted columns are
    the raw ones).  0.0 forces the compacted-input prologue, 2.0
    select-first."""
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_BELOW",
                        select_first_below)
    n = 200_003
    cols = _table(n, ["i64", "s25"], seed=int(keep * 1000) + 1)
    _check(cols, 0, _mask(n, keep, seed=21), 777, 32, count=True)


def test_raw_columns_without_the_count_raise():
    n = 12_000
    cols = _table(n, ["i64"], seed=31)
    mask = _mask(n, 0.4, seed=32)
    with pytest.raises(ValueError, match="need the mask's kept rows"):
        partition(cols, 0, mask, 0, 16)


@pytest.mark.parametrize("keep", [0.0, 0.01, 0.095, 0.3, 0.63, 1.0])
@pytest.mark.parametrize("raw_below", [0.0, 2.0])
def test_raw_columns_with_the_mask_count(keep, raw_below, monkeypatch):
    """The folded sort pipe's contract: raw columns and the mask's kept
    rows.  0.0 forces the fold, 2.0 select-first (the raw rows gathered)."""
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_RAW_BELOW", raw_below)
    n = 150_001
    cols = _table(n, ["i64", "s25", "i16"], seed=int(keep * 1000) + 3)
    _check(cols, 0, _mask(n, keep, seed=33), 99, 64,
           compact=[False] * 4, count=True)


def test_a_count_the_compacted_rows_contradict_raises():
    n = 10_000
    cols = _table(n, ["i64"], seed=34)
    mask = _mask(n, 0.5, seed=35)
    packed = [c[mask] for c in cols]
    with pytest.raises(RuntimeError, match="compacted columns hold"):
        partition(packed, 0, mask, 0, 32, n_valid=int(mask.sum()) + 1)


@pytest.mark.parametrize("raw_below", [0.0, 2.0])
def test_raw_and_compacted_columns_mixed(raw_below, monkeypatch):
    """A folded tick mixing raw and zero-copy (compacted) columns: 0.0
    forces the fold with the compacted-input prologue, 2.0 select-first."""
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_RAW_BELOW", raw_below)
    n = 90_000
    cols = _table(n, ["i64", "s25", "i32"], seed=41)
    mask = _mask(n, 0.4, seed=42)
    for compact in ([False, True, False, True], [True, False, True, False]):
        _check(cols, 0, mask, 5, 64, compact=compact, count=True)


@pytest.mark.parametrize("offset", [3, 5])
def test_offset_mask_views(offset, monkeypatch):
    """A mask and columns that start off a 16-byte boundary, in every mode:
    compacted columns both ways, raw ones folded without the count, and
    with it both ways."""
    n = 64_000
    cols = [c[offset:] for c in _table(n, ["i64"], seed=51)]
    mask = _mask(n, 0.2, seed=52)[offset:]
    for below in (0.0, 2.0):
        monkeypatch.setattr(partition_sort, "SELECT_FIRST_BELOW", below)
        _check(cols, 0, mask, 0, 32)
    for below in (0.0, 2.0):
        monkeypatch.setattr(partition_sort, "SELECT_FIRST_RAW_BELOW", below)
        _check(cols, 0, mask, 0, 32, compact=[False, False], count=True)


def test_small_masked_ticks():
    for n in (1, 25, 2049):
        cols = _table(n, ["i64"], seed=n)
        _check(cols, 0, _mask(n, 0.5, seed=n), 0, 8)
        _check(cols, 0, _mask(n, 0.0, seed=n), 0, 8)


def test_repeated_calls_reuse_the_count_buffer():
    # 512 takes K3's opt-in shared memory, and the calls after it do not
    for i, B in enumerate((64, 8, 512, 1, 128, 256, 16)):
        _check(_table(30_000 + i, ["i32"], seed=60 + i), 0, None, i, B)


@pytest.mark.parametrize("case", ["five widths", "seven columns", "int32 key",
                                  "B=1024", "B=24", "B=0"])
def test_ticks_outside_the_limits_raise(case):
    n = 1000
    key = _keys(n, 70)
    cols, B = {
        "five widths": ([key] + [_column(n, k, 71)
                                 for k in ("i32", "i16", "i8", "s25")], 32),
        "seven columns": ([key] * 7, 32),
        "int32 key": ([key.to(torch.int32)], 32),
        "B=1024": ([key], 1024),
        "B=24": ([key], 24),
        "B=0": ([key], 0),
    }[case]
    with pytest.raises(ValueError, match="outside the kernel's limits"):
        partition(cols, 0, None, 0, B)


def test_the_bucket_limit_is_the_joins_cap():
    # The B partition() accepts, the kernel's BMAX (and its log2) and the
    # largest B a join takes are one number.
    assert partition_sort._B_MAX == JOIN_B_MAX
    cu = (Path(__file__).resolve().parents[2] / "operators" / "kernels"
          / "partition_sort.cu").read_text()
    bmax = re.search(r"constexpr int BMAX = (\d+);", cu)
    logbmax = re.search(r"constexpr int LOGBMAX = (\d+);", cu)
    assert bmax and int(bmax.group(1)) == JOIN_B_MAX
    assert logbmax and 1 << int(logbmax.group(1)) == JOIN_B_MAX


@pytest.mark.parametrize("raw_below", [0.0, 2.0])
@pytest.mark.parametrize("off", [-37, 41])
def test_a_count_the_mask_contradicts_raises(raw_below, off, monkeypatch):
    """Raw columns with a wrong count: the fold (0.0) and select-first (2.0)
    both fail, and nothing was read or written past what the count sized, so
    the GPU is still usable."""
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_RAW_BELOW", raw_below)
    n = 100_003
    cols = _table(n, ["i64", "s25"], seed=91)
    mask = _mask(n, 0.1, seed=92)
    with pytest.raises(RuntimeError, match="the mask keeps"):
        partition(list(cols), 0, mask, 0, 32, n_valid=int(mask.sum()) + off)
    torch.cuda.synchronize()
    _check(cols, 0, mask, 0, 32, compact=[False] * 3, count=True)


@pytest.mark.parametrize("select_first_below", [0.0, 2.0])
def test_compacted_columns_of_another_mask_raise(select_first_below,
                                                 monkeypatch):
    """Compacted columns that another mask kept: compacted-input (0.0) and
    select-first (2.0) count the given mask's rows and fail, for a mask
    keeping fewer and more rows than the columns hold."""
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_BELOW",
                        select_first_below)
    n = 100_003
    cols = _table(n, ["i64", "s25"], seed=93)
    kept = _mask(n, 0.3, seed=94)
    packed = [c[kept] for c in cols]
    for other in (0.2, 0.4):
        with pytest.raises(RuntimeError, match="the mask keeps"):
            partition(list(packed), 0, _mask(n, other, seed=95), 0, 16)
    torch.cuda.synchronize()
    _check(cols, 0, kept, 0, 16)


@pytest.mark.parametrize("B", [16, 512])
@pytest.mark.parametrize("mode", ["plain", "compacted-input", "select-first",
                                  "fold", "raw-select-first"])
def test_a_tick_past_a_thousand_tiles(mode, B, monkeypatch):
    """5,000,011 rows: 2442 tiles and 1221 select-first blocks, past the
    1024 that K2's scan chunk and k_exscan1 take in one step."""
    n = 5_000_011
    cols = _table(n, ["i32"], seed=96)
    if mode == "plain":
        _check(cols, 0, None, 7, B)
        return
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_BELOW",
                        2.0 if mode == "select-first" else 0.0)
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_RAW_BELOW",
                        2.0 if mode == "raw-select-first" else 0.0)
    raw = mode in ("fold", "raw-select-first")
    _check(cols, 0, _mask(n, 0.45, seed=97), 7, B, compact=[not raw] * 2,
           count=raw)


@pytest.mark.parametrize("B", [1, 128, 256, 512])
@pytest.mark.parametrize("below", [0.0, 2.0])
def test_masked_ticks_at_one_to_512_buckets(below, B, monkeypatch):
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_BELOW", below)
    monkeypatch.setattr(partition_sort, "SELECT_FIRST_RAW_BELOW", below)
    n = 80_000
    cols = _table(n, ["i64", "s25"], seed=98)
    mask = _mask(n, 0.37, seed=99)
    _check(cols, 0, mask, 0, B)
    _check(cols, 0, mask, 0, B, compact=[False] * 3, count=True)
