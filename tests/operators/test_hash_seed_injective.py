"""Guard the fixed grouping-hash seed (operators.aggregate._COEFF_BASE_SEED).

The two-pass aggregator groups a wide/composite key by a LINEAR hash
``sum_word(packed_word * coeff)``.  For structured TPC-H keys (s_name, the
Q16 (p_brand,p_type,p_size) composite, c_name, nation composites) that linear
form collides distinct keys for ~3-4% of *random* coefficient draws — the
historical Q16/Q21 wrong-rows flake (e.g. seed 33 merges
``Supplier#000005042`` with ``Supplier#000005842`` and 8 other +400 pairs).

The fix is a DETERMINISTIC, VALIDATED seed: coefficients are a pure function of
the fixed base seed + key shape, and that seed is proven collision-free over
every structured grouping-key domain the 22-query suite reaches.  Injectivity
over a key's full domain is a global property of the coeff vs the domain's
byte-difference lattice, so it holds for every filtered subset, scale factor,
and plan (a subset is a subset of the domain).

These tests fail if someone changes ``_COEFF_BASE_SEED`` to a colliding value,
or breaks the determinism that makes "validated seed" meaningful (coeffs must
NOT depend on global RNG consumption / query order).  Requires CUDA; the shared
custom allocator is bootstrapped in tests/conftest.py.
"""
from __future__ import annotations

import itertools

import pytest
import torch

import operators.aggregate as agg

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"

# Full supplier / customer domains at the largest supported scale (SF700 has
# 7M suppliers, ~10.5M customers).  Structured-key collisions are structural,
# not cardinality-driven, but validating the full domain certifies the
# scale-independent guarantee directly.  Override for memory-constrained GPUs.
SF700_SUPPLIERS = 7_000_000
SF700_CUSTOMERS = 10_500_000

NATIONS = [
    "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA", "FRANCE",
    "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN",
    "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA",
    "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"]
BRANDS = [f"Brand#{a}{b}" for a in range(1, 6) for b in range(1, 6)]  # 25
_SYL = (
    ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"],
    ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"],
    ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"])
PTYPES = [" ".join(c) for c in itertools.product(*_SYL)]  # 150


def _prefixed_names(prefix: str, n_ids: int, width: int) -> torch.Tensor:
    """(n_ids, width) int8: ``prefix`` + 9-digit zero-padded id (s_name/c_name)."""
    k = torch.arange(1, n_ids + 1, device=DEV, dtype=torch.int64)
    pb = torch.tensor([ord(c) for c in prefix], device=DEV, dtype=torch.uint8)
    digits = torch.empty(n_ids, 9, device=DEV, dtype=torch.uint8)
    for i in range(9):
        digits[:, i] = ((k // (10 ** (8 - i))) % 10).to(torch.uint8) + ord("0")
    body = torch.cat([pb.unsqueeze(0).expand(n_ids, len(prefix)), digits], dim=1)
    buf = torch.zeros(n_ids, width, dtype=torch.uint8, device=DEV)
    buf[:, :body.shape[1]] = body[:, :width]
    return buf.view(torch.int8)


def _str_col(strings, width: int) -> torch.Tensor:
    buf = torch.zeros(len(strings), width, dtype=torch.uint8, device=DEV)
    for i, s in enumerate(strings):
        b = s.encode()[:width]
        buf[i, :len(b)] = torch.tensor(list(b), dtype=torch.uint8, device=DEV)
    return buf.view(torch.int8)


def _structured_domains():
    """Each entry: label -> (key column list, expected distinct count).

    Column dtypes/widths match the schema (s_name/n_name fixed text 25 -> 4
    words, p_brand fixed 10 -> 2, p_type variable 25 -> 4, sizes/years int64).
    """
    d = {}
    d["s_name"] = ([_prefixed_names("Supplier#", SF700_SUPPLIERS, 25)],
                   SF700_SUPPLIERS)
    d["c_name"] = ([_prefixed_names("Customer#", SF700_CUSTOMERS, 25)],
                   SF700_CUSTOMERS)
    combos = list(itertools.product(BRANDS, PTYPES, range(1, 51)))  # Q16
    d["q16_composite"] = ([
        _str_col([c[0] for c in combos], 10),
        _str_col([c[1] for c in combos], 25),
        torch.tensor([c[2] for c in combos], dtype=torch.int64,
                     device=DEV).unsqueeze(1),
    ], len(combos))
    d["n_name"] = ([_str_col(NATIONS, 25)], len(NATIONS))
    pair = list(itertools.product(range(len(NATIONS)), range(len(NATIONS)), range(7)))
    d["q7_nation_nation_year"] = ([
        _str_col([NATIONS[i] for i, _, _ in pair], 25),
        _str_col([NATIONS[j] for _, j, _ in pair], 25),
        torch.tensor([1992 + y for *_, y in pair], dtype=torch.int64,
                     device=DEV).unsqueeze(1),
    ], len(pair))
    return d


@requires_cuda
@pytest.mark.parametrize("label", [
    "s_name", "c_name", "q16_composite", "n_name", "q7_nation_nation_year"])
def test_shipped_seed_is_injective_over_structured_domain(label):
    """The shipped seed must hash every structured grouping-key domain without
    a single collision (0 merged groups)."""
    cols, n_distinct = _structured_domains()[label]
    coeffs = agg._make_key_coeffs(cols, DEV)
    h = agg._hash_key_columns(cols, coeffs)
    n_unique = int(torch.unique(h).numel())
    assert n_unique == n_distinct, (
        f"seed {agg._COEFF_BASE_SEED} collides on {label}: "
        f"{n_distinct - n_unique} distinct keys merged "
        f"({n_unique}/{n_distinct} survive) — pick a validated seed")


@requires_cuda
def test_shipped_seed_is_injective_over_q16_dedup_domain():
    """Q16's count(DISTINCT ps_suppkey) first groups by (p_brand, p_type,
    p_size, ps_suppkey).  That key's hash is the Q16 composite's hash h plus
    c * ps_suppkey, so two keys can collide only where two composites' h
    differ (mod 2**64) by a multiple of c below c times the supplier count.
    The domain is too large to hash (1.3e12 keys at SF700), but such close
    pairs of h are few."""
    cols, n_combos = _structured_domains()["q16_composite"]
    suppkey = torch.ones(n_combos, 1, dtype=torch.int64, device=DEV)
    coeffs = agg._make_key_coeffs(cols + [suppkey], DEV)
    h = agg._hash_key_columns(cols, coeffs[:3])
    assert torch.equal(agg._hash_key_columns(cols + [suppkey], coeffs), h + coeffs[3])
    c = int(coeffs[3])
    h = torch.sort(h).values
    for step in range(1, n_combos):
        # Distance to the step-th next hash around the int64 circle (wraps).
        gap = h.roll(-step) - h
        near = gap[(gap >= 0) & (gap <= c * SF700_SUPPLIERS)]
        if near.numel() == 0:
            break
        assert not bool((near % c == 0).any()), (
            f"seed {agg._COEFF_BASE_SEED} collides on the Q16 dedup key — pick a validated seed")


@requires_cuda
def test_coeffs_are_deterministic_across_calls():
    """Two calls with the same key shapes yield byte-identical coefficients —
    the property that makes the validated seed reproducible at runtime."""
    cols = [_prefixed_names("Supplier#", 1000, 25)]
    a = agg._make_key_coeffs(cols, DEV)
    b = agg._make_key_coeffs(cols, DEV)
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert torch.equal(ca, cb)


@requires_cuda
def test_coeffs_independent_of_global_rng():
    """Coefficients must NOT depend on the process-global RNG stream, or the
    coeff that runs would differ from the coeff that was validated (it would
    depend on how much randomness earlier queries consumed)."""
    cols = [_prefixed_names("Supplier#", 1000, 25),
            torch.zeros(1000, 1, dtype=torch.int64, device=DEV)]
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    a = agg._make_key_coeffs(cols, DEV)
    # Consume a different amount of global randomness, reseed differently.
    _ = torch.randint(0, 1_000_000, (7,), device=DEV)
    torch.manual_seed(99)
    torch.cuda.manual_seed_all(99)
    _ = torch.rand(13, device=DEV)
    b = agg._make_key_coeffs(cols, DEV)
    for ca, cb in zip(a, b):
        assert torch.equal(ca, cb), "coeffs leaked global RNG state"
