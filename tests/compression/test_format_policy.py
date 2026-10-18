"""Unit tests for compression.format_policy.CompressionPolicy.

Pure CPU; no CUDA / engine bootstrap required.  Covers both compression modes
(fixed / adaptive) and scan-materialization narrowing (formats_for).  The
codec map singleton is pinned to the defaults by tests/conftest.py.
"""
import pytest

from compression.format_policy import (
    CompressionPolicy,
    _DEFAULT_BEST,
    MAPPED_TYPES,
)


# --------------------------------------------------------------------------- #
# Adaptive mode: profiled (type, sorted) verdict
# --------------------------------------------------------------------------- #

def test_adaptive_matches_part1_verdict():
    p = CompressionPolicy.adaptive(best=dict(_DEFAULT_BEST))
    # The eight cells from fig_01_compression_pareto.
    assert p.compressed_format("int", False) == "BITCOMP"
    assert p.compressed_format("int", True) == "CASCADED"
    assert p.compressed_format("float", False) == "ANS"
    assert p.compressed_format("float", True) == "CASCADED"
    assert p.compressed_format("date", False) == "BITCOMP"
    assert p.compressed_format("date", True) == "CASCADED"
    assert p.compressed_format("string", False) == "ANS"
    assert p.compressed_format("string", True) == "ANS"


def test_adaptive_sortedness_flips_codec_for_numeric():
    p = CompressionPolicy.adaptive(best=dict(_DEFAULT_BEST))
    for t in ("int", "float", "date"):
        assert p.compressed_format(t, True) != p.compressed_format(t, False), t
    # Strings are order-insensitive: ANS both ways.
    assert p.compressed_format("string", True) == p.compressed_format("string", False)


def test_default_for_is_unsorted_class_best():
    p = CompressionPolicy.adaptive(best=dict(_DEFAULT_BEST))
    for t in MAPPED_TYPES:
        assert p.default_for(t) == p.compressed_format(t, False)
    # Never a sorted-only codec.
    assert p.default_for("int") == "BITCOMP"  # not CASCADED


def test_adaptive_unknown_type_degrades_to_raw():
    p = CompressionPolicy.adaptive(best=dict(_DEFAULT_BEST))
    assert p.compressed_format("widget", False) is None
    assert p.formats_for("widget", False) == ["RAW"]


# --------------------------------------------------------------------------- #
# formats_for: scan materialization narrowing
# --------------------------------------------------------------------------- #

def test_formats_for_raw_plus_at_most_one():
    p = CompressionPolicy.adaptive(best=dict(_DEFAULT_BEST))
    for t in MAPPED_TYPES:
        for s in (False, True):
            fmts = p.formats_for(t, s)
            assert fmts[0] == "RAW"
            assert len(fmts) <= 2
            if len(fmts) == 2:
                assert fmts[1] == p.compressed_format(t, s)


def test_int_sorted_materializes_cascaded():
    p = CompressionPolicy.adaptive(best=dict(_DEFAULT_BEST))
    assert p.formats_for("int", True) == ["RAW", "CASCADED"]
    assert p.formats_for("int", False) == ["RAW", "BITCOMP"]


# --------------------------------------------------------------------------- #
# Fixed mode: reproduces --compression exactly
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("codec", ["ANS", "BITCOMP", "CASCADED"])
def test_fixed_forces_single_codec(codec):
    p = CompressionPolicy.fixed(codec)
    for t in MAPPED_TYPES:
        for s in (False, True):
            assert p.compressed_format(t, s) == codec
            assert p.formats_for(t, s) == ["RAW", codec]
    assert p.default_for("int") == codec


def test_fixed_raw_is_raw_only():
    for raw in ("RAW", None):
        p = CompressionPolicy.fixed(raw)
        assert p.compressed_format("int", True) is None
        assert p.formats_for("int", True) == ["RAW"]
        assert p.default_for("string") is None


def test_fixed_rejects_unknown_codec():
    with pytest.raises(ValueError):
        CompressionPolicy.fixed("ZSTD")


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        CompressionPolicy(mode="bogus")
