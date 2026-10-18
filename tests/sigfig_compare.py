"""Significant-figure comparator for engine result files.

``results.out`` is pandas-rendered text: per query, a header line and then
every row of the result, each led by its index.  The checked-in reference
holds DuckDB's answers rendered the same way.  Numeric tokens are compared
after rounding to ``n`` significant figures, text tokens after whitespace
normalisation (pandas right-justifies columns, so inter-column spacing
differs for identical data).  In a result's rows, integer tokens are
compared exactly: ``main.py`` prints every float with six decimals, so an
integer token is a key, a count or digits inside text, never a float.  The
headers are compared in order; each result's rows are compared without
their index and in any order, since rows that tie on a query's ORDER BY can
come out in any order.  So the row order itself is not checked, only which
rows a LIMIT keeps.

Two tolerances are in use:

* :data:`SIGFIGS_THRESHOLD` (10) — same-geometry runs.  Float partial sums are
  reduced in the same order, so results agree to 10 significant figures
  (not to the bit: Q1's aggregation drifts at the last ulp between two
  identical runs — atomic float adds, A/A-verified 2026-07-20).
* :data:`SIGFIGS_GEOMETRY` (6) — runs at a different chunk geometry (or with
  live sizing, which re-chunks mid-run).  Reordered partial sums drift past
  10 significant figures; 6 is the anchor the SF1/SF100 live cells hold.
"""
from __future__ import annotations

import math
import re
from collections import Counter

SIGFIGS_THRESHOLD = 10
SIGFIGS_GEOMETRY = 6

# Match ints/floats incl. scientific notation (not inf/nan)
NUM_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
# A result row's leading index.
INDEX_RE = re.compile(r'\d+')
# A numeric token with no point and no exponent.
INT_RE = re.compile(r'[-+]?\d+')


def _tokenize(line: str):
    """Yield (is_number: bool, token: str) covering the whole line."""
    pos = 0
    for m in NUM_RE.finditer(line):
        if m.start() > pos:
            yield (False, line[pos:m.start()])
        yield (True, m.group(0))
        pos = m.end()
    if pos < len(line):
        yield (False, line[pos:])


def _first_diff_idx(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def _sigfigs_in_token(tok: str) -> int:
    """Infer the significant-figure count from a numeric token's *string*.

    Only the mantissa matters; leading zeros are not significant; with a
    decimal point trailing zeros are significant, without one they are not;
    an all-zero number counts as 1.
    """
    s = tok.lstrip("+-")
    mantissa = re.split(r'[eE]', s, maxsplit=1)[0]
    if '.' in mantissa:
        left, right = mantissa.split('.', 1)
        digits = re.sub(r'\D', '', left.lstrip('0') + right).lstrip('0')
        if digits == '':
            return max(1, len(re.sub(r'\D', '', right)))
        return len(digits)
    s2 = mantissa.lstrip('0')
    if s2 == '':
        return 1
    return len(s2.rstrip('0'))


def _round_to_sigfigs(x: float, n: int) -> float:
    if x == 0 or math.isnan(x) or math.isinf(x):
        return x
    return float(f"{x:.{max(n - 1, 0)}e}")


def _compare_numbers_by_sigfig(tok_got: str, tok_ref: str, line_no: int,
                               token_idx: int, fixed_sigfigs: int | None = None):
    """Compare two numeric tokens at ``fixed_sigfigs`` (or the reference
    token's own precision when None)."""
    v1 = float(tok_got)
    v2 = float(tok_ref)
    if any(map(math.isnan, (v1, v2))) or any(map(math.isinf, (v1, v2))):
        assert v1 == v2, (
            f"Line {line_no}, numeric token {token_idx}: special value mismatch.\n"
            f"  got: {tok_got}\n  ref: {tok_ref}")
        return
    n = fixed_sigfigs if fixed_sigfigs is not None else _sigfigs_in_token(tok_ref)
    r1 = _round_to_sigfigs(v1, n)
    r2 = _round_to_sigfigs(v2, n)
    if not math.isclose(r1, r2, rel_tol=0, abs_tol=0):
        raise AssertionError(
            f"Line {line_no}, numeric token {token_idx} mismatch at {n} significant figures:\n"
            f"  got: {tok_got} -> rounded({n} sf) = {r1}\n"
            f"  ref: {tok_ref} -> rounded({n} sf) = {r2}")


def _compare_lines(line1: str, line2: str, line_no: int, fixed_sigfigs: int | None = None):
    t1 = list(_tokenize(line1))
    t2 = list(_tokenize(line2))
    assert len(t1) == len(t2), (
        f"Line {line_no}: token count differs.\n  got: {t1}\n  ref: {t2}")
    for j, ((isnum1, tok1), (isnum2, tok2)) in enumerate(zip(t1, t2), start=1):
        if isnum1 and isnum2:
            _compare_numbers_by_sigfig(tok1, tok2, line_no, j, fixed_sigfigs=fixed_sigfigs)
        elif isnum1 != isnum2:
            raise AssertionError(
                f"Line {line_no}, token {j}: numeric/text kind mismatch:\n"
                f"  got: {tok1!r} ({'num' if isnum1 else 'text'})\n"
                f"  ref: {tok2!r} ({'num' if isnum2 else 'text'})")
        else:
            norm1, norm2 = " ".join(tok1.split()), " ".join(tok2.split())
            if norm1 != norm2:
                k = _first_diff_idx(norm1, norm2)
                raise AssertionError(
                    f"Line {line_no}, text token {j} differs (whitespace-normalized) at char {k}:\n"
                    f"  got: {tok1!r} -> {norm1!r}\n"
                    f"  ref: {tok2!r} -> {norm2!r}")


def _split_results(text: str):
    """``[(line_no, header, rows)]``, one entry per result.  A header line
    starts with whitespace (the blank above the index column) and each row
    under it with its index; a row is ``(line_no, text after the index)``.
    Any other line (``Q16 Output Skipped``) stands alone, its rows None."""
    results = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        index = INDEX_RE.match(line)
        if index and results and results[-1][2] is not None:
            results[-1][2].append((line_no, line[index.end():]))
        else:
            results.append((line_no, line, [] if line[:1].isspace() else None))
    return results


def _number_key(tok: str, sigfigs: int):
    """An integer token exactly, any other number rounded to ``sigfigs``."""
    if INT_RE.fullmatch(tok):
        return int(tok)
    return _round_to_sigfigs(float(tok), sigfigs)


def _row_key(row: str, sigfigs: int):
    """A row's comparison key: integers exact, other numbers rounded to
    ``sigfigs``, text whitespace-normalised.  Two rows compare equal iff
    their keys do."""
    return tuple((0, _number_key(tok, sigfigs)) if isnum
                 else (1, " ".join(tok.split()))
                 for isnum, tok in _tokenize(row))


def _rows_with(side: str, rows, keys: Counter, sigfigs: int, limit: int = 5):
    """Up to ``limit`` of ``rows`` whose keys are in ``keys``, one line each."""
    keys, out = Counter(keys), []
    for line_no, row in rows:
        key = _row_key(row, sigfigs)
        if keys[key] > 0 and len(out) < limit:
            keys[key] -= 1
            out.append(f"    {side} line {line_no}: {' '.join(row.split())}")
    return out


def _compare_result(got, ref, sigfigs: int) -> None:
    """Assert one result of each side agrees: the header, then the rows."""
    (g_no, g_head, g_rows), (r_no, r_head, r_rows) = got, ref
    _compare_lines(g_head, r_head, g_no, fixed_sigfigs=sigfigs)
    assert (g_rows is None) == (r_rows is None), (
        f"Line {g_no}: got {g_head!r}, ref line {r_no} {r_head!r}")
    if g_rows is None:
        return
    got_keys = Counter(_row_key(row, sigfigs) for _, row in g_rows)
    ref_keys = Counter(_row_key(row, sigfigs) for _, row in r_rows)
    if got_keys != ref_keys:
        only_got, only_ref = got_keys - ref_keys, ref_keys - got_keys
        raise AssertionError("\n".join([
            f"Line {g_no} (ref line {r_no}): the rows under "
            f"{' '.join(g_head.split())!r} differ at {sigfigs} significant "
            f"figures: got {len(g_rows)} rows, ref {len(r_rows)}; "
            f"{sum(only_got.values())} only in got, "
            f"{sum(only_ref.values())} only in ref (up to 5 of each):",
            *_rows_with("got", g_rows, only_got, sigfigs),
            *_rows_with("ref", r_rows, only_ref, sigfigs)]))


def compare_results(got: str, ref: str, sigfigs: int = SIGFIGS_THRESHOLD) -> None:
    """Assert two result texts hold the same results at ``sigfigs``: the
    same headers in the same order, and under each header the same rows in
    any order (compared as multisets of row keys).  Every result is
    checked; the error lists each one that differs."""
    got_res, ref_res = _split_results(got), _split_results(ref)
    assert len(got_res) == len(ref_res), (
        f"Result count differs: got {len(got_res)} vs ref {len(ref_res)}\n"
        f"  got: {[' '.join(h.split()) for _, h, _ in got_res]}\n"
        f"  ref: {[' '.join(h.split()) for _, h, _ in ref_res]}")
    errors = []
    for g, r in zip(got_res, ref_res):
        try:
            _compare_result(g, r, sigfigs)
        except AssertionError as e:
            errors.append(str(e))
    assert not errors, (
        f"{len(errors)} of {len(got_res)} results differ:\n" + "\n".join(errors))


def results_of(text: str, qids) -> str:
    """The results of ``qids`` from a results text of all 22 queries in
    order, as text: each its header line (led by the blank above the index
    column) and its rows (each led by its index), or its one line
    (``Q16 Output Skipped``)."""
    results = []
    for line in text.splitlines(keepends=True):
        if not line[:1].isdigit():
            results.append("")
        results[-1] += line
    assert len(results) == 22, len(results)
    return "".join(results[q - 1] for q in qids)
