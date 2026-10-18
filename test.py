#!/usr/bin/env python
"""Run the TQP-Vortex test suites under pytest.

The suites live under ``tests/`` (a shared ``tests/conftest.py`` bootstraps
the custom CUDA allocator once for the whole run); ``tests/README.md`` lists
them all. The two end-to-end ones:

  * ``tests/smoke/``      — SF1 end-to-end correctness (all 22 TPC-H queries
                            vs reference) in the shipped default mode. Also
                            runnable via ``main.py --smoke``.
  * ``tests/e2e/``        — the same, as ``main.py`` subprocesses up a
                            ladder: everything off, + compression search,
                            + sizing search, + each cache policy.
                            Marker ``e2e``; ~2.5 min.

GPU suites skip on a CPU-only box; the SF1 suites need the SF1 dataset
(``make data SF=1``) and fail without it (``tests/conftest.py::sf1_parquet``).

Usage:
    python test.py                  # run all suites
    python test.py tests/allocator  # run one suite (or any path)
    python test.py -k governor      # forward any pytest flags
"""
import sys

import pytest

if __name__ == "__main__":
    args = sys.argv[1:]
    # Default the collection target to tests/ when only flags (or nothing)
    # are given, so flag-only invocations stay scoped to the suites.
    if not args or all(a.startswith("-") for a in args):
        args = ["tests/", *args]
    sys.exit(pytest.main(["-q", *args]))
