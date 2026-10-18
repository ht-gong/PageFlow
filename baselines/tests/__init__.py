"""Unit tests of the spill pass's record builders (CPU only; `make spill-test`)."""
import importlib.util
import os
import sys

BASELINES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BASELINES)


def load(rel, name):
    """A builder module by its path under baselines/: each engine's is
    spill_stats.py, so they cannot share one import name."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASELINES, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
