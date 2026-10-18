"""Shared pytest setup for the TQP-Vortex test suites.

The custom CUDA allocator (``libcustom_allocator.so``) can only be swapped
in **once per process**, and only before any CUDA tensor exists.  When each
test module did its own ``change_current_allocator`` (as the old layout did),
running more than one file in a single ``pytest`` invocation raised
``RuntimeError: Can't swap an already initialized allocator``.

This conftest centralises that bootstrap: its top-level code runs once, before
any test module is imported, so every suite shares a single custom-allocator
arena, which is also RMM's device resource (``utility/rmm_arena.py``), so
cuDF allocates from it too.  Individual tests should NOT swap the allocator themselves — they use
the ``cuda_allocator`` fixture (or the module-level skip marks) to gate on
availability.

The ``tests/smoke/`` suite runs the full engine (``main.exec``) on real SF1
data, so the shared arena is sized to the production default (22 GB) rather
than the few GB the allocator/page-cache unit tests need.  The same suite is
also reachable via ``python main.py --smoke``, which bootstraps in main's
``__main__`` first; this conftest's idempotent bootstrap then no-ops.

The SF1 suites (smoke, e2e) take the SF1 dataset from the ``sf1_parquet``
session fixture, the data root holding ``sf1/parquet``, which fails on a GPU
box without it and says how to make it (``make data SF=1``).
"""
import os
import sys

import pytest

from utility.config import tables_path

# Make the repo root importable (``import main``, ``from IO... import``) no
# matter which directory pytest is launched from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Pool sizing for the shared test arena.  Sized to the production default so
# the tests/smoke/ suite (full engine, 7 GB slot pool) fits; the allocator and
# page-cache unit tests use only a small fraction of it.  Its 1 MiB blocks
# are main.py's.
_POOL_BYTES = 22 * 1024 ** 3  # 22 GB
_BLOCK_BYTES = 1024 ** 2  # 1 MiB

_alloc_state = None  # None = untried, True = custom allocator active, False = unavailable


def _ensure_custom_allocator() -> bool:
    """Idempotently install the custom CUDA allocator. Returns True if the
    custom allocator is active (or already was), False if CUDA/torch is
    unavailable."""
    global _alloc_state
    if _alloc_state is not None:
        return _alloc_state

    try:
        import torch
    except Exception:
        _alloc_state = False
        return _alloc_state

    if not torch.cuda.is_available():
        _alloc_state = False
        return _alloc_state

    import TQPlib.tqpmemory as cmp
    from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator

    try:
        alloc = CUDAPluggableAllocator(
            path_to_so_file="libcustom_allocator.so",
            alloc_fn_name="pa_cuda_malloc",
            free_fn_name="pa_cuda_free",
        )
        change_current_allocator(alloc)
        cmp.init(_POOL_BYTES, _BLOCK_BYTES)
    except Exception:
        # The allocator was already swapped earlier in this process (e.g. by
        # main.py's __main__ before it invoked pytest).  The custom allocator
        # is already active, so the suites can still run.
        pass

    # cuDF allocates through RMM: from the arena too (idempotent, so a no-op
    # after main.py's install).
    from utility import rmm_arena
    rmm_arena.install(_BLOCK_BYTES)

    _alloc_state = True
    return _alloc_state


# Bootstrap once, as early as possible: conftest top-level executes before any
# test module is collected/imported, i.e. before any CUDA tensor exists.
_ensure_custom_allocator()


@pytest.fixture(autouse=True)
def _default_codec_map():
    """Every test sees the built-in codec map (class table + chunk sizes),
    whatever ``profiling/results/codec_map.json`` says on this host: the
    nvCOMP manager tests hard-code the default chunks and the policy tests
    assert the default verdict.  Re-armed (``None``) afterwards so a test
    that wants the lazy file load can ask for it explicitly."""
    from compression import format_policy
    format_policy.set_codec_map(format_policy.default_codec_map())
    yield
    format_policy.set_codec_map(None)


@pytest.fixture(autouse=True)
def _telemetry_switch_restored():
    """Every test leaves the telemetry and debug switches
    (``IO/telemetry/gates.py``) as it found them, so a test that sets one
    cannot leak it into the next."""
    from IO.telemetry import gates
    saved = {name: value for name, value in vars(gates).items()
             if name.isupper() and isinstance(value, bool)}
    yield
    for name, value in saved.items():
        setattr(gates, name, value)


@pytest.fixture
def telemetry():
    """Set the telemetry switch for one test: ``telemetry(experiment=True)``.
    Restored afterwards."""
    from IO.telemetry import gates

    def configure(*, experiment):
        gates.EXPERIMENT = bool(experiment)

    return configure


@pytest.fixture(scope="session")
def cuda_allocator() -> bool:
    """Session fixture: True when the custom CUDA allocator is active.

    GPU-dependent tests depend on this and ``pytest.skip`` when it is False
    (no GPU / torch), so the suites stay green on CPU-only machines.
    """
    return _ensure_custom_allocator()


# ── SF1 data ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sf1_parquet(cuda_allocator) -> str:
    """Session fixture: the data root (``"tables path"``) holding the SF1
    dataset, ``sf1/parquet``.  The engine-image services mount
    ``$TQP_DATA_DIR`` there.

    Skips on a CPU-only box (every consumer needs the GPU anyway); on a GPU
    box a missing dataset is a failure, not a skip, so a misconfigured
    environment cannot silently pass as "nothing to test".
    """
    from IO.parquet_reader import table_dir
    if not cuda_allocator:
        pytest.skip("SF1 suites require CUDA")
    root = tables_path()
    if not root:
        pytest.fail("no tables path configured (config.json or TQP_TABLES_PATH)")
    if not os.path.isdir(table_dir(root, 1, "lineitem")):
        pytest.fail(f"no SF1 parquet under {root!r}: run `make data SF=1`; the "
                    "engine-image services mount $TQP_DATA_DIR there (make's "
                    "default $DATA_DIR; a bare docker compose mounts ./data)")
    return root
