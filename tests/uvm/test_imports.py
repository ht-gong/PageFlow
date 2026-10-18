"""The UVM executor's own modules (``uvm/``) import no streaming machinery.

Every import of every module in ``uvm/`` is checked against an allowlist
of exactly what they import (every entry is used): torch and numpy, leaf
functions of the streaming engine's operator modules (``operators``, but
no ``tqp_*`` operator), ``constants``, ``variable``'s type names, the
logger and tensor utilities, the NUMA and RMM helpers, the scan's parquet
table, occurrence count, type lookup and dtypes (``IO.data_scanner``) and
the native memory module.  Anything else (``VortexPipeline``, the rearrange,
region producers, record tables, the pinned and slot pools, the cache,
transfers, the searches, codecs) fails.  The check covers ``uvm/``'s own
imports only: the operator modules it imports leaf functions from also
define the streaming operators and import that machinery themselves.
Nothing in ``uvm/`` names what reads the arena's occupancy or fits a
structure to it (the arena stats, free bytes, the slot governor, the
aggregation slabs, the in-place sorted-run merge, key narrowing): every
size comes from the chunk budget, the partition budget or a fixed cap.
And ``cmp.prefetch``, the one hint the mode gives the driver, is called
from the UVM scan only, outside the timed region.  CPU-only.
"""
from __future__ import annotations

import ast
import pathlib
import re

_REPO = pathlib.Path(__file__).resolve().parents[2]

ALLOWED_MODULES = (
    "__future__", "collections", "contextlib",
    "torch", "numpy",
    "operators", "constants",
    "utility.logger", "utility.tensor_utils", "utility.numa",
    "utility.rmm_arena",
    "TQPlib.tqpmemory",
    "uvm",
)
# Modules from which only these names may be imported.
ALLOWED_NAMES = {
    "variable": {"torch_to_type", "type_to_torch"},
    "IO.data_scanner": {"column_type", "engine_dtype", "parquet_table",
                        "scan_occurrences"},
}


# Names that read arena occupancy or fit a structure to it, rejected
# anywhere in ``uvm/`` (as a name, an attribute, an imported name or a
# string key).
OCCUPANCY_NAMES = frozenset({
    "get_arena_stats", "free_b", "largest_free_run_b", "mem_get_info",
    "lendable_b", "lend_stats",
    "DenseAggregationSlab", "FixedColumnSlab", "aggregation_slab",
    "merge_sorted_unique_runs_inplace",
})


def _occupancy_uses(path: pathlib.Path):
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [part for alias in node.names
                     for part in alias.name.split(".")]
            if isinstance(node, ast.ImportFrom) and node.module:
                names += node.module.split(".")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        out += [f"{path.name}:{node.lineno}: {name}" for name in names
                if name in OCCUPANCY_NAMES]
    return out


def _under(module: str, m: str) -> bool:
    return module == m or module.startswith(m + ".")


def _module_allowed(module: str) -> bool:
    return any(_under(module, m) for m in ALLOWED_MODULES)


def _imports(path: pathlib.Path):
    """``(module, name)`` of each import in ``path`` (name None for a
    plain ``import``)."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from ((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield from ((node.module, alias.name) for alias in node.names)


def _violations(path: pathlib.Path):
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed(alias.name):
                    out.append(f"{path.name}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}:{node.lineno}: relative import"
            module = node.module
            for alias in node.names:
                name = alias.name
                if module in ALLOWED_NAMES:
                    ok = name in ALLOWED_NAMES[module]
                else:
                    # ``from utility import numa`` imports the module
                    # ``utility.numa``.
                    ok = (_module_allowed(module)
                          or _module_allowed(f"{module}.{name}"))
                if ok and module.split(".")[0] == "operators":
                    ok = not name.startswith("tqp_")
                if not ok:
                    out.append(f"{path.name}:{node.lineno}: from {module} "
                               f"import {name}")
    return out


def test_uvm_imports_only_leaf_modules():
    files = sorted((_REPO / "uvm").glob("*.py"))
    assert {f.name for f in files} >= {
        "__init__.py", "aggregate.py", "column.py", "executor.py",
        "buckets.py", "gather.py", "join.py", "morsel.py", "scan.py"}
    bad = [v for f in files for v in _violations(f)]
    assert not bad, "uvm/ imports outside its allowlist:\n" + "\n".join(bad)
    # The allowlist is exact: every entry is imported somewhere.
    imports = [i for f in files for i in _imports(f)]
    unused = [m for m in ALLOWED_MODULES if not any(
        _under(mod, m) or (name and _under(f"{mod}.{name}", m))
        for mod, name in imports)]
    unused += [f"{mod}.{name}" for mod, names in ALLOWED_NAMES.items()
               for name in names if (mod, name) not in imports]
    assert not unused, f"allowlist entries uvm/ does not import: {unused}"


def test_the_allowlist_catches_streaming_imports(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from IO.pipeline.vortex_pipeline import VortexPipeline\n"
        "from operators.filter import tqp_filter\n"
        "from variable import TableVariable\n"
        "import IO.pages.pinned_mem\n"
        "from adaptive import sizing_controller\n"
        "from utility import numa\n"
        "from operators.operator_utils import evaluate_filter\n",
        encoding="utf-8")
    assert [v.split(": ", 1)[1] for v in _violations(probe)] == [
        "from IO.pipeline.vortex_pipeline import VortexPipeline",
        "from operators.filter import tqp_filter",
        "from variable import TableVariable",
        "import IO.pages.pinned_mem",
        "from adaptive import sizing_controller",
    ]


def test_uvm_reads_no_arena_occupancy():
    files = sorted((_REPO / "uvm").glob("*.py"))
    bad = [v for f in files for v in _occupancy_uses(f)]
    assert not bad, "uvm/ reads arena occupancy:\n" + "\n".join(bad)


def test_the_occupancy_check_catches_its_names(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import TQPlib.tqpmemory as cmp\n"
        "from operators.aggregation_slab import FixedColumnSlab\n"
        "from operators.sorted_run_merge import reduce_tick_slice\n"
        "free = cmp.get_arena_stats()['largest_free_run_b']\n"
        "lend = cmp.get_arena_stats()['lendable_b']\n"
        "torch.cuda.mem_get_info()\n",
        encoding="utf-8")
    assert sorted(v.split(": ", 1)[1] for v in _occupancy_uses(probe)) == [
        "FixedColumnSlab", "aggregation_slab", "get_arena_stats",
        "get_arena_stats", "largest_free_run_b", "lendable_b",
        "mem_get_info"]


def test_prefetch_is_called_from_the_uvm_scan_only():
    call = re.compile(r"\bcmp\.prefetch\(|tqpmemory\.prefetch\(|MemAdvise")
    skip = {"tests", "baselines", "experiments", "out", "build", ".git",
            "pybind11"}
    callers = sorted(
        str(p.relative_to(_REPO)) for p in _REPO.rglob("*.py")
        if not skip & set(p.relative_to(_REPO).parts[:1])
        and call.search(p.read_text(encoding="utf-8", errors="replace")))
    assert callers == ["uvm/scan.py"]
