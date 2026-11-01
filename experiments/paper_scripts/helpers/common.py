"""What the experiment drivers (``paper_scripts/``, ``self-verification/``)
share: the repository root, the data root and GPU defaults, the engine code's
commit, and running a command in the ``dev`` container.

The self-verification drivers run on the host and their engine work in the
``dev`` service (``docker compose run``, the working tree mounted at
``/opt/TQP-Vortex``), pinned to one GPU and, when its NUMA node is found, to
that node's CPUs; E1 runs the baselines' ``run.sh``, and E2-E4 run
``main.py`` natively or in the ``experiments`` container (AGENTS.md,
"Experiments").
Paths handed to a container are relative to the repository root; a
container writes root-owned files, so a driver creates each output directory
on the host before a container writes into it, and writes its own summaries
beside the container's subdirectories, never inside them.

The data root and the GPU default as in the Makefile: ``TQP_DATA_DIR``,
else ``$DATA_DIR``; ``TQP_GPU``, else ``$GPU``, else 0.  ``DATA_DIR`` and
``GPU`` come from the environment, else from ``local.env`` at the repository
root.  With no data root a driver stops, rather than let compose mount its
own default, ``./data``.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.pardir, os.pardir, os.pardir))
QUERIES = list(range(1, 23))



def box_setting(key: str) -> str | None:
    """``key`` from the environment, else its first value in ``local.env``
    (as the Makefile and ``bench/run_tqp.sh`` read it); None when unset."""
    if key in os.environ:
        return os.environ[key]
    try:
        with open(os.path.join(ROOT, "local.env")) as f:
            for line in f:
                k, sep, v = line.rstrip("\n").partition("=")
                if sep and k == key:
                    return v
    except OSError:
        pass
    return None


def default_gpu() -> str:
    """A driver's ``--gpu`` default: ``TQP_GPU``, else ``GPU``, else 0."""
    return box_setting("TQP_GPU") or box_setting("GPU") or "0"


def data_dir() -> str:
    """The host data root: ``TQP_DATA_DIR``, else ``$DATA_DIR``."""
    path = box_setting("TQP_DATA_DIR") or box_setting("DATA_DIR")
    if not path:
        sys.exit("set DATA_DIR in local.env (copy local.env.example) or "
                 "TQP_DATA_DIR")
    return path


# The four baseline engines.  Each benchmarks on its own format-map datasets,
# under $DATA_DIR/<engine>, which a suite's loop makes and drops one SF at a time
# (baselines/datagen/engine_data.py); the experiment drivers only read them.
BASELINE_ENGINES = ("sirius", "rapids", "polars_gpu", "duckdb_cpu")


def engine_root(engine: str) -> str | None:
    """``$DATA_DIR/<engine>`` (``DATA_DIR`` from the environment or
    ``local.env``); None when no data root is set."""
    top = box_setting("DATA_DIR")
    return os.path.join(top, engine) if top else None


def dataset_problems(root: str, sf: int, engine: str) -> list[str]:
    """Why ``<root>/sf<sf>`` and the warm pass's ``<root>/sf1`` are not ready
    for ``engine``, or [] when they are: a dataset is missing, or was not
    generated from ``engine``'s format map (its ``NDSGEN.json``)."""
    out = []
    for s in sorted({1, sf}):
        d = os.path.join(root, f"sf{s}")
        if not os.path.isdir(os.path.join(d, "parquet")):
            out.append(f"{d}: no dataset")
            continue
        try:
            with open(os.path.join(d, "NDSGEN.json")) as f:
                fmt = json.load(f).get("format", "")
        except (OSError, ValueError):
            fmt = ""
        if not (fmt.startswith("map ") and fmt.split()[-1] == engine):
            out.append(f"{d}: not {engine}'s format-map dataset "
                       f"({fmt or 'no NDSGEN.json'})")
    return out


def source_commit() -> str:
    """The engine code's commit, for the drivers' ``commit`` column: the
    checkout's HEAD, ``+dirty`` when the tree differs from it outside
    experiments/.  In the experiments container (``make experiments``), which
    has no .git and runs the image's copy of the engine, the commit the image
    was built from (``TQP_IMAGE_COMMIT``, which ``make build`` sets)."""
    if not os.path.exists(os.path.join(ROOT, ".git")):
        return os.environ.get("TQP_IMAGE_COMMIT") or "unknown"
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "HEAD", "--", ".",
                       ":!experiments"], capture_output=True).returncode:
        commit += "+dirty"
    return commit


def default_out(name: str) -> str:
    """``out/<name>/<YYYYmmdd-HHMM>``, relative to the tree root."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return os.path.join("out", name, stamp)


def host_dir(rel: str) -> str:
    """Create ``ROOT/rel`` on the host (so the host can write beside what a
    container writes there) and return its absolute path."""
    path = os.path.join(ROOT, rel)
    os.makedirs(path, exist_ok=True)
    return path


def node_cpus(gpu: str) -> str | None:
    """The CPU list of ``gpu``'s NUMA node, from its PCI address; None when
    it cannot be found."""
    try:
        bus = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=pci.bus_id",
             "--format=csv,noheader"], capture_output=True, text=True,
            check=True).stdout.strip().lower()
        # nvidia-smi prints an 8-digit PCI domain; sysfs uses 4.
        dom, rest = bus.split(":", 1)
        dev = f"{int(dom, 16):04x}:{rest}"
        with open(f"/sys/bus/pci/devices/{dev}/numa_node") as f:
            node = int(f.read())
        if node < 0:
            return None
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            return f.read().strip()
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def compose(cmd: list[str], *, gpu: str, log: str,
            env: dict | None = None, cpus: str | None = None) -> int:
    """Run ``cmd`` in the ``dev`` container on ``gpu``, its CPUs
    limited to ``cpus`` (``node_cpus(gpu)`` when None), with the engine
    variables ``env``.  Output goes to the file ``log``; returns the exit
    code."""
    cpus = cpus or node_cpus(gpu)
    host_env = dict(os.environ, TQP_GPU=str(gpu), TQP_DATA_DIR=data_dir())
    files = ["-f", os.path.join(ROOT, "docker-compose.yml")]
    override = None
    if cpus:
        with tempfile.NamedTemporaryFile("w", suffix=".yml",
                                         delete=False) as f:
            f.write(f'services:\n  dev:\n    cpuset: "{cpus}"\n')
            override = f.name
        files += ["-f", override]
    args = ["docker", "compose", "-p", "tqp-vortex", "--project-directory",
            ROOT, *files, "run", "--rm", "-T"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += ["dev", *cmd]
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(cmd)}  (log {log})",
          flush=True)
    try:
        with open(log, "w") as f:
            rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT,
                                 env=host_env)
    finally:
        if override:
            os.unlink(override)
    print(f"[{time.strftime('%H:%M:%S')}] exit {rc}", flush=True)
    return rc


def main_cmd(sf: int, queries: list[int], perf_report: str,
             extra: list[str] = ()) -> list[str]:
    """``python3 main.py`` at the engine's defaults, debug output off."""
    return ["python3", "main.py", "--SF", str(sf),
            "--q", *map(str, queries), "--no_debug", *extra,
            "--perf_report", perf_report]


def split_extra(argv: list[str]) -> tuple[list[str], list[str]]:
    """``(driver args, main.py args)``: everything after ``--`` goes to
    main.py."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []
