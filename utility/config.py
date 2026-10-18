"""Repo configuration loading, with environment overrides.

``config.json`` holds the checked-in defaults.  The field that is
genuinely host-specific is where the datagen parquet lives, and editing a
tracked file to point at it makes every container (and every machine) carry
a local diff.  ``TQP_TABLES_PATH`` overrides it instead; the engine image and
``docker-compose.yml`` set it to the mounted data root.

Paths are anchored on the repo root rather than the CWD, so a caller that runs
from a subdirectory still resolves the same file.
"""
from __future__ import annotations

import json
import os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "config.json"

# env var -> config.json key
_ENV_OVERRIDES = {
    "TQP_TABLES_PATH": "tables path",
}


def load_config(path=None) -> dict:
    """Return config.json as a dict, with environment overrides applied."""
    cfg_path = pathlib.Path(path) if path is not None else _CONFIG_PATH
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for env_var, key in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            cfg[key] = value
    return cfg


def tables_path() -> str:
    """The data root: the datagen parquet, ``sf<n>/parquet/<table>/`` (may
    be empty)."""
    return load_config().get("tables path", "")
