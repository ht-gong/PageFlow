"""Regenerate the plan text and plan pickles: ``make plans [Q="..."]``.

``queries/plan-q<N>.txt`` is the artifact of record: Spark's formatted physical
plan for ``queries/tpch-q<N>.sql``.  ``queries/pickles/planobj-q<N>.pkl`` is
lowered from that text by :meth:`ParsingLayer.lower`, which needs no Spark.

The column ids in a plan (``l_quantity#242``) are Spark expression ids, drawn
from one counter in the JVM.  They therefore depend on the pyspark version and
on everything the JVM ran before the query.  This tool pins both:

* it refuses to capture under any pyspark but the one ``requirements.txt`` pins;
* it captures each query in its own Python process, so its own JVM, which runs
  only ``ParsingLayer``'s table setup before the query.

Regenerating a query therefore reproduces the same ids, whichever other queries
are regenerated with it and in whatever order.

    python3 -m parsing.regenerate [--q N ...]               capture, then lower
    python3 -m parsing.regenerate --lower-only [--q N ...]  re-lower existing text
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUERIES = REPO / "queries"
ALL_QUERIES = list(range(1, 23))

# First line of every captured plan file.  ParsingLayer.lower ignores it (the
# plan tree starts at the first "+-" line).
HEADER = "# Captured by parsing/regenerate.py with pyspark {version}. Do not edit; regenerate with `make plans Q={qid}`.\n"
_HEADER_RE = re.compile(r"# Captured by parsing/regenerate\.py with pyspark (\S+)\.")


def pinned_pyspark_version() -> str:
    """The exact pyspark version ``requirements.txt`` pins (``pyspark==X``)."""
    for line in (REPO / "requirements.txt").read_text().splitlines():
        match = re.fullmatch(r"\s*pyspark\s*==\s*([^\s#;]+)\s*(#.*)?", line)
        if match:
            return match.group(1)
    raise RuntimeError(
        "requirements.txt must pin pyspark exactly (pyspark==X.Y.Z): the plan "
        "column ids depend on the pyspark version")


def text_path(qid: int) -> Path:
    return QUERIES / f"plan-q{qid}.txt"


def pickle_path(qid: int) -> Path:
    return QUERIES / "pickles" / f"planobj-q{qid}.pkl"


def captured_version(text: str) -> str | None:
    """The pyspark version recorded in a plan file's header, if any."""
    match = _HEADER_RE.match(text)
    return match.group(1) if match else None


def _capture_one(qid: int) -> None:
    """Capture one query's plan text.  Runs in a fresh interpreter (see capture)."""
    import pyspark

    pinned = pinned_pyspark_version()
    if pyspark.__version__ != pinned:
        raise SystemExit(
            f"pyspark {pyspark.__version__} is installed but requirements.txt pins "
            f"{pinned}; capturing under another version renumbers every column id. "
            "Rebuild the image (make build) or change the pin deliberately.")

    from parsing.layer import ParsingLayer

    sql = (QUERIES / f"tpch-q{qid}.sql").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="tqp-spark-") as scratch:
        layer = ParsingLayer(scratch_dir=scratch)
        try:
            plan = layer.explain(sql)
        finally:
            layer.spark.stop()
    text_path(qid).write_text(
        HEADER.format(version=pinned, qid=qid) + plan, encoding="utf-8")


def capture(qid: int) -> None:
    """Capture ``qid`` in its own process, so in a JVM that ran nothing else."""
    subprocess.run(
        [sys.executable, "-m", "parsing.regenerate", "--capture-one", str(qid)],
        cwd=REPO, check=True)


def lower_text(qid: int):
    """Lower the checked-in plan text for ``qid`` (the lowering's debug output is
    discarded)."""
    from parsing.layer import ParsingLayer

    text = text_path(qid).read_text(encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return ParsingLayer.lower(text, qid)


def write_pickle(qid: int) -> None:
    plans = lower_text(qid)
    with open(pickle_path(qid), "wb") as f:
        pickle.dump(plans, f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--q", type=int, nargs="+", default=ALL_QUERIES,
                        help="query ids (default: all 22)")
    parser.add_argument("--lower-only", action="store_true",
                        help="skip Spark; re-lower the checked-in plan text")
    parser.add_argument("--capture-one", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.capture_one is not None:
        _capture_one(args.capture_one)
        return 0

    for qid in args.q:
        if not args.lower_only:
            capture(qid)
        write_pickle(qid)
        print(f"Q{qid}: {text_path(qid).relative_to(REPO)} -> "
              f"{pickle_path(qid).relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
