#!/usr/bin/env python3
"""E6 -- Ablation: what does each mechanism (adaptive sizing, the dataflow
cache, adaptive compression) buy?  The paper's figures 1-5.

Runs ``helpers/ablation.py`` in the working tree's ``dev`` container (the
four-arm ladder, timed passes with every telemetry switch off, then one
untimed telemetry pass per arm), then draws the figures on the host
(``helpers/ablation_figs.py``, ``helpers/buffer_traffic_fig.py``).
Arguments after ``--`` go to ``ablation.py`` (``--reps``, ``--arms``,
``--slot_pool_gb``, ``--chunk_size_mb``, ``--extra <main.py flags>``).

    python3 experiments/paper_scripts/e6_ablation.py \\
        [--gpu 5] --sf 700 [-- --reps 3 --extra --pinned_pool_gb 500]
    python3 experiments/paper_scripts/e6_ablation.py figures <out dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "helpers"))
import common  # noqa: E402


def figures(out_dir: str) -> None:
    """The ablation figures into ``<out_dir>/plot``."""
    for script, args in (("ablation_figs.py", [out_dir]),
                         ("buffer_traffic_fig.py",
                          ["--data", out_dir, "--out",
                           os.path.join(out_dir, "plot")])):
        subprocess.run([sys.executable, os.path.join(HERE, "helpers", script),
                        *args], check=False)


def main() -> int:
    argv, passthrough = common.split_extra(sys.argv[1:])
    if argv[:1] == ["figures"]:
        figures(os.path.abspath(argv[1]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default=common.default_gpu(),
                    help="default: TQP_GPU, else GPU (local.env), else 0")
    ap.add_argument("--sf", type=int, default=100)
    ap.add_argument("--q", type=int, nargs="+", default=common.QUERIES)
    ap.add_argument("--out", default=common.default_out("e6_ablation"),
                    help="relative to the repository root")
    args = ap.parse_args(argv)
    out_dir = common.host_dir(args.out)
    # Host-owned, so the host's figures can join the container's in it.
    common.host_dir(os.path.join(args.out, "plot"))
    cmd =["python3", "experiments/paper_scripts/helpers/ablation.py",
           "--SF", str(args.sf), "--q", *map(str, args.q), "--out", args.out,
           *passthrough]
    rc = common.compose(cmd, gpu=args.gpu,
                        log=os.path.join(out_dir, "ablation.log"))
    if rc == 0:
        figures(out_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
