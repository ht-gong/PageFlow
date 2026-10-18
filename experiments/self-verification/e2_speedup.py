#!/usr/bin/env python3
"""E2 -- End-to-end speedup: how much faster is the new code, per query and
over the suite?

An in-process A/B: one ``main.py`` in the working tree's ``dev`` container
alternates the arms per query occurrence (``helpers/switch_ab.py``), so the
shared host's drift cancels.  An arm is ``name:module.path=value[,...]``
over engine switches, the first arm being the base, so the old path must
still sit behind a switch, e.g. a module global ``NEW_PATH`` in
``IO/some/module.py``:

  --arm old:IO.some.module.NEW_PATH=False
  --arm new:IO.some.module.NEW_PATH=True

Times are ``query_times.csv`` rows (AGENTS.md, "TIMING"); every telemetry
switch stays off.  Prints per query and arm the median over cycles, and for
the suite and the plan's query groups A-E (``helpers/common.py``) the change
against the base with a 95% paired-cycle bootstrap interval.

    python3 experiments/self-verification/e2_speedup.py \\
        [--gpu 5] --arm ... --arm ... [--cycles 3] [-- <main.py flags>]
    python3 experiments/self-verification/e2_speedup.py --summarise <out dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Its own helpers, then the ones shared with paper_scripts/ (common, perf_json).
sys.path[:0] = [os.path.join(HERE, "helpers"),
                os.path.join(HERE, os.pardir, "paper_scripts", "helpers")]
import common  # noqa: E402
import switch_ab_analyze  # noqa: E402


def summarise(out_dir: str) -> None:
    res = switch_ab_analyze.analyse(out_dir, common.GROUPS)
    switch_ab_analyze.print_summary(out_dir, res)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


def main() -> int:
    argv, extra = common.split_extra(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default=common.default_gpu(),
                    help="default: TQP_GPU, else GPU (local.env), else 0")
    ap.add_argument("--sf", type=int, default=100)
    ap.add_argument("--q", type=int, nargs="+", default=common.QUERIES)
    ap.add_argument("--arm", action="append", default=[],
                    help="name:module.path=value[,...]; the first is the base")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default=common.default_out("e2_speedup"),
                    help="relative to the repository root")
    ap.add_argument("--summarise", metavar="DIR",
                    help="re-print a finished run instead of running")
    args = ap.parse_args(argv)
    if args.summarise:
        summarise(os.path.abspath(args.summarise))
        return 0
    if len(args.arm) < 2:
        ap.error("give two or more --arm")

    n_occ = args.warm + args.cycles * len(args.arm)
    qids = [q for q in args.q for _ in range(n_occ)]
    out_dir = common.host_dir(args.out)
    cmd = ["python3", "experiments/self-verification/helpers/switch_ab.py",
           "--out", args.out, "--warm", str(args.warm)]
    for arm in args.arm:
        cmd += ["--arm", arm]
    perf = os.path.join(args.out, "perf")
    cmd += ["--", *common.main_cmd(args.sf, qids, perf, extra)[2:]]
    common.compose(cmd, gpu=args.gpu,
                   log=os.path.join(out_dir, "switch_ab.log"))
    summarise(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
