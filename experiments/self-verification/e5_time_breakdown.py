#!/usr/bin/env python3
"""E5 -- Time breakdown: where does each query's time go?

One Nsight trace of a ``main.py`` run with NVTX ranges per pipe and per
host step (``helpers/nvtx_ranges.py``), then ``helpers/trace_analyze.py`` on
the host: kernel / copy-only / idle time per pipe class and host step, GPU
idle in the output steps, exposed D2H by the step that issued it, and CUDA
call counts.  A trace inflates host time (about 8% at SF100): use it to
attribute time, never as a timing claim (E2 is the timing).

With ``--arm`` (as in E2's ``switches``) the arms alternate per occurrence in
the one traced process and each arm is analysed on its own (``ONLY_ARM``);
without, the hot occurrence of each query is analysed.  ``--before`` takes
an earlier run's ``analysis.json`` and prints the ratios to it.

    python3 experiments/self-verification/e5_time_breakdown.py \\
        [--gpu 5] [--q 2 3 ... 22] [--arm old:... --arm new:...] \\
        [--before <analysis.json>] [-- <main.py flags>]
    python3 experiments/self-verification/e5_time_breakdown.py analyse <out dir> [--before <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Its own helpers, then the ones shared with paper_scripts/ (common, perf_json).
sys.path[:0] = [os.path.join(HERE, "helpers"),
                os.path.join(HERE, os.pardir, "paper_scripts", "helpers")]
import common  # noqa: E402

NSYS = ["nsys", "profile", "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--force-overwrite=true"]


def trace(args, extra) -> None:
    common.host_dir(args.out)
    perf = os.path.join(args.out, "perf")
    rep = os.path.join(args.out, "trace")
    main_args = ["--cuda_profile", *extra]
    if args.arm:
        n_occ = args.warm + args.cycles * len(args.arm)
        qids = [q for q in args.q for _ in range(n_occ)]
        runner = ["experiments/self-verification/helpers/switch_ab.py",
                  "--out", args.out, "--warm", str(args.warm), "--nvtx"]
        for arm in args.arm:
            runner += ["--arm", arm]
    else:
        qids = [q for q in args.q for _ in range(2)]  # cold, hot
        runner = ["experiments/self-verification/helpers/nvtx_ranges.py"]
    cmd = [*NSYS, "-o", rep, "python3", *runner, "--",
           *common.main_cmd(args.sf, qids, perf, main_args)[2:]]
    common.compose(cmd, gpu=args.gpu,
                   log=os.path.join(common.ROOT, args.out, "trace.log"))
    common.compose(["nsys", "export", "--type", "sqlite",
                    "--force-overwrite=true", "-o", rep + ".sqlite",
                    rep + ".nsys-rep"], gpu=args.gpu,
                   log=os.path.join(common.ROOT, args.out, "export.log"))
    with open(os.path.join(common.ROOT, args.out, "arms.json"), "w") as f:
        json.dump([a.split(":", 1)[0] for a in args.arm or []], f)


def analyse(out_dir: str, before: str | None) -> None:
    with open(os.path.join(out_dir, "arms.json")) as f:
        arms = json.load(f)
    for arm in arms or [None]:
        name = f"analysis_{arm}" if arm else "analysis"
        env = dict(os.environ)
        if arm:
            env["ONLY_ARM"] = arm
        if before:
            env["BEFORE"] = before
        print(f"\n===== {arm or 'all occurrences'} =====", flush=True)
        res = subprocess.run(
            [sys.executable, os.path.join(HERE, "helpers", "trace_analyze.py"),
             os.path.join(out_dir, "trace.sqlite"),
             os.path.join(out_dir, name + ".json"),
             os.path.join(out_dir, "perf")],
            env=env, capture_output=True, text=True)
        print(res.stdout, end="")
        print(res.stderr, end="", file=sys.stderr)
        with open(os.path.join(out_dir, name + ".txt"), "w") as f:
            f.write(res.stdout)


def main() -> int:
    argv, extra = common.split_extra(sys.argv[1:])
    if argv[:1] == ["analyse"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("dir")
        ap.add_argument("--before")
        a = ap.parse_args(argv[1:])
        analyse(os.path.abspath(a.dir), a.before)
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", default=common.default_gpu(),
                    help="default: TQP_GPU, else GPU (local.env), else 0")
    ap.add_argument("--sf", type=int, default=100)
    ap.add_argument("--q", type=int, nargs="+",
                    default=[q for q in common.QUERIES if q != 1],
                    help="default: Q2-Q22 (Q1 traces on its own)")
    ap.add_argument("--arm", action="append",
                    help="name:module.path=value[,...], as in E2")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--before", help="an earlier run's analysis.json")
    ap.add_argument("--out", default=common.default_out("e5_time_breakdown"),
                    help="relative to the repository root")
    args = ap.parse_args(argv)
    if args.arm and len(args.arm) < 2:
        ap.error("give two or more --arm, or none")
    trace(args, extra)
    analyse(os.path.join(common.ROOT, args.out), args.before)
    return 0


if __name__ == "__main__":
    sys.exit(main())
