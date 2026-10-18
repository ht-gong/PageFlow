#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_times(log_dir):
    rows = []
    for path in log_dir.glob("q*.txt"):
        match = re.fullmatch(r"q(\d+)", path.stem)
        time = re.search(r"^time:\s+avg=([0-9.]+)", path.read_text(errors="replace"), re.M)
        if match and time:
            rows.append((int(match.group(1)), path.stem, float(time.group(1))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pruning_off")
    parser.add_argument("pruning_on")
    parser.add_argument("--out")
    parser.add_argument("--sf")
    args = parser.parse_args()

    off = Path(args.pruning_off)
    on = Path(args.pruning_on)
    out = Path(args.out) if args.out else on.parent / "pruning_runtime.png"

    data = {"query": {}, "runtime_off_ms": {}, "runtime_on_ms": {}}
    for number, query, runtime in read_times(off):
        data["query"][number] = query
        data["runtime_off_ms"][number] = runtime
    for number, query, runtime in read_times(on):
        data["query"][number] = query
        data["runtime_on_ms"][number] = runtime

    df = pd.DataFrame(data).rename_axis("query_n").sort_index()
    df = df.dropna()
    if df.empty:
        raise SystemExit("No completed query pairs found")
    df["speedup"] = df["runtime_off_ms"] / df["runtime_on_ms"]
    df.to_csv(out.with_suffix(".csv"), index=False)

    ax = df.set_index("query")[["runtime_off_ms", "runtime_on_ms"]].plot(
        kind="bar", figsize=(8, 4)
    )
    ax.set_xlabel("TPC-H query")
    ax.set_ylabel("Mean runtime (ms)")
    title = "GOLAP pruning"
    if args.sf:
        title += f" at SF{args.sf}"
    ax.set_title(title)
    ax.legend(["pruning off", "pruning on"])
    plt.tight_layout()
    plt.savefig(out, dpi=200)

    print(out)
    print(out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
