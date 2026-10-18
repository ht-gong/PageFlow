#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def get_metric(text, name):
    m = re.search(rf"^{re.escape(name)}:\s+([0-9.]+)", text, re.M)
    return float(m.group(1)) if m else None


def get_avg_ms(text):
    m = re.search(r"^time:\s+avg=([0-9.]+)", text, re.M)
    return float(m.group(1)) if m else None


def plot(df, cols, title, ylabel, out):
    ax = df.set_index("mode")[cols].plot(kind="bar", figsize=(7, 4))
    ax.set_xlabel("mode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_io(df, out):
    df = df.sort_values("query_n")
    x = range(len(df))
    fig, left = plt.subplots(figsize=(10, 4.8))
    right = left.twinx()

    left.bar([i - 0.2 for i in x], df["read_mb"], width=0.4, label="read MB")
    left.bar([i + 0.2 for i in x], df["uncompressed_read_mb"], width=0.4, label="logical MB")
    right.plot(x, df["effective_throughput_gbs"], marker="o", label="effective GB/s")
    right.plot(x, df["io_throughput_gbs"], marker="o", label="io GB/s")

    left.set_xticks(list(x))
    left.set_xticklabels(df["query"])
    left.set_xlabel("TPC-H query")
    left.set_ylabel("read volume MB")
    right.set_ylabel("throughput GB/s")
    left.set_title("GOLAP read volume and throughput")

    lines1, labels1 = left.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    left.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--out", default="tpch_runtime.png")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out = Path(args.out)

    # read query logs
    files = list(log_dir.glob("q*.txt")) + list(log_dir.glob("sf*/*/q*.txt"))
    rows = []

    for path in files:
        if not re.fullmatch(r"q\d+", path.stem):
            continue

        text = path.read_text(errors="replace")
        avg_ms = get_avg_ms(text)
        if avg_ms is None:
            continue

        flat = path.parent == log_dir
        rows.append(
            {
                "sf": "1" if flat else path.parts[-3].removeprefix("sf"),
                "mode": log_dir.parent.name if flat else path.parts[-2],
                "query": path.stem,
                "avg_ms": avg_ms,
                "read_mb": get_metric(text, "read_mb"),
                "uncompressed_read_mb": get_metric(text, "uncompressed_read_mb"),
                "effective_throughput_gbs": get_metric(text, "effective_throughput_gbs"),
                "io_throughput_gbs": get_metric(text, "io_throughput_gbs"),
            }
        )

    if not rows:
        raise SystemExit(f"No query timings found under {args.log_dir}")

    # save parsed data
    df = pd.DataFrame(rows)
    df["query_n"] = df["query"].str.extract(r"q(\d+)").astype(int)
    df = df.sort_values(["query_n", "mode"])
    df.to_csv(out.with_suffix(".csv"), index=False)

    # plot runtime
    order = df.drop_duplicates("query").sort_values("query_n")["query"]
    wide = df.pivot(index="query", columns="mode", values="avg_ms").reindex(order)
    ax = wide.plot(kind="bar", figsize=(9, 4.5))
    ax.set_xlabel("TPC-H query")
    ax.set_ylabel("Average runtime (ms)")
    ax.set_title("GOLAP TPC-H")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    # plot io details
    io_df = df.dropna(subset=["read_mb", "uncompressed_read_mb", "effective_throughput_gbs", "io_throughput_gbs"])
    if not io_df.empty:
        plot_io(io_df, out.with_name("tpch_io_summary.png"))


if __name__ == "__main__":
    main()
