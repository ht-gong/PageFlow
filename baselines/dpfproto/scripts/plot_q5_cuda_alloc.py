#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ALLOC_RE = re.compile(
    r"\[cuda_alloc(?P<failed>_failed)?\]\s+"
    r"request_mb=(?P<request>[0-9.]+)\s+"
    r"free_mb=(?P<free>[0-9.]+)\s+"
    r"total_mb=(?P<total>[0-9.]+)"
)


def q5_path(path):
    path = Path(path)
    return path / "q5.txt" if path.is_dir() else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out")
    args = parser.parse_args()

    log = q5_path(args.path)
    out = Path(args.out) if args.out else log.with_name("q5_cuda_alloc.png")

    rows = []
    for m in ALLOC_RE.finditer(log.read_text(errors="replace")):
        rows.append(
            {
                "alloc": len(rows) + 1,
                "request_mb": float(m.group("request")),
                "free_mb": float(m.group("free")),
                "total_mb": float(m.group("total")),
                "failed": bool(m.group("failed")),
            }
        )

    if not rows:
        raise SystemExit(f"No cuda allocation logs found in {log}")

    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False)

    ok = df[~df["failed"]]
    failed = df[df["failed"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["alloc"], df["free_mb"], marker="o", label="free GPU memory")
    ax.scatter(ok["alloc"], ok["request_mb"], marker="x", label="request")
    if not failed.empty:
        ax.scatter(failed["alloc"], failed["free_mb"], color="red", zorder=4, label="OOM")

    ax.set_xlabel("allocation number")
    ax.set_ylabel("MB")
    ax.set_title("Q5 SF100 GPU allocation pressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)

    print(out)
    print(out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
