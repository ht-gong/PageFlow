#!/usr/bin/env python
"""sirius column probes (see probe_common.py). One Sirius duckdb process runs
every probe of the list from one SQL script: views as the runner's
(sirius/run_tpch_sirius.py) and a warm-up query, then per column a marker and
the probe query, timed by the CLI's `.timer on`. DuckDB CPU fallback is
disabled for the session, so a probe Sirius cannot run on the GPU fails instead
of silently running on the CPU; such errors are recorded as FALLBACK. `.bail
off` keeps the process going after a failed probe, as the other engines'
workers do.

Env: SIRIUS_DUCKDB, SIRIUS_ENVLIB, SIRIUS_CONFIG_FILE (as the runner).
"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402

MARK = "__PROBE__"
RUN_TIME_RE = re.compile(r"Run Time \(s\): real ([0-9]+\.[0-9]+)")
ERROR_RE = re.compile(r"(Error|Out of Memory|std::bad_alloc|CUDA error|RMM error)")
FALLBACK_RE = re.compile(r"fallback|not supported|unsupported", re.I)


def script(parquet, cols):
    lines = [".bail off"]
    for t in pc.TABLES:
        files = sorted(glob.glob(f"{parquet}/{t}/*.parquet"))
        lst = ", ".join(f"'{f}'" for f in files)
        lines.append(f"CREATE VIEW {t} AS SELECT * FROM read_parquet([{lst}]);")
    lines += ["SELECT n_regionkey, count(*) FROM nation GROUP BY n_regionkey;",
              "SET enable_duckdb_fallback = false;",
              ".mode csv", ".headers off", ".timer on"]
    prime = [] if os.environ.get("PROBE_NO_PRIME") == "1" else pc.PRIME
    for t, c in prime + cols:
        lines += [f".print {MARK} {t} {c}", pc.probe_sql(t, c) + ";"]
    return "\n".join(lines) + "\n"


class Lines:
    """duckdb's merged stdout/stderr, one line of lookahead."""
    def __init__(self, stream):
        self.stream, self.back = stream, None

    def next(self):
        if self.back is not None:
            line, self.back = self.back, None
            return line
        line = self.stream.readline()
        return None if line == "" else line.rstrip("\n")


def main():
    parquet, cols, out_csv = pc.args()
    sql = os.path.splitext(out_csv)[0] + ".sql"
    with open(sql, "w") as f:
        f.write(script(parquet, cols))
    # Line-buffered so markers reach us (and the driver's timeout) as they run.
    env = dict(os.environ, LD_LIBRARY_PATH=os.environ["SIRIUS_ENVLIB"])
    proc = subprocess.Popen(["stdbuf", "-oL", "-eL", os.environ["SIRIUS_DUCKDB"], "-f", sql],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    out = Lines(proc.stdout)

    def probe(table, column):
        while True:                       # up to this probe's marker
            line = out.next()
            if line is None:
                raise pc.Stop(f"duckdb exited (rc={proc.wait()})")
            if line.startswith(MARK):
                if tuple(line.split()[1:3]) != (table, column):
                    raise RuntimeError(f"probe order mismatch at {line!r}")
                break
            if ERROR_RE.search(line):
                print(line, flush=True)   # setup / warm-up error
        err = None
        while True:                       # up to its timer line, or the next marker
            line = out.next()
            if line is None or line.startswith(MARK):
                out.back = line
                if line is None:
                    err = err or f"duckdb exited during this probe (rc={proc.wait()})"
                return ("FALLBACK" if err and FALLBACK_RE.search(err) else "FAIL"), None, (err or "no timing line")[:160]
            m = RUN_TIME_RE.search(line)
            if m:
                if err:
                    return ("FALLBACK" if FALLBACK_RE.search(err) else "FAIL"), float(m.group(1)), err[:160]
                return "OK", float(m.group(1)), ""
            if err is None and ERROR_RE.search(line):
                err = line.strip().replace(",", ";")

    pc.run(cols, out_csv, probe)
    proc.wait()


if __name__ == "__main__":
    main()
