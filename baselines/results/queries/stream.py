"""The qgen query stream, with q11's FRACTION for a scale factor.

stream_qualification.sql is qgen's stream with its default substitutions. The
one parameter that depends on the scale factor, q11's FRACTION, is the
placeholder :FRACTION there; queries(sf) fills it in. The rapids, duckdb_cpu
and sirius runners read their queries through it, and polars_gpu's native q11
takes fraction(sf).
"""
import os
import re

STREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_qualification.sql")
PLACEHOLDER = ":FRACTION"


def fraction(sf):
    """q11's FRACTION at scale factor sf as qgen prints it: 0.0001/SF to 10
    decimals (varsub.c, "%11.10f"). SF500 gives 0.0000002000, SF700
    0.0000001429."""
    return "%.10f" % (0.0001 / float(sf))


def stream_text(sf, path=STREAM):
    """The stream with q11's FRACTION for scale factor sf."""
    with open(path) as f:
        text = f.read()
    if text.count(PLACEHOLDER) != 1:
        raise ValueError(f"{path}: expected one {PLACEHOLDER}, found {text.count(PLACEHOLDER)}")
    return text.replace(PLACEHOLDER, fraction(sf))


def queries(sf, path=STREAM):
    """{query number: [statements]} at scale factor sf. Each template's body is
    split on ';' and only fragments with real SQL are kept, so q15 has three
    statements (create view, select, drop view)."""
    pat = re.compile(r"-- Template file: (\d+)\n\n(.*?)(?=(?:-- Template file: \d+)|\Z)", re.S)
    return {int(n): [s.strip() for s in body.split(";")
                     if re.search(r"\b(select|create|drop|with)\b", s, re.I)]
            for n, body in pat.findall(stream_text(sf, path))}
