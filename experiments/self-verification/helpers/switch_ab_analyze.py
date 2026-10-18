"""Summarise a ``switch_ab.py`` run (E2).  Standard library only.

Only complete cycles count: a cycle of a qid in which every arm has a time.
Per query and arm: the median over cycles.  Per arm against the first arm
(the base): the paired change of each cycle, and for the suite and each
group the sum of the per-query medians, with a 95% interval from resampling
each query's cycles (paired cycle bootstrap, docs/sf100-bottleneck-study.md
§6).  Failed occurrences (no ``query_times.csv`` row) are listed and left
out.
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from switch_ab import CMDLINE_NAME, OUT_NAME  # noqa: E402


def load(d):
    with open(os.path.join(d, CMDLINE_NAME), encoding='utf-8') as f:
        cmd = json.load(f)
    with open(os.path.join(d, OUT_NAME), encoding='utf-8') as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [a['name'] for a in cmd['driver']['arms']], recs


def cycles(recs, arms, exclude=()):
    """``({qid: [{arm: time} per complete cycle]}, failed occurrences)``."""
    by = defaultdict(dict)
    failed = []
    for r in recs:
        if r['warm'] or r['qid'] in exclude:
            continue
        if r['approx_time'] is None:
            failed.append((r['qid'], r['occ'], r['arm']))
            continue
        by[(r['qid'], r['cycle'])][r['arm']] = r['approx_time']
    out = defaultdict(list)
    for (q, c), t in sorted(by.items()):
        if all(a in t for a in arms):
            out[q].append(t)
    return dict(out), failed


def summarise(cyc, arms, groups, n_boot=2000, seed=7):
    base = arms[0]
    med = {q: {a: st.median(c[a] for c in cs) for a in arms} for q, cs in cyc.items()}
    sets = {'suite': sorted(cyc)}
    for name, qs in groups.items():
        members = [q for q in qs if q in cyc]
        if members:
            sets[name] = members
    rnd = random.Random(seed)
    boot = {s: {a: [] for a in arms[1:]} for s in sets}
    for _ in range(n_boot):
        draw = {q: [rnd.choice(cs) for _ in cs] for q, cs in cyc.items()}
        tq = {q: {a: st.median(c[a] for c in d) for a in arms} for q, d in draw.items()}
        for s, qs in sets.items():
            tb = sum(tq[q][base] for q in qs)
            for a in arms[1:]:
                boot[s][a].append(sum(tq[q][a] for q in qs) / tb - 1 if tb else 0.0)
    res = {'arms': arms, 'queries': {}, 'sets': {}}
    for q, cs in sorted(cyc.items()):
        res['queries'][q] = {
            'cycles': len(cs), 'median': med[q],
            'cycle_change': {a: [c[a] / c[base] - 1 for c in cs] for a in arms[1:]}}
    for s, qs in sets.items():
        tot = {a: sum(med[q][a] for q in qs) for a in arms}
        entry = {'queries': qs, 'sum_of_medians': tot, 'change': {}, 'interval95': {}}
        for a in arms[1:]:
            b = sorted(boot[s][a])
            entry['change'][a] = tot[a] / tot[base] - 1 if tot[base] else 0.0
            entry['interval95'][a] = ([b[int(0.025 * len(b))], b[min(len(b) - 1, int(0.975 * len(b)))]]
                                      if b else [0.0, 0.0])
        res['sets'][s] = entry
    return res


def print_summary(title, res):
    arms, base = res['arms'], res['arms'][0]
    cyc = [e['cycles'] for e in res['queries'].values()]
    print(f'{title}: arms {arms} (base {base}); {len(cyc)} queries; '
          f'cycles per query {min(cyc)}-{max(cyc)}; failed {res["failed"] or "none"}')
    print(f"\n{'':10s} " + ''.join(f'{a:>12s}' for a in arms)
          + ''.join(f'{a + " vs " + base:>22s}' for a in arms[1:]))
    for s, e in res['sets'].items():
        print(f'{s:10s} ' + ''.join(f"{e['sum_of_medians'][a]:12.3f}" for a in arms)
              + ''.join(f"{100 * e['change'][a]:+8.2f}% [{100 * e['interval95'][a][0]:+.1f},"
                        f"{100 * e['interval95'][a][1]:+.1f}]" for a in arms[1:]))
    print(f"\n{'Q':>4s} {'cyc':>3s} " + ''.join(f'{a:>10s}' for a in arms) + '   change per cycle (%)')
    for q, e in res['queries'].items():
        print(f'Q{q:<3d} {e["cycles"]:3d} ' + ''.join(f"{e['median'][a]:10.3f}" for a in arms)
              + '   ' + '  '.join(a + ' ' + ','.join(f'{100 * x:+.1f}' for x in e['cycle_change'][a])
                                  for a in arms[1:]))


def analyse(d, groups, exclude=(), n_boot=2000):
    """Summary of the switch_ab.py output directory ``d`` (raises
    SystemExit when no cycle is complete)."""
    arms, recs = load(d)
    cyc, failed = cycles(recs, arms, set(exclude))
    if not cyc:
        sys.exit(f'{d}: no complete cycle')
    res = summarise(cyc, arms, groups, n_boot)
    res['failed'] = failed
    return res
