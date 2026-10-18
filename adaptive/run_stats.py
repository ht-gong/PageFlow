"""Per-query observability records for the perf report.

Two record families, both process-global and reset by ``main.py`` at the top
of every query iteration (exactly like the cache/insert counters), then
folded into the per-query perf JSON via ``perf_logger().add_section``:

* **Sizing-model samples** — one record per predicted-vs-observed transient
  comparison the sizing plane makes (gather-window sizing: every gather
  iteration; tick-pipe sizing: every trained pipe sample).  This is the raw
  material for the paper's "memory-model accuracy" evaluation; the engine
  previously logged only breaches, discarding the (predicted, observed)
  pair on every healthy iteration.

* **Compression-search registry** — the live (pipe, driver, controller,
  menu) tuples the ``--compression_search`` observer factories create.
  ``snapshot_compression_choices()`` projects them to a JSON-safe summary
  of what the search actually chose (k*, k_w*, gate reasons, and the final
  per-column codec vectors).  Nothing else in the engine retains the
  drivers; the closures are otherwise unreachable after construction.

Recorders never raise: observability must not fail a query.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# ── sizing-model samples ────────────────────────────────────────────────

# Cap protects the perf JSON against pathological streams; drops are counted
# loudly rather than silently truncated.
_SIZING_SAMPLE_CAP = 20000

_sizing = {
    'samples': [],          # list[dict], see record_sizing_sample
    'dropped': 0,
    # Tick-pipe samples that never reach the predicted-vs-observed compare
    # (coverage accounting — the compare is skipped for these by design).
    'skipped_mixed': 0,     # write-flipped pipes
    'skipped_partial': 0,   # input under tick_full_frac x chunk_b
    'skipped_no_pending': 0,  # no decision-time prediction to compare against
    'skipped_sectioned': 0,   # a section session trained the pipe tick-by-tick
}


def record_sizing_sample(plane: str, pipe: str, c_b: int, predicted_b: int,
                         observed_b: int, seeded: bool, breach: bool,
                         budget_b: Optional[int] = None) -> None:
    """One predicted-vs-observed transient comparison.

    ``plane`` is ``"SZ-A"`` (a tick-pipe sizing sample) or ``"SZ-B"`` (a
    gather-window sample); ``pipe`` identifies the stream (gather-window
    instance) or ``pipe_class:instance`` (tick pipe).  ``predicted_b`` is
    the envelope prediction that gated this size, evaluated BEFORE the
    observation entered the fit; ``budget_b`` (tick pipes only) is the
    decision-time safety budget the prediction was held against.
    """
    if len(_sizing['samples']) >= _SIZING_SAMPLE_CAP:
        _sizing['dropped'] += 1
        return
    _sizing['samples'].append({
        'plane': plane,
        'pipe': str(pipe),
        'c_b': int(c_b),
        'predicted_b': int(predicted_b),
        'observed_b': int(observed_b),
        'seeded': bool(seeded),
        'breach': bool(breach),
        'budget_b': int(budget_b) if budget_b is not None else None,
    })


def record_sizing_skip(reason: str) -> None:
    key = f'skipped_{reason}'
    if key in _sizing:
        _sizing[key] += 1


def get_sizing_stats() -> dict:
    return {
        'samples': list(_sizing['samples']),
        'dropped': _sizing['dropped'],
        'skipped_mixed': _sizing['skipped_mixed'],
        'skipped_partial': _sizing['skipped_partial'],
        'skipped_no_pending': _sizing['skipped_no_pending'],
        'skipped_sectioned': _sizing['skipped_sectioned'],
    }


def reset_sizing_stats() -> None:
    _sizing['samples'] = []
    for k in _sizing:
        if k != 'samples':
            _sizing[k] = 0


# ── compression-search registry ─────────────────────────────────────────

# (pipe_name, drv, ctrl, menu) appended by the observer factories in
# main.py.  Entries registered during warmup are flushed by the reset at
# the top of the first real query iteration.
_search_registry: List[Tuple] = []


def register_compression_search(pipe_name: str, drv, ctrl, menu) -> None:
    _search_registry.append((str(pipe_name), drv, ctrl, menu))


def reset_compression_search() -> None:
    _search_registry.clear()


def _read_leg_summary(read, ctrl, n_read: int) -> dict:
    final_vec = None
    if read.emitted:
        final_vec = list(read.emitted[-1][2])
    elif read.k_star is not None:
        final_vec = list(ctrl.pick_vector(read.k_star))
    return {
        'k_star': read.k_star,
        'n_decision_cols': n_read,
        'converged': bool(read.converged),
        'pregate_skipped': bool(getattr(read, 'pregate_skipped', False)),
        'probes_spent': int(getattr(read, 'probes_spent', 0)),
        'final_input_algs': final_vec,
    }


def _write_leg_summary(write, ctrl, n_write: int) -> dict:
    final_vec = None
    if write is not None and write.k_star is not None:
        try:
            final_vec = list(ctrl.write_pick_vector(
                write.k_star, getattr(write, '_ranked', None)))
        except Exception:
            final_vec = None
    if write is None:
        return {'k_star': None, 'n_decision_cols': n_write,
                'converged': False, 'searched': False}
    return {
        'k_star': write.k_star,
        'n_decision_cols': n_write,
        'converged': bool(write.converged),
        'searched': True,
        'mode': getattr(write, 'mode', None),
        'write_policy': getattr(write, 'write_policy', None),
        'pregate_skipped': bool(getattr(write, 'pregate_skipped', False)),
        'benefit_gated': bool(getattr(write, 'benefit_gated', False)),
        'encode_gated': bool(getattr(write, 'encode_gated', False)),
        'readback_gated': bool(getattr(write, 'readback_gated', False)),
        'readback_price_ms': {str(i): round(v, 4) for i, v in
                              getattr(write, 'price', {}).items()},
        'probes_spent': int(getattr(write, 'probes_spent', 0)),
        'windows': [list(w) for w in getattr(write, 'windows', [])],
        'final_output_algs': final_vec,
    }


def snapshot_compression_choices() -> dict:
    """JSON-safe projection of every registered pipe's final search state.

    Static (non-searched) pipes are absent: their codecs are the applier's
    policy resolution, constant per run, recorded by ``cmdline.json`` and
    the per-pipe ``ratios`` in the same perf JSON.
    """
    pipes = []
    for pipe_name, drv, ctrl, menu in _search_registry:
        try:
            entry = {
                'pipe': pipe_name,
                'total_ticks': int(menu.total_ticks),
                # Static defaults the search moves away from: each input /
                # output column's full legal range from the published menu.
                'menu_inputs': [list(c.choices) for c in menu.inputs],
                'menu_outputs': [list(c.choices) for c in menu.outputs],
            }
            if hasattr(drv, 'read'):          # ReadWriteSearchDriver
                entry['read'] = _read_leg_summary(
                    drv.read, ctrl, ctrl.decision_count())
                entry['write'] = _write_leg_summary(
                    drv.write, ctrl, ctrl.write_decision_count())
            else:                             # CompressionSearchDriver
                entry['read'] = _read_leg_summary(
                    drv, ctrl, ctrl.decision_count())
                entry['write'] = None
            pipes.append(entry)
        except Exception as e:                # observability must not fail
            pipes.append({'pipe': pipe_name,
                          'error': f'{type(e).__name__}: {e}'})
    return {'pipes': pipes}
