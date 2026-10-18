"""Unit tests for the tick-pipe sizing leg (``--sizing_search per_pipe``,
the learned jump): ``TickSizingController`` (each pipe's start size, through
the resolver's decision layer, and its store) and ``TickJumpSession`` (the
in-pipe measure, walk, fit and jump).

CPU-only, no CUDA.  The controller is installed against a synthetic
arena-stats dict and driven through the real ``sizing_resolver.resolve`` and
``adaptive_driver.make_tick_sizing_session`` surfaces (the channels the
pipeline uses); a session is fed chunks the way the event loop feeds it,
the chunks already issued at the old size included, and torn down per test.
"""
import math

import pytest

import constants
import sizing_resolver
from adaptive import adaptive_driver, sizing_controller
from adaptive.sizing_controller import (
    SizingSearchConfig, install_per_pipe_sizing, uninstall_per_pipe_sizing,
    load_config, _budget_from_stats)

MB = 1024 ** 2
GB = 1024 ** 3


@pytest.fixture(autouse=True)
def _logger():
    from utility.logger import set_message_logger
    set_message_logger(None, False)


@pytest.fixture(autouse=True)
def _clean():
    sizing_resolver.reset()
    sizing_controller.reset_run_store()
    adaptive_driver.set_gather_sizing_enabled(True)
    constants.set_query_context(100, 2)
    orig = constants.get_chunk_size_b()
    constants.set_chunk_size_b(512 * MB)
    yield
    constants.set_chunk_size_b(orig)
    uninstall_per_pipe_sizing()
    sizing_controller.reset_run_store()
    sizing_resolver.reset()
    adaptive_driver.set_gather_sizing_enabled(True)


LIVE = 8 * GB
ROOMY = {"capacity_b": 29 * GB, "allocated_b": LIVE,
         "free_b": 21 * GB, "largest_free_run_b": 21 * GB}
C0 = 512 * MB
K = SizingSearchConfig().tick_samples


def _install(stats=None, cfg=None):
    stats = dict(ROOMY) if stats is None else stats
    cfg = cfg or SizingSearchConfig()
    ctl = install_per_pipe_sizing(cfg, arena_stats_fn=lambda: dict(stats),
                                  codec_epoch="BITCOMP/off")
    return ctl, stats, cfg


def _limit(stats, cfg):
    return cfg.grow_budget_frac * _budget_from_stats(stats, cfg)


def _session(cls="filter", inst="filter 3 filter pipe", *, n_ticks=400,
             input_b=400 * GB, sectionable=None, quantum_b=0):
    c = sizing_resolver.resolve(cls, inst).chunk_b
    return c, adaptive_driver.make_tick_sizing_session(
        f"{inst}", cls, inst, c, n_ticks, input_b, sectionable=sectionable,
        quantum_b=quantum_b)


def _drive(s, q_ms, transient, total_b, in_flight=2):
    """Feed ``s`` the chunks of a pipe of ``total_b`` input bytes, cut at
    the size it asks for (``in_flight`` chunks already issued at the old
    size after each re-cut), ``q_ms(c)`` the tick time and
    ``transient(c)`` the transient at size ``c``.  Returns the sizes the
    pipe ran, in order, one entry per run of equal sizes.  The session
    learns the pipe's input bytes from ``total_b`` (as the pipeline passes
    them when it opens the session; it reads them only from the first
    measured size on)."""
    s.input_b = total_b
    size, pending, issued, path = s.cur, [], 0, [s.cur]
    while issued < total_b:
        c = pending.pop(0) if pending else size
        b = min(c, total_b - issued)
        issued += b
        got = s.observe(c, b, q_ms(c), LIVE + transient(c), total_b - issued)
        if got is not None:
            pending = [c] * in_flight
            size = got
            if path[-1] != got:
                path.append(got)
    s.close()
    return path


def host_bound(F=20.0, t_per_mb=0.001):
    """A tick time dominated by a fixed cost (host bound)."""
    return lambda c: F + t_per_mb * c / MB


def proportional(t_per_mb=0.04):
    """A tick time proportional to the chunk (no fixed cost to spread)."""
    return lambda c: t_per_mb * c / MB


def linear_t(slope=3.0, icpt=256 * MB):
    return lambda c: int(icpt + slope * c)


# --------------------------------------------------------------------------- #
# Off-mode identity, class scoping, warmup
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", sizing_resolver.TICK_CLASSES)
def test_off_mode_identity_every_tick_class(cls):
    d = sizing_resolver.resolve(cls, f"{cls} pipe")
    assert d.chunk_b == constants.get_chunk_size_b()
    assert d.source == "constants"


def test_rearrange_not_decided_by_tick_controller():
    _install()
    d = sizing_resolver.resolve("rearrange", "join 4 rearrange")
    assert d.source == "constants"


@pytest.mark.parametrize("cls", sizing_resolver.TICK_CLASSES)
def test_first_sight_starts_at_the_off_mode_budget(cls):
    _install()
    constants.set_chunk_size_b(768 * MB)
    d = sizing_resolver.resolve(cls, f"{cls} pipe")
    assert d.chunk_b == 768 * MB and d.source == "controller"


def test_a_configured_c0_wins_over_the_chunk_flag():
    _install(cfg=SizingSearchConfig(tick_c0_b=256 * MB))
    assert sizing_resolver.resolve("sort", "s").chunk_b == 256 * MB


def test_warmup_decides_nothing_and_opens_no_session():
    _install()
    adaptive_driver.set_gather_sizing_enabled(False)
    assert sizing_resolver.resolve("filter", "f").source == "constants"
    assert adaptive_driver.make_tick_sizing_session(
        "f", "filter", "f", C0, 400, 400 * GB) is None
    assert not adaptive_driver.has_tick_sizing()


@pytest.mark.parametrize("cls,sectionable,opens", [
    ("filter", None, True), ("project", None, True), ("sort", None, True),
    ("agg", None, False), ("agg", True, True), ("core", None, False),
    ("filter", False, False)])
def test_only_a_re_cuttable_pipe_gets_a_session(cls, sectionable, opens):
    _install()
    _, s = _session(cls, f"{cls} 1", sectionable=sectionable)
    assert (s is not None) == opens


def test_a_pipe_too_short_to_measure_gets_no_session():
    _install()
    assert _session(n_ticks=K + 2)[1] is None
    assert _session(inst="other", n_ticks=K + 3)[1] is not None


# --------------------------------------------------------------------------- #
# The measure: skips, attribution, the walk, the jump
# --------------------------------------------------------------------------- #

def test_the_first_chunk_at_a_size_and_other_sizes_are_not_measured():
    _install()
    _, s = _session()
    assert s.observe(C0, C0, 9.0, LIVE + C0, 100 * GB) is None   # skipped
    for _ in range(K - 1):
        assert s.observe(C0, C0, 1.0, LIVE + C0, 100 * GB) is None
        assert s.observe(2 * C0, C0, 99.0, LIVE + C0, 100 * GB) is None
    got = s.observe(C0, C0, 1.0, LIVE + C0, 100 * GB)
    assert got == 2 * C0                                         # the walk
    assert s.points[C0][0] == 1.0


def test_a_host_bound_pipe_walks_then_jumps_to_its_memory_bound():
    """The walk (x2) wins; the fit's optimum is far past the memory bound,
    so the pipe jumps to the size its transient per byte allows on the
    room, and fixes there (nothing is left to step to)."""
    ctl, stats, cfg = _install()
    _, s = _session()
    path = _drive(s, host_bound(), linear_t(slope=6.0, icpt=0), 400 * GB)
    bound = int(_limit(stats, cfg) / 6.0)
    assert bound < cfg.tick_c_max_b
    assert path[:2] == [C0, 2 * C0]
    assert path[2] == bound            # learned: rho = 6 over the room
    assert s.fixed and s.inc == bound
    assert ctl.accepts == 2 and ctl.jumps == 2


def test_a_light_pipe_jumps_to_the_size_cap():
    _, _, cfg = _install()
    _, s = _session()
    path = _drive(s, host_bound(), linear_t(slope=2.0, icpt=0), 400 * GB)
    assert path == [C0, 2 * C0, cfg.tick_c_max_b]
    assert "c_max cap" in " ".join(s.path)


def test_the_jump_is_the_fitted_optimum_inside_the_bounds():
    """q = F + t·c whose whole-pipe optimum sqrt(D·F/t) lies between the
    walk and both caps: the pipe jumps there (F/t = 200 MB over D = 32 GB:
    2.5 GB)."""
    _, stats, cfg = _install()
    _, s = _session()
    t_per_mb, total = 0.05, 32 * GB
    F = 200 * t_per_mb
    path = _drive(s, host_bound(F, t_per_mb), linear_t(1.0, 0), total)
    assert path[:2] == [C0, 2 * C0]
    want = math.sqrt(total * F / (t_per_mb / MB))
    assert abs(path[2] - want) <= 0.02 * want
    assert 2 * C0 < path[2] < cfg.tick_c_max_b
    assert "predicted" in " ".join(s.path)


def test_a_step_measured_at_the_pipe_end_is_judged_on_the_whole_pipe():
    """A walk whose measurement ends the pipe (no bytes left to use it) is
    kept when it is faster per byte with the drain counted over the pipe's
    input: the verdict is for the next occurrence, which runs the pipe
    whole."""
    _install()
    _, s = _session()
    total = (K + 1) * C0 + 2 * C0 + (K + 2) * 2 * C0
    path = _drive(s, host_bound(), linear_t(3.0, 0), total)
    assert path[:2] == [C0, 2 * C0]
    assert s.inc == 2 * C0 and "won" in " ".join(s.path)


def test_a_pipe_with_no_fixed_cost_loses_the_walk_and_stays():
    ctl, _, _ = _install()
    _, s = _session()
    path = _drive(s, proportional(), linear_t(), 100 * GB)
    assert path == [C0, 2 * C0, C0]
    assert s.fixed and s.inc == C0 and ctl.reverts == 1


def test_a_walk_inside_epsilon_is_reverted():
    _install()
    _, s = _session()
    # The walk saves under 1% of the time left.
    q = lambda c: 10.0 * (c / C0) * 0.995 if c > C0 else 10.0
    path = _drive(s, q, linear_t(), 100 * GB)
    assert path[-1] == C0 and s.fixed


def test_a_jump_whose_transient_breaches_the_room_is_reverted():
    ctl, stats, cfg = _install()
    _, s = _session()
    limit = _limit(stats, cfg)
    # Linear up to the walk, then a cliff: the jump's transient is over.
    tr = lambda c: 3 * c if c <= 2 * C0 else int(limit) + GB
    path = _drive(s, host_bound(), tr, 400 * GB)
    assert path[-1] == 2 * C0 and s.fixed and ctl.breaches == 1


def test_memory_per_byte_that_leaves_no_room_fixes_at_the_start():
    _, stats, cfg = _install()
    _, s = _session()
    # rho so large that the bound is under the start size.
    rho = _limit(stats, cfg) / (C0 * 0.9)
    path = _drive(s, host_bound(), lambda c: int(rho * c), 100 * GB)
    assert path == [C0] and s.fixed
    assert "no step" in s.path[-1]


def test_a_pipe_too_short_for_the_walk_stays_at_its_start():
    """The walk must leave k + 2 chunks of the new size to measure and
    use: a short pipe never re-cuts."""
    _install()
    _, s = _session(n_ticks=K + 4)
    path = _drive(s, host_bound(), linear_t(), (K + 4) * C0)
    assert path == [C0]


def test_whole_quanta_round_the_jump():
    _install()
    q = 96 * MB
    _, s = _session(quantum_b=q)
    _drive(s, host_bound(), linear_t(3.0, 0), 400 * GB)
    assert len(s.points) >= 3
    assert all(c % q == 0 for c in s.points if c != C0)


def test_a_refused_re_cut_fixes_the_session():
    _install()
    _, s = _session()
    for _ in range(K + 1):
        got = s.observe(C0, C0, 20.0, LIVE + C0, 100 * GB)
    assert got == 2 * C0
    s.recut_refused()
    assert s.fixed
    assert s.observe(2 * C0, C0, 1.0, LIVE + C0, 100 * GB) is None


# --------------------------------------------------------------------------- #
# The store: repeated occurrences
# --------------------------------------------------------------------------- #

def test_a_repeated_occurrence_starts_where_the_last_one_fixed():
    _install()
    _, s = _session()
    _drive(s, host_bound(), linear_t(3.0, 0), 400 * GB)
    settled = s.inc
    c, again = _session()
    assert c == settled and again is None        # fixed: no new session


def test_a_pipe_that_ends_while_measuring_its_start_searches_again():
    _install()
    _, s = _session()
    for _ in range(K - 2):
        s.observe(C0, C0, 20.0, LIVE + C0, 100 * GB)
    s.close()
    c, again = _session()
    assert c == C0 and again is not None and not again.points


def test_a_won_walk_with_no_room_left_to_jump_is_kept_and_fixed():
    """The walk wins but too few bytes are left to measure a jump: the
    pipe keeps the walk, fixed, and its next occurrence starts there."""
    _install()
    _, s = _session()
    _drive(s, host_bound(), linear_t(3.0, 0),
           (K + 3) * C0 + (K + 1) * 2 * C0 + 4 * C0)
    assert s.fixed and s.inc == 2 * C0
    c, again = _session()
    assert c == 2 * C0 and again is None


def test_the_start_size_is_lowered_when_the_room_shrank():
    ctl, stats, cfg = _install()
    _, s = _session()
    _drive(s, host_bound(), linear_t(3.0, 0), 400 * GB)
    big = s.inc
    stats["allocated_b"] = 20 * GB               # much less room now
    c, _ = _session()
    fit = int(_limit(stats, cfg) / 3.0)
    assert c < big and c == max(C0, min(fit, big)) and ctl.clamps == 1


def test_the_query_report_lists_each_session():
    ctl, _, _ = _install()
    _, s = _session()
    _drive(s, host_bound(), linear_t(3.0, 0), 400 * GB)
    report = ctl.take_report()
    assert len(report) == 1 and report[0].startswith("filter:filter 3")
    assert "walk" in report[0] and "fixed at" in report[0]
    assert ctl.take_report() == []


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_load_config_tick_keys_and_retired_keys(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"tick": {"c0_mb": 256, "c_max_mb": 2048, "samples": 4,'
                 ' "jump_mb": {"filter": 2048}, "sections": true,'
                 ' "seed_slope": 3.0},'
                 ' "march": {"k_seg": 8, "k_sense": 2, "delta_mb": 128}}')
    cfg = load_config(str(p))
    assert cfg.tick_c0_b == 256 * MB and cfg.tick_c_max_b == 2048 * MB
    assert cfg.tick_samples == 4 and cfg.march_delta_b == 128 * MB


def test_default_config_starts_at_the_chunk_flag():
    cfg = load_config(None)
    assert cfg.tick_c0_b is None and cfg.tick_samples >= 2


def test_the_cache_region_s_lendable_bytes_do_not_widen_the_bound():
    """The memory bound plans on the operator space's own headroom: what
    the cache region could lend is the fallback, not room to grow into."""
    lending = dict(ROOMY, lendable_b=20 * GB)
    ctl, stats, cfg = _install(stats=lending)
    _, s = _session()
    _drive(s, host_bound(), linear_t(slope=6.0, icpt=0), 400 * GB)
    assert s.room_b == _budget_from_stats(dict(ROOMY), cfg)
    assert s.inc == int(_limit(dict(ROOMY), cfg) / 6.0)
