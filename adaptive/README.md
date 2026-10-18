# `adaptive/` — runtime feedback planes for TQP-Vortex

This directory holds the engine's runtime decision logic: code that watches a
running pipeline's measured per-tick timings and arena usage and retunes the
pipeline while the query runs. It has two independent **planes**:

- the **compression plane** (`--compression_search`) decides, per pipe, which
  columns actually move compressed and which move RAW;
- the **sizing plane** (`--sizing_search`) decides each pipe's chunk budget,
  counting what the granule cache's region could lend the operators (the
  arena lends it on a miss, `memory/README.md`, "Lending").

It also has one module of static execution settings, `instage_optimization.py`,
which every run uses: the global compression mode, the zero-copy mask
threshold and the rep selector.

Three rules hold across the package:

1. **Off means absent.** With both flags `off` (the default) no factory,
   observer or decision function is installed. The pipeline checks a cheap
   predicate (`has_tick_observer_factory()`, `has_tick_sizing()`, …)
   and runs byte-identical to a build without this package.
2. **Adaptive code never touches bytes.** Nothing here owns memory or launches
   kernels. Each plane returns *decisions* through one channel: an observer
   return value (compression), a `sizing_resolver` override or decision or a
   re-cut request (sizing). The pipeline
   (`IO/pipeline/`) does all sequencing and actuation.
3. **No torch at import time.** No module here imports torch, and
   `instage_optimization` imports the zero-copy stages lazily. The planes' logic therefore unit-tests CPU-only.

Consumers: `IO/pipeline/vortex_pipeline.py` (menu publication, the per-tick
observer call, pick actuation, sizing samples and re-cuts),
`IO/pipeline/rearrange_pipeline.py` (gather-window sessions),
`IO/transfer/data_exchange.py` (`TransferTimes`), `IO/pipeline/common.py`
and `IO/data_scanner.py` (the static settings in
`instage_optimization`), and `main.py`, which installs everything from the CLI.
`sizing_resolver.py` at the repo root is the sizing plane's actuation channel,
and the operators call it for their budgets. Design notes:
`docs/adaptive_compression_switching.md` and
`docs/adaptive_intermediate_sizing.md`.

## The compression plane

`--compression_search read-only` searches the read leg (which input columns
arrive compressed); `read-and-write` also searches the write leg (which output
columns are written compressed). `fixed` runs no search:
`instage_optimization.set_fixed_search_json` records the path and the dataflow
applier applies its pins. Each pipe goes through these steps:

1. **Menu.** At construction, when an observer factory is installed and the
   pipe has at least `tick_observer_min_ticks()` ticks, the pipeline publishes
   a `decision_menu.DecisionMenu`. It holds one `ColumnChoices` per input
   column and per output slot, plus the pipe's tick count. A range is
   `("RAW",)`, `(codec,)` or `("RAW", codec)`, and only a two-entry range is a
   decision. An input range lists the reps the column actually holds: a scan
   column under `--compression adaptive` holds RAW plus one compressed rep,
   while an operator output holds one. An output range comes from the codec the
   dataflow applier resolved. No tags or column names cross the boundary. Picks
   are positional, and the menu is publish-only.
2. **Factory.** `adaptive_driver.make_tick_observer_for(pipe_name, menu)` calls
   the factory that `main.py` installed (`_read_search_factory` or
   `_rw_search_factory`). The factory wraps the menu in a
   `CompressionController`, builds a driver, and returns `None` when the driver
   is inert. An inert pipe pays nothing per tick. A driver is inert when the
   pipe has no decision columns, too few ticks to finish a search, or when its
   worst-case probe samples would exceed 25% of the ticks. The read leg needs a
   pipe of at least 16 ticks and the write leg at least 14, so at the default
   256 MB chunk few SF1 pipes search at all.
3. **Per-tick view.** After every tick the pipeline passes its
   `ResourceMonitor` to the observer. The view is only valid during the call.
   It holds `last_tick`, `last_transfer` (h2d, d2d and d2h ms), `last_compute`
   (`None` on the transfer-only prologue tick) and the objective `f_last`: the
   slowest transfer lane or the full compute bracket, whichever is larger.
   Every lane is a span with host gaps: a transfer lane runs from the first
   copy batch's start to the last one's end on its stream
   (`IO/telemetry/copy_events.py`), 0 when the stream copied nothing, and
   `compute` (from the tick's input decompress to the end of its output
   step) and, inside it, `compress_out` (the output step) are spans on the
   compute stream.
4. **Search.** `CompressionSearchDriver` (the read leg) ranks the decision
   columns by the bytes compression saves (`raw_bytes − stored_bytes`). It then
   ternary-searches *k*, the number of top-ranked columns to compress, because
   `f(k)` is unimodal. Before probing it watches the incumbent (the static
   default, *k = N*). If transfer already outweighs compute there, *k = N* is
   proven without a single probe. `ReadWriteSearchDriver` runs the read leg
   first, then a `WriteWindowSearchDriver` over the ticks that remain.
5. **Actuation (pipeline side).** A non-`None` return is a full pick vector.
   `validate_pick_vector` checks it against the menu. Input picks take effect
   at tick *t+2* through the pipeline's just-in-time in-stage rebuild. Output
   picks, carried in a `PickPair`, take effect at *t+1*: each tick writes its
   own region per column, so a flip starts with the next region. The drivers only *assume* these lags when they attribute a sample
   to the *k* that built its tick.
6. **Report.** Each factory registers its live drivers with
   `run_stats.register_compression_search`. At query end,
   `snapshot_compression_choices()` becomes the perf JSON's `compression`
   section.

### The write leg

`WriteWindowSearchDriver` searches *k_w*, a prefix of the output columns
ranked once, the dearest to write RAW first (by readback price, below; by
measured output bytes without one). Each evaluation is a probe window.
The driver emits the new output vector, skips the settling ticks, collects
`window_ticks` samples, then emits a null seal so the window's own bytes drain.
The window's score is the largest per-lane mean over the window and drain
ticks, plus the readback price of the columns the window writes RAW.

**The readback price.** A RAW output's extra bytes cross PCIe again after
the producing pipe: home, and once per consumer read. None of that lands in
the producer's lanes, and with the granule cache not even its own d2h lane
(an admitted output goes home only on eviction), so a producer-local `f`
sees only the encode leaving its compute lane. On SF300 Q21 three join sort
pipes flipped to RAW took the query from 6.4 to 9.0 s, spilling and
restoring 47 GB each way. Each write decision column is priced in ms per
tick, `raw/tick × (1 − 1/ratio) × (reads + 1) ÷ PCIE_B_PER_MS`, from the
pipe's byte counters on the incumbent (a compressed output's stored bytes
are counted at its pack, a tick after its raw bytes, so the ratio pairs a
view's stored bytes with the previous view's raw ones), its consumer reads
on the menu (the dataflow's expected reads) and the host-to-device rate
`main.py` probes at startup (`data_exchange.h2d_b_per_ms`).

Three pre-gates run on the incumbent first:

- **P1** holds the incumbent (*k_w = n*) when transfer already outweighs
  compute.
- **P1b** (the encode gate) holds it when compute minus `compress_out`, the
  tick's output-compress span, still exceeds the slowest transfer lane. There,
  dropping compression cannot move the bottleneck. On the SF700 suite, a RAW
  output's larger pages ate the encode saving: the compute lane got 0.127 s
  *slower* over 35 flipped pipes. The gate reads that default span, host gaps
  included, not a codec-kernel timer of its own.
- **P3** (the readback gate) holds it when `compress_out` is no larger than
  the cheapest single column's readback price: writing any subset RAW saves
  at most that span and costs at least that price.

Pipes too short for a ternary use a greedy fallback. The policy for that
fallback comes from the environment. The default, `measured`, probes *k_w = 0*
for one window and flips to RAW only if that window's `f` plus the price of
writing every column RAW is below the incumbent's. `TQP_WRITE_PREDICTIVE`
selects benefit floors instead (net of that price), and `TQP_WRITE_UNGATED`
flips blindly with no floors and no price. `TQP_WRITE_NO_ENCODE_GATE` turns
off P1b.

A driver's `settled()` is true once it emits no more picks (inert, or both
legs converged); `main.py`'s observers carry it, and from that tick on the
pipe's remaining ticks leave the tick barrier for event-ordered ticks
(`IO/pipeline/README.md`, "The hand-over").

## The sizing plane

Every sized site gets its chunk budget from
`sizing_resolver.resolve(pipe_class, instance)`. The answer is the first that
exists of: a controller override, a static value from `--sizing_config`, or
`--chunk_size_mb`. The controllers in `sizing_controller.py` are the only code
that writes overrides or decisions. The plane has two legs:

- **SZ-A: tick-loop pipes** (`sort`, `core`, `agg`, `filter`, `project`).
  Both `per_pipe` and `mid_stream` arm it. `install_per_pipe_sizing`
  installs a `TickSizingController` as the resolver's decision function and
  the pipeline's tick sizing session factory.
  - At construction, `TickSizingController.decide` gives each pipe its start
    size: `c0` (`--chunk_size_mb` unless the config sets `c0_mb`) the first
    time, else the size its last occurrence settled on, lowered (never under
    `c0`) until the transient per byte it measured there fits the budget.
  - A pipe that may be re-cut mid-run (sort, filter and project by class,
    aggregation pass 1 per pipe; uniform geometry) gets a `TickJumpSession`,
    the learned jump (`docs/adaptive_intermediate_sizing.md` §8): on its
    event-ordered ticks it measures `samples` chunks at a size (tick time,
    input bytes, transient), doubles once (the walk), then fits the tick
    time as `F + t·x` and moves to the size that minimizes the pipe's time,
    capped by the size its transient per byte fits on the budget; a step is
    kept only on a measured gain of `epsilon`, else the pipe is re-cut back
    and fixed. Join cores (their geometry is fixed by `div`) and the other
    aggregation pipes (whose operators read the construction-time budget
    into contracts a re-split would break) run their start size.
- **SZ-B: rearrange gather windows** (`mid_stream` only).
  `install_gather_sizing` gives each multi-iteration gather stream a
  `GatherSizingSession`. The stream re-resolves its window every iteration and
  passes the session one `GatherIterSample`. The session writes resolver
  overrides only between iterations.

Both legs' budget counts what the cache region could lend: a size over the
operator space alone runs, its allocations borrowing from the region inside
the allocator.

The gather leg walks (`docs/adaptive_intermediate_sizing.md` §6):

```
SENSE   small δ probes (default 64 MB), downward before upward: c0−δ, c0
        [, c0+δ, c0+2δ]
        memory only: fits this stream's transient model before any upward step
MARCH   c += step (Δ = 256 MB, escalating Δ, 2Δ, 4Δ … up to 2 GB on
        consecutive accepts)
        keep a step only if measured time per byte improves by ≥ ε (2%),
        one sub-ε step tolerated (patience 2) against the same incumbent
FROZEN  converged (the knee) | line_blocked (memory) | breach
        the shrink-to-safety check still runs at every iteration;
        line_blocked re-opens when the budget recovers by r_worst·Δ
```

The tick leg learns instead (§8):

```
MEASURE  k chunks at the size (the first at a size skipped): tick time q
         (compute-end gap + readback), bytes b, transient T
WALK     2× the start size, capped (memory, c_max, measurable)
JUMP     fit q = F + t·x; x* = sqrt(D·F/t) for the pipe's D input bytes,
         capped by the memory bound ρ·x ≤ 0.9·budget (ρ = T/b at the
         largest size) and the bytes left to measure in, taken when the fit
         predicts ≥ ε less pipe time
KEEP     a measured step with ≥ ε less pipe time (q·(D/b + 1)); otherwise
         re-cut back and fixed
```

The gather leg's `c0` is 1 GB unless the sizing config sets `c0_mb`. A growth
step must pass two memory gates. First, the model's envelope at the new size
must fit `grow_budget_frac` (0.9) of the safety budget. Second, the increment
bound must hold: `T_obs(c) + r_worst·step ≤ budget`, where `T_obs` is the
largest transient seen at the current size and `r_worst` is the worst
transient-per-byte ratio this stream has shown. The tick leg's one memory
gate is its measured `ρ` against `grow_budget_frac` of the budget. Safety is
purely predictive, and nothing recovers from an OOM.

**The budget** is the whole arena's room: `capacity + lendable − allocated −
M_abs`, where `lendable` is what an allocation that misses could borrow from
the cache region right now (the arena stats' `lendable_b`: the attached
cache's lendable blocks, 0 without a region or a cache). With the contiguity
belt it is capped at the larger of the largest free run and the free run
that ends at the boundary plus `lendable` (lent blocks are the region's
lowest and join that run). A lend moves blocks from the region to the
operator space without changing it, so it decides every size and is every
level recorded or compared across time: the growth gates, the sense probes,
the entry check and its clamp, the growth clamp, the shrink-to-safety check,
`blocked_budget_b` and the re-open tests, the tick leg's memory bound and
start clamp, and the `budget_b` of the sizing telemetry. A gather growth
step over it clamps to the largest size it vouches for; a running window
over it shrinks to what it vouches for.

**Warmup** is excluded: `main.warmup` calls `set_gather_sizing_enabled(False)`,
which silences both legs, so warmup runs static budgets and trains nothing.

**State lives for the process.** The gather leg's fitted models are keyed
`(sf, query, pipe_class, instance, codec_epoch)`, with a class-level prior
`(sf, pipe_class, codec_epoch)` that is read only while a stream's own fit is
too thin; the tick leg's store keeps, under the same key, each pipe's
settled size, whether it is fixed, and its `ρ`. `codec_epoch` is
`"<--compression>/<--compression_search>"`. Repeating a qid in `--q`
therefore starts each pipe where the last occurrence left it.
`reset_run_store()` drops the pipe state and converged windows (the fitted
models stay), and `main.py` calls it
when `--slot_pool_gb_list` resizes the cache region. `select_run_store(key)`
keeps that state per cache-region size: `main.py` selects the occurrence's
region size, so occurrences with and without the cache (`TQP_CACHE_CORE`)
walk separately, each in its own operator space. Nothing persists across
processes. The budget and transients read the arena's operator space (the
arena below the cache region): the budget is `capacity − M_abs − allocated`
of that space plus what the cache could lend (the stats' `lendable_b`), and
a transient is a window peak over the operator space minus its allocated
bytes at entry.
They had the same values when the cache was a slab allocated from the
arena; only the perf report's absolute `peak_gpu_bytes` lost the cache's
bytes.

### Lending, not leases

The slot governor (`slot_governor.py`), which lent the cache region's blocks
to the sizing search, the region producer and aggregation pass 2 on leases,
is retired: the arena lends them itself to any allocation that misses, in
C++, in every mode (`memory/README.md`, "Lending"; `IO/cache/README.md`).
The sizing plane only reads `lendable_b` into its budget and moves nothing.

## When decisions take effect

**Read picks: tick *t+2*.** Within one steady-state iteration *i*,
`VortexPipeline.do_exchange` runs these steps in order:

1. `exchange.launch(ts[i].in_stage)`: H2D of chunk *i*;
2. compute of tick *i* (after packing tick *i−1*'s regions);
3. the pre-plan for tick *i+1* (cache pins for `ts[i+1].in_stage`), before
   the sync;
4. the sync;
5. `collect_transfer` / `collect_compute`, so the sample for tick *i* lands,
   and then the observer call.

`ts[i+1]` is consumed at step 3, before sample *i* exists, so feedback from
tick *t* can change only ticks ≥ *t+2*. `_emit_tick` rebuilds `ts[t+2]`'s input
stages right after step 5 (`_jit_rebuild`), and only its input stages. The
rule holds through the prologue too: sample 0 (transfer only) arrives before
the `ts[2]` pre-plan. With `TQP_PICK_DEBUG` set, the pipeline asserts at launch
that each tick's input stages match the pick state recorded at its rebuild
horizon.

**Write picks: tick *t+1*.** Sample *t* lands after tick *t*'s outputs were
compressed into their regions, so a write flip can take effect at tick
*t+1*'s post-compute without a race (`_actuate_write_picks`): the next region
of each flipped column carries the new codec. Write picks never change inside
a probe window.

**Sizing:**
- **Tick pipes** resolve their budget once, at construction.
- **Re-cut pipes** (the tick sizing session) re-cut at the end of an
  event-ordered tick, from the next unissued tick (`_resection`, which
  re-splits only ticks that have not been issued).
- **Gather windows** change only between gather iterations, each of which is
  internally synchronized.

Where the code falls short of these rules, and of the rest of this README, is
tracked in `docs/bug_backlog.md`.

## Files

| File | Purpose |
|------|---------|
| `adaptive_driver.py` | The package's registration point: the process-wide hooks the pipeline polls. The tick-observer factory (`set_tick_observer_factory`, `has_tick_observer_factory`, `tick_observer_min_ticks`, `make_tick_observer_for`); the gather-sizing session factory (`set_gather_sizing_factory`, `make_gather_sizing_session`); the tick sizing session factory (`set_tick_sizing_factory`, `has_tick_sizing`, `make_tick_sizing_session`); and the warmup switch `set_gather_sizing_enabled` / `sizing_enabled`, which gates both sizing legs despite its name. Defines the feedback record `GatherIterSample` (one gather iteration: window, rows, CUDA-event legs, arena peak and entry, host-bracketed `iter_ms`). The shipped factories return plain closures. |
| `decision_menu.py` | The compression plane's boundary objects. `ColumnChoices` is a legal range of one or two codecs (at most one non-RAW) plus RAW and stored bytes; `DecisionMenu` is a pipe's input and output ranges and `total_ticks`; `PickPair` is the `read-and-write` return (input picks, output picks, `seal`). `build_input_choices` builds a range from the reps a column holds and `build_output_choices` from the resolved codec; with `locked=True` it makes the non-base slots of a B-partitioned slot group singletons, so the group flips together. `validate_pick_vector` rejects a wrong-length or out-of-range vector before any state changes. `PickState` is used only by `tests/adaptive/test_menu_surface.py`; the pipeline keeps its own live pick list. |
| `resource_monitor.py` | `TransferTimes` (per-tick h2d, d2d and d2h ms) and `ComputeTimes` (the compute bracket, and inside it the output-compress span). `do_exchange` builds one `ResourceMonitor` per pipe run; it accumulates whole-pipe averages (`get_trans_t`, `get_comp_t`) and keeps the last tick as the observer's view, with `f_last` as the objective. Under `read-and-write` the pipeline also wires in live per-output-slot raw and stored byte counters, the write search's readback-price inputs. |
| `compression_controller.py` | `CompressionController`, a stateless view over one menu. The decision columns are the two-entry ranges; `benefit(i) = raw_bytes − stored_bytes[candidate]`; `ranked_indices()` orders them highest benefit first, position as tiebreak; `pick_vector(k)` compresses the top *k* and sets the other decision columns RAW. Write-leg counterparts: `write_decision_indices`, `ranked_write_indices(weights)` (weights are measured output bytes) and `write_pick_vector(k, ranked)`. |
| `compression_search.py` | The search math and the drivers. `ternary_argmin_gen` is integer ternary search over `k ∈ [0, n]` as a generator (memoized, ties to the smaller *k*), with `argmin_unimodal` and `max_evaluations`; `min_searchable_ticks` and `min_write_searchable_ticks` are the minimum pipe lengths `main.py` passes to the factory. `CompressionSearchDriver` is the read leg: two samples per *k*, their median, attribution under the *t+2* lag, epilogue drain ticks ignored. `WriteWindowSearchDriver` is the write leg (see above). `ReadWriteSearchDriver` runs the two in sequence and budgets the write leg from its actual start tick. |
| `run_stats.py` | Per-query perf-report records, reset by `main.py` at the top of every query; experiment telemetry, kept only under `TQP_EXPERIMENT_TELEMETRY` (the sizing controller and `main.py`'s search factories test the switch before recording; no decision reads them). Sizing-model samples (one predicted-vs-observed transient per sizing decision, capped at 20,000, plus skip counters the gather leg uses) become the `sizing_model` section. The compression-search registry becomes the `compression` section through `snapshot_compression_choices()`: per searched pipe, the menus, *k\**, which gates fired, the probes spent and the final codec vectors. Recorders never raise. |
| `sizing_controller.py` | The sizing plane's controllers. `SizingSearchConfig` holds every knob and default (`validate()` checks them); `load_config(path)` reads the `--sizing_config` JSON, is the schema of record and ignores unknown keys (an old config's `governor` section among them). `_budget_from_stats` (the budget) reads the arena stats the controllers get, `lendable_b` included. `GatherSizingSession` (SZ-B) walks one rearrange stream: it scores host-bracketed iteration time (`objective: "iter"`, the default) or the three CUDA legs (`"legs"`); shrinks on entry and at any iteration where the safety line moved; grows by trying, in order, a plain step, a clamp to the budget by model inversion, and finally `line_blocked`; and on `close()` stores the converged window for same-key streams and clears its override. A stream whose key already converged only re-runs the down-probe. `TickSizingController` (SZ-A) has two entry points: `decide` (the resolver decision function: each pipe's start size from its store) and `session`, the factory for `TickJumpSession` (the learned jump: measure, walk ×2, fit the tick time, jump to its optimum within the memory bound, keep a step only on a measured gain), whose outcomes go to the store and whose paths make the query's `[sizing]` line (`take_report`). `SECTIONABLE_CLASSES` (sort, filter, project) are the classes re-cut by default. Wiring: `install_gather_sizing`, `install_per_pipe_sizing`, their `uninstall_*` counterparts and `reset_run_store`. |
| `sizing_memory_model.py` | The memory model. It answers one question: how large will the arena transient (peak minus live-at-entry) be at budget `c`? `TransientEnvelopeModel` is a least-squares line (slope clamped at zero or above) whose prediction is the fit plus a pad, the larger of the worst positive residual and `pad_min`. The fit is used only with at least two distinct sizes and for `c` at most one Δ past the largest size observed; otherwise the analytic seed `seed_slope·c + pad_min` answers (slope 2.0 for rearrange; the gather leg is its only user). `invert(budget)` gives the largest safe `c`; `r_worst` is the worst transient/`c` ratio ever seen and never decreases. The fit window keeps 160 samples, at most 8 per size. `ModelRegistry` holds one model per key tuple, with per-class seed slopes. |
| `instage_optimization.py` | Static settings `main.py` sets from the CLI. `set_fixed_compression` / `set_adaptive_compression` (from `--compression`), and `active_compression_policy()`, the one `CompressionPolicy` both the scan and the applier read, so the rep the scan materializes always matches the rep the applier selects. `set_zero_copy_selectivity` (from `--mask_selectivity`), and `decide_column_type`, called once per input column at pipe construction: `zero_copy_mask` when the column is RAW, entirely in pinned host memory and its mask keeps fewer rows than the threshold, else `direct` with the resolved codec. `find_rep_by_algorithm`, the rep selector the pipeline uses; under `read-and-write`, `set_mixed_rep_acceptance(True)` lets it return the only rep of a single-rep column whose producer flipped its write picks mid-run. `set_fixed_search_json` / `get_fixed_search_json` for `--compression_search fixed`. |

There is no `__init__.py`. `adaptive` is a namespace package, and each
consumer imports the module it needs.

## Tests

- **Unit tier** (no GPU work; pytest collects it): `tests/adaptive/`
  `test_controller.py`, `test_search.py`, `test_write_search.py`,
  `test_sizing_model.py`, `test_sizing_controller.py` and
  `test_tick_sizing.py`. Lending is tested on a small arena in
  `tests/allocator/test_arena_lend.py` (GPU). Run:
  `TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/adaptive`.
- **Standalone GPU probes** (not collected; each runs SF1 queries, e.g.
  `TQP_GPU=<idle> docker compose run --rm -T dev python3 tests/adaptive/pick_flip_probe.py`):
  - `pick_flip_probe.py`: forced per-tick read flips under the *t+2* rebuild;
  - `write_flip_probe.py`: forced write flips (an aggregation pipe, a
    partition sort, the sort under the LRU granule cache, null and alternating
    boundaries) give identical results.
- **End to end**: the `tests/e2e` ladder adds `--compression_search
  read-and-write`, then `--sizing_search mid_stream`, and requires 22/22 at
  each rung.

## Known gaps

Open issues in both planes are tracked in `docs/bug_backlog.md`. The gaps
below are specific to this package's settings and docs.

- `instage_optimization.py`'s comments name a `--fixed_compression` flag (the
  flag is `--compression`).
- The schema in `load_config`'s docstring omits `safety.contiguity_gate` and
  `safety.grow_budget_frac`. The loader reads both, and `AGENTS.md` lists them.
- SZ-B's `objective: "iter"` uses host-bracketed time (each iteration is
  synchronized on both sides); the tick leg reads only the GPU timeline's
  compute-end gaps, which the event-ordered ticks record anyway.
