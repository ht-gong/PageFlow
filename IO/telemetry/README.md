# `IO/telemetry/` — copy timings, the movement ledger, the residency peaks and the switches

Four small modules. One records what engine decisions read, in every run:
`copy_events.py` times each tick's host↔GPU copies for the compression
searches. The searches read no timing of their own: besides these copy spans
they read the compute-stream spans the pipeline records every tick
(`compute`, `compress_out`; `adaptive/resource_monitor.py`). Two count
bytes for experiments, written to the perf report: `movement.py`, the
movement ledger, and `residency.py`, the most each memory tier held.
`gates.py` holds the switch the ledger and the
other experiment numbers sit behind, and the debug validators' switches. The subpackage imports nothing from
the rest of `IO/`; the recording sites are in `IO/cache/`, `IO/codec/`,
`IO/transfer/`, `IO/pipeline/` and `operators/output.py`, and the counters
`residency.py` reads in `IO/pages/` and `memory/`.

Overlap and idle time (did a copy run under compute or with the GPU idle?)
are not measured here: take an Nsight trace with
`experiments/self-verification/e5_time_breakdown.py`.

## `copy_events.py`: one event pair per copy batch

A *copy batch* is the copies one site issues on one stream in one launch:
the tick's H2D pack and D2D pack (`DirectIn`) and the zero-copy stage's mask
copies. The site records a start event on its stream right before its first
copy and an end event right after its last (`begin(lane)` / `end(batch)`),
so a batch's time is its copies plus any host gap between them, never the
planning before the first copy. The copies home of the region producer are timed by
the home writer's own event pairs (`IO/codec/home_writer.py`, `take_lane`),
which the pipeline adds to the tick's d2h lane.

Batches go into a `CopyLog`. `NaiveExchange.launch` opens one per tick and
`sync` closes it. With no log open, `begin` records nothing. Once the log's
streams are synchronized:

- `lane_ms()` are the tick's input lanes of `TransferTimes`: per stream
  (`h2d`, `d2d`), the span from its first batch's start to its last batch's
  end, 0 when the stream copied nothing.

Event pairs are pooled and reused once a log is released. `reset()` (top of
each query) drops any log a failed query left open.

## `movement.py`: the movement ledger

Every copy of record bytes between host and GPU memory, and every copy the
granule cache avoided, counted once at the site that issues (or skips) it, per query.
A record has a *kind* and three counts: `pages` (0 for zero-copy reads,
which read rows in place), `logical_b` (decompressed bytes) and `wire_b`
(stored bytes). A copied read (`bind_tick_sources`) counts what the tick
selects of each region rather than the region: one run per region (its
selected records, `IO/transfer/README.md`), the selected records' decoded
bytes as `logical_b` (for a RAW record read by rows, the rows copied) and the
run's bytes as `wire_b`.

| Kind | What moved | Site |
|---|---|---|
| `scan_h2d` | a base-table run copied H2D | `bind_tick_sources` (for a zero-copy stage's mask or index too, read through a `DirectIn`) |
| `scan_uva` | base-table rows read zero-copy over UVA | the zero-copy stages; a gather over UVA (a streamed gather's load is its pipe's `scan_h2d` / `restore_h2d`) |
| `spill_bypass` | an intermediate written home without being offered to the cache | the home writer's copies of the region producer's regions that no cache holds (`IO/codec/region_producer.py`; a page there is a region) |
| `spill_rejected` | a cached region's refused suffix copied home | the producer's pack (a page there is a region) |
| `spill_evict` | a dirty granule copied home by a reclaim, or by a lend of the cache region to an allocation that misses | `NativeGranuleCache._victims`; a lend's from `NativeGranuleCache.take_lend_stats` at the query's end |
| `spill_mirror` | a cleaning copy | the producer, after a tick's admissions |
| `restore_h2d` / `restore_uva` | an intermediate read back, copied or zero-copy (a producer's record table is an intermediate; a scanned table, `RecordTable.base`, is a base table) | the `scan_*` sites |
| `restore_evicted_h2d` | a cached region's home piece copied H2D: granules the cache evicted, or never admitted | `bind_tick_sources` (one page per home piece) |
| `cache_hit` | a cached region's slot piece read D2D | `bind_tick_sources` (one page per slot piece of a run) |
| `resident_d2d` | a GPU region outside the cache, read D2D | `bind_tick_sources` |
| `cache_fill` | the granules a pack (or a RAW output) wrote into slots | the region producer |
| `avoided_dead` / `avoided_clean` | a reclaimed (or lent away) dead or clean granule, released with no copy | `NativeGranuleCache._victims`; `release_dead` (dead granules once their reads settle); `take_lend_stats` |
| `result_d2h` | the output step's copy of the result columns to the host (a page there is a column) | `operators/output.py` |

A page of `cache_fill`, `spill_evict`, `spill_mirror` and the `avoided`
kinds is a granule: `wire_b` is its bytes (the last granule of a region is
shorter), `logical_b` the decoded bytes of the records in it, each record's
decoded bytes spread evenly over its stored bytes (the pack for
`cache_fill`; `NativeGranuleCache.ledger_bytes`, from the decoded sizes the
producer registers under experiment telemetry, for the others). A read's
home pieces are `restore_evicted_h2d`, one page per piece.

Each consumer of a region reads it, and a read leaves the bytes where they
are (a home read admits nothing): the read kinds count a region once per
consumer and can exceed what was written (SF1 q6 reads its intermediates
three times). A granule enters a slot only by admission (`cache_fill`) and
leaves it once at most, by `spill_evict`, `avoided_clean` or `avoided_dead`
(or unrecorded, when its region is freed): the spill pass checks that their
bytes stay within the fill's.

A record table's host reads are restores or scans by `RecordTable.base`,
its GPU reads `resident_d2d`. Records go to the current *phase*: `query`,
or `output` while `main.py` runs the output step (`set_phase`, around
`tqp_output`): its reads of the result columns and its `result_d2h`. The
timer (`approx_time`) leaves the main plan's output step out, so the query
phase is the timed run's movement; a subquery plan's output step (a scalar)
is in the output phase too. `main.py` calls `reset()` at the top of each
query (which also returns to the query phase) and writes `report()` as the
perf report's `movement` section: `kinds` (the query phase) and `output`,
each with every kind, zeros included. The experiment scripts read it through
`experiments/paper_scripts/helpers/perf_json.py` (`mv(..., phase="output")`
for the output step). A report written before `restore_evicted_h2d` and
`result_d2h` (2026-10-10) has neither, and no output phase: `perf_json`
reads the two kinds as 0 there (its `LATER_KINDS`; a cached region's home
pieces were `restore_h2d` then, so the sums are unchanged).

## `residency.py`: what each memory tier held at most

The perf report's `residency` section, under the same switch: per query, the
most each tier held during the timed run.

| Section | What | Counter |
|---|---|---|
| `gpu_arena` | `capacity_b`, `at_start_b` (allocated when the timer starts), `peak_b` | the operator space's high-water mark (`TQPlib.tqpmemory.get_peak_bytes`, reset with `reset_peak` where the timer starts), in whole arena blocks (1 MiB); cuDF's allocations included, the cache region not (it is never an allocation) |
| `host_pool` | `capacity_b`, `block_b`, and per tag (`intermediate`: the chunks the pipes' regions are carved from, the unwritten rest of each open chunk included, up to 16 MiB per open column; `scan`: the scanned tables) `at_start_b` and `peak_b`; `peak_total_b` | the pinned pool's block counts (`MemoryPool.usage`, whole 0.5 MiB blocks) |
| `slot_pool` | the granule cache's slots, None without a cache: `capacity_b` and `slab_b`, the cache region's bytes when the timer starts (the arena may lend blocks of it to the operators later, back at a later pipeline start); `slot_b`, a slot (an arena block); `at_start_used_b` and `peak_used_b`, the slots in use then and at most since | `NativeGranuleCache.used_bytes` / `peak_used_bytes` (the core's count of slots handed out, kept whatever the switch says) |

`main.py` calls `begin(pinned_pool, cache)` (the query's granule cache, or
None) right after the arena's
`reset_peak` where each query's timer starts (it rebaselines the pools' peaks,
so the scan and the warmup are not in them), and `report(...)` once the
query's operators have run, before the pools are released. The peaks cover
the output step. A failed query's partial report has the residency where it
stopped (`timed` False when it failed before its timer started). The pinned
pool's counters are counted only under the switch.

## The spill pass

`bench/run_tqp.sh` with `SPILL=1` (`baselines/AGENTS.md`, "Spill pass") runs
each query under the switch and turns its perf report into the comparison's
record (`bench/tqp_spill_stats.py`, `baselines/results/spill_profile.csv`):
the ledger's kinds as they are, `born` (`cache_fill` + `spill_bypass` +
`spill_rejected`), `result_d2h` and the output step's reads from `output`,
and the residency peaks.

## `gates.py`: the switches

Each switch is read once at import, on-word (`1`, `true`, `yes`, `on`; `0`
is off), and off by default. Import the module, never a name, so a test can
set `gates.<NAME>`.

- `EXPERIMENT`, from `TQP_EXPERIMENT_TELEMETRY`. It turns on the ledger's
  hooks, the residency counters, each pipe's `peak_gpu_bytes`, the sizing
  and compression searches' records, the perf report's `movement`,
  `residency`, `sizing_model` and `compression` sections, a failed query's
  partial report (`wall.status` FAIL), the resolved settings, versions,
  GPU and CPU pin in `cmdline.json`. Every hook tests it at its call site, before
  computing its arguments, so with it off (every timing run) no hook does any
  work.
- The debug validators, which raise on a mismatch: `VARCHK` (a compressed
  read's record index as its decode template compiles;
  `transfer/tick_layout.py`), `PICK_DEBUG`
  (each tick's in-stages at launch; `pipeline/vortex_pipeline.py`) and
  `TIMED_GC_VALIDATE` (`main.py`).

## Tests

`tests/telemetry/`: `test_copy_events.py` (lane spans over fake events,
and the log stack); `test_movement.py` (the record/report contract and the
output phase, `perf_json`'s reader, scans, spills, restores, evicted
granules' read-back and UVA reads on real pipes, nothing recorded with the
switch off, and SF1 Q6's perf report both ways with its output step and
residency); `test_gates.py` (the
switches' parsing, each read once from its own variable, main.py refusing
the experiment switch without `--perf_report`, and the
runtime inputs reaching the searches and the sizing session with it off);
and the pipeline and gather integration tests. The residency counters:
`tests/allocator/test_pool_peaks.py`. The spill pass's record builder:
`baselines/tests/test_spill_tqp.py`. The write search's reading of `compute` and
`compress_out` is covered by `tests/adaptive/test_write_search.py`.
