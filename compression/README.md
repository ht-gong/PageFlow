# `compression/` — GPU column compression for TQP-Vortex

This directory holds everything the engine needs to compress and decompress
columns on the GPU: the nvCOMP-backed CUDA extension, the Python-side
codec registry that wraps it, and the policy that decides which codec a column
gets. Compressed records let the scan path move fewer bytes over PCIe and keep
more of a table resident on the device, at the cost of a decompress step
before an operator can read the data.

The pieces fit together like this:

1. **`nvcomp_manager.cu` + `compression.cu`** are compiled by
   `CMakeLists.txt` into one Python extension module, `TQPlib.tqpcompress`.
   The manager file drives nvCOMP's high-level manager API (self-describing
   `NVCOMP_NATIVE` objects, nvCOMP-managed chunking and scratch); the other
   file holds the copy and pack CUDA kernels the IO layer uses to stage records.
2. **`compression_algorithms.py`** is the registry the rest of the engine
   imports. It names the codecs (`RAW`, `ANS`, `CASCADED`, `BITCOMP`, `LZ4`,
   `SNAPPY`, `GDEFLATE`, `DEFLATE`), assigns them stable integer indices (the
   `kCodec*` ids in `nvcomp_manager.cu`); a compressed record's index lives
   in its record table (`IO/pages/record_table.py`). It imports no CUDA, so
   plan-time code can use it.
3. **`format_policy.py`** decides which codec a given column should be stored
   in, keyed on `(mapped_type, is_sorted)`, in either a fixed or an adaptive
   mode, and holds every codec's nvCOMP manager chunk size. Both come from
   **the codec map**, `profiling/results/codec_map.json`, written by the
   offline sweep (`make profiling`, `profiling/sweep.py`) and loaded once per
   process on first use; this module is its only reader. It is pure Python
   with no CUDA or torch dependency.

Consumers live mostly in `IO/` (`page.py`, `batched_decompress.py`,
`manager_compress.py`, `record_table.py`), in
`adaptive/`, `dataflow/compression_applier.py`, `profiling/sweep.py` and
`main.py`. Tests are under `tests/compression/`. The offline sweep that
writes the codec map is described in `profiling/README.md`.

## Manager rules

Each of these fixed a real failure during the nvCOMP manager integration:

- **`mgr_shutdown()` runs immediately before any arena reset**
  (`cmp.free_all()`). Managers keep their scratch in the Torch arena.
  Managers created during warmup that survived the post-warmup reset kept
  using scratch that the real run's tensors were then handed: silent
  corruption, and an illegal memory access at SF100. The real run
  re-creates the managers lazily.
- **`mgr_release_scratch()` runs after every query** (`_release_pools` in
  `main.py`). Managers persist across queries for their cached configs, and
  their retained scratch otherwise accumulates and fragments the arena: Q13,
  Q17, Q18, Q21 and Q22 ran out of memory at SF100 without it.
- **One manager call compresses at most 256 MiB of worst-case output**
  (`_MAX_OUT_BUF_BYTES` in `IO/codec/batched_compress.py`). An uncapped 4 GB
  sort tick made nvCOMP request a single 5.5 GB scratch block.
- **Record sizes are queried, never assumed.** Compression can expand data:
  ANS at 128 KiB chunks turns 8 MiB into 11.17 MB. `IO/codec/manager_compress.py`
  binary-searches `mgr_max_compressed_size` for the largest record whose
  worst case fits a page.
- **Every decompress derives private `DecompressionConfig`s** from the cached
  `CompressionConfig`s, because nvCOMP forbids reusing one config across
  overlapping operations.

Codec options are fixed in `nvcomp_manager.cu` and are never swept. ANS,
CASCADED and BITCOMP keep the engine's tuned options, and the other four use
nvCOMP's defaults. Zstd ships in the same nvCOMP build (`zstd.hpp`) but is
not wired in; adding it follows "Codec changes" in `AGENTS.md`.

**After an nvCOMP version change**, a reconfigure can fail with "imported
target references ...", because the CMake cache still holds the old
`nvcomp_*` paths. Reconfigure with `cmake -U "nvcomp_*"`, or drop the `dev`
service's build volume with `make clean`.

The integration's running log is in git history:
`git show 20052ca:docs/nvcomp_manager_integration_notes.md`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Empty marker that makes `compression` an importable Python package. |
| `CMakeLists.txt` | Builds `compression.cu` and `nvcomp_manager.cu` into the `tqpcompress` Python extension module. Links nvCOMP, the CUDA runtime, Torch and the Torch Python library; sets `-G` for Debug/RelWithDebInfo and `-O3 --use_fast_math` for Release; strips the `lib` prefix and applies the CPython extension suffix so the `.so` is importable directly. |
| `compression.cu` | Copy and pack kernels for the IO layer, plus the `PYBIND11_MODULE` definition: `batched_memcpy` (device-side scatter copy used to pack compressed slabs), `batched_memcpy_abs` (the same with an absolute source and destination per chunk: the region producer's pack into a dense buffer and the granule cache's slots; its address arrays may sit on the GPU or in pinned host memory, which the kernel reads over UVA), `batched_memcpy_abs_staged` (the same from one pinned staging buffer by its address, destinations offset from a base, with an event recorded after it: one call for `IO/pages/abs_copy.py`'s `launch`), `batched_d2h_to_ptrs` and `batched_h2d_to_ptrs` (async copies between scattered pointers), and the `batched_h2d_pack` / `batched_d2d_pack` stage packers, which take per-page sources, `agg_buf` destination offsets and sizes, validate every destination before copying, and issue one copy per page payload. Calls into the manager component to register its bindings. |
| `nvcomp_manager.cu` | nvCOMP 5.2 high-level manager component for the seven codecs (ANS, CASCADED, BITCOMP with the engine's tuned options; LZ4, SNAPPY, GDEFLATE, DEFLATE with nvCOMP's defaults; byte-oriented codecs share one manager across element sizes). Caches one manager per `(codec, dtype, chunk_bytes, stream)` for the process lifetime, routes manager scratch through Torch's caching allocator (override with `TQP_MGR_SCRATCH=cuda`), and exposes an enqueue-only batched surface: `mgr_compress_begin` / `mgr_compress_launch` / `mgr_compress_finish` for compression with a ticket registry (the launch stores the actual sizes into pinned memory by one block's UVA stores, not a copy, which would queue on the copy engine behind the copies home in flight), `mgr_decompress` from cached config handles, `mgr_ingest_configs` as a counted, synchronizing header-parse fallback, plus `mgr_max_compressed_size` for output sizing, counters, scratch release and `mgr_shutdown`, and `uva_store`, the same store for any small readback (the event-ordered ticks' mask counts). Its contract is no stream synchronization on hot paths. Each batched call (`mgr_compress_begin`'s configure, `mgr_compress_launch`, `mgr_decompress`) goes in slices of at most 65,535 buffers (`kMaxBatch`): nvCOMP cannot launch a longer batch ("invalid configuration argument"), which SF1000 Q1's aggregation pass 2 reaches at chunks of 128 MiB or less. |
| `compression_algorithms.py` | Codec registry: static tables give codec names and indices, and the accessors `algo_names`, `name_to_int_map` and `mgr_chunk_size` (which delegates to `format_policy.chunk_bytes`). A compressed record's index (offsets, sizes, `DecompressionConfig` handles, the manager `codec_key`) is its record table's, held on the host only (`IO/pages/record_table.py`). |
| `format_policy.py` | The codec map and the per-`(mapped_type, is_sorted)` compression format policy. `codec_map()` loads `profiling/results/codec_map.json` once per process (missing file: built-in defaults plus one stderr warning; broken file: a hard error), `chunk_bytes(codec)` reads its per-codec chunk, and `CompressionPolicy` resolves the codec for a column in `fixed` mode (one codec everywhere) or `adaptive` mode (the map's best per class). `formats_for` returns the representations the scan path materializes (`RAW` always, plus at most one compressed codec). Dependency-free (stdlib only) so it can be imported from plan-time code. |
