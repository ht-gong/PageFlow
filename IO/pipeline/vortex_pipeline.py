from __future__ import annotations
import os
import numpy as np
import torch
from typing import List, Dict
import IO.transfer.data_exchange as data_exchange
from utility.logger import perf_logger, message_logger
from IO.transfer.pre_comp_stage import PreCompStage
from IO.transfer.tick_stages import TickStages
from IO.transfer.in_stage import DirectIn
from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
from IO.codec.batched_decompress import BatchedDecompress
from adaptive.instage_optimization import (
    find_rep_by_algorithm, _find_all_cpu_rep, decide_column_type,
)
from IO.pages.combined_column_rep import CombinedColumnRep
from adaptive import resource_monitor as _rm
from adaptive.resource_monitor import ResourceMonitor, TransferTimes, ComputeTimes
from adaptive.adaptive_driver import (
    make_tick_observer_for, has_tick_observer_factory,
    tick_observer_min_ticks,
    has_tick_sizing, make_tick_sizing_session,
)
from adaptive.decision_menu import (
    DecisionMenu, PickPair, build_input_choices, build_output_choices,
    validate_pick_vector,
)
import TQPlib.tqpcompress as tqpcompress
import TQPlib.tqpmemory as cmp
from IO.cache import config as _cache_cfg
from IO.telemetry import gates
from IO.pipeline.common import (
    PartitionedTensor, _WindowPeakTracker, _is_raw_algo,
)
from utility import cuda_streams
from IO.pipeline import tick_cutter
from IO.codec.home_writer import writer_stream

# A tick that reads no cached granule (``_pin_granules``).
_NO_GRANULES = np.zeros(0, dtype=np.int64)

# ``TQP_SERIAL_TICKS`` (on-word, an experiment switch): a pipe built while
# it is on runs under the tick barrier with nothing overlapped: a tick's
# input copies end before its compute starts, and the copies home (and the
# pack's) end before the next copy or compute.  Read at construction, so an
# in-process A/B can set the attribute.
SERIAL_TICKS: bool = (os.environ.get("TQP_SERIAL_TICKS") or ""
                      ).strip().lower() in ("1", "true", "yes", "on")

class _ComputeEvents:
    """One tick's compute-stream spans: the operator's (start to end), the
    output compress's and the region pack's, and whether it packed."""

    __slots__ = ("start", "end", "cprs_start", "cprs_end", "pack_start",
                 "pack_end", "packed")

    def __init__(self):
        (self.start, self.end, self.cprs_start, self.cprs_end,
         self.pack_start, self.pack_end) = (
            torch.cuda.Event(enable_timing=True) for _ in range(6))
        self.packed = False


class OutputLayout:
    """How a pipeline's physical output slots map onto its node's logical
    output instances: ``slots[s] = (instance index, use_order_hint)``.

    The graph describes logical instances only.  A join's sort pipe and a
    partitioned aggregation's pass 1 emit B partition slots per logical
    column (B is chosen per operator at run time, ``operators/partitions.py``,
    and the graph never learns it), so the pipeline derives every per-slot
    fact — codec, next-use position, expected reads, residency, order hint
    and slot group — from ``(instance, b)``.  ``flat(n)`` is every other
    pipe.
    """

    def __init__(self, slots):
        self.slots = list(slots)
        # Base slot of each slot's run (the first slot of the same instance):
        # a B-partitioned column's slots form one write-decision group.
        self._group = {}
        base_of = {}
        for s, (i, _b) in enumerate(self.slots):
            self._group[s] = base_of.setdefault(i, s)

    @classmethod
    def flat(cls, n: int) -> "OutputLayout":
        return cls((i, 0) for i in range(n))

    @classmethod
    def partitioned(cls, n: int, B: int) -> "OutputLayout":
        # use_order_hint = radix partition index b: the consumer (the join
        # core, or the aggregation's pass 2) reads partitions in increasing-b
        # order, so lower b = consumed sooner = retained longer in cache.
        return cls((i, b) for i in range(n) for b in range(B))

    @property
    def n_instances(self) -> int:
        return 1 + max((i for i, _ in self.slots), default=-1)

    @property
    def is_flat(self) -> bool:
        return all(b == 0 for _, b in self.slots) and len(self.slots) == self.n_instances

    def slot_group(self, s: int) -> int:
        return self._group[s]

    def collapse(self, results):
        """Per instance, its slots' results in slot order."""
        per_instance = [[] for _ in range(self.n_instances)]
        for (i, _b), r in zip(self.slots, results):
            per_instance[i].append(r)
        return per_instance


class VortexPipeline:
    """
    Creates a Vortex Pipeline that operates on input_columns in batches using a specified operator,
    either using a pre-specified division or a fixed chunk_size,
    then the results are returned in batches back to the output_columns.
    REQUIRES: Both the input_columns (as Variables) and the output_columns are allocated on the host (CPU).
              The output types are known apriori.
    """
    def __init__(self, input_columns: Dict[int, CombinedColumnRep] = None, output_columns_type: List[torch.dtype] = None, operator=None, cpu_mem_pool=None,\
                 div=None, chunk_size=None, pass_mask=False, mask_map: Dict[int, int] = {}, name="VortexPipe", applier=None,
                 page_cache=None,
                 output_compression_algs: List[str] = None,
                 tick_observer=None,
                 sizing_key=None,
                 sectionable=None,
                 *,
                 node=None,
                 inputs=None,
                 layout: OutputLayout = None,
                 fold_mask: bool = False):
        """A pipeline is built from its dataflow node.

        ``node`` is the graph's ``PipeNode``; ``inputs`` are ``(key,
        ColumnInstance)`` pairs (the key is whatever the operator closure
        wants to see in its column dict; the instance supplies the payload,
        the codec and the cache bookkeeping); ``layout`` maps the physical
        output slots onto ``node.outputs`` (flat by default).  Without a
        node, ``input_columns`` (``{key: CombinedColumnRep}``) and ``name``
        describe a bare pipeline with the fixed codec everywhere (tests).
        """
        from adaptive.instage_optimization import get_fixed_compression as _gfc

        # chunk_size=None means "the uniform budget" — resolved here, at
        # construction, rather than baked into the signature as a literal.
        # A default frozen at def-time would silently bypass --chunk_size_mb
        # for any caller that omits the argument (project.py did exactly that,
        # at a hardcoded 256 MB); resolving live makes the bypass structurally
        # impossible.  Normalized before ANY use — the boundary computation
        # below reads the local, not the attribute.
        if chunk_size is None:
            import constants as _constants
            chunk_size = _constants.get_chunk_size_b()

        # The node and its (key, instance) inputs.  The instances hold the
        # payloads; the same objects are what the producer wrote and what
        # the applier resolves, so nothing is looked up by name.
        self.__node = node
        if inputs is not None:
            self.__input_pairs = list(inputs)
            for key, inst in self.__input_pairs:
                assert inst.ccr is not None, (
                    f"VortexPipeline {name!r}: input {key!r} ({inst.name}) has no payload")
            input_columns = {key: inst.ccr for key, inst in self.__input_pairs}
            assert len(input_columns) == len(self.__input_pairs), (
                f"VortexPipeline {name!r}: duplicate input keys")
        else:
            self.__input_pairs = None
            assert input_columns is not None, "VortexPipeline needs inputs or input_columns"
        if node is not None:
            name = node.name
            if layout is None:
                layout = OutputLayout.flat(len(node.outputs))
            assert len(output_columns_type) == len(layout.slots), (
                f"VortexPipeline {name!r}: {len(output_columns_type)} output types "
                f"for {len(layout.slots)} layout slots")
            assert layout.n_instances == len(node.outputs), (
                f"VortexPipeline {name!r}: layout covers {layout.n_instances} "
                f"instances, node has {len(node.outputs)}")
        else:
            assert layout is None, "an OutputLayout needs a node"
        self.__layout = layout
        self.__outputs_published = False

        self.__input_cols = input_columns
        self.__input_col_keys = list(input_columns.keys())
        self.__out_host = []
        self.__output_columns_type = output_columns_type

        # Resolve per-column compression algorithms once at construction,
        # off the instances: the applier answers for producer and consumer
        # alike.  Without an applier (RAW mode, bare pipelines) the fixed
        # codec applies everywhere.
        if applier is not None:
            assert self.__input_pairs is not None and node is not None, (
                f"VortexPipeline {name!r}: an applier needs (key, instance) inputs and a node")
            input_compression_algs = [applier.codec(inst) for _, inst in self.__input_pairs]
            if output_compression_algs is None:
                output_compression_algs = [applier.codec(node.outputs[i])
                                           for i, _ in layout.slots]
        else:
            input_compression_algs = [_gfc()] * len(self.__input_col_keys)
            if output_compression_algs is None:
                output_compression_algs = [_gfc()] * len(output_columns_type)
        assert len(input_compression_algs) == len(self.__input_col_keys), (
            f"VortexPipeline {name!r}: input_compression_algs length "
            f"{len(input_compression_algs)} != {len(self.__input_col_keys)} input columns"
        )
        assert len(output_compression_algs) == len(output_columns_type), (
            f"VortexPipeline {name!r}: output_compression_algs length "
            f"{len(output_compression_algs)} != {len(output_columns_type)} output columns"
        )
        self.__input_compression_algs = input_compression_algs
        self.__output_compression = list(output_compression_algs)
        self.__strict = applier is not None
        self.__operator = operator
        self.name = name

        # Output slot groups (write leg): base slot index per output slot.
        # A B-partitioned logical column spans B consecutive slots that must
        # carry one codec (the partition-uniformity invariant), so only each
        # group's base slot publishes a write decision range and a write
        # flip propagates base -> group.  From the layout; every slot its
        # own group for a bare pipeline.
        n_out_slots = len(output_columns_type)
        if layout is None:
            self.__out_slot_group = list(range(n_out_slots))
        else:
            self.__out_slot_group = [layout.slot_group(s) for s in range(n_out_slots)]
        # Flip hygiene: set on the first sanctioned write flip
        # (codec_epoch 'mixed' — excluded from cross-run sizing fits).
        self.__write_flipped = False
        # In-run write-leg ranking + penalty bytes: cumulative RAW
        # and STORED (wire) output bytes per BASE slot, exposed to the
        # observer through the monitor view.  Accumulated only when a
        # write-capable observer exists.
        self.__out_raw_bytes_by_slot = [0] * n_out_slots
        self.__out_stored_bytes_by_slot = [0] * n_out_slots

        # Per-tick feedback hook: resolved AFTER the tick geometry below,
        # because the published menu carries the pipe's tick count
        # (``total_ticks``).  Latched here so the resolution branch stays in
        # one place.
        self.__requested_tick_observer = tick_observer

        # Per-slot cache bookkeeping, read off the output instances once at
        # construction (trust-the-graph):
        # next_use_op is the first consumer's position (the terminal rank
        # with no in-DAG consumer), expected_reads the consumer count and
        # uncacheable the instance's host residency.  LRU/MRU ignore the
        # positions.  A bare pipeline (no node) carries placeholders no
        # admission ever observes.
        n_out_for_keys = len(output_columns_type)
        if layout is None:
            self.__output_next_use_ops = [0] * n_out_for_keys
            self.__output_expected_reads = [1] * n_out_for_keys
            self.__output_uncacheable = [False] * n_out_for_keys
        else:
            self.__output_next_use_ops = [node.outputs[i].next_use_op for i, _ in layout.slots]
            self.__output_expected_reads = [node.outputs[i].expected_reads for i, _ in layout.slots]
            self.__output_uncacheable = [node.outputs[i].host_resident for i, _ in layout.slots]
        self.cpu_mem_pool = cpu_mem_pool
        self.__pass_mask = pass_mask
        # With ``fold_mask`` the operator applies the masks itself: a
        # direct column under a mask reaches it uncompacted (a zero-copy one
        # is compacted by its read), and it is called as
        # ``operator(columns, counts)``, ``counts`` the kept rows of each
        # mask by its key (hashjoin's sort pipes).
        assert pass_mask or not fold_mask, "a folded mask must reach the operator"
        self.__fold_mask = fold_mask
        # Every pipe's outputs are written as regions, one per tick and
        # output instance, RAW outputs included, by the region producer
        # (IO/codec/region_producer.py), which offers each cacheable
        # instance's regions to the granule cache (``page_cache``) when one
        # is configured.  A bare pipeline (no node) publishes nothing on
        # instances: its tables are ``result_tables()``.
        assert cpu_mem_pool is not None, (
            f"VortexPipeline {name!r}: regions need a pinned host pool")
        self.__granule_cache = page_cache
        from IO.codec.region_producer import RegionProducer
        self.__producer = RegionProducer(cpu_mem_pool, name, cache=page_cache)
        if page_cache is not None and layout is not None:
            # Each instance by its base slot; a gather source stays a host
            # region (the gather reads it over UVA).
            self.__producer.cached_outputs = {
                s: (self.__output_next_use_ops[s],
                    self.__output_expected_reads[s])
                for s in range(len(layout.slots))
                if layout.slot_group(s) == s
                and not self.__output_uncacheable[s]}
        # The tick's compute spans (``_ComputeEvents``): one set under the
        # tick barrier, a ring of four with event-ordered ticks (read two
        # ticks late, with the previous tick's end for the tick time);
        # ``__cev`` is the tick's own.
        self.__cev_ring = [_ComputeEvents()]
        self.__cev = self.__cev_ring[0]
        # A new stream per pipe, none of the process's kept streams
        # (utility/cuda_streams.py).
        self.__compute_stream = cuda_streams.compute_stream('cuda:0')
        self.__producer.stream = self.__compute_stream
        self.__batched_decompress = BatchedDecompress()

        self.__chunk_size_b = chunk_size
        # Tick-pipe sizing identity: ``(pipe_class, instance)`` from the
        # construction site whose budget came through the sizing resolver.
        # ``None`` (any un-sized caller) ⇒ no sizing session is ever made —
        # byte-identical to the pre-sizing build.
        self.__sizing_key = sizing_key
        # Whether the tick sizing session may re-cut this pipe mid-run: None
        # defers to the class default (SECTIONABLE_CLASSES); False vetoes
        # (budget has out-of-pipeline consumers — contract-bearing agg pipes);
        # True admits a pipe from an excluded class (agg pass-1).
        self.__sectionable = sectionable

        self.__mask_map = {}
        cols = list(self.__input_cols.keys())
        for k, v in mask_map.items():
            assert v in cols, "mask col not passed in"
            self.__mask_map[cols.index(k)] = cols.index(v)

        column_boundaries = []

        # Re-cut state for the tick sizing session: only uniform-geometry
        # pipes (no ``div``) can be re-cut mid-run — the re-split needs the
        # shared row-boundary list and the per-row byte width saved here.
        # ``None``/0 marks a pipe that cannot.
        self.__uniform_geometry = False
        self.__boundaries = None
        self.__elem_size = 0
        self.__col_size = 0
        # The tick sizing session (``_exchange_ticks``), its per-tick arena
        # peaks, and each chunk's outputs' stored bytes times their later
        # crossings (``_charge``) by the chunk computing (``__computing``)
        # and the chunk packing (``__packing``).
        self.__sizing = None
        self.__peaks = None
        self.__chunk_cross = {}
        self.__computing = -1
        self.__packing = -1
        # The tick cutter (``IO/pipeline/tick_cutter.py``): ticks are whole
        # quanta of the records every input column shares (``__points``,
        # ``__quantum`` rows), and a tick that reads whole records selects
        # them without a trim.
        from IO.codec import region_producer
        self.__points = None
        self.__quantum = 1
        # Under a per-pipe sizing search, the class floor: a budget whose
        # whole quanta fall under it is rounded up a quantum
        # (``sizing_controller.whole_quanta``), as the search's own budgets
        # are.
        self.__floor_b = 0
        if sizing_key is not None and has_tick_sizing():
            import sizing_resolver
            self.__floor_b = sizing_resolver.bounds(sizing_key[0])[0]

        if div:
            self.stages = len(div[0]) + 2
            column_boundaries = [d + [d[-1], d[-1]] for d in div]
        else:
            col_size = {s.col_length() for s in input_columns.values()}
            assert len(col_size) == 1, "Column size is not equal for parititoning"
            col_size = next(iter(col_size))

            elem_size = sum(t.orig_elem_size() * t.col_width() for t in input_columns.values())
            cols = list(input_columns.values())
            bounds = tick_cutter.common_bounds(cols)
            if bounds is not None:
                piece = tick_cutter.input_record_rows(cols)
                # The masks the pipe writes (1 byte per row) start a
                # record every quantum of their own.
                if any(t == torch.bool for t in output_columns_type):
                    piece_out = region_producer.record_rows(1)
                else:
                    piece_out = 1
                self.__quantum = max(piece, piece_out)
                self.__points = tick_cutter.quantum_points(
                    bounds, piece, self.__quantum)
            ticks = tick_cutter.cut(self.__points, 0, col_size, elem_size,
                                    chunk_size, self.__floor_b)
            num_chunks = len(ticks)
            self.stages = num_chunks + 2  # Prologue and epilogue stages
            boundaries = ticks.ends.tolist() + [col_size, col_size]
            column_boundaries = [boundaries] * len(input_columns)
            self.__uniform_geometry = True
            self.__boundaries = boundaries
            self.__elem_size = elem_size
            self.__col_size = col_size

        # Per-tick built budget (the sizing session's sample attribution):
        # constant until a re-cut (``_resection``) changes the tail.
        self.__tick_budget_b = [chunk_size] * self.stages

        # Each tick's input slices; per tick, whether it reads whole records
        # of every column (``tick_whole``; False for a tick cut by rows).
        self.__in_host = [[] for _ in range(self.stages)]
        self.tick_whole = [True] * self.stages
        for j, col in enumerate(input_columns.values()):
            bounds_j = column_boundaries[j][:self.stages]
            self._slice_inputs(col, [0] + bounds_j[:-1], bounds_j, 0)
        message_logger().debug(lambda: (
            f'[ticks] {name}: {self.stages - 2} ticks, '
            f'{sum(not w for w in self.tick_whole[:self.stages - 2])} not '
            f'whole records'))

        # The pipe's decision menu (published once here — one-directional,
        # no reply) + the per-tick observer.  The menu is built only when an
        # observer will exist: explicit observer, else the process-wide
        # per-pipe factory (adaptive_driver).  With neither, menu and
        # emission sites are dead code.  Runs after the geometry above
        # because the menu publishes the tick count (``total_ticks``, the
        # search's probe-budget input).
        self.__decision_menu = None
        if self.__requested_tick_observer is not None:
            self.__decision_menu = self._build_decision_menu()
            self.__tick_observer = self.__requested_tick_observer
        elif (has_tick_observer_factory()
                and self.stages >= tick_observer_min_ticks()):
            # Pipes below the factory's declared tick floor skip the menu
            # build wholesale — their driver would be budget-inert anyway,
            # and the menu's byte accounting is real host work on big-column
            # pipes.
            self.__decision_menu = self._build_decision_menu()
            self.__tick_observer = make_tick_observer_for(
                name, self.__decision_menu)
        else:
            self.__tick_observer = None

        # Event-ordered ticks (``_event_ticks``): the host issues tick i+1
        # while the GPU computes tick i, wherever nothing reads a quiet
        # tick, with or without the granule cache.  A pipe with no tick
        # observer runs every tick ordered by events; one with an observer
        # (a compression search's) runs the tick barrier until the observer
        # settles (``_searches_settled``), then events.  An explicit
        # observer keeps the barrier throughout.
        self.__serial = SERIAL_TICKS
        self.__events_ok = (not self.__serial
                            and self.__requested_tick_observer is None)
        # Whether the ticks run on events: from the first tick for a pipe
        # without an observer, from the hand-over for one with one.
        self.__events = (self.__events_ok
                         and self.__tick_observer is None)
        self.__count_ring = None
        self.__count_np = None
        if self.__events_ok:
            self.__cev_ring += [_ComputeEvents() for _ in range(3)]
            # H_i (after tick i's input issue, on the h2d stream) and T_i
            # (the tick event, on the compute stream), by tick mod 3.
            self.__h_events = [torch.cuda.Event() for _ in range(3)]
            self.__t_events = [torch.cuda.Event() for _ in range(3)]
            if self.__mask_map:
                # The mask counts, by tick mod 3 and input column, stored
                # from the GPU over UVA at the input issue and read a tick
                # later.
                self.__count_ring = torch.zeros(
                    (3, len(self.__input_col_keys)), dtype=torch.int64,
                    pin_memory=True)
                self.__count_np = self.__count_ring.numpy()

        # Accumulate per-slot output bytes only when an observer could
        # actually consume them (write decision ranges published) — zero
        # cost with the search off or read-only.
        self.__track_out_bytes = (
            self.__tick_observer is not None
            and self.__decision_menu is not None
            and any(c.is_decision for c in self.__decision_menu.outputs))

        # --- Static decisions (once per column, not per chunk) ---
        self.__col_decisions = self._decide_column_types()

        # Live per-column input codecs.  Initialized to the static
        # resolution baked into the column decisions; observer-returned pick
        # vectors (validated against the menu) overwrite the *direct* columns'
        # entries mid-run, and the JIT rebuild below re-selects reps from
        # them.  Zero-copy columns keep their structural ``None`` — a pick on
        # such a column is legal but moot, so it is never stored.
        self.__live_input_algs = [d.algorithm for d in self.__col_decisions]

        # --- Pre-allocate all tick stages ---
        self._build_tick_stages()

        # Per-tick record of the pick state each tick's in-stages were built
        # from (the JIT-rebuild skip test + the debug epoch surface).
        built = self._effective_pick_state()
        self.__stage_built_algs = [built] * self.stages
        # Debug epoch asserts (TQP_PICK_DEBUG): at launch, a tick's in-stages
        # must match the pick state recorded at its t+2 rebuild horizon.
        self.__pick_debug = gates.PICK_DEBUG
        self.__debug_expected = {}

    def _slice_inputs(self, col, starts, ends, first_tick):
        """Append column ``col``'s slices ``[starts[t], ends[t])`` to ticks
        ``first_tick + t`` (``tick_cutter.slice_column``), noting the ticks
        that do not read whole records."""
        slices, exact = tick_cutter.slice_column(
            col, np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64))
        for t, (sl, ok) in enumerate(zip(slices, exact.tolist())):
            i = first_tick + t
            self.__in_host[i].append(sl)
            if not ok and ends[t] > starts[t]:
                self.tick_whole[i] = False

    def _quantum_b(self, budget_b: int) -> int:
        """The raw bytes of the pipe's largest piece (one quantum where the
        inputs' boundaries nest; more where they do not) when its ticks are
        whole pieces at ``budget_b`` (``tick_cutter.cut``), else 0 (cut by
        rows): the unit in which a budget changes its ticks."""
        if self.__points is None or self.__elem_size <= 0:
            return 0
        piece_b = tick_cutter.max_piece(self.__points) * self.__elem_size
        return piece_b if piece_b <= budget_b else 0

    def _build_decision_menu(self) -> DecisionMenu:
        """Build this pipe's decision menu.

        Input ranges are enumerated from the reps each column ACTUALLY holds
        (ground truth — a choice is offered iff taking it is pure rep
        selection): a scan column carrying both reps publishes a two-entry
        range; an operator-output-fed column carries its producer's single
        rep and publishes a singleton.  Output ranges come from the latched
        static resolution: RAW-resolved (untyped / RAW-policy)
        publishes a singleton, codec-resolved publishes ``(RAW, candidate)``
        — the resolved codec IS the typed-policy candidate, so no applier
        round-trip is needed and the escape-hatch construction paths
        (pre-resolved alg lists) publish identically.  Byte counts ride the
        input entries (the benefit-ranking input, per-rep stored bytes).
        """
        inputs = []
        for key in self.__input_col_keys:
            ccr = self.__input_cols[key]
            # Codec identification is O(1) per rep: a rep is single-codec by
            # construction, so its first record speaks for all.
            reps = []                     # (algo, rep, compressed?)
            for rep in ccr.reps:
                name = rep.leading_codec_name()
                if name is None:
                    reps.append(("RAW", rep, False))
                else:
                    reps.append((name, rep, True))
            raw_b = ccr.reps[0].data_original() if ccr.reps else 0
            if len({a for a, _, _ in reps}) <= 1:
                # Singleton range: nothing to rank, so skip ALL per-record
                # byte walks (they were a measured +3-35 % wall on
                # join/rearrange-heavy queries with zero searches run).
                rep_bytes = {a: (0 if c else raw_b) for a, _, c in reps}
            else:
                # Two-entry (decision) column: the candidate's ACTUAL stored
                # bytes are the benefit-ranking input — the one
                # remaining O(records) walk, on scan columns only.
                rep_bytes = {}
                for algo, rep, compressed in reps:
                    rep_bytes.setdefault(
                        algo, rep.data_to_transfer() if compressed else raw_b)
            inputs.append(build_input_choices(rep_bytes, raw_b))
        # Non-base slots of a slot group publish LOCKED singletons (their
        # resolved codec with no alternative): a vector that desyncs a
        # B-partition group is inexpressible by construction, and
        # actuation propagates the base slot's pick to the whole group.
        outputs = tuple(
            build_output_choices(None if _is_raw_algo(a) else a,
                                 locked=(self.__out_slot_group[j] != j),
                                 reads=self.__output_expected_reads[j])
            for j, a in enumerate(self.__output_compression))
        return DecisionMenu(pipe_name=self.name, inputs=tuple(inputs),
                            outputs=outputs, total_ticks=self.stages)

    def _decide_column_types(self):
        """Decide transfer type once per column using the full column representation."""
        cols = list(self.__input_cols.values())
        decisions = []
        for col_idx, data_col in enumerate(cols):
            mask_col = cols[self.__mask_map[col_idx]] if col_idx in self.__mask_map else None
            decision = decide_column_type(
                data_col, mask_col, self.__input_compression_algs[col_idx])
            decisions.append(decision)
        return decisions

    def _pin_granules(self, in_stages):
        """Pin, in the granule cache, every cached granule ``in_stages``
        read (``NativeGranuleCache.replace_consume_pins``): a DirectIn's column,
        a zero-copy stage's mask and source."""
        self.__granule_cache.replace_consume_pins(
            gids=self._read_granules(in_stages))

    def _read_granules(self, in_stages):
        """The cached granules ``in_stages`` read (``_pin_granules``)."""
        from IO.transfer.tick_layout import cached_read_granules
        gids = []
        seen = None
        for stage in in_stages:
            if isinstance(stage, ZeroCopyMaskIn):
                cols = (stage.mask_col, stage.src_col)
            else:
                layout = getattr(stage, '_compiled_tick_layout', None)
                if layout is not None and _cache_cfg.PLAN_AT_COMPILE:
                    # The layout compiled the stage's cached granules
                    # (``tick_layout._CachedRuns``; ``()`` for a stage that
                    # reads nothing).
                    binding = layout.per_stage_bindings[stage._layout_index]
                    tc = layout.tick_cached
                    if tc is not None and getattr(binding, 'in_tick', False):
                        # The tick's fully cached stages' granules, once.
                        if seen is None:
                            seen = set()
                        if id(layout) not in seen:
                            seen.add(id(layout))
                            gids.append(tc.gids)
                        continue
                    plan = getattr(binding, 'cached', None)
                    if plan is not None:
                        gids.append(plan.gids)
                    continue
                cols = (stage.src_col,)
            for col in cols:
                got = cached_read_granules(col)
                if got is not None:
                    gids.append(got)
        return ((gids[0] if len(gids) == 1 else np.concatenate(gids))
                if gids else _NO_GRANULES)

    def _materialize_mask_counts(self, tick_stage, h2d_stream):
        """Read GPU mask_count scalars as Python ints.

        Called right after exchange.sync() so h2d_stream is fully drained.
        Performs the .item() D2H copy on h2d_stream (not the legacy default
        stream 0) so that no implicit ordering is created between the
        legacy default stream and subsequently launched non-blocking streams.
        """
        for pc in tick_stage.pre_comp:
            in_s = pc.in_stage_ref
            if in_s.mask_count is not None:
                assert isinstance(in_s.mask_count, torch.Tensor), \
                    "mask_count already materialized — double call?"
                with torch.cuda.stream(h2d_stream):
                    in_s.mask_count = int(in_s.mask_count.item())

    def _ring_mask_counts(self, tick, h2d_stream):
        """Event-ordered ticks, at the input issue of ``tick``: store each
        in-stage's mask count into the count ring's slot for ``tick`` over
        UVA (``tqpcompress.uva_store``), behind the sum on ``h2d_stream``
        (``H`` of the tick follows).  A copy would queue on the copy engine
        behind the copies home in flight, and ``H`` with it."""
        if self.__count_ring is None:
            return
        row = self.__count_ring[tick % 3]
        with torch.cuda.stream(h2d_stream):
            for j, in_s in enumerate(self.__tick_stages[tick].in_stage):
                if isinstance(in_s.mask_count, torch.Tensor):
                    tqpcompress.uva_store(in_s.mask_count.view(1),
                                          row[j:j + 1])

    def _read_mask_counts(self, tick):
        """Event-ordered ticks: the mask counts of the chunk ``tick``
        computes (issued at ``tick - 1``), from the count ring once that
        issue's ``H`` has fired."""
        in_stages = [pc.in_stage_ref for pc in self.__tick_stages[tick].pre_comp
                     if isinstance(pc.in_stage_ref.mask_count, torch.Tensor)]
        if not in_stages:
            return
        slot = (tick - 1) % 3
        self.__h_events[slot].synchronize()
        row = self.__count_np[slot]
        for pc in self.__tick_stages[tick].pre_comp:
            in_s = pc.in_stage_ref
            if isinstance(in_s.mask_count, torch.Tensor):
                in_s.mask_count = int(row[pc.col_idx])

    def _create_in_stages(self, tick):
        """One InStage per input column for the given tick (column order)."""
        stages = [self._build_in_stage(tick, j, ccr)
                  for j, ccr in enumerate(self.__in_host[tick])]
        self._mark_mask_sum_stages(stages)
        return stages

    def _build_in_stage(self, tick, j, ccr):
        """Construct one column's InStage based on its static stage_type.

        Empty-chunk slices (prologue / epilogue) take a trivial DirectIn
        so ``raw_numel() == 0`` short-circuits compute.
        """
        if ccr.col_length() == 0:
            return DirectIn(ccr.reps[0])
        decision = self.__col_decisions[j]
        if decision.stage_type == "direct":
            return self._build_direct_in(j, ccr)
        if decision.stage_type == "zero_copy_mask":
            return self._build_zero_copy_mask_in(j, ccr, tick)
        raise ValueError(
            f"VortexPipeline {self.name!r} col {j}: unsupported stage_type "
            f"{decision.stage_type!r}")

    def _build_direct_in(self, j, ccr):
        try:
            rep = find_rep_by_algorithm(ccr, self.__live_input_algs[j],
                                        strict=self.__strict)
        except RuntimeError as e:
            raise RuntimeError(
                f"VortexPipeline {self.name!r} col_key={self.__input_col_keys[j]!r} j={j}: {e}"
            ) from None
        return DirectIn(rep)

    def _build_zero_copy_mask_in(self, j, ccr, tick):
        cpu_data = _find_all_cpu_rep(ccr)
        mask_idx = self.__mask_map[j]
        mask_ccr = self.__in_host[tick][mask_idx]
        mask_rep = find_rep_by_algorithm(mask_ccr, "RAW", strict=False)
        assert mask_rep is not None, f"mask column {mask_idx} has no RAW rep"
        return ZeroCopyMaskIn(cpu_data, mask_rep, cpu_data.col_width())

    def _mark_mask_sum_stages(self, stages):
        """Flag InStages whose data feeds another column's mask so
        ``DirectIn.phase_finalize`` queues a sum on h2d_stream — the
        popcount is then ready when the operator reads it.
        """
        for j in set(self.__mask_map.values()):
            stages[j]._compute_mask_sum = True

    def _build_tick_stages(self):
        """Pre-allocate every TickStages with forward refs wired up."""
        # Append per-tick (not a list comprehension) so ``_create_pre_comps``
        # can read ``self.__tick_stages[tick - 1]`` while building ``tick``.
        self.__tick_stages = []
        for t in range(self.stages):
            self.__tick_stages.append(self._build_one_tick(t))

    def _build_one_tick(self, tick):
        """One tick's stages: its in-stages, and from tick 1 the pre-comp
        stages computing the chunk transferred at tick-1."""
        tick_stages = TickStages(
            in_stage=self._create_in_stages(tick),
            pre_comp=self._create_pre_comps(tick) if tick >= 1 else [],
        )
        # Immutable DirectIn geometry belongs to the tick and is
        # compiled outside NaiveExchange.launch.  Exchange groups by exact
        # stage class, so this filtered order is the launch order too.
        direct_stages = [
            stage for stage in tick_stages.in_stage
            if isinstance(stage, DirectIn)
        ]
        if direct_stages:
            DirectIn.install_compiled_tick_layout(direct_stages)
        return tick_stages

    def _built(self, tick):
        """``ts[tick]``, built from the live pick state if a re-cut
        (``_resection``) left it unbuilt, with every unbuilt tick before it,
        in order: a tick's pre-comps point at the previous tick's
        in-stages."""
        ts = self.__tick_stages
        if ts[tick] is None:
            first = tick
            while ts[first - 1] is None:
                first -= 1
            for t in range(first, tick + 1):
                ts[t] = self._build_one_tick(t)
                self.__stage_built_algs[t] = self._effective_pick_state()
        return ts[tick]

    def _chunk_b(self, chunk):
        """The input bytes of chunk ``chunk`` (uniform geometry)."""
        lo = self.__boundaries[chunk - 1] if chunk else 0
        return (self.__boundaries[chunk] - lo) * self.__elem_size

    def _create_pre_comps(self, tick):
        prev_in = self.__tick_stages[tick - 1].in_stage
        out = []
        for j, decision in enumerate(self.__col_decisions):
            is_zc = decision.stage_type != "direct"
            mask_target = self.__mask_map.get(j) if not is_zc else None
            out.append(PreCompStage(j, prev_in[j], is_zc, mask_target))
        return out

    # ------------------------------------------------------------------ #
    # Read-leg JIT actuation (the t+2 in-stage rebuild)
    # ------------------------------------------------------------------ #

    def _effective_pick_state(self):
        """The build-relevant projection of the live input codecs: direct
        columns carry their current codec (RAW normalized), zero-copy columns
        carry ``None`` (structure preempts the pick).  Two equal
        projections build byte-identical in-stages, which is what makes the
        rebuild skip test below sound."""
        return tuple(
            ("RAW" if _is_raw_algo(self.__live_input_algs[j]) else
             self.__live_input_algs[j])
            if d.stage_type == "direct" else None
            for j, d in enumerate(self.__col_decisions))

    def _jit_rebuild(self, target_tick):
        """(Re)build ``ts[target_tick].in_stage`` just-in-time from the live
        pick state.

        Called once per tick from ``_emit_tick`` with ``target_tick =
        last_tick + 2`` — i.e. at the rebuild point right after sample *t*
        lands and before the iteration that pre-plans (cache-pin walk) or
        launches the rebuilt tick, so the t+2 rule holds by construction.
        Touches ONLY the tick's in-stages (plus the next tick's pre-comp
        back-references, which point at them).  A tick whose in-stages already
        match the live pick state is left untouched — with an inert observer
        (all-``None`` replies) no stage is ever rebuilt and the run is the
        static build.
        """
        if (target_tick >= self.stages
                or self.__tick_stages[target_tick] is None):
            return       # not built yet: it will be, from the live picks
        want = self._effective_pick_state()
        if want == self.__stage_built_algs[target_tick]:
            return
        new_in = self._create_in_stages(target_tick)
        # Compiled-input-plan invariant: rebuilt DirectIns must
        # carry a compiled tick layout too, or every rebuilt tick falls back
        # to compiling inside the launch-critical path.
        rebuilt_direct = [s for s in new_in if isinstance(s, DirectIn)]
        if rebuilt_direct:
            DirectIn.install_compiled_tick_layout(rebuilt_direct)
        self.__tick_stages[target_tick].in_stage = new_in
        self.__stage_built_algs[target_tick] = want
        # PreCompStage[t+1] holds per-column references to tick t's
        # in-stages (the compute of chunk t runs at tick t+1) — re-point
        # them at the rebuilt objects or the compute would read the stale
        # (never-launched) stages.
        if (target_tick + 1 < self.stages
                and self.__tick_stages[target_tick + 1] is not None):
            for pc in self.__tick_stages[target_tick + 1].pre_comp:
                pc.in_stage_ref = new_in[pc.col_idx]

    # ------------------------------------------------------------------ #
    # Write-leg boundary actuation (read-and-write probe windows)
    # ------------------------------------------------------------------ #

    def _actuate_write_picks(self, out_picks, seal, resource_monitor):
        """One probe-window boundary (``read-and-write``).

        Runs inside ``_emit_tick`` — i.e. after tick *t*'s compute sync and
        collect, before tick *t+1* compresses its outputs — so write
        actuation is **t+1** and race-free: tick *t*'s outputs were already
        compressed under the outgoing codecs.  Steps: validate the output
        vector against the menu, propagate each slot group's BASE pick to
        the whole group (partition uniformity by construction), and
        retarget the live output codecs.  Every tick writes regions of its
        own, so a ``seal``-only call (``outputs`` matching the current
        state, a null A-A boundary) changes nothing.
        """
        menu = self.__decision_menu
        n_out = len(self.__output_compression)
        eff = None
        if out_picks is not None:
            out_picks = validate_pick_vector(
                menu.outputs, out_picks,
                what=f"output picks for {self.name!r}")
            eff = ["RAW" if _is_raw_algo(out_picks[self.__out_slot_group[j]])
                   else out_picks[self.__out_slot_group[j]]
                   for j in range(n_out)]
        cur = ["RAW" if _is_raw_algo(a) else a
               for a in self.__output_compression]
        changed = ([j for j in range(n_out) if eff[j] != cur[j]]
                   if eff is not None else [])
        if not changed and not seal:
            return

        for j in changed:
            self.__output_compression[j] = eff[j]
        if changed:
            self.__write_flipped = True   # flip hygiene: codec_epoch 'mixed'
            message_logger().info(
                "[write_flip] %s: tick %d slots %s -> %s (window boundary)",
                self.name, resource_monitor.last_tick, changed,
                [eff[j] for j in changed])

    def _debug_assert_stage_current(self, tick):
        """Debug epoch assert (env ``TQP_PICK_DEBUG``): at launch, tick
        *t*'s in-stages must match the pick state recorded at its rebuild
        horizon (sample *t−2*'s emit).  Catches a refactor that moves the
        rebuild point after the pre-plan/launch of its target."""
        expected = self.__debug_expected.get(tick)
        if expected is not None:
            assert self.__stage_built_algs[tick] == expected, (
                f"VortexPipeline {self.name!r}: tick {tick} launched with "
                f"in-stages built from {self.__stage_built_algs[tick]} but "
                f"the t+2 rebuild horizon recorded {expected} (JIT rebuild "
                f"sequencing violation)")

    def _execute_pre_comp_and_operator(self, tick, pack=False,
                                       pack_after=False):
        """Decompress/zero-copy, apply masks, run operator. Returns list of result tensors.

        With ``pack``, first pack the previous tick's regions (the region
        producer), so their copies home start before the operator; with
        ``pack_after`` (event-ordered ticks), pack them once the operator
        is launched, so it runs while the host waits for their sizes."""
        pre_comps = tick.pre_comp

        self.__cev.start.record(self.__compute_stream)
        self.__cev.packed = False
        if pack and self.__producer.has_pending():
            self._pack_regions()
            if self.__serial:
                self.__compute_stream.synchronize()
                writer_stream().synchronize()

        if all(pc.in_stage_ref.raw_numel() == 0 for pc in pre_comps):
            res = [torch.tensor([], device='cuda') for _ in self.__output_columns_type]
        else:
            res = self._run_operator(pre_comps)
        if pack_after and self.__producer.has_pending():
            self._pack_regions()
        return res

    def _run_operator(self, pre_comps):
        # Phase 1: coalesce direct-path decompress into one nvCOMP batch,
        # then run zero-copy kernels (per-stage, not batchable here).
        direct_stages = [pc.in_stage_ref for pc in pre_comps
                         if not pc.is_zero_copy and
                         getattr(pc.in_stage_ref, '_has_compression', False)]
        if direct_stages:
            self.__batched_decompress.decompress_all(direct_stages)
        for pc in pre_comps:
            pc.execute_zero_copy()

        # Phase 1b: extract raw tensors (None for zero-copy columns)
        all_raw = [pc.extract_raw() for pc in pre_comps]

        # Phase 1c: collect mask counts (pre-computed on DirectIn, materialized after sync).
        mask_counts = {}
        for pc in pre_comps:
            if not pc.is_zero_copy and pc.mask_map_target is not None:
                mt = pc.mask_map_target
                if mt not in mask_counts:
                    mask_counts[mt] = pre_comps[mt].in_stage_ref.mask_count

        # Phase 2: apply mask via compact_masked (sync-free), unless the
        # operator folds the masks
        for pc in pre_comps:
            pc.apply_mask(all_raw, mask_counts, fold=self.__fold_mask)

        # Build input_dict (filtering mask columns same as old safe_apply_op)
        input_dict = {}
        for (col_name, old_var), (i, pc) in zip(self.__input_cols.items(), enumerate(pre_comps)):
            if i not in self.__mask_map.values():
                # Not a mask column — include in input_dict
                input_dict[col_name] = pc.ready_tensor
            else:
                # IS a mask column
                if self.__pass_mask:
                    input_dict[col_name] = pc.ready_tensor

        if self.__fold_mask:
            counts = {col_name: pre_comps[i].in_stage_ref.mask_count
                      for i, col_name in enumerate(self.__input_cols)
                      if i in self.__mask_map.values()}
            res = self.__operator(input_dict, counts)
        else:
            res = self.__operator(input_dict)
        if isinstance(res, torch.Tensor) and res.ndim == 0:
            res = [res]

        # Free consumed inputs immediately to reclaim GPU memory for the
        # outputs' compress.
        for pc in pre_comps:
            pc.in_stage_ref.gpu_col = None
            pc.ready_tensor = None

        return list(res)

    def _pack_regions(self):
        """Pack the region producer's pending records on the compute stream,
        bracketed by the pack events (the tick's ``compress_out`` adds their
        span)."""
        self.__cev.pack_start.record(self.__compute_stream)
        stored = self.__producer.pack()
        self.__cev.pack_end.record(self.__compute_stream)
        self.__cev.packed = True
        if self.__sizing is not None:
            self._charge(self.__packing, stored)
        if self.__track_out_bytes:
            for base, nbytes in stored.items():
                self.__out_stored_bytes_by_slot[base] += nbytes

    def _execute_post_comp(self, result_buffer):
        """Hand the operator's outputs to the region producer: compressed
        now, packed and shipped next tick.  A ``PartitionedTensor`` is one
        instance's B buckets."""
        self.__cev.cprs_start.record(self.__compute_stream)
        feed = []  # (out_idx_base, tensor, counts, algo)
        track_bytes = self.__track_out_bytes
        out_idx = 0
        for j, item in enumerate(result_buffer):
            if isinstance(item, PartitionedTensor):
                B = len(item.partition_row_counts)
                algo = self.__output_compression[out_idx]
                if track_bytes:
                    t = item.tensor
                    rb = (t.shape[1] if t.dim() >= 2 else 1) * t.element_size()
                    self.__out_raw_bytes_by_slot[out_idx] += (
                        int(sum(item.partition_row_counts)) * rb)
                # Sanity: all B slots of this fat tensor must agree on algo
                # (they're partitions of one logical col).
                for k in range(1, B):
                    assert self.__output_compression[out_idx + k] == algo, (
                        f"PartitionedTensor at out_idx={out_idx} expects "
                        f"uniform compression across {B} slots; saw "
                        f"{self.__output_compression[out_idx:out_idx + B]}"
                    )
                feed.append((out_idx, item.tensor.contiguous(),
                             item.partition_row_counts, algo))
                out_idx += B
            else:
                algo = self.__output_compression[out_idx]
                item_c = item.contiguous()
                if track_bytes and item_c.shape[0] > 0:
                    rb = ((item_c.shape[1] if item_c.dim() >= 2 else 1)
                          * item_c.element_size())
                    self.__out_raw_bytes_by_slot[out_idx] += (
                        int(item_c.shape[0]) * rb)
                # All-empty ticks emit n_out plain 0-row tensors instead of
                # ncols PartitionedTensors (operator short-circuit), so the
                # out_idx→bucket mapping differs from a data tick: a 0-row
                # tensor is never fed (it would mis-key the column's B).
                if item_c.shape[0] > 0:
                    feed.append((out_idx, item_c, [item_c.shape[0]], algo))
                out_idx += 1
            result_buffer[j] = None  # Free raw operator output

        if feed:
            stored = self.__producer.compress(feed)
            if self.__sizing is not None:
                self._charge(self.__computing, stored)
            if track_bytes:
                for base, nbytes in stored.items():
                    self.__out_stored_bytes_by_slot[base] += nbytes
        self.__cev.cprs_end.record(self.__compute_stream)

    def _charge(self, chunk, stored):
        """Add to chunk ``chunk``'s readback the stored bytes of its
        outputs (``stored``, by base slot: the RAW ones as the region
        producer ships them, the compressed ones as it packs them) times
        their later crossings: home, and each consumer read (the dataflow's
        expected reads, at least one)."""
        reads = self.__output_expected_reads
        self.__chunk_cross[chunk] = self.__chunk_cross.get(chunk, 0) + sum(
            int(n) * (max(1, reads[b]) + 1) for b, n in stored.items())

    def do_exchange(self):
        """Run the pipe's ticks (``_exchange_ticks``).  If they fail with a
        region producer's copies home still in flight, wait for those copies
        before the failure leaves the pipe: the buffers they read and the
        homes they write are freed once it has (the region producer's
        buffers stay referenced until their copy's event has fired)."""
        self.__exchange = None
        try:
            return self._exchange_ticks()
        except BaseException:
            if self.__events:
                self._abort_event_ticks()
            self.__producer.abort()
            raise

    def _abort_event_ticks(self) -> None:
        """Event-ordered ticks failed: wait for the compute and exchange
        streams before the hold lists go, since the GPU may still read the
        buffers they keep."""
        try:
            self.__compute_stream.synchronize()
            if self.__exchange is not None:
                self.__exchange.sync()
        finally:
            for t in self.__tick_stages:
                if t is not None:
                    t.holds = []

    def _exchange_ticks(self):
        gcache = self.__granule_cache
        if gcache is not None:
            gcache.advance(self.__node)
        perf_logger().start(f"{self.name}")
        # Bracket this pipeline's peak over the arena's operator space (the
        # resident operator allocations + this op's transient buffers; never
        # the cache region).  Independent of the global reset_peak() that
        # baselines the OOM-diagnostic peak.
        cmp.reset_window_peak()
        # The learned jump (``sizing_controller.TickJumpSession``): a sized
        # pipe of uniform geometry that may be re-cut mid-run measures its
        # tick time and memory on its event-ordered ticks and re-cuts itself
        # to the sizes they point at.  Its per-tick arena brackets keep the
        # whole-pipe high-water mark for the perf record (``final_peak``).
        session = None
        if (self.__events_ok and self.__sizing_key is not None
                and self.__uniform_geometry and self.__elem_size > 0):
            _pc, _inst = self.__sizing_key
            session = make_tick_sizing_session(
                self.name, _pc, str(_inst), int(self.__chunk_size_b),
                self.stages - 2, self.__col_size * self.__elem_size,
                sectionable=self.__sectionable,
                quantum_b=self._quantum_b(int(self.__chunk_size_b)))
        self.__sizing = session
        self.__peaks = _WindowPeakTracker() if session is not None else None
        self.__chunk_cross = {}

        exchange = data_exchange.exchange()
        self.__exchange = exchange
        resource_monitor = ResourceMonitor()
        if self.__track_out_bytes:
            # Live references (write-leg ranking + penalty inputs): the
            # observer sees the accumulators' current values at each tick.
            resource_monitor.out_slot_raw_bytes = self.__out_raw_bytes_by_slot
            resource_monitor.out_slot_stored_bytes = (
                self.__out_stored_bytes_by_slot)

        if self.__events:
            self._event_ticks(exchange, resource_monitor)
        else:
            start = self._barrier_ticks(exchange, resource_monitor, gcache)
            if start is not None:
                self._event_ticks(exchange, resource_monitor, start)

        # The final protected input has drained at the last exchange.sync().
        # Release it before the drain's admissions and before handing the
        # cache to the next pipeline.
        if gcache is not None:
            # The last reads have ended: count them, before the drain's
            # pack admits.
            gcache.replace_consume_pins()
            gcache.settle_reads()

        self._drain_regions()
        if gcache is not None:
            # The drain's pack holds no slot address past its return: its
            # granules may move from here on, so an allocation between
            # pipes can borrow their blocks too.
            gcache.advance_tick()

        if session is not None:
            session.close()
        window_peak_b = (self.__peaks.final_peak() if self.__peaks is not None
                         else cmp.get_window_peak_bytes())
        if gates.EXPERIMENT:
            perf_logger().record(f"{self.name}", peak_gpu_bytes=window_peak_b)
        perf_logger().stop(f"{self.name}")

    def _tick_collect(self, resource_monitor, lanes, cev):
        """One tick's telemetry once its events have fired: its input
        lanes ``(h2d, d2d)`` ms and the home writer's lane, its compute
        spans ``cev`` (``None`` on the transfer-only prologue), then
        ``_emit_tick``."""
        h2d_ms, d2d_ms = lanes
        # The home writer's background copies are this pipe's d2h lane:
        # each tick's copies once they have ended, normally one tick later.
        home_ms, _ = self.__producer.collect_lane()
        resource_monitor.collect_transfer(
            TransferTimes(h2d_ms, d2d_ms, home_ms))
        if cev is not None:
            compress_out = cev.cprs_start.elapsed_time(cev.cprs_end)
            if cev.packed:
                # The region pack runs outside the output-compress events;
                # it is output compression.
                compress_out += cev.pack_start.elapsed_time(cev.pack_end)
            resource_monitor.collect_compute(ComputeTimes(
                compute=cev.start.elapsed_time(cev.end),
                compress_out=compress_out))
        self._emit_tick(resource_monitor)

    def _emit_tick(self, resource_monitor):
        """Deliver the tick's feedback view — this pipe's own monitor,
        whose last-tick fields ``_tick_collect`` just set — synchronously
        on the host thread, before the next iteration's pre-plan.  The view
        is ephemeral: valid only inside the callback.  A non-None return is
        a full input-pick vector: boundary-validated, applied to the live
        pick state, and actuated by the JIT rebuild of
        ``ts[t+2].in_stage`` (the rebuild runs before the next iteration's
        pre-plan/launch of that tick by construction).  The rebuild call
        runs every tick so a vector applied at sample *t* keeps building
        ticks > t+2 until superseded; with all-None replies it never
        rebuilds anything."""
        observer = self.__tick_observer
        if observer is None:
            return
        ret = observer(resource_monitor)
        picks, out_picks, seal = None, None, False
        if isinstance(ret, PickPair):
            # read-and-write: the PickPair return.  Output picks actuate at
            # t+1; input picks stay on the t+2 rule.
            picks, out_picks, seal = ret.inputs, ret.outputs, ret.seal
        elif ret is not None:
            picks = ret
        if picks is not None:
            picks = validate_pick_vector(
                self.__decision_menu.inputs, picks,
                what=f"input picks for {self.name!r}")
            for j, pick in enumerate(picks):
                if self.__col_decisions[j].stage_type == "direct":
                    self.__live_input_algs[j] = pick
        if out_picks is not None or seal:
            self._actuate_write_picks(out_picks, seal, resource_monitor)
        target = resource_monitor.last_tick + 2
        self._jit_rebuild(target)
        if self.__pick_debug and target < self.stages:
            self.__debug_expected[target] = self._effective_pick_state()

    def _tick_wait(self, i, released):
        """Event-ordered tick ``i``, before its input is allocated: wait
        for ``T`` of tick ``i - 2`` (normally fired), then release
        ``ts[i - 3]``, whose chunk that tick computed: its hold list and
        intermediates (unless it is among the first ``released`` ticks,
        which the tick barrier already released).  Three ticks' inputs
        stay alive at most."""
        self.__t_events[(i - 2) % 3].synchronize()
        if i >= 3 and i - 3 >= released:
            with torch.cuda.stream(self.__compute_stream):
                self.__tick_stages[i - 3].clean_intermediates()

    def _searches_settled(self) -> bool:
        """Whether the pipe's tick observer, if any, has settled (it emits
        no more picks: ``settled``), so its remaining ticks can run on
        events."""
        if not self.__events_ok:
            return False
        observer = self.__tick_observer
        if observer is not None:
            settled = getattr(observer, "settled", None)
            if settled is None or not settled():
                return False
        return True

    def _hand_over_to_events(self, start):
        """The barrier has ended tick ``start - 1`` and the observer has
        settled: from ``start`` on the ticks run on the final pick state
        (each was built at its own t+2 horizon, which the event loop's late
        telemetry no longer meets).  Tick ``start`` is rebuilt now; a later
        tick built from other picks is dropped and built when the loop
        reaches it (``_built``).  The observer is dropped."""
        self._jit_rebuild(start)
        want = self._effective_pick_state()
        ts, built = self.__tick_stages, self.__stage_built_algs
        for t in range(start + 1, self.stages):
            if ts[t] is not None and built[t] != want:
                ts[t] = None
                built[t] = None
        if self.__pick_debug:
            for t in range(start, self.stages):
                self.__debug_expected[t] = want
        self.__tick_observer = None
        message_logger().debug(lambda: (
            f'[ticks] {self.name}: searches settled, events from tick '
            f'{start} of {self.stages - 2}'))

    def _observe_chunk(self, session, t, peak_b, remaining_b):
        """Chunk ``t - 1``, computed at tick ``t`` (collected, so its
        events have fired), for the tick sizing session: its tick time is
        the gap between its compute's end and the previous chunk's on the
        GPU timeline, plus its readback (``_charge``) at the probed PCIe
        rate; its transient's peak is the arena's while tick ``t`` ran.
        Returns the budget the session asks the rest of the pipe to be
        re-cut at, or None."""
        chunk = t - 1
        nbytes = self._chunk_b(chunk)
        if nbytes <= 0:
            return None      # the epilogue's empty chunk
        ring = self.__cev_ring
        q_ms = (ring[(t - 1) % 4].end.elapsed_time(ring[t % 4].end)
                + self.__chunk_cross.pop(chunk, 0) / _rm.PCIE_B_PER_MS)
        return session.observe(self.__tick_budget_b[chunk], nbytes, q_ms,
                               peak_b, remaining_b)

    def _event_ticks(self, exchange, resource_monitor, start=0):
        """The ticks ordered by events (``IO/pipeline/README.md``).  Tick i
        issues chunk i's input, computes chunk i-1 and packs chunk i-2's
        outputs after that operator is launched.  Two events per tick:
        ``H_i`` on the h2d stream after the input issue, and ``T_i`` on the
        compute stream after chunk i-1's compress and a wait on ``H_i``.
        The host waits on ``T`` two ticks old before allocating (step 2),
        reads the mask counts behind ``H`` one tick old, and reads tick
        i-2's telemetry at the end of tick i.  The pipe ends with one full
        wait.  With the granule cache, each tick first advances the cache's
        tick and pins the granules of chunks i and i-1 (below).

        ``start`` > 0 takes over from the tick barrier
        (``_barrier_ticks``), which ended tick ``start - 1`` with both
        streams synchronized, its telemetry collected and every tick before
        it released: chunk ``start - 1``'s mask counts are read from the
        GPU directly, and its granules are pinned with chunk ``start``'s.

        With a tick sizing session (``sizing_controller.TickJumpSession``),
        each chunk computed on events after the first goes to it when its
        tick is collected (``_observe_chunk``, the peak from the tick's
        arena bracket), and a re-cut it asks for applies from the next
        unissued tick (``_resection``)."""
        self.__events = True
        if start:
            self._hand_over_to_events(start)
        ts = self.__tick_stages
        cs = self.__compute_stream
        h2d = exchange.h2d_stream
        logs = [None] * 3
        session = self.__sizing
        peaks = self.__peaks
        peak_at = {}         # tick -> the arena's peak while it ran
        # The first tick whose chunk the session sees: the previous tick's
        # compute end must be on events too (the barrier's is not).
        first_seen = max(2, start + 1)
        state = {"collected": start, "released": max(0, start - 1),
                 "recut": None, "remaining": 0}

        def collect_upto(stop):
            while state["collected"] < stop:
                t = state["collected"]
                log = logs[t % 3]
                lanes = log.lane_ms()
                self._tick_collect(
                    resource_monitor,
                    (lanes['h2d'], lanes['d2d']),
                    self.__cev_ring[t % 4] if t >= 1 else None)
                log.release()
                logs[t % 3] = None
                state["collected"] = t + 1
                peak_b = peak_at.pop(t, 0)
                if session is not None and t >= first_seen:
                    got = self._observe_chunk(session, t, peak_b,
                                              state["remaining"])
                    if got is not None:
                        state["recut"] = got

        gcache = self.__granule_cache
        prev_gids = _NO_GRANULES
        if start:
            # Chunk start-1, issued under the barrier: its mask counts are
            # read now, with the h2d stream idle, and its granules are
            # pinned with chunk start's.
            self._materialize_mask_counts(self._built(start), h2d)
            if gcache is not None:
                prev_gids = self._read_granules(ts[start - 1].in_stage)
        i = start
        while i < self.stages:
            n = self.stages
            if peaks is not None:
                peaks.iter_begin()
            self._built(i)
            if gcache is not None:
                # With the granule cache: the reads tick i-1 bound settle
                # (``advance_tick``), then the granules of chunk i (its
                # copies run beside this tick's pack, and a lend before its
                # launch would move them) and of chunk i-1 (its compute,
                # launched below, reads them in place) are pinned before
                # the bind.  The reads of chunk i-2 and earlier need no
                # pin: every slot write from here on that could reuse their
                # slots is on the compute stream behind T_{i-2}, which
                # waited on their copies (H_{i-2}), or behind a lend's
                # device sync.
                gcache.advance_tick()
                gids = self._read_granules(ts[i].in_stage)
                gcache.replace_consume_pins(
                    gids=np.concatenate([prev_gids, gids]))
                prev_gids = gids
            # 1. bind chunk i's input (host only).
            bound = exchange.bind(ts[i].in_stage)
            # 2. wait for T two ticks old; release the chunk it computed.
            if i >= 2:
                self._tick_wait(i, state["released"])
                state["released"] = max(state["released"], i - 2)
            # 3. issue chunk i's input; its mask counts into the ring; H_i.
            if self.__pick_debug:
                self._debug_assert_stage_current(i)
            ts[i].holds = []
            # The issue's copies wait for the compute work queued so far
            # (chunk i-2's compute, chunk i-3's pack); chunk i-1's compute,
            # launched next, overlaps them.  Without it SF100 Q16
            # (aggregation 15 pass 1) faults with an illegal memory access
            # even with the allocator's force-drain guarded; the race is
            # between this issue and chunk i-2's compute, not yet pinned
            # down (``IO/pipeline/README.md``).
            h2d.wait_stream(cs)
            logs[i % 3] = exchange.launch(ts[i].in_stage, bound,
                                          holds=ts[i].holds)
            self._ring_mask_counts(i, h2d)
            self.__h_events[i % 3].record(h2d)
            if i >= 1:
                # 4. chunk i-1's mask counts, behind H_{i-1}.
                self._read_mask_counts(i)
                if i >= 2:
                    self.__producer.ticks_left = n - i
                self.__cev = self.__cev_ring[i % 4]
                self.__computing, self.__packing = i - 1, i - 2
                with torch.cuda.stream(cs):
                    # 5-6. compute chunk i-1, then pack chunk i-2 (not at
                    # the last tick: the drain packs).
                    result = self._execute_pre_comp_and_operator(
                        ts[i], pack_after=2 <= i < n - 1)
                    # 7. compress chunk i-1.
                    self._execute_post_comp(result)
                    self.__cev.end.record(cs)
            # T_i: chunk i-1's compute and chunk i's input copies.
            cs.wait_event(self.__h_events[i % 3])
            self.__t_events[i % 3].record(cs)
            if peaks is not None:
                peak_at[i] = peaks.iter_end()
            # 8. tick i-2's telemetry (T_{i-2} fired at step 2).
            if i >= 2:
                if session is not None:
                    state["remaining"] = ((self.__col_size
                                           - self.__boundaries[i])
                                          * self.__elem_size)
                collect_upto(i - 1)
            if state["recut"] is not None:
                # Ticks up to i are issued; the rest is re-cut.
                if not self._resection(i + 1, state["recut"],
                                       build_ahead=session.k + 3):
                    session.recut_refused()
                state["recut"] = None
            i += 1

        # The last barrier: everything has ended; the ticks left are
        # collected and released in order.
        n = self.stages
        cs.synchronize()
        exchange.sync()
        collect_upto(n)
        with torch.cuda.stream(cs):
            for t in range(state["released"], n):
                ts[t].clean_intermediates()

    def _barrier_ticks(self, exchange, resource_monitor, gcache):
        """The ticks under the tick barrier: every tick ends with the
        compute and exchange streams synchronized on the host.  Returns
        the tick the event loop takes over from once the pipe's observer
        has settled (``_searches_settled``), or None when the barrier ran
        every tick."""
        ts = self.__tick_stages

        # Tick 0: prologue — just H2D, no compute

        if gcache is not None:
            # A tick's reads are pinned while it can admit (its pack and
            # its RAW outputs reclaim slots mid-tick).
            gcache.advance_tick()
            self._pin_granules(ts[0].in_stage)

        exchange.launch(ts[0].in_stage)
        exchange.sync()
        # prologue: transfer-only (monitor last_compute None)
        self._tick_collect(resource_monitor, exchange.get_last_times(), None)
        self._materialize_mask_counts(ts[1], exchange.h2d_stream)

        # Tick 1: H2D chunk 1 + compute chunk 0 (no D2H yet)
        if gcache is not None:
            gcache.advance_tick()
            self._pin_granules(ts[1].in_stage)
        exchange.launch(ts[1].in_stage)
        if self.__serial:
            exchange.sync()
        with torch.cuda.stream(self.__compute_stream):
            result = self._execute_pre_comp_and_operator(ts[1])
            self._execute_post_comp(result)
            self.__cev.end.record(self.__compute_stream)
        # With the cache, pin ts[2]'s cached granules behind the compute's
        # GPU tail, before the next tick's admissions can pick them as
        # victims.
        if 2 < self.stages and gcache is not None:
            self._pin_granules(ts[2].in_stage)
        self.__compute_stream.synchronize()
        if self.__serial:
            writer_stream().synchronize()
        exchange.sync()
        self._tick_collect(resource_monitor, exchange.get_last_times(),
                           self.__cev)

        # Ticks 2..N-1: H2D + compute + D2H (steady state), until the
        # observer settles.
        i = 2
        while i < self.stages:
            if gcache is not None:
                gcache.advance_tick()
            self._materialize_mask_counts(ts[i], exchange.h2d_stream)
            if self.__pick_debug:
                self._debug_assert_stage_current(i)
            exchange.launch(ts[i].in_stage)
            if self.__serial:
                exchange.sync()
            # The ticks still producing output, this tick's input included:
            # the producer sizes its home chunks by it.
            self.__producer.ticks_left = self.stages - i
            with torch.cuda.stream(self.__compute_stream):
                # The last tick computes nothing; its predecessor's regions
                # are packed by the drain.
                result = self._execute_pre_comp_and_operator(
                    ts[i], pack=i < self.stages - 1)
                self._execute_post_comp(result)
                self.__cev.end.record(self.__compute_stream)
            # Pin ts[i+1]'s cached granules behind the compute's GPU tail
            # (host state only), as for ts[2] above.
            if i + 1 < self.stages and gcache is not None:
                self._pin_granules(ts[i + 1].in_stage)
            self.__compute_stream.synchronize()
            if self.__serial:
                writer_stream().synchronize()
            exchange.sync()
            self._tick_collect(resource_monitor, exchange.get_last_times(),
                               self.__cev)
            with torch.cuda.stream(self.__compute_stream):
                ts[i - 1].clean_intermediates()
            i += 1
            if i < self.stages and self._searches_settled():
                return i

        # Clean up the last tick's intermediates — the loop only covers ts[1]
        # through ts[stages-2].
        if self.stages >= 2:
            with torch.cuda.stream(self.__compute_stream):
                ts[-1].clean_intermediates()
        return None

    def _resection(self, from_tick: int, new_chunk_b: int,
                   build_ahead: int = 0) -> bool:
        """Re-cut the ticks ≥ ``from_tick`` at ``new_chunk_b`` (the tick
        sizing session): re-split the remaining rows; those ticks are built
        when the event loop reaches them (``_built``), the first
        ``build_ahead`` now, so the chunks the session measures after the
        re-cut carry no build of their own, as the ticks built with the
        pipe did not.

        Called on event-ordered ticks at the end of tick ``from_tick − 1``,
        whose input is issued and whose in-stages tick ``from_tick``'s
        pre-comps keep pointing at: no tick before ``from_tick`` has its
        geometry touched.

        Returns False (no-op) when only the epilogue remains.
        """
        assert self.__uniform_geometry and self.__elem_size > 0, (
            f"VortexPipeline {self.name!r}: resection on non-uniform "
            f"geometry")
        assert from_tick >= 2, "resection inside the t+2 horizon"
        if from_tick > self.stages - 2:
            return False
        start = self.__boundaries[from_tick - 1]
        remaining = self.__col_size - start
        if remaining <= 0:
            return False
        ticks = tick_cutter.cut(self.__points, start, self.__col_size,
                                self.__elem_size, int(new_chunk_b),
                                self.__floor_b)
        m = len(ticks)
        bounds = (self.__boundaries[:from_tick] + ticks.ends.tolist()
                  + [self.__col_size, self.__col_size])
        old_stages = self.stages
        self.__boundaries = bounds
        self.stages = from_tick + m + 2
        self.__chunk_size_b = int(new_chunk_b)
        # Remaining input, re-sliced at the new budget (mask columns share
        # the row boundaries by construction, exactly as at __init__).
        del self.__in_host[from_tick:]
        del self.tick_whole[from_tick:]
        self.__in_host.extend([] for _ in range(from_tick, self.stages))
        self.tick_whole.extend([True] * (self.stages - from_tick))
        for col in self.__input_cols.values():
            self._slice_inputs(col, bounds[from_tick - 1:self.stages - 1],
                               bounds[from_tick:self.stages], from_tick)
        del self.__tick_stages[from_tick:]
        self.__tick_stages.extend([None] * (self.stages - from_tick))
        del self.__stage_built_algs[from_tick:]
        self.__stage_built_algs.extend([None] * (self.stages - from_tick))
        if build_ahead > 0:
            self._built(min(from_tick + build_ahead, self.stages) - 1)
        del self.__tick_budget_b[from_tick:]
        self.__tick_budget_b.extend(
            [int(new_chunk_b)] * (self.stages - from_tick))
        for k in [k for k in self.__debug_expected if k >= from_tick]:
            del self.__debug_expected[k]
        message_logger().info(
            "[sizing] %s: resection @tick %d -> %d MB (%d ticks remain, "
            "was %d)", self.name, from_tick,
            int(new_chunk_b) // (1024 ** 2), self.stages - from_tick - 2,
            old_stages - from_tick - 2)
        return True

    def _drain_regions(self) -> None:
        """The region producer's end of pipe: pack the last tick's records
        and wait for every copy home (``RegionProducer.drain``)."""
        with torch.cuda.stream(self.__compute_stream):
            stored = self.__producer.drain()
        if self.__track_out_bytes:
            for base, nbytes in stored.items():
                self.__out_stored_bytes_by_slot[base] += nbytes

    def result_tables(self):
        """The region producer's outputs: ``{output instance index: frozen
        RecordTable}`` for the instances that wrote rows (by base slot for a
        bare pipeline)."""
        if self.__layout is None:
            return dict(self.__producer.tables())
        return {self.__layout.slots[base][0]: table
                for base, table in self.__producer.tables().items()}

    def get_result(self) -> None:
        """After ``do_exchange``: synchronize and publish a flat layout's
        outputs on their instances (``output_variable``); a partitioned
        layout's operator reads ``result_tables``."""
        torch.cuda.synchronize()
        self._publish_outputs()

    def output_variable(self, i: int, tensor_type: str = None, **attrs):
        """Output instance ``i``'s column after ``get_result``, for an
        operator to publish under its own type name (``tensor_type``) and
        attributes (``selectivity``): the table-backed Variable
        ``get_result`` published when the region producer wrote the
        instance, else an empty Variable.  Needs a flat layout."""
        from variable import Variable, torch_to_type
        if tensor_type is None:
            tensor_type = torch_to_type(self.__output_columns_type[i])
        if i in self.result_tables():
            assert self.__node is not None and self.__layout.is_flat, (
                f"VortexPipeline {self.name!r}: output {i} of a partitioned "
                f"layout")
            self._publish_outputs()
            var = self.__node.outputs[i].ccr.reps[0]
            assert var.table_backed, (
                f"VortexPipeline {self.name!r}: output {i}'s published "
                f"column was replaced")
            var.tensor_type = tensor_type
            for name, value in attrs.items():
                setattr(var, name, value)
            return var
        return Variable(tensor_type=tensor_type, **attrs)

    def _publish_outputs(self) -> None:
        """Set each output instance's payload for a flat layout: its record
        table as a ``TableVariable``, or an empty Variable when it wrote no
        row, typed from the declared dtype.  A partitioned layout is
        collapsed by its operator from ``result_tables``.  An operator may
        then re-wrap an instance's payload when it knows a finer type (a
        date column, a mask's selectivity)."""
        if self.__outputs_published or self.__node is None or not self.__layout.is_flat:
            return
        self.__outputs_published = True
        from variable import TableVariable, Variable, torch_to_type
        tables = self.result_tables()
        for i, inst in enumerate(self.__node.outputs):
            tensor_type = torch_to_type(self.__output_columns_type[i])
            table = tables.get(i)
            inst.ccr = CombinedColumnRep([
                TableVariable(table, tensor_type) if table is not None
                else Variable(tensor_type=tensor_type)])
