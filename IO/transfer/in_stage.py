import torch
from utility.tensor_utils import dtype_size
import TQPlib.tqpcompress as tqpcompress
from IO.telemetry import copy_events
from IO.transfer.tick_layout import (
    _launch_mgr_group, bind_tick_sources, compile_tick_layout,
    launch_slot_pieces)


class InStage:
    """
    Base class for different IO transfer modes (input stage).

    All InStage subclasses share a single per-tick transfer interface,
    orchestrated by ``NaiveExchange.launch``:

      ``plan_input_data_movement(stages)``  -> ``(state, plan)``
            CPU-side; build a per-class batched plan across all stages
            of this class in the tick.  Returns ``(None, None)`` when
            there is no work.

      ``launch_h2d(state, plan)``
            Issue all CPU→GPU work for the tick on the current stream
            (the orchestrator runs this on h2d_stream).

      ``launch_d2d(state, plan)``
            Issue all GPU→GPU work (cached granules, GPU-resident
            sources) for the tick on the current stream (the
            orchestrator runs this on d2d_stream in parallel with
            launch_h2d).

      ``build_decompress_metadata(state, plan)``
            Build nvCOMP metadata views into ``state``.  Default no-op.

      ``phase_finalize(d2d_stream)``
            Per-stage finalize: build the gpu_col Variable, run any
            mask-sum kernel, etc.  Runs on h2d_stream after the
            orchestrator has done ``h2d_stream.wait_stream(d2d_stream)``
            so launch_d2d's writes are visible.

    Subclasses override only the methods they need; defaults are
    no-ops so subclasses with trivial transfer paths can leave most of
    the interface alone.
    """
    def __init__(self, src_col):
        self.src_col = src_col
        self.gpu_col = None
        self.mask_count = None
        self._mask_count_buf = None  # pre-allocated 1-element GPU buffer for .sum(out=)

    def transferred_bytes(self):
        return self.src_col.data_to_transfer()

    def raw_numel(self):
        return self.src_col.raw_numel() if self.src_col else 0

    @classmethod
    def plan_input_data_movement(cls, stages):
        """CPU-side per-tick plan builder.  Default: no batched plan."""
        return None, None

    @classmethod
    def bind_input(cls, stages):
        """The host half of ``plan_input_data_movement``: whatever the class
        can bind before the tick's buffers are allocated, handed back to
        ``alloc_input``.  Default: nothing to bind."""
        return None

    @classmethod
    def alloc_input(cls, stages, bound, holds=None):
        """The allocating half: ``(state, plan)`` as
        ``plan_input_data_movement`` returns them, on the current (h2d)
        stream.  ``holds``, when given, collects the buffers allocated here
        that the compute stream reads (the tick's hold list).  Default: the
        whole plan, which allocates nothing."""
        return cls.plan_input_data_movement(stages)

    def issue_holds(self):
        """The buffers this stage allocated during the input issue (after
        ``phase_finalize``) that another stream reads: the mask count."""
        return [self._mask_count_buf] if self._mask_count_buf is not None \
            else []

    @staticmethod
    def launch_h2d(state, plan):
        """Issue all CPU→GPU work for the tick on the current stream.
        Default no-op."""
        pass

    @staticmethod
    def launch_d2d(state, plan):
        """Issue all GPU→GPU work for the tick on the current stream.
        Default no-op."""
        pass

    @staticmethod
    def build_decompress_metadata(state, plan):
        """Build nvCOMP metadata views into state.  Default no-op."""
        pass

    def phase_finalize(self, d2d_stream):
        pass


class DirectIn(InStage):
    """
    Represents a direct data transfer without any special handling.
    Creates a GPU-side copy of the source column.

    The stage reads a selection of its column's record table
    (``TableVariable.record_selection``): the records its rows overlap, one
    run (one copy) per stretch of them in a region, into the tick's shared
    agg_buf.  Compressed stages decompress only those records from there,
    through their codec group's compiled manager template.  A tick cut in whole quanta
    reads whole records; a tick cut by rows has the operator's rows trimmed
    inside the first and last record (``gpu_col.row_range``).
    """

    def __init__(self, src_col):
        super().__init__(src_col)
        # Shared by every DirectIn in one tick.  VortexPipeline installs this
        # at stage construction; standalone callers compile on demand.
        self._compiled_tick_layout = None
        # This stage's index in that layout's per-stage arrays.
        self._layout_index = None
        # Inter-phase state (populated by plan_input_data_movement +
        # launch_h2d/launch_d2d, consumed by phase_finalize and decompress).
        self._transfer_buf = None   # the tick's agg_buf
        self._data_buf = None       # uint8 view of this stage's payload region
        self._has_compression = False
        self._total_rows = 0
        self._data_width = 0
        self._data_type = None
        # Rows of the decoded records the operator keeps (``None``: all).
        self._row_range = None
        # Unified-plan state (Stage 1+2): per-tick codec-group sharing.  When
        # set, phase_finalize plumbs these into the Variable so Level B batched
        # decompress covers every stage in the group in one nvCOMP launch.
        self._group_ctx = None
        self._shared_output = None
        self._byte_offset_in_shared_output = 0

    # ------------------------------------------------------------------
    # Phase interface (called by NaiveExchange under batched stream ctx).
    # H2D goes through ``plan_input_data_movement`` + ``launch_h2d``;
    # ``phase_finalize`` runs the per-stage Variable build + optional
    # mask sum after the batched H2D completes.
    # ------------------------------------------------------------------

    def phase_finalize(self, d2d_stream):
        """Phase 3 — assumes h2d_stream is current.
        Build the empty gpu_col Variable; for RAW columns also point its
        ``_contiguous_data`` at the H2D buffer.  Compression scratch stays
        on ``self`` and is consumed by ``self.decompress()``."""
        from variable import Variable

        if self.gpu_col is not None:
            return  # an empty stage's, set by bind_input

        self.gpu_col = Variable(
            tensor_type=self.src_col.tensor_type,
            row_range=self._row_range,
        )

        if not self._has_compression:
            buf = self._data_buf.view(
                self._data_type).view(self._total_rows, self._data_width)
            # Pre-normalize: width-1 numeric columns collapse to 1-D so
            # downstream data() can skip the .squeeze dispatch.
            if self._data_width == 1:
                buf = buf.squeeze(1)
            self.gpu_col._contiguous_data = buf

        if getattr(self, '_compute_mask_sum', False):
            # H2D fills run on h2d_stream, D2D fills on d2d_stream.
            # The exchange driver inserts h2d_stream.wait_stream(
            # d2d_stream) after launch_d2d, so by the time
            # phase_finalize runs on h2d_stream both batches are
            # already visible.
            if self._has_compression:
                self.decompress()
            mask_data = self.gpu_col._contiguous_data
            if self.gpu_col.row_range is not None:
                mask_data = mask_data[self.gpu_col.row_range.start:self.gpu_col.row_range.stop]
            self._mask_count_buf = torch.zeros((), dtype=torch.int64, device='cuda:0')
            torch.sum(mask_data.view(-1), 0, out=self._mask_count_buf)
            self.mask_count = self._mask_count_buf

        # Inter-phase scratch (transfer_buf, data_buf, group_ctx,
        # shared_output) stays alive on self until ``decompress`` consumes
        # it (or ``clean_intermediates`` fires for RAW / mask-only stages).

    def decompress(self):
        """Run nvCOMP decompress for this stage's H2D'd data and write the
        result into ``self.gpu_col._contiguous_data``.  No-op for RAW
        stages (their _contiguous_data is already wired to ``_data_buf``
        in phase_finalize) and for empty stages.

        Always goes through the group-batched path: the unified per-tick
        plan in ``_alloc_agg_buf_and_bind_views`` binds every compressed
        stage to a codec-group ctx, so the first member of each group
        runs one nvCOMP call covering every member; later members narrow
        their own view into the shared output.

        After running, scratch refs are dropped so the agg_buf can be
        freed by GC / clean_intermediates.
        """
        if not getattr(self, '_has_compression', False):
            return
        if self.gpu_col is None or self._group_ctx is None:
            return
        self._decompress_group_batched()

    def _decompress_group_batched(self):
        """Level B: one nvCOMP call covers every stage in the codec group.
        First caller through the group runs the kernel; later callers
        just narrow their slice of the shared output buffer.

        ``self._group_ctx`` is a mutable dict shared across all stages in
        the group (set up by ``plan_input_data_movement``).
        """
        import TQPlib.tqpcompress as tqpcompress

        device = 'cuda:0'
        # Degenerate path: a non-raw codec group whose stages all have
        # total_rows == 0 gets shared_output_size == 0 in the plan, so
        # group_ctxs[gi] is left as None.  No data to decompress —
        # materialize an empty output and bail.
        if self._group_ctx is None:
            self.gpu_col._contiguous_data = torch.empty(
                (self._total_rows, self._data_width),
                dtype=self._data_type, device=device)
            self._clear_decompress_state()
            return

        ctx = self._group_ctx
        shared_output = self._shared_output
        byte_off = self._byte_offset_in_shared_output
        total_rows = self._total_rows
        data_width = self._data_width
        data_type = self._data_type
        row_bytes = data_width * dtype_size(data_type)
        stage_bytes = total_rows * row_bytes

        if not ctx['decompressed']:
            _launch_mgr_group(ctx, shared_output)
            ctx['decompressed'] = True

        if stage_bytes > 0:
            buf = shared_output.narrow(
                0, byte_off, stage_bytes).view(data_type).view(total_rows, data_width)
        else:
            buf = torch.empty(
                (total_rows, data_width), dtype=data_type, device=device)
        if data_width == 1:
            buf = buf.squeeze(1)
        self.gpu_col._contiguous_data = buf
        self._clear_decompress_state()

    def _clear_decompress_state(self):
        """Drop refs to the agg_buf views so GC / clean_intermediates can
        actually reclaim them.  Called at the tail of decompress."""
        self._transfer_buf = None
        self._data_buf = None
        self._group_ctx = None
        self._shared_output = None

    def get_transferred(self):
        return self.gpu_col

    @staticmethod
    def install_compiled_tick_layout(stages):
        """Compile once and attach one tick-owned layout to all *stages*."""
        layout = compile_tick_layout(stages)
        for i, stage in enumerate(stages):
            stage._compiled_tick_layout = layout
            stage._layout_index = i
        return layout

    @staticmethod
    def plan_input_data_movement(stages):
        """Plan the per-tick input data movement for a batch of
        ``DirectIn`` stages: ``bind_input`` then ``alloc_input``.

        Steps performed here (all CPU-side except the agg_buf alloc):
          * Empty stages get an empty ``gpu_col`` Variable wired up.
          * ``bind_tick_sources`` patches only current host/slot/GPU
            source pointers plus cache-read accounting.
          * One ``agg_buf`` is allocated on the current CUDA stream and
            per-codec-group shared output buffers are allocated.
          * Per-stage buffer refs (``_data_buf`` / ``_group_ctx`` /
            ``_shared_output``) are bound off the agg_buf so finalize /
            decompress can find their slices.

        Returns ``(agg_buf, plan)`` for the caller to feed into
        ``launch_h2d`` / ``launch_d2d`` / ``build_decompress_metadata``
        in order.  Returns ``(None, None)``
        when the tick has no work (every stage was empty).

        Caller should be inside the target h2d_stream context so the
        agg_buf alloc records that stream with the caching allocator.
        ``launch_d2d`` is the only sibling that runs on a different
        stream (d2d_stream); the alloc is host-side so cross-stream
        use is safe with an ``agg_buf.record_stream(d2d_stream)``
        hint at the call site.
        """
        return DirectIn.alloc_input(stages, DirectIn.bind_input(stages))

    @staticmethod
    def bind_input(stages):
        """The host half: empty stages' ``gpu_col``, and the tick's source
        pointers bound into its compiled layout (``bind_tick_sources``).
        Returns the bound plan, or
        ``None`` when the tick has nothing to read.  Allocates nothing, so
        it may run before the wait that bounds the tick's allocation."""
        from variable import Variable

        any_non_empty = False
        for stage in stages:
            if stage.src_col.reads_nothing():
                stage.gpu_col = Variable(
                    tensor_type=stage.src_col.tensor_type,
                    row_range=stage.src_col.row_range,
                )
            else:
                any_non_empty = True
        if not any_non_empty:
            return None

        layout = stages[0]._compiled_tick_layout
        if (layout is None or any(
                stage._compiled_tick_layout is not layout for stage in stages)):
            layout = DirectIn.install_compiled_tick_layout(stages)
        return bind_tick_sources(stages, layout)

    @staticmethod
    def alloc_input(stages, bound, holds=None):
        """The allocating half, on the current (h2d) stream: the tick's
        agg_buf and decoded outputs (``_alloc_agg_buf_and_bind_views``).
        Returns ``(agg_buf, plan)``, or ``(None, None)`` when ``bound`` is
        ``None``."""
        if bound is None:
            return None, None
        agg_buf = DirectIn._alloc_agg_buf_and_bind_views(stages, bound, holds)
        return agg_buf, bound

    @staticmethod
    def _alloc_agg_buf_and_bind_views(stages, plan, holds=None):
        """Allocate the tick's agg_buf, allocate one shared decompressed
        output buffer per compressed codec group, and bind per-stage
        ``_data_buf`` / ``_group_ctx`` / ``_shared_output`` references off
        agg_buf.  Returns ``agg_buf``.  ``holds``, when given, gets the
        agg_buf and the decoded outputs: the compute stream reads them, so
        they stay referenced until the tick that computes them has ended
        (the tick's hold list).
        """
        agg_buf = torch.empty(
            plan.layout.agg_buf_size, dtype=torch.uint8, device='cuda:0')
        if holds is not None:
            holds.append(agg_buf)

        # One shared decompressed output buffer per compressed codec group.
        # RAW groups alias their data_buf as the output view, so no
        # separate allocation.  The group ctx carries the compiled manager
        # template (_launch_mgr_group only rebases its two base pointers)
        # and the RAW-record row copies of mixed-codec stages.
        group_shared_outputs = []
        group_ctxs = []
        for group in plan.layout.codec_groups:
            if group['is_raw'] or group['shared_output_size'] == 0:
                group_shared_outputs.append(None)
                group_ctxs.append(None)
                continue
            shared_output = torch.empty(
                group['shared_output_size'], dtype=torch.uint8, device='cuda:0')
            if holds is not None:
                holds.append(shared_output)
            group_shared_outputs.append(shared_output)
            group_ctxs.append({
                'decompressed': False,
                'agg_buf': agg_buf,
                'mgr_template': group['mgr_template'],
                'mgr_raw_rows': group['mgr_raw_rows'],
            })

        for stage, view in zip(stages, plan.layout.per_stage_views):
            if view is None:
                continue
            gi = view['codec_group_idx']
            d0, d1 = view['data_buf_slice']

            stage._transfer_buf = agg_buf
            stage._data_buf = agg_buf[d0:d1]
            stage._total_rows = view['total_rows']
            stage._row_range = view['row_range']
            stage._data_width = view['data_width']
            stage._data_type = view['data_type']
            stage._has_compression = not view['is_raw']
            stage._group_ctx = group_ctxs[gi]
            stage._shared_output = group_shared_outputs[gi]
            stage._byte_offset_in_shared_output = view['byte_offset_in_output']

        return agg_buf

    @staticmethod
    def launch_h2d(agg_buf, plan):
        """Issue every CPU→GPU run copy for the tick in one C++ call.

        ``plan.pack_args`` carries one row per run across every
        non-empty stage.  Rows for runs of GPU regions
        have ``src_ptr=0`` so the C++ pack call skips them — those runs are
        filled by ``launch_d2d``.

        Issued on h2d_stream (PCIe / H2D copy engine).  GPU-resident fills
        run separately on d2d_stream — see
        ``launch_d2d``.  The pack is one copy batch (``copy_events``)
        when any run crosses H2D.
        """
        if plan.home_waits:
            # Home pieces of cached regions: behind the copies into their
            # homes (``BoundTickPlan.home_waits``).
            stream = torch.cuda.current_stream()
            for e in plan.home_waits:
                stream.wait_event(e)
        batch = (copy_events.begin(copy_events.H2D)
                 if plan.h2d_pack_pages else None)
        tqpcompress.batched_h2d_pack(agg_buf, *plan.pack_args)
        copy_events.end(batch)

    @staticmethod
    def launch_d2d(agg_buf, plan):
        """Issue every GPU→GPU fill for the tick in one C++ call.

        Sources are GPU regions (e.g. a sort's output, ``gpu_region_column``),
        and the slot pieces of cached granules.  Destinations are slots in the tick's agg_buf that
        ``launch_h2d`` left empty.  No-op when the tick has none.

        Issued on d2d_stream so the GPU dispatches it to the D2D copy
        engine in parallel with ``launch_h2d`` on the H2D engine — the
        two writes target disjoint regions of agg_buf, so they have
        no inter-launch dependency.  Caller is responsible for
        ``agg_buf.record_stream(d2d_stream)`` (caching-allocator
        deferred-reuse hint) and for
        ``h2d_stream.wait_stream(d2d_stream)`` after the call so any
        later reader on h2d_stream (``build_decompress_metadata``,
        ``phase_finalize``, ``decompress``) sees the d2d fills.
        ``d2d_stream.wait_stream(h2d_stream)`` is NOT needed — the
        agg_buf alloc is host-side (no GPU work).
        """
        if plan.d2d_pack_pages == 0:
            return
        batch = copy_events.begin(copy_events.D2D)
        tqpcompress.batched_d2d_pack(agg_buf, *plan.d2d_pack_args)
        launch_slot_pieces(agg_buf, plan)
        copy_events.end(batch)
