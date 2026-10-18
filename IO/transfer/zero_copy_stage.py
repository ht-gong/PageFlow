import torch
import TQPlib.tqpcompress as tqpcompress
import TQPlib.zero_copy_reader as zero_copy_reader
from IO.telemetry import copy_events, gates, movement
from IO.transfer.in_stage import DirectIn, InStage
from IO.transfer.tick_layout import launch_slot_pieces
from variable import gpu_region_column


class ZeroCopyMaskIn(InStage):
    """
    Represents a zero-copy transfer using a boolean mask.
    The mask is moved to GPU during the unified launch_h2d path;
    execute_zero_copy() runs the kernel against the GPU mask.

    The mask is read like a DirectIn column, through its own one-stage
    layout, compiled here: only the mask records the stage's rows overlap
    are copied and decoded (whole records for a tick cut in whole quanta;
    a tick cut by rows trims the decoded rows to its own).
    """
    def __init__(self, src_col, mask_col, dst_col_width):
        super().__init__(src_col)
        self.mask_col = mask_col
        self.dst_col_width = dst_col_width
        self.mask_gpu = None
        self.dst_numel = 0
        self._mask_in = None
        if not mask_col.reads_nothing():
            self._mask_in = DirectIn(mask_col)
            DirectIn.install_compiled_tick_layout([self._mask_in])

    @classmethod
    def plan_input_data_movement(cls, stages):
        # Passthrough: launch_h2d iterates the stage list and runs each
        # stage's legacy per-page H2D body.  A real batched plan sharing
        # one agg_buf across all ZeroCopyMaskIn stages in the tick would
        # replace this.
        return None, list(stages)

    @staticmethod
    def launch_h2d(state, plan):
        """Every stage's mask copies and decode, as one copy batch
        (``copy_events``).  The movement ledger records each mask's runs at
        its bind, as a DirectIn's."""
        batch = copy_events.begin(copy_events.H2D)
        for s in plan:
            s._read_mask()
        copy_events.end(batch)

    def _read_mask(self):
        """Copy the mask's selected records (one copy per run, from host or
        GPU memory) and decode them on the current stream.  A single
        batched_h2d_pack call across all ZeroCopyMaskIn stages would replace
        the per-stage calls."""
        stage = self._mask_in
        if stage is None:
            self.mask_gpu = self.mask_col.col_to_gpu()
            return
        agg_buf, plan = DirectIn.plan_input_data_movement([stage])
        if plan is not None:
            if plan.h2d_pack_pages:
                tqpcompress.batched_h2d_pack(agg_buf, *plan.pack_args)
            if plan.d2d_pack_pages:
                tqpcompress.batched_d2d_pack(agg_buf, *plan.d2d_pack_args)
                launch_slot_pieces(agg_buf, plan)
            stage.phase_finalize(None)
            stage.decompress()
            # The copies and the decode are issued: the layout can be bound
            # again should the stage be read again.
            plan.release()
        self.mask_gpu = stage.gpu_col.data()
        stage.gpu_col = None

    def issue_holds(self):
        """The decoded mask (read by the zero-copy kernel on the compute
        stream) and the mask count; the mask's own agg_buf is used on the
        h2d stream only."""
        held = super().issue_holds()
        if self.mask_gpu is not None:
            held.append(self.mask_gpu)
        return held

    def phase_finalize(self, d2d_stream):
        """Allocate the mask-count scratch and run the sum on
        h2d_stream after launch_d2d's fills are visible.  The
        orchestrator does h2d_stream.wait_stream(d2d_stream) before
        this is called."""
        if self.mask_gpu is None:
            return
        self._mask_count_buf = torch.zeros((), dtype=torch.int64, device='cuda:0')
        torch.sum(self.mask_gpu.view(-1), 0, out=self._mask_count_buf)
        self.mask_count = self._mask_count_buf

    def execute_zero_copy(self):
        """Run the zero-copy mask kernel over the source's host rows
        (``uva_source``: its table's segments) and the stored
        mask_gpu; the output is a GPU region (``gpu_region_column``)."""
        self.dst_numel = self.mask_count

        src = self.src_col.uva_source()
        dst = torch.empty((self.dst_numel, self.dst_col_width),
                          dtype=src.dtype, device='cuda:0')
        zero_copy_reader.zero_copy_mask(src.seg_ptrs, src.seg_rows,
                                        self.mask_gpu, dst, src.row_start,
                                        src.row_stop)
        if gates.EXPERIMENT:
            movement.record_uva_read(self.src_col, dst.nbytes)
        self.gpu_col = gpu_region_column(dst, self.src_col.tensor_type)

    def get_transferred(self):
        return self.gpu_col
