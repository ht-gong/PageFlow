from typing import List
from IO.transfer.in_stage import InStage
from IO.transfer.pre_comp_stage import PreCompStage


class TickStages:
    """All columns through the input stages of one tick: in, pre_comp."""
    def __init__(self, in_stage: List[InStage] = None,
                       pre_comp: List[PreCompStage] = None):
        self.in_stage = in_stage or []
        self.pre_comp = pre_comp or []
        # Event-ordered ticks only: the buffers the tick's input issue
        # allocated on the h2d stream that the compute stream reads (the
        # agg_buf, the decoded outputs, the mask counts, a zero-copy
        # stage's decoded mask).  Kept until the tick that computes this
        # chunk has ended on the GPU, then released with the rest
        # (``clean_intermediates``).  Empty under the tick barrier.
        self.holds = []

    def clean_intermediates(self):
        """Clear all intermediate GPU tensors and references held by this TickStages.

        SAFETY: Only call after ALL of the following have completed:
          1. compute_stream.synchronize() — the operator has finished reading
             in_stage gpu_col tensors.
          2. exchange.sync() — the tick's H2D/D2D copies have completed.

        Calling before these conditions are met risks use-after-free: async
        CUDA kernels or DMA engines may still be reading the GPU memory
        backing these tensors.  With event-ordered ticks, the event after
        the tick that computed this chunk stands for both.
        """
        self.holds = []
        # Release GPU-side Variables produced by InStage.transfer().
        for in_s in self.in_stage:
            in_s.gpu_col = None
            # DirectIn: _transfer_buf is the H2D backing storage;
            # _data_buf is a view of it.
            # gpu_col._contiguous_data may be a view of _data_buf (RAW),
            # so dropping gpu_col alone does not free the underlying GPU memory.
            # decompress() clears most of these, but RAW / mask-only
            # stages skip decompress, so clean them defensively here.
            if hasattr(in_s, '_clear_decompress_state'):
                in_s._clear_decompress_state()
            if hasattr(in_s, 'mask_gpu'):
                in_s.mask_gpu = None
            if hasattr(in_s, '_mask_count_buf'):
                in_s._mask_count_buf = None

        # Release PreCompStage ready tensors.
        for pc in self.pre_comp:
            pc.cleanup()
