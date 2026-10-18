import torch
from utility.tensor_utils import compact_masked
from variable import type_to_torch


# Per-dtype cached zero-length GPU tensors. Returned by extract_raw when the
# input partition is empty, so we don't hit Variable.data()'s torch.empty(0)
# allocator path on every empty partition. Safe to share across callers
# because consumers (compact_masked, operator inputs) treat raw tensors as
# read-only views.
_EMPTY_GPU_CACHE: dict = {}

def _empty_gpu(torch_dtype):
    t = _EMPTY_GPU_CACHE.get(torch_dtype)
    if t is None:
        t = torch.empty(0, dtype=torch_dtype, device="cuda")
        _EMPTY_GPU_CACHE[torch_dtype] = t
    return t


class PreCompStage:
    """Pre-computation stage: decompresses a tick's records (Direct) or runs zero-copy kernel."""
    def __init__(self, col_idx, in_stage_ref, is_zero_copy, mask_map_target=None):
        self.col_idx = col_idx
        self.in_stage_ref = in_stage_ref    # the InStage that feeds this PreComp
        self.is_zero_copy = is_zero_copy
        self.mask_map_target = mask_map_target  # index of mask col in pre_comps list
        self.ready_tensor = None
    def decompress(self):
        """Phase 1: decompress the tick's records (no-op for zero-copy)."""
        if not self.is_zero_copy:
            self.in_stage_ref.decompress()

    def execute_zero_copy(self):
        """Phase 1 (zero-copy path): launch zero-copy kernel on in_stage_ref."""
        if self.is_zero_copy:
            self.in_stage_ref.execute_zero_copy()

    def extract_raw(self):
        """Phase 1b: get raw tensor after decompress (None for zero-copy).

        Fast path: when the GPU column carries no materialized contiguous
        buffer, return a per-dtype cached zero-length tensor
        instead of routing through Variable.data() (which would allocate a
        fresh torch.empty(0) on every empty partition).
        """
        if self.is_zero_copy:
            return None
        gpu_col = self.in_stage_ref.gpu_col
        if gpu_col._contiguous_data is None:
            return _empty_gpu(type_to_torch(gpu_col.tensor_type))
        return gpu_col.data()

    def apply_mask(self, all_raw_tensors, mask_counts=None, fold=False):
        """Phase 2: apply mask (Direct) or extract zero-copy result.

        Args:
            mask_counts: dict mapping mask column index -> Python int count
                of True elements, pre-computed on the DirectIn mask column's
                own data (same tensor that all_raw_tensors[mt] references).
            fold: the operator applies the mask itself, so a Direct column
                stays uncompacted.
        """
        if self.is_zero_copy:
            self.ready_tensor = self.in_stage_ref.gpu_col.data()
            return
        raw = all_raw_tensors[self.col_idx]
        if self.mask_map_target is not None and not fold:
            mask = all_raw_tensors[self.mask_map_target]
            count = mask_counts[self.mask_map_target]
            self.ready_tensor = compact_masked(raw, mask, count)
        else:
            self.ready_tensor = raw

    def cleanup(self):
        self.ready_tensor = None
