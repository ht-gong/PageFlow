import torch


def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def row_bytes(t: torch.Tensor) -> int:
    """The bytes of one row of ``t``: its element size times its width (1
    for a 1-D tensor, ``shape[1]`` for a 2-D one)."""
    return t.element_size() * (t.shape[1] if t.dim() == 2 else 1)

def compact_masked(src: torch.Tensor, mask: torch.Tensor, n_true: int) -> torch.Tensor:
    """
    Equivalent to src[mask], but with no D2H sync.
    n_true = number of True elements (already known on CPU).
    mask must be 1-D bool with mask.numel() == src.shape[0].
    src can be 1-D or 2-D; selection is along dim 0 (rows).
    """
    n = mask.numel()

    # 1. Destination index for every row (0-indexed).
    #    True rows get sequential output positions; False values are garbage.
    #    Use int64 throughout — scatter_ requires LongTensor indices.
    #    Fuse the bool→int64 cast into cumsum to avoid an intermediate
    #    allocation that would go through the pool and potentially pick up
    #    cross-stream events from h2d temporaries.
    dest = mask.cumsum(0, dtype=torch.int64) - 1

    # 2. Build the inverse mapping: for each output slot, which input row?
    #    Allocate n_true + 1 so False rows scatter to a trash slot.
    source_idx = torch.empty(n_true + 1, dtype=torch.long, device=src.device)
    all_idx    = torch.arange(n, dtype=torch.long, device=src.device)
    safe_dest  = torch.where(mask, dest, n_true)
    safe_src   = torch.where(mask, all_idx, 0)
    source_idx.scatter_(0, safe_dest, safe_src)

    # 3. Index into src along dim 0 using the clean inverse map (drop trash).
    #    src[int_indices] works for any src rank — no sync needed.
    return src[source_idx[:n_true]]