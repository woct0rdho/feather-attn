"""BSHD <-> HND layout conversions for the FeatherAttn backward.

`transpose(1, 2).contiguous()` drops to ~20 GB/s once a Q/K/V sized tensor leaves L2
on gfx1151 (0.24 ms at 4096 tokens, 3.7 ms at 9216), which is more than the attention
kernel it feeds. Here each program reads `[BLOCK_L, heads * head_dim]` slabs
contiguously and writes one contiguous `[BLOCK_L, head_dim]` slab per head, i.e. a
plain strided copy that runs at ~200 GB/s.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _pack_hnd(
    src,
    dst,
    seqlen,
    stride_sb,
    stride_sl,
    stride_db,
    stride_dh,
    stride_dl,
    BLOCK_DMODEL: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask = offs_l < seqlen
    src_ptrs = src + pid_b * stride_sb + offs_l[:, None] * stride_sl + offs_d[None, :]
    dst_ptrs = dst + pid_b * stride_db + offs_l[:, None] * stride_dl + offs_d[None, :]
    for h in tl.static_range(HEADS):
        tile = tl.load(src_ptrs + h * BLOCK_DMODEL, mask=mask[:, None])
        tl.store(dst_ptrs + h * stride_dh, tile, mask=mask[:, None])


@triton_op("feather_attn::pack_hnd_out", mutates_args={"out"})
def pack_hnd_out(x: torch.Tensor, out: torch.Tensor) -> None:
    """Copy `x` (BSHD logical, BSHD contiguous) into `out` (HND contiguous)."""
    batch, seqlen, heads, head_dim = x.shape

    def grid(meta):
        return (triton.cdiv(seqlen, meta["BLOCK_L"]), batch)

    wrap_triton(_pack_hnd)[grid](
        x,
        out,
        seqlen,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_DMODEL=int(head_dim),
        HEADS=int(heads),
        BLOCK_L=2,
    )


def to_hnd(t: torch.Tensor) -> torch.Tensor:
    """`(B, L, H, D)` -> `(B, H, L, D)` carrying the same data, contiguous in HND."""
    hnd = t.transpose(1, 2)
    if hnd.is_contiguous():
        return hnd
    batch, seqlen, heads, head_dim = t.shape
    if t.stride(3) != 1 or t.stride(2) != head_dim or t.stride(1) != heads * head_dim or t.stride(0) != seqlen * heads * head_dim:
        return hnd.contiguous()
    out = torch.empty((batch, heads, seqlen, head_dim), device=t.device, dtype=t.dtype)
    pack_hnd_out(t, out)
    return out
