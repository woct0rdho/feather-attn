"""bf16 D128 forward attention built on the native FeatherAttn kernel.

This is the low-level entry point for the bf16 kernel and is intentionally not wired into
`feather_attn.ops.feather_attn_func`. The FlashAttention-compatible path uses the
Triton kernel. `layout` selects the physical head/sequence order (`"HND"` is
`[B,H,N,D]`, `"NHD"` is `[B,N,H,D]`) and `config` pins a tile shape, with 0
selecting the heuristic.

The operator validates every operand in C++, so a bad shape or dtype raises from there
rather than being re-checked here.
"""

import torch

from .layout import LAYOUT_IDS
from .reference import default_softmax_scale

# `config` selects one of the tile configurations of the kernel, 0 being the heuristic that
# the host resolves per layout. The ids are the `kTileConfigs` rows in
# csrc/fwd/bf16/bf16_fwd_d128.cpp. `tests/test_attn_fwd_bf16_d128.py` launches each of them.


def feather_attn_bf16_d128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    config: int = 0,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """out = softmax(q @ k^T * softmax_scale) @ v for bf16 operands with head_dim 128."""
    out = torch.empty_like(q)
    torch.ops.feather_attn.attn_fwd_bf16_d128_feather.default(
        q,
        k,
        v,
        out,
        LAYOUT_IDS[layout],
        default_softmax_scale(q.shape[-1], softmax_scale),
        config,
    )
    return out
