"""bf16 D128 forward attention built on the native FeatherAttn kernel.

`layout` selects the physical head/sequence order (`"HND"` is `[B,H,N,D]`, `"NHD"` is
`[B,N,H,D]`) and `config` pins a tile shape, with 0 selecting the heuristic. The operator
writes the natural-log log-sum-exp the backward kernels consume, so `out` and `lse` are
allocated here, outside the operator boundary, where `torch.compile` can see them.

The operator validates every operand in C++, so a bad shape or dtype raises from there rather
than being re-checked here.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(out, softmax_lse)` for bf16 operands with head_dim 128."""
    out = torch.empty_like(q)
    if layout == "NHD":
        batch, n_q, heads = q.shape[0], q.shape[1], q.shape[2]
    else:
        batch, heads, n_q = q.shape[0], q.shape[1], q.shape[2]
    softmax_lse = torch.empty((batch, heads, n_q), device=q.device, dtype=torch.float32)
    torch.ops.feather_attn.attn_fwd_bf16_d128_feather.default(
        q,
        k,
        v,
        out,
        softmax_lse,
        LAYOUT_IDS[layout],
        default_softmax_scale(q.shape[-1], softmax_scale),
        config,
    )
    return out, softmax_lse
