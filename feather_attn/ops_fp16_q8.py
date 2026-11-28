"""fp16 forward attention with an E5M2-quantized Q tile.

The low-level entry point for the fp16 family, mirroring `ops_bf16`: it is not wired into
`feather_attn.ops.feather_attn_func`, which keeps the vendored Triton kernel as the
public forward. Its error is about 5e-2 relative L2, a consequence of the E5M2 Q
quantization, which is why it is not the default forward.

The operator validates every operand in C++, so a bad shape or dtype raises from there.
"""

import torch

from .layout import LAYOUT_IDS
from .reference import default_softmax_scale


def feather_attn_fp16_q8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """out = softmax(q @ k^T * softmax_scale) @ v for fp16 operands, D64 or D128."""
    out = torch.empty_like(q)
    torch.ops.feather_attn.attn_fwd_fp16_q8_feather.default(q, k, v, out, LAYOUT_IDS[layout], default_softmax_scale(q.shape[-1], softmax_scale))
    return out
