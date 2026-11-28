"""bf16 D128 forward attention built on the native FeatherAttn kernel.

This is the low-level entry point for the new kernel and is intentionally not
wired into `feather_attn.ops.feather_attn`. `layout` selects the physical
head/sequence order (`"HND"` is `[B,H,N,D]`, `"NHD"` is `[B,N,H,D]`) and
`config` pins a tile shape (0 picks the default heuristic).
"""

import torch

HND = 0
NHD = 1

CONFIG_NAMES = {
    0: "auto",
    1: "br256_bc64_swz",
    2: "br256_bc32_pad",
    3: "br256_bc64_pad_buf1",
    4: "br128_bc32_pad",
}


def _layout_id(layout: str) -> int:
    if layout == "HND":
        return HND
    if layout == "NHD":
        return NHD
    raise ValueError(f"layout must be 'HND' or 'NHD', got {layout}")


def feather_attn_bf16_d128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    config: int = 0,
) -> torch.Tensor:
    """out = softmax(q @ k^T / sqrt(128)) @ v for bf16 inputs and head_dim 128."""
    out = torch.empty_like(q)
    torch.ops.feather_attn.attn_bf16_d128_feather.default(q, k, v, out, _layout_id(layout), config)
    return out
