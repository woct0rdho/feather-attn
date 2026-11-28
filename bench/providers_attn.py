"""Provider builders for the forward attention benchmarks.

One builder per implementation family, each writing out what that implementation needs: the
AITER control takes no output buffer, the vendored Triton kernel takes one plus a
log-sum-exp tensor, the native fp16 kernel has no tile knob, and the native bf16 kernel is
pinned per tile configuration. Provider names are the implementation name, with `@cN`
appended when more than one tile configuration of the same kernel is in the matrix.
"""

import torch

from feather_attn.aiter_vendored import fwd_out as vendored_fwd_out
from feather_attn.layout import to_bshd
from feather_attn.ops_bf16 import feather_attn_bf16_d128
from feather_attn.ops_fp16_q8 import feather_attn_fp16_q8
from feather_attn.reference import aiter_fwd


def add_aiter_fwd_providers(providers: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str) -> None:
    providers["aiter"] = lambda: aiter_fwd(q, k, v, layout)


def add_vendored_fwd_providers(providers: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str) -> None:
    """The public API's Triton forward: it takes an output buffer and a log-sum-exp tensor."""
    out = torch.empty_like(q)
    bs_q = to_bshd(q, layout)
    bs_k, bs_v, bs_out = to_bshd(k, layout), to_bshd(v, layout), to_bshd(out, layout)
    lse = torch.empty((bs_q.shape[0], bs_q.shape[2], bs_q.shape[1]), device=q.device, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5

    def run() -> torch.Tensor:
        vendored_fwd_out(bs_q, bs_k, bs_v, bs_out, lse, scale)
        return out

    providers["vendored"] = run


def add_fp16_q8_providers(providers: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str) -> None:
    """fp16 storage with an E5M2-quantized Q tile. D64 and D128, no tile knob."""
    if q.dtype != torch.float16:
        return
    providers["feather_fp16_q8"] = lambda: feather_attn_fp16_q8(q, k, v, layout=layout)


def add_bf16_d128_providers(providers: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str, configs: tuple[int, ...]) -> None:
    """bf16 D128. The tile configuration is an operator argument, so each one is a provider."""
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        return
    for config in configs:
        providers[f"feather_bf16_d128@c{config}"] = lambda config=config: feather_attn_bf16_d128(q, k, v, layout=layout, config=config)
