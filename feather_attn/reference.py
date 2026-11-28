"""The AITER control and the softmax scale, for the benchmarks, the tests, and the fallback.

The one place in this repo that calls `interface_v2.fwd`, so the argument order and the
defaults cannot drift between a benchmark and a test. The control takes and returns the
caller's physical layout. AITER is BSHD-only, so an HND call hands it a transposed view,
which is free.

The fp32 reconstruction and the error metrics are test-only and live in
`tests/reference.py`.
"""

import torch
from aiter.ops.triton._triton_kernels.attention.dao_ai import interface_v2 as flash_attn_2

from .layout import to_bshd


def default_softmax_scale(head_dim: int, softmax_scale: float | None = None) -> float:
    """The scale a caller gets when it does not pass one, `1/sqrt(head_dim)`."""
    return head_dim**-0.5 if softmax_scale is None else float(softmax_scale)


def aiter_fwd_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
    window: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AITER's Triton forward in BSHD, with its full option set."""
    out, lse, _, _ = flash_attn_2.fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=alibi_slopes,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window[0],
        window_size_right=window[1],
        softcap=0.0,
        return_softmax=False,
    )
    return out, lse


def aiter_fwd_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AITER's dense forward in the caller's layout, plus its log-sum-exp in `[B, H, N]`."""
    out, lse = aiter_fwd_bshd(
        to_bshd(q, layout),
        to_bshd(k, layout),
        to_bshd(v, layout),
        default_softmax_scale(q.shape[-1], softmax_scale),
    )
    return (out.transpose(1, 2) if layout == "HND" else out), lse.contiguous()


def aiter_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """AITER's dense forward in the caller's layout: the control for the benchmarks."""
    return aiter_fwd_lse(q, k, v, layout, softmax_scale)[0]
