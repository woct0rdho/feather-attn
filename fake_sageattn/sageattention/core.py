"""
Fake ``sageattention`` shim that redirects SageAttention calls to FeatherAttention.

This module provides a drop-in replacement for the public ``sageattn`` entry point
of the real `SageAttention <https://github.com/SageAttention/SageAttention>`_ package.
ComfyUI does::

    from sageattention import sageattn
    out = sageattn(q, k, v, **sage_kwargs)

where ``sage_kwargs`` is ``{"is_causal": False, "tensor_layout": "HND"|"NHD",
"sm_scale": None|float, "smooth_k": False}`` plus an optional ``attn_mask``.

Instead of running the (NVIDIA-only) SageAttention kernels, every call is forwarded to
FeatherAttention (``feather_attn`` / ``feather_attn_backward``), which is a ROCm-native
attention kernel for RDNA3/3.5.

Behaviour notes / compatibility decisions:

* The top-level signature deliberately mirrors the real ``sageattn`` and does **not**
  declare ``attn_mask`` as a named parameter. This keeps
  ``inspect.signature(sageattn).parameters`` identical to the real package, so ComfyUI's
  ``SAGE_ATTENTION_SUPPORTS_MASK`` stays ``False`` (masked attention falls back to
  PyTorch attention, exactly like the real package).
* FeatherAttention is FP16-only; bf16 inputs are round-tripped through FP16.
* FeatherAttention supports head_dim 64/128 (and values that pad to those sizes).
  Other head_dims raise ``NotImplementedError`` so ComfyUI can fall back.
* ``is_causal``, ``smooth_k`` and ``attn_mask`` are not supported by FeatherAttention
  and raise ``NotImplementedError`` (ComfyUI always passes ``is_causal=False``,
  ``smooth_k=False`` and, given the mask-support flag above, never passes a mask).
* If the ``feather_attn`` extension is not importable (non-ROCm build, not installed),
  calling ``sageattn`` raises ``RuntimeError`` so ComfyUI's exception handler gracefully
  falls back to PyTorch attention.
* Autograd is supported via ``feather_attn_backward``. When gradients are not enabled
  (the common inference case), the log-sum-exp computation is skipped entirely.
"""

from typing import Any, Optional, Union

import math

import torch
import torch.nn.functional as F

try:
    from feather_attn import feather_attn as _feather_attn
    from feather_attn import feather_attn_backward as _feather_attn_backward
    _FEATHER_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the build environment
    _feather_attn = None
    _feather_attn_backward = None
    _FEATHER_AVAILABLE = False

#: Head dims that the FeatherAttention kernels natively support.
_FEATHER_SUPPORTED_HEAD_DIMS = (64, 128)


def _require_feather() -> None:
    """Raise a friendly error when the FeatherAttention extension is unavailable."""
    if not _FEATHER_AVAILABLE:
        raise RuntimeError(
            "FeatherAttention is not importable in this environment (the `feather_attn` "
            "extension is only built for a ROCm PyTorch build). Falling back to PyTorch "
            "attention."
        )


def _check_cuda(q: torch.Tensor) -> None:
    """Validate the inputs live on a GPU. Overridable by tests via monkeypatching."""
    assert q.is_cuda, "Input tensors must be on cuda."


class _FeatherAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the raw FeatherAttention forward/backward ops.

    ``q``, ``k``, ``v`` are expected to be FP16, contiguous and already (optionally)
    padded and scaled so that FeatherAttention's internal ``1 / sqrt(head_dim)`` scale
    produces the desired softmax scale. The log-sum-exp tensor is computed in HND order
    ``[batch, num_heads, seqlen]`` (which is what ``feather_attn_backward`` expects
    regardless of the data layout).
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: str,
        head_dim_padded: int,
    ) -> torch.Tensor:
        out = _feather_attn(q, k, v, layout=layout)

        if any(ctx.needs_input_grad):
            inv_sqrt_d = head_dim_padded ** -0.5
            if layout == "HND":  # q, k are [B, H, L, D]
                scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * inv_sqrt_d
                lse = scores.logsumexp(dim=-1)  # [B, H, Lq]
            else:  # q, k are [B, L, H, D]
                scores = torch.matmul(q.float().transpose(1, 2), k.float().transpose(1, 2).transpose(-1, -2)) * inv_sqrt_d
                lse = scores.logsumexp(dim=-1)  # [B, H, Lq]
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.layout = layout
            ctx.head_dim_padded = head_dim_padded

        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv, _delta = _feather_attn_backward(
            grad_out.contiguous(),
            q,
            k,
            v,
            out,
            lse,
            softmax_scale=ctx.head_dim_padded ** -0.5,
            layout=ctx.layout,
        )
        return dq, dk, dv, None, None


def _compute_lse(q: torch.Tensor, k: torch.Tensor, layout: str, head_dim_padded: int) -> torch.Tensor:
    """Reference log-sum-exp of the (already scaled + padded) attention scores, HND order."""
    inv_sqrt_d = head_dim_padded ** -0.5
    if layout == "HND":  # q, k are [B, H, L, D]
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * inv_sqrt_d
        return scores.logsumexp(dim=-1)  # [B, H, Lq]
    # q, k are [B, L, H, D]
    scores = torch.matmul(q.float().transpose(1, 2), k.float().transpose(1, 2).transpose(-1, -2)) * inv_sqrt_d
    return scores.logsumexp(dim=-1)  # [B, H, Lq]


def _plan_padding(head_dim: int) -> tuple[int, int]:
    """Return ``(head_dim_padded, pad)`` for a given original head dim.

    Head dims 64/128 map to themselves. Anything below 64 pads to 64, anything below 128
    pads to 128 flavour. Anything else is unsupported.
    """
    if head_dim in _FEATHER_SUPPORTED_HEAD_DIMS:
        return head_dim, 0
    if head_dim < 64:
        return 64, 64 - head_dim
    if head_dim < 128:
        return 128, 128 - head_dim
    raise NotImplementedError(
        f"FeatherAttention shim: unsupported head_dim {head_dim} (supports 64/128, or "
        f"values that pad to 64/128)."
    )


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Drop-in replacement for ``sageattention.sageattn`` that forwards to FeatherAttention.

    See the real SageAttention docstring for the shape/parameter contract. In short:
    ``tensor_layout`` is ``"HND"`` (``[B, H, L, D]``) or ``"NHD"`` (``[B, L, H, D]``);
    ``is_causal`` is unsupported (raises); ``sm_scale`` defaults to ``1 / sqrt(head_dim)``.
    """
    _require_feather()

    _check_cuda(q)
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"Input tensors must be of dtype torch.float16 or torch.bfloat16, got {q.dtype}"
    )
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if is_causal:
        raise NotImplementedError("FeatherAttention shim: causal attention is not supported.")

    smooth_k = kwargs.get("smooth_k", False)
    attn_mask = kwargs.get("attn_mask", None)
    if smooth_k:
        raise NotImplementedError("FeatherAttention shim: smooth_k is not supported.")
    if attn_mask is not None:
        raise NotImplementedError("FeatherAttention shim: attn_mask is not supported.")

    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"tensor_layout must be 'HND' or 'NHD', got {tensor_layout!r}")

    orig_dtype = q.dtype
    head_dim_og = q.size(-1)
    head_dim_padded, pad = _plan_padding(head_dim_og)

    if sm_scale is None:
        sm_scale = head_dim_og ** -0.5

    # Scale q,k so that FeatherAttention's internal 1/sqrt(head_dim_padded) yields the
    # requested softmax scale: c = sqrt(sm_scale * sqrt(head_dim_padded)).
    c = (sm_scale * (head_dim_padded ** 0.5)) ** 0.5
    use_default_scale = math.isclose(sm_scale, head_dim_padded ** -0.5, rel_tol=1.0e-6)

    # --- prepare inputs (dtype cast / scale / pad / contiguity) -----------------------
    if q.dtype != torch.float16:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    if not use_default_scale:
        q = q * c
        k = k * c
    if pad:
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # --- run FeatherAttention ----------------------------------------------------------
    out = _FeatherAttentionFunction.apply(q, k, v, tensor_layout, head_dim_padded)

    if pad:
        out = out[..., :head_dim_og]
    if orig_dtype != torch.float16:
        out = out.to(orig_dtype)

    if return_lse:
        lse = _compute_lse(q, k, tensor_layout, head_dim_padded)
        return out, lse
    return out


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Variable-length attention is not implemented by FeatherAttention."""
    raise NotImplementedError(
        "FeatherAttention shim: `sageattn_varlen` is not supported (variable-length "
        "attention is not implemented by FeatherAttention)."
    )