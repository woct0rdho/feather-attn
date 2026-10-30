import torch
from torch.library import custom_op

from .aiter_kernels import bwd_aiter_out, fwd_aiter_out
from .aiter_vendored import fwd_out
from .layout import to_hnd


def feather_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    if layout == "HND":
        layout_id = 0
    elif layout == "NHD":
        layout_id = 1
    else:
        raise ValueError(f"layout must be 'HND' or 'NHD', got {layout}")

    out = torch.empty_like(q)
    torch.ops.feather_attn.attn_fp16_feather.default(q, k, v, out, layout_id)
    return out


# Benchmarked including all transpose and copy costs
def _use_force_hnd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
) -> bool:
    _batch, n_q, heads, head_dim = q.shape
    same_shape = q.shape == k.shape == v.shape == out.shape == dout.shape
    if not same_shape:
        # TODO: Heuristic for cross attention
        return False

    if head_dim == 64:
        if heads < 8:
            return False
        if heads < 16:
            min_seq_len = 16384
        elif heads < 24:
            min_seq_len = 24576
        elif heads < 40:
            min_seq_len = 10240
        else:
            min_seq_len = 8192
    elif head_dim == 128:
        if heads < 8:
            return False
        min_seq_len = 4096
    else:
        return False
    return n_q >= min_seq_len


def _is_bhsd_contiguous(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> bool:
    # A generator expression here breaks Dynamo tracing of the autograd Function.
    for tensor in (q, k, v, out):
        if not tensor.transpose(1, 2).is_contiguous():
            return False
    return True


@custom_op("feather_attn::bwd_out", mutates_args={"dq", "dk", "dv", "delta", "dq_acc"})
def _bwd_out(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    delta: torch.Tensor,
    dq_acc: torch.Tensor | None,
    softmax_scale: float,
    layout_id: int,
) -> None:
    # The fused kernel is only built for head_dim 128, and measured slower than the split
    # kernels for few heads / short keys. The key length is a runtime value under
    # `torch.compile`, so the choice is made here instead of branching in the traced graph.
    if q.shape[-1] == 128 and (layout_id == 0 or q.shape[2] >= 32 or k.shape[1] >= 8192):
        torch.ops.feather_attn.attn_bwd_feather_fused.default(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, dq_acc, softmax_scale, layout_id)
    else:
        torch.ops.feather_attn.attn_bwd_feather.default(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, softmax_scale, layout_id)


def _feather_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
):
    use_hnd = _is_bhsd_contiguous(q, k, v, out) or _use_force_hnd_bwd(q, k, v, out, dout)

    delta = torch.empty(softmax_lse.shape, dtype=torch.float32, device=softmax_lse.device)
    softmax_lse = softmax_lse.contiguous()

    if use_hnd:
        dout_h = to_hnd(dout)
        q_h = to_hnd(q)
        k_h = to_hnd(k)
        v_h = to_hnd(v)
        out_h = to_hnd(out)
        dq = torch.empty_like(q_h)
        dk = torch.empty_like(k_h)
        dv = torch.empty_like(v_h)
        dq_acc = torch.empty_like(q_h, dtype=torch.float32) if q_h.shape[-1] == 128 else None
        _bwd_out(dout_h, q_h, k_h, v_h, out_h, softmax_lse, dq, dk, dv, delta, dq_acc, softmax_scale, 0)
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), delta
    else:
        dout = dout.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = out.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq_acc = torch.empty_like(q, dtype=torch.float32) if q.shape[-1] == 128 else None
        _bwd_out(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, dq_acc, softmax_scale, 1)
        return dq, dk, dv, delta


def feather_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float | None = None,
    layout: str = "NHD",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    if layout == "HND":
        dq, dk, dv, delta = _feather_bwd(
            dout.transpose(1, 2),
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            out.transpose(1, 2),
            softmax_lse,
            float(softmax_scale),
        )
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), delta

    if layout != "NHD":
        raise ValueError(f"layout must be 'HND' or 'NHD', got {layout}")
    return _feather_bwd(dout, q, k, v, out, softmax_lse, float(softmax_scale))


def _supports_feather_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int],
    alibi_slopes: torch.Tensor | None,
) -> bool:
    """Features and dtypes both attention kernels accept, independent of the head layout."""
    return (
        dropout_p == 0.0
        and not causal
        and tuple(window_size) == (-1, -1)
        and alibi_slopes is None
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and q.shape[-1] in (64, 128)
    )


def _batch_and_head_dim_match(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return q.shape[0] == k.shape[0] == v.shape[0] and q.shape[-1] == k.shape[-1] == v.shape[-1]


def _can_use_feather_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int],
    alibi_slopes: torch.Tensor | None,
) -> bool:
    return (
        _batch_and_head_dim_match(q, k, v)
        and k.shape[2] == v.shape[2]
        and q.shape[2] % k.shape[2] == 0
        and _supports_feather_bwd(q, k, v, dropout_p, causal, window_size, alibi_slopes)
    )


def _aiter_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int],
    alibi_slopes: torch.Tensor | None,
):
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.zeros_like(softmax_lse, dtype=torch.float32)
    bwd_aiter_out(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        delta,
        alibi_slopes,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        dropout_p,
    )
    return dq, dk, dv


class _FeatherAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
        alibi_slopes: torch.Tensor | None,
        return_attn_probs: bool,
    ):
        is_grad = q.requires_grad or k.requires_grad or v.requires_grad
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        batch, seqlen_q, nheads_q, _ = q.shape
        out = torch.empty_like(q)
        softmax_lse = torch.empty((batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32)
        needs_scores = return_attn_probs or dropout_p > 0.0
        if needs_scores or causal or alibi_slopes is not None or window_size != (-1, -1):
            sd_mask = torch.zeros((batch, nheads_q, seqlen_q, k.shape[1]), device=q.device, dtype=torch.float32) if needs_scores else None
            fwd_aiter_out(
                q,
                k,
                v,
                out,
                softmax_lse,
                sd_mask,
                alibi_slopes,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                dropout_p,
                return_attn_probs,
            )
        else:
            sd_mask = None
            fwd_out(q, k, v, out, softmax_lse, softmax_scale)

        use_feather_bwd = is_grad and _can_use_feather_bwd(q, k, v, dropout_p, causal, window_size, alibi_slopes)
        if use_feather_bwd:
            use_feather_bwd = out.shape == q.shape and softmax_lse.shape == (q.shape[0], q.shape[2], q.shape[1])

        if is_grad:
            ctx.save_for_backward(q, k, v, out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.dropout_p = dropout_p
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.use_feather_bwd = use_feather_bwd

        if return_attn_probs:
            return out, softmax_lse, sd_mask
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout = grad_outputs[0]
        q, k, v, out, softmax_lse = ctx.saved_tensors
        if ctx.use_feather_bwd:
            group_size = q.shape[2] // k.shape[2]
            if group_size > 1:
                k_expanded = k.repeat_interleave(group_size, dim=2)
                v_expanded = v.repeat_interleave(group_size, dim=2)
                dq, dk_expanded, dv_expanded, _ = _feather_bwd(dout, q, k_expanded, v_expanded, out, softmax_lse, ctx.softmax_scale)
                batch, seqlen_k, _, head_dim = dk_expanded.shape
                dk = dk_expanded.view(batch, seqlen_k, k.shape[2], group_size, head_dim).sum(3)
                dv = dv_expanded.view(batch, seqlen_k, v.shape[2], group_size, head_dim).sum(3)
            else:
                dq, dk, dv, _ = _feather_bwd(dout, q, k, v, out, softmax_lse, ctx.softmax_scale)
        else:
            dq, dk, dv = _aiter_bwd(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                ctx.softmax_scale,
                ctx.dropout_p,
                ctx.causal,
                ctx.window_size,
                ctx.alibi_slopes,
            )
        return dq, dk, dv, *((None,) * (len(ctx.needs_input_grad) - 3))


def feather_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """FlashAttention 2-compatible dense attention with FeatherAttn backward.

    Q/K/V use FlashAttention's BSHD logical layout. The forward keeps GQA natively
    (k/v may have fewer heads), the backward expands k/v for the FeatherAttn kernels
    when the requested features are supported and otherwise falls back to AITER.
    """
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported")
    if deterministic:
        raise NotImplementedError("deterministic=True is not supported")
    return _FeatherAttnFunc.apply(
        q,
        k,
        v,
        float(dropout_p),
        float(softmax_scale) if softmax_scale is not None else None,
        bool(causal),
        (int(window_size[0]), int(window_size[1])),
        alibi_slopes,
        return_attn_probs,
    )
