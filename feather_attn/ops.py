import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from .aiter_tuning import install_aiter_tuned_configs


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
def _use_force_hnd_backward(
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


def _force_hnd_backward(
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
    softmax_scale: float,
    fused: bool,
) -> None:
    q_hnd = q.transpose(1, 2).contiguous()
    k_hnd = k.transpose(1, 2).contiguous()
    v_hnd = v.transpose(1, 2).contiguous()
    out_hnd = out.transpose(1, 2).contiguous()
    dout_hnd = dout.transpose(1, 2).contiguous()
    dq_hnd = torch.empty_like(q_hnd)
    dk_hnd = torch.empty_like(k_hnd)
    dv_hnd = torch.empty_like(v_hnd)
    if fused:
        dq_acc_hnd = torch.empty_like(q_hnd, dtype=torch.float32)
        torch.ops.feather_attn.attn_bwd_feather_fused.default(
            dout_hnd,
            q_hnd,
            k_hnd,
            v_hnd,
            out_hnd,
            softmax_lse,
            dq_hnd,
            dk_hnd,
            dv_hnd,
            delta,
            dq_acc_hnd,
            softmax_scale,
            0,
        )
    else:
        torch.ops.feather_attn.attn_bwd_feather.default(
            dout_hnd,
            q_hnd,
            k_hnd,
            v_hnd,
            out_hnd,
            softmax_lse,
            dq_hnd,
            dk_hnd,
            dv_hnd,
            delta,
            softmax_scale,
            0,
        )
    dq.copy_(dq_hnd.transpose(1, 2))
    dk.copy_(dk_hnd.transpose(1, 2))
    dv.copy_(dv_hnd.transpose(1, 2))


def feather_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float | None = None,
    layout: str = "NHD",
    fused: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    if layout == "HND":
        layout_id = 0
    elif layout == "NHD":
        layout_id = 1
    else:
        raise ValueError(f"layout must be 'HND' or 'NHD', got {layout}")

    force_hnd = layout_id == 1 and _use_force_hnd_backward(q, k, v, out, dout)
    if fused is None:
        fused = q.shape[-1] == 128
        if fused and layout == "NHD" and not force_hnd and q.shape[2] < 32 and k.shape[1] < 8192:
            fused = False
    if fused and q.shape[-1] != 128:
        fused = False

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(softmax_lse, dtype=torch.float32)
    softmax_scale = float(softmax_scale)
    if force_hnd:
        _force_hnd_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, softmax_scale, fused=fused)
    elif fused and q.shape[-1] == 128:
        dq_acc = torch.empty_like(q, dtype=torch.float32)
        torch.ops.feather_attn.attn_bwd_feather_fused.default(
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
            dq_acc,
            softmax_scale,
            layout_id,
        )
    else:
        torch.ops.feather_attn.attn_bwd_feather.default(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, softmax_scale, layout_id)
    return dq, dk, dv, delta


def _can_use_feather_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
) -> bool:
    return (
        dropout_p == 0.0
        and not causal
        and window_size == (-1, -1)
        and softcap == 0.0
        and alibi_slopes is None
        and not deterministic
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and q.shape[-1] == k.shape[-1] == v.shape[-1]
        and q.shape[-1] in (64, 128)
    )


def _aiter_attn_backward(
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
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    rng_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    install_aiter_tuned_configs(q.device)
    flash_attn_2.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        deterministic=deterministic,
        rng_state=rng_state,
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
        softcap: float,
        alibi_slopes: torch.Tensor | None,
        deterministic: bool,
        return_attn_probs: bool,
        is_grad_enabled: bool,
    ):
        if softcap != 0.0:
            raise NotImplementedError("AITER FlashAttention Triton forward does not support softcap")

        is_grad = is_grad_enabled and any(tensor.requires_grad for tensor in (q, k, v))
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        softmax_scale = float(softmax_scale)
        window_size = (int(window_size[0]), int(window_size[1]))
        install_aiter_tuned_configs(q.device)
        out, softmax_lse, s_dmask, rng_state = flash_attn_2.fwd(
            q,
            k,
            v,
            out=None,
            alibi_slopes=alibi_slopes,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            return_softmax=return_attn_probs or dropout_p > 0.0,
        )

        use_feather_backward = is_grad and _can_use_feather_backward(
            q,
            k,
            v,
            dropout_p,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
        )
        if use_feather_backward:
            use_feather_backward = out.shape == q.shape and softmax_lse.shape == (q.shape[0], q.shape[2], q.shape[1])

        if is_grad:
            backward_layout = None
            if use_feather_backward:
                if all(tensor.transpose(1, 2).is_contiguous() for tensor in (q, k, v, out)):
                    backward_layout = "HND"
                    softmax_lse = softmax_lse.contiguous()
                else:
                    backward_layout = "NHD"
                    q = q.contiguous()
                    k = k.contiguous()
                    v = v.contiguous()
                    out = out.contiguous()
                    softmax_lse = softmax_lse.contiguous()
            ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
            ctx.softmax_scale = softmax_scale
            ctx.dropout_p = dropout_p
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.use_feather_backward = use_feather_backward
            ctx.backward_layout = backward_layout

        if return_attn_probs:
            return out, softmax_lse, s_dmask
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        dout = grad_outputs[0]
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        if ctx.use_feather_backward:
            if ctx.backward_layout == "HND":
                dout = dout.transpose(1, 2).contiguous()
                dq, dk, dv, _ = feather_attn_backward(
                    dout,
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    out.transpose(1, 2),
                    softmax_lse,
                    softmax_scale=ctx.softmax_scale,
                    layout="HND",
                )
                dq = dq.transpose(1, 2)
                dk = dk.transpose(1, 2)
                dv = dv.transpose(1, 2)
            else:
                dout = dout.contiguous()
                dq, dk, dv, _ = feather_attn_backward(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    softmax_scale=ctx.softmax_scale,
                    layout="NHD",
                )
        else:
            dq, dk, dv = _aiter_attn_backward(
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
                ctx.softcap,
                ctx.alibi_slopes,
                ctx.deterministic,
                rng_state,
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

    Q/K/V use FlashAttention's BSHD logical layout.
    The forward pass uses AITER's FlashAttention Triton kernel. The backward
    pass uses the FeatherAttn kernels when the requested features are
    supported, otherwise it falls back to AITER's backward.
    """
    return _FeatherAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )
