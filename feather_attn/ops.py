import torch


def feather_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str = "NHD",
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
    fused: bool = False,
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
