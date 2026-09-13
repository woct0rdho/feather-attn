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
    torch.ops.feather_attn_fp16.attn_fp16_feather.default(q, k, v, out, layout_id)
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
        if heads < 16:
            min_seq_len = 16384
        elif heads < 24:
            min_seq_len = 8192
        elif heads < 32:
            min_seq_len = 16384
        elif heads < 48:
            min_seq_len = 8192
        else:
            min_seq_len = 5120
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
) -> None:
    q_hnd = q.transpose(1, 2).contiguous()
    k_hnd = k.transpose(1, 2).contiguous()
    v_hnd = v.transpose(1, 2).contiguous()
    out_hnd = out.transpose(1, 2).contiguous()
    dout_hnd = dout.transpose(1, 2).contiguous()
    dq_hnd = torch.empty_like(q_hnd)
    dk_hnd = torch.empty_like(k_hnd)
    dv_hnd = torch.empty_like(v_hnd)
    torch.ops.feather_attn_fp16.attn_bwd_fp16_feather.default(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if layout == "HND":
        layout_id = 0
    elif layout == "NHD":
        layout_id = 1
    else:
        raise ValueError(f"layout must be 'HND' or 'NHD', got {layout}")
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(softmax_lse, dtype=torch.float32)
    softmax_scale = float(softmax_scale)
    if layout_id == 1 and _use_force_hnd_backward(q, k, v, out, dout):
        _force_hnd_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, softmax_scale)
    else:
        torch.ops.feather_attn_fp16.attn_bwd_fp16_feather.default(dout, q, k, v, out, softmax_lse, dq, dk, dv, delta, softmax_scale, layout_id)
    return dq, dk, dv, delta
