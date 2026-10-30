"""Minimal dense FlashAttention 2 forward kernel for gfx1151 (vendored from AITER).

Only what FeatherAttn needs for LoRA training is kept: dense BSHD, fp16/bf16,
non-causal, native GQA (`HK <= HQ`). Dropout, bias, fp8, alibi, varlen,
sliding window and return_scores are dropped (the AITER launcher handles those
cases, see `kernels.fwd_aiter_out`).
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# Tile configs tuned for gfx1151
_CONFIGS = {
    64: {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1},
    128: {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 8, "num_stages": 1, "waves_per_eu": 6},
}


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    LSE,
    Out,
    seqlen_q,
    seqlen_k,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471824645996
    GROUP_SIZE: tl.constexpr = HQ // HK

    off_h_q = tl.program_id(0)
    start_m = tl.program_id(1)
    off_z = tl.program_id(2)
    if GROUP_SIZE != 1:  # GQA / MQA: query head -> shared kv head
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    # Shape-derived values are runtime arguments (not constexpr), so the kernel does
    # not specialize on the sequence length: one compiled kernel serves every bucket.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    m_mask = offs_m < seqlen_q

    q_ptrs = Q + off_z * stride_qz + off_h_q * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in tl.range(0, seqlen_k, BLOCK_N, num_stages=1):
        kv_offs_n = start_n + offs_n
        n_mask = kv_offs_n < seqlen_k
        k_ptrs = k_base + offs_d[:, None] * stride_kk + kv_offs_n[None, :] * stride_kn
        v_ptrs = v_base + kv_offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)

        qk = tl.dot(q, k) * SM_SCALE
        qk = tl.where(n_mask[None, :] & m_mask[:, None], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted * RCP_LN2)
        l_ij = tl.sum(p, 1)

        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff * RCP_LN2)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.dtype), v, acc=acc)

    # Rows without any visible key (fully masked) keep acc = 0 and LSE = -inf.
    invalid = m_i == float("-inf")
    l_safe = tl.where(invalid, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = (m_i * RCP_LN2 + tl.math.log2(l_i)) * LN2
    lse = tl.where(invalid, float("-inf"), lse)

    l_ptrs = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + offs_m * stride_lse_m
    tl.store(l_ptrs, lse, mask=m_mask)

    o_ptrs = Out + off_z * stride_oz + off_h_q * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask[:, None])


@triton_op("feather_attn::fwd_out", mutates_args={"out", "softmax_lse"})
def fwd_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
) -> None:
    # Head count / head dim are model constants and become constexpr tile parameters.
    # The sequence lengths stay runtime values so one kernel serves every bucket.
    head_dim = int(q.shape[-1])
    cfg = _CONFIGS.get(head_dim)
    if cfg is None:
        raise NotImplementedError(f"feather_attn forward supports head_dim 64/128, got {head_dim}")
    batch, seqlen_q, heads_q, _ = q.shape
    seqlen_k = k.shape[1]

    def grid(meta):
        return (int(heads_q), triton.cdiv(seqlen_q, meta["BLOCK_M"]), batch)

    wrap_triton(_attn_fwd)[grid](
        q,
        k,
        v,
        softmax_lse,
        out,
        seqlen_q,
        seqlen_k,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        q.stride(3),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        k.stride(3),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        HQ=int(heads_q),
        HK=int(k.shape[2]),
        BLOCK_DMODEL=head_dim,
        SM_SCALE=softmax_scale,
        **cfg,
    )
