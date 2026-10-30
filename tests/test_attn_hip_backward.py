from itertools import product

import pytest
import torch

from feather_attn import feather_attn_backward

flash_attn_amd = pytest.importorskip(
    "aiter.ops.triton._triton_kernels.flash_attn_triton_amd",
    reason="AITER is required to construct the saved forward state",
)
flash_attn_2 = flash_attn_amd.flash_attn_2
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU is required")

LAYOUTS = ("HND", "NHD")
DTYPES = ((torch.float16, "fp16"), (torch.bfloat16, "bf16"))
D64_SHAPES = (
    (1, 16, 33, 35),
    (1, 32, 65, 67),
    (1, 56, 65, 129),
    (2, 2, 65, 129),
    (1, 1, 4095, 4097),
    (1, 1, 4097, 4099),
    (1, 1, 8191, 67),
    (1, 1, 65, 8193),
    (1, 1, 16383, 67),
    (1, 1, 65, 16385),
    (1, 32, 65, 16385),
)
HEAD_DIMS = (64, 128)


def _backward_cases() -> tuple[object, ...]:
    return tuple(
        pytest.param(
            dtype,
            layout,
            batch,
            heads,
            n_q,
            n_kv,
            head_dim,
            id=f"{dtype_name}-{layout}-B{batch}-H{heads}-NQ{n_q}-NKV{n_kv}-D{head_dim}",
        )
        for dtype, dtype_name in DTYPES
        for layout, head_dim, (batch, heads, n_q, n_kv) in product(LAYOUTS, HEAD_DIMS, D64_SHAPES)
    )


def _hnd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "HND" else tensor.transpose(1, 2)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    scale: float,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = (_hnd(tensor, layout) for tensor in (q, k, v))
    out, dout = (_hnd(tensor, layout) for tensor in (out, dout))
    qf, kf, vf = q.float(), k.float(), v.float()
    outf, doutf = out.float(), dout.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    probabilities = torch.exp(scores - lse.float().unsqueeze(-1))
    delta = (outf * doutf).sum(-1)
    d_probability = torch.matmul(doutf, vf.transpose(-1, -2))
    d_score = probabilities * (d_probability - delta.unsqueeze(-1))
    dq = torch.matmul(d_score, kf) * scale
    dk = torch.matmul(d_score.transpose(-1, -2), qf) * scale
    dv = torch.matmul(probabilities.transpose(-1, -2), doutf)
    if layout == "NHD":
        dq, dk, dv = (tensor.transpose(1, 2) for tensor in (dq, dk, dv))
    return dq, dk, dv, delta


def _saved_state(
    batch: int,
    heads: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
    seed: int,
    layout: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q_hnd = torch.randn((batch, heads, n_q, head_dim), device="cuda", dtype=dtype, generator=generator)
    k_hnd = torch.randn((batch, heads, n_kv, head_dim), device="cuda", dtype=dtype, generator=generator)
    v_hnd = torch.randn((batch, heads, n_kv, head_dim), device="cuda", dtype=dtype, generator=generator)
    dout_hnd = torch.randn((batch, heads, n_q, head_dim), device="cuda", dtype=dtype, generator=generator)
    scale = head_dim**-0.5
    out_bshd, lse, _, _ = flash_attn_2.fwd(
        q_hnd.transpose(1, 2),
        k_hnd.transpose(1, 2),
        v_hnd.transpose(1, 2),
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    if layout == "HND":
        return q_hnd, k_hnd, v_hnd, out_bshd.transpose(1, 2).contiguous(), lse.contiguous(), dout_hnd, scale
    return (
        q_hnd.transpose(1, 2).contiguous(),
        k_hnd.transpose(1, 2).contiguous(),
        v_hnd.transpose(1, 2).contiguous(),
        out_bshd.contiguous(),
        lse.contiguous(),
        dout_hnd.transpose(1, 2).contiguous(),
        scale,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, n_kv: int) -> None:
    tolerance_factor = 0.10 if n_kv < 1024 else 0.05
    tolerance = tolerance_factor + tolerance_factor * expected.float().abs()
    difference = actual.float() - expected.float()
    tolerance_ratio = float((difference.abs() / tolerance).max())
    assert torch.isfinite(actual).all()
    assert torch.all(difference.abs() <= tolerance), f"tol_ratio={tolerance_ratio:.3g}"


@pytest.mark.parametrize(
    ("dtype", "layout", "batch", "heads", "n_q", "n_kv", "head_dim"),
    _backward_cases(),
)
def test_feather_attn_backward_matches_saved_state_reference(
    dtype: torch.dtype,
    layout: str,
    batch: int,
    heads: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> None:
    with torch.inference_mode():
        state = _saved_state(batch, heads, n_q, n_kv, head_dim, seed=0, layout=layout, dtype=dtype)
        q, k, v, out, lse, dout, scale = state
        expected = _reference(q, k, v, out, lse, dout, scale, layout)
        actual = feather_attn_backward(dout, q, k, v, out, lse, softmax_scale=scale, layout=layout)
        assert all(result.dtype == dtype for result in actual[:3])
        assert actual[3].dtype == torch.float32
        for result, reference in zip(actual, expected):
            _assert_close(result, reference, n_kv)


@pytest.mark.parametrize("dtype", [pytest.param(dtype, id=name) for dtype, name in DTYPES])
@pytest.mark.parametrize("head_dim", HEAD_DIMS, ids=lambda head_dim: f"D{head_dim}")
def test_nhd_hnd_backward_dispatch_is_exact(head_dim: int, dtype: torch.dtype) -> None:
    with torch.inference_mode():
        q, k, v, out, lse, dout, scale = _saved_state(1, 32, 8192, 8192, head_dim, seed=0, layout="NHD", dtype=dtype)
        direct = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
        direct_delta = torch.empty_like(lse)
        if head_dim == 128:
            direct_dq_acc = torch.empty_like(q, dtype=torch.float32)
            torch.ops.feather_attn.attn_bwd_feather_fused.default(
                dout,
                q,
                k,
                v,
                out,
                lse,
                *direct,
                direct_delta,
                direct_dq_acc,
                scale,
                1,
            )
        else:
            torch.ops.feather_attn.attn_bwd_feather.default(
                dout,
                q,
                k,
                v,
                out,
                lse,
                *direct,
                direct_delta,
                scale,
                1,
            )
        selected = feather_attn_backward(dout, q, k, v, out, lse, softmax_scale=scale)
        if head_dim == 128:
            _assert_close(selected[0], direct[0], q.shape[1])
            assert torch.equal(selected[1], direct[1])
            assert torch.equal(selected[2], direct[2])
            assert torch.equal(selected[3], direct_delta)
        else:
            for actual, expected in zip(selected, (*direct, direct_delta)):
                assert torch.equal(actual, expected)


def test_backward_rejects_mixed_attention_dtypes() -> None:
    with torch.inference_mode():
        q, k, v, out, lse, dout, scale = _saved_state(1, 8, 128, 128, 64, seed=0, layout="HND", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="same dtype"):
            feather_attn_backward(
                dout,
                q,
                k.to(torch.bfloat16),
                v,
                out,
                lse,
                softmax_scale=scale,
                layout="HND",
            )
