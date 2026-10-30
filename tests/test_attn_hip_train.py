import inspect
from unittest.mock import patch

import pytest
import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from feather_attn import feather_attn_func


def _aiter_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    alibi_slopes: torch.Tensor | None,
):
    return flash_attn_2.fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=alibi_slopes,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=0.0,
        return_softmax=False,
    )


def _aiter_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
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
        0.0,
        softmax_scale,
        causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=0.0,
        deterministic=deterministic,
    )
    return dq, dk, dv


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=5.0e-3,
        rtol=2.0e-2,
    )


@pytest.mark.parametrize("head_dim", (64, 128), ids=lambda value: f"D{value}")
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float16, id="fp16"), pytest.param(torch.bfloat16, id="bf16")],
)
def test_feather_attn_func_backward_matches_aiter(head_dim: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch, seq_len, heads = 1, 512, 8
    shape = (batch, seq_len, heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    out = feather_attn_func(q, k, v)
    dout = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), dout)
    q_ref, k_ref, v_ref = (tensor.detach() for tensor in (q, k, v))
    out_ref, lse_ref, _, _ = _aiter_forward(
        q_ref,
        k_ref,
        v_ref,
        head_dim**-0.5,
        False,
        (-1, -1),
        None,
    )
    expected = _aiter_backward(
        dout,
        q_ref,
        k_ref,
        v_ref,
        out_ref,
        lse_ref,
        head_dim**-0.5,
        False,
        (-1, -1),
        None,
        False,
    )
    for actual_grad, expected_grad in zip(actual, expected):
        _assert_close(actual_grad, expected_grad)


def test_feather_attn_func_gqa_backward_matches_aiter(monkeypatch) -> None:
    torch.manual_seed(0)
    batch, heads_q, heads_kv, seq_len, head_dim = 1, 8, 2, 256, 128
    q = torch.randn((batch, seq_len, heads_q, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((batch, seq_len, heads_kv, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((batch, seq_len, heads_kv, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def fail_aiter_backward(*args, **kwargs):
        raise AssertionError("GQA should use the FeatherAttn backward")

    monkeypatch.setattr("feather_attn.ops._aiter_backward", fail_aiter_backward)
    out = feather_attn_func(q, k, v)
    dout = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), dout)

    q_ref, k_ref, v_ref = (tensor.detach() for tensor in (q, k, v))
    out_ref, lse_ref, _, _ = _aiter_forward(q_ref, k_ref, v_ref, head_dim**-0.5, False, (-1, -1), None)
    expected = _aiter_backward(dout, q_ref, k_ref, v_ref, out_ref, lse_ref, head_dim**-0.5, False, (-1, -1), None, False)
    _assert_close(out, out_ref)
    for actual_grad, expected_grad in zip(actual, expected):
        _assert_close(actual_grad, expected_grad)


def test_feather_attn_func_falls_back_for_causal_gqa() -> None:
    torch.manual_seed(0)
    batch, heads_q, heads_kv, seq_len, head_dim = 1, 4, 2, 128, 64
    q = torch.randn((batch, seq_len, heads_q, head_dim), device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn((batch, seq_len, heads_kv, head_dim), device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn((batch, seq_len, heads_kv, head_dim), device="cuda", dtype=torch.float16, requires_grad=True)
    out = feather_attn_func(q, k, v, causal=True)
    dout = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), dout)
    q_ref, k_ref, v_ref = (tensor.detach() for tensor in (q, k, v))
    out_ref, lse_ref, _, _ = _aiter_forward(
        q_ref,
        k_ref,
        v_ref,
        head_dim**-0.5,
        True,
        (-1, -1),
        None,
    )
    expected = _aiter_backward(
        dout,
        q_ref,
        k_ref,
        v_ref,
        out_ref,
        lse_ref,
        head_dim**-0.5,
        True,
        (-1, -1),
        None,
        False,
    )
    for actual_grad, expected_grad in zip(actual, expected):
        _assert_close(actual_grad, expected_grad)


def test_feather_attn_func_return_attn_probs() -> None:
    torch.manual_seed(0)
    q = torch.randn((1, 128, 2, 64), device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out, softmax_lse, probabilities = feather_attn_func(q, k, v, return_attn_probs=True)
    assert out.shape == q.shape
    assert softmax_lse.shape == (q.shape[0], q.shape[2], q.shape[1])
    assert probabilities.shape == (q.shape[0], q.shape[2], q.shape[1], q.shape[1])
    assert torch.isfinite(probabilities).all()
    torch.autograd.grad(out, (q, k, v), torch.randn_like(out))


def test_feather_attn_func_rejects_softcap() -> None:
    q = torch.randn((1, 128, 2, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(NotImplementedError, match="softcap"):
        feather_attn_func(q, k, v, softcap=0.5)


def test_feather_attn_func_falls_back_for_window() -> None:
    torch.manual_seed(0)
    shape = (1, 128, 2, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    window_size = (32, 0)
    out = feather_attn_func(q, k, v, window_size=window_size)
    dout = torch.randn_like(out)
    actual = torch.autograd.grad(out, (q, k, v), dout)
    q_ref, k_ref, v_ref = (tensor.detach() for tensor in (q, k, v))
    out_ref, lse_ref, _, _ = _aiter_forward(
        q_ref,
        k_ref,
        v_ref,
        64**-0.5,
        False,
        window_size,
        None,
    )
    expected = _aiter_backward(
        dout,
        q_ref,
        k_ref,
        v_ref,
        out_ref,
        lse_ref,
        64**-0.5,
        False,
        window_size,
        None,
        False,
    )
    for actual_grad, expected_grad in zip(actual, expected):
        _assert_close(actual_grad, expected_grad)


def test_feather_attn_func_dropout_backward_is_finite() -> None:
    torch.manual_seed(0)
    shape = (1, 128, 2, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    out = feather_attn_func(q, k, v, dropout_p=0.1)
    torch.autograd.grad(out, (q, k, v), torch.randn_like(out))


def test_feather_attn_func_matches_flash_attn_2_signature() -> None:
    parameters = inspect.signature(feather_attn_func).parameters
    assert list(parameters) == [
        "q",
        "k",
        "v",
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size",
        "softcap",
        "alibi_slopes",
        "deterministic",
        "return_attn_probs",
    ]
    assert parameters["dropout_p"].default == 0.0
    assert parameters["softmax_scale"].default is None
    assert parameters["causal"].default is False
    assert parameters["window_size"].default == (-1, -1)
    assert parameters["softcap"].default == 0.0
    assert parameters["alibi_slopes"].default is None
    assert parameters["deterministic"].default is False
    assert parameters["return_attn_probs"].default is False


@pytest.mark.parametrize("head_dim", (64, 128), ids=lambda value: f"D{value}")
def test_feather_attn_func_physical_bhsd_uses_hnd_without_layout_copy(head_dim: int) -> None:
    torch.manual_seed(0)
    batch, seq_len, heads = 1, 512, 8
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16, requires_grad=True).transpose(1, 2)
    k = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16, requires_grad=True).transpose(1, 2)
    v = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16, requires_grad=True).transpose(1, 2)
    with patch("feather_attn.ops._aiter_backward", side_effect=AssertionError("unexpected AITER fallback")):
        out = feather_attn_func(q, k, v)
        dout = torch.randn_like(out)
        actual = torch.autograd.grad(out, (q, k, v), dout)
        assert out.stride() == q.stride()
        assert all(gradient.stride() == q.stride() for gradient in actual)
        q_ref, k_ref, v_ref = (tensor.detach().contiguous() for tensor in (q, k, v))
        out_ref, lse_ref, _, _ = _aiter_forward(
            q_ref,
            k_ref,
            v_ref,
            head_dim**-0.5,
            False,
            (-1, -1),
            None,
        )
        expected = _aiter_backward(
            dout.contiguous(),
            q_ref,
            k_ref,
            v_ref,
            out_ref,
            lse_ref,
            head_dim**-0.5,
            False,
            (-1, -1),
            None,
            False,
        )
        for actual_grad, expected_grad in zip(actual, expected):
            _assert_close(actual_grad, expected_grad)
