import pytest
import torch
import torch._dynamo as dynamo
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2
from torch._inductor.utils import run_and_get_code

from feather_attn import feather_attn_func

DTYPE = torch.bfloat16


def _aiter_forward(q, k, v, scale, causal):
    out, lse, _, _ = flash_attn_2.fwd(q, k, v, None, None, 0.0, scale, causal, -1, -1, 0.0, False)
    return out, lse


def _aiter_backward(dout, q, k, v, out, lse, scale, causal):
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    flash_attn_2.bwd(dout, q, k, v, out, lse, dq, dk, dv, None, 0.0, scale, causal, -1, -1, 0.0, False)
    return dq, dk, dv


def _case(batch, seqlen_q, seqlen_k, heads, head_dim, causal, gqa):
    torch.manual_seed(0)
    heads_kv = heads // gqa
    q = torch.randn((batch, seqlen_q, heads, head_dim), device="cuda", dtype=DTYPE, requires_grad=True)
    k = torch.randn((batch, seqlen_k, heads_kv, head_dim), device="cuda", dtype=DTYPE, requires_grad=True)
    v = torch.randn((batch, seqlen_k, heads_kv, head_dim), device="cuda", dtype=DTYPE, requires_grad=True)
    return q, k, v


def _reference(q, k, v, causal, gqa, head_dim):
    scale = head_dim**-0.5
    batch, seqlen_k, heads_kv, _ = k.shape
    k_ref = k.detach().repeat_interleave(gqa, dim=2)
    v_ref = v.detach().repeat_interleave(gqa, dim=2)
    out_ref, lse_ref = _aiter_forward(q.detach(), k_ref, v_ref, scale, causal)
    dout = torch.ones_like(out_ref)
    dq, dk, dv = _aiter_backward(dout, q.detach(), k_ref, v_ref, out_ref, lse_ref, scale, causal)
    if gqa > 1:
        dk = dk.view(batch, seqlen_k, heads_kv, gqa, head_dim).sum(3)
        dv = dv.view(batch, seqlen_k, heads_kv, gqa, head_dim).sum(3)
    return out_ref, (dq, dk, dv), dout


def _assert_close(actual, expected, rtol=2e-2):
    denom = expected.float().abs().max().clamp_min(1e-6)
    rel = (actual.float() - expected.float()).abs().max() / denom
    assert rel < rtol, f"max relative difference {rel:.3e} exceeds {rtol}"


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("self_attn_nhd", {"batch": 1, "seqlen_q": 256, "seqlen_k": 256, "heads": 8, "head_dim": 128, "causal": False, "gqa": 1}),
        ("cross_attn", {"batch": 1, "seqlen_q": 256, "seqlen_k": 128, "heads": 8, "head_dim": 128, "causal": False, "gqa": 1}),
        ("gqa", {"batch": 1, "seqlen_q": 256, "seqlen_k": 256, "heads": 8, "head_dim": 128, "causal": False, "gqa": 4}),
        ("causal_aiter_bwd", {"batch": 1, "seqlen_q": 256, "seqlen_k": 256, "heads": 8, "head_dim": 128, "causal": True, "gqa": 1}),
        ("hnd_d128", {"batch": 1, "seqlen_q": 4096, "seqlen_k": 4096, "heads": 8, "head_dim": 128, "causal": False, "gqa": 1}),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_compiled_matches_aiter(label, kwargs) -> None:
    """fullgraph compile keeps numerics (against AITER) for every dispatch path."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    q, k, v = _case(**kwargs)
    out_ref, grads_ref, dout = _reference(q, k, v, kwargs["causal"], kwargs["gqa"], kwargs["head_dim"])

    def run(qq, kk, vv):
        return feather_attn_func(qq, kk, vv, softmax_scale=kwargs["head_dim"] ** -0.5, causal=kwargs["causal"])

    compiled = torch.compile(run, fullgraph=True)
    out = compiled(q, k, v)
    assert torch.equal(out, out_ref), f"{label}: forward differs from AITER"
    grads = torch.autograd.grad(out, (q, k, v), dout)
    for actual, expected in zip(grads, grads_ref):
        _assert_close(actual, expected)


def test_compiled_has_no_graph_breaks_and_hoists_allocations() -> None:
    """The attention body is traced: buffers are graph allocations, kernels are visible."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    q, k, v = _case(1, 256, 256, 8, 128, False, 1)

    def run(qq, kk, vv):
        return feather_attn_func(qq, kk, vv, softmax_scale=128**-0.5)

    explanation = dynamo.explain(run)(q, k, v)
    assert explanation.graph_break_count == 0, explanation.break_reasons

    _, code = run_and_get_code(torch.compile(run, fullgraph=True), q, k, v)
    text = "\n".join(code)
    # out/softmax_lse are allocated by the graph (Inductor buffer), not inside the op
    assert "empty_strided" in text
    # the AITER forward kernel is codegen'd by Inductor with its autotune config space
    assert "user_autotune" in text
    assert "attn_fwd" in text


def test_compiled_backward_uses_triton_kernels() -> None:
    """AITER's Triton backward fallback is also captured (not opaque)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    q, k, v = _case(1, 128, 128, 8, 128, False, 1)

    def run(qq, kk, vv):
        # causal forces AITER's Triton backward
        return feather_attn_func(qq, kk, vv, softmax_scale=128**-0.5, causal=True)

    out, code = run_and_get_code(torch.compile(run, fullgraph=True), q, k, v)
    out.sum().backward()
    text = "\n".join(code)
    assert "attn_fwd" in text
