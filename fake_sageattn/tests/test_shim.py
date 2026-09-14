"""Smoke tests for the fake ``sageattention`` shim (no GPU / no feather_attn needed).

These tests substitute a tiny CPU "feather" stub for the real native kernels so the
shim's scale absorption, padding and autograd wiring can be verified end-to-end on any
machine. The real ``feather_attn`` extension is never required.
"""

import math
import sys
from pathlib import Path

import torch

# Make the shim importable (repo root is two levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sageattention.core as core


def _to_hnd(t, layout):
    return t if layout == "HND" else t.transpose(1, 2)  # -> [B, H, L, D]


def _from_hnd(t, layout):
    return t if layout == "HND" else t.transpose(1, 2)  # -> [B, L, H, D]


def _reference_attn(q, k, v, layout, scale):
    """Exact fp32 attention used as ground truth for the forward."""
    qh, kh, vh = _to_hnd(q, layout), _to_hnd(k, layout), _to_hnd(v, layout)
    qf, kf, vf = qh.float(), kh.float(), vh.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    w = scores.softmax(dim=-1)
    return _from_hnd(torch.matmul(w, vf), layout).to(q.dtype)


def _install_stub(layout, head_dim):
    """Fake the native feather kernels with a CPU fp16 reference that keeps the same
    scale convention (softmax(qk^T / sqrt(head_dim))). ``head_dim`` is the padded dim."""
    inv = head_dim ** -0.5

    def _fa(q, k, v, layout=layout):
        assert q.dtype == torch.float16, q.dtype
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert q.stride(-1) == 1
        return _reference_attn(q, k, v, layout, inv)

    def _fab(dout, q, k, v, out, lse, softmax_scale=None, layout=layout):
        scale = softmax_scale if softmax_scale is not None else inv
        qh, kh, vh = _to_hnd(q, layout), _to_hnd(k, layout), _to_hnd(v, layout)
        outh, douth = _to_hnd(out, layout), _to_hnd(dout, layout)
        qf, kf, vf, outf, doutf = (t.float() for t in (qh, kh, vh, outh, douth))
        # lse is already in HND order [B, H, Lq]
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        probs = (scores - lse.float().unsqueeze(-1)).exp()
        delta = (outf * doutf).sum(-1)
        dp = torch.matmul(doutf, vf.transpose(-1, -2))
        ds = probs * (dp - delta.unsqueeze(-1))
        dq = torch.matmul(ds, kf) * scale
        dk = torch.matmul(ds.transpose(-1, -2), qf) * scale
        dv = torch.matmul(probs.transpose(-1, -2), doutf)
        return (
            _from_hnd(dq, layout).to(torch.float16),
            _from_hnd(dk, layout).to(torch.float16),
            _from_hnd(dv, layout).to(torch.float16),
            delta.to(torch.float32),
        )

    core._feather_attn = _fa
    core._feather_attn_backward = _fab
    core._FEATHER_AVAILABLE = True


def _run_case(layout, head_dim, batch, heads, n_q, n_kv, scale=None, dtype=torch.float16):
    head_dim_padded, _pad = core._plan_padding(head_dim)
    _install_stub(layout, head_dim_padded)  # stub's internal scale uses the PADDED dim
    core._check_cuda = lambda _t: None  # CPU smoke test bypass
    torch.manual_seed(0)
    if layout == "HND":
        q = torch.randn(batch, heads, n_q, head_dim, dtype=dtype)
        k = torch.randn(batch, heads, n_kv, head_dim, dtype=dtype)
        v = torch.randn(batch, heads, n_kv, head_dim, dtype=dtype)
    else:
        q = torch.randn(batch, n_q, heads, head_dim, dtype=dtype)
        k = torch.randn(batch, n_kv, heads, head_dim, dtype=dtype)
        v = torch.randn(batch, n_kv, heads, head_dim, dtype=dtype)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    sm_scale = head_dim ** -0.5 if scale is None else scale

    # forward
    with torch.no_grad():
        out = core.sageattn(q, k, v, tensor_layout=layout, sm_scale=sm_scale, smooth_k=False)
    expected = _reference_attn(q, k, v, layout, sm_scale)
    err = float((out.float() - expected.float()).abs().max())
    assert out.shape == expected.shape, (out.shape, expected.shape)
    assert out.dtype == dtype
    assert err < 0.02, (layout, head_dim, err)

    # autograd (backward via feather_attn_backward) vs. exact reference autograd.
    # Use a deterministic upstream gradient so both paths are compared on the same values.
    torch.manual_seed(1)
    up = torch.randn_like(q)

    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    outg = core.sageattn(qg, kg, vg, tensor_layout=layout, sm_scale=sm_scale, smooth_k=False)
    (outg * up).sum().backward()
    dq_got, dk_got, dv_got = qg.grad, kg.grad, vg.grad

    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    vr = v.clone().requires_grad_(True)
    outr = _reference_attn(qr, kr, vr, layout, sm_scale)
    (outr * up).sum().backward()
    dq_want, dk_want, dv_want = qr.grad, kr.grad, vr.grad

    # fp32-based reference grads vs fp16 shim grads: allow a small relative tolerance.
    for name, got, want in (
        ("dq", dq_got.float(), dq_want.float()),
        ("dk", dk_got.float(), dk_want.float()),
        ("dv", dv_got.float(), dv_want.float()),
    ):
        scale_n = want.abs().max().clamp_min(1.0)
        rel_err = float((got - want).abs().max() / scale_n)
        assert rel_err < 0.05, (layout, head_dim, name, rel_err)


def test_forward_and_backward():
    for layout in ("HND", "NHD"):
        for head_dim, n_q, n_kv in ((64, 33, 35), (128, 16, 16), (40, 20, 20)):
            for dtype in (torch.float16, torch.bfloat16):
                _run_case(layout, head_dim, 1, 4, n_q, n_kv, dtype=dtype)


def test_signature_mask_flag():
    import inspect
    params = list(inspect.signature(core.sageattn).parameters)
    assert "attn_mask" not in params, params


def test_graceful_degradation():
    core._FEATHER_AVAILABLE = False
    try:
        core.sageattn(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64))
        assert False, "should have raised"
    except RuntimeError:
        pass
    core._FEATHER_AVAILABLE = True


def test_unsupported_paths():
    _install_stub("HND", 64)
    q = torch.randn(1, 4, 8, 64, dtype=torch.float16).contiguous()
    for kwargs, exc in (
        (dict(is_causal=True), NotImplementedError),
        (dict(smooth_k=True), NotImplementedError),
        (dict(attn_mask=torch.ones(1, 4, 8, 8, dtype=torch.float16)), NotImplementedError),
        (dict(tensor_layout="XYZ"), ValueError),
    ):
        try:
            core.sageattn(q, q, q, **kwargs)
            assert False, f"should have raised for {kwargs}"
        except exc:
            pass
    # head_dim 256 (would pad past 128) unsupported
    q = torch.randn(1, 4, 8, 256, dtype=torch.float16).contiguous()
    try:
        core.sageattn(q, q, q)
        assert False, "should have raised"
    except NotImplementedError:
        pass


if __name__ == "__main__":
    test_forward_and_backward()
    test_signature_mask_flag()
    test_graceful_degradation()
    test_unsupported_paths()
    print("all fake-sageattn smoke tests passed")