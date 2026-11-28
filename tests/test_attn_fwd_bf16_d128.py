"""bf16 D128 forward contract.

The kernel keeps Q and K exact and rounds only its own probability tile to bf16, with a
truncating conversion whose half-ulp bias is cancelled in beta, so its expected error is a
few 1e-3 relative L2: the gate below is `atol=rtol=0.02` plus `rel_l2 <= 1e-2`, measured at
5.0e-3 against AITER and 4.4e-3 against an fp32 reconstruction. Cosine is not part of the
forward contract. It is the backward's primary metric.
"""

import pytest
import torch

from feather_attn.contract import LAYOUTS
from feather_attn.layout import to_hnd
from feather_attn.ops_bf16 import feather_attn_bf16_d128
from feather_attn.reference import aiter_fwd
from tests.reference import elementwise_ratio, fp32_fwd, rel_l2
from tests.shapes import SMALL_SHAPES, contract_shapes

BATCH = 1
HEAD_DIM = 128
DTYPE = torch.bfloat16
RTOL = 0.02
REL_L2 = 1.0e-2
# The log-sum-exp is an fp32 online recurrence over the whole key range: measured 5.3e-6
# against fp64 at 4096 keys, so the gate carries an order of magnitude of headroom.
LSE_ATOL = 5.0e-5
CONFIGS = (0, 1, 2, 3, 4)
# Config 0 is the heuristic, which resolves to a different tile per layout on purpose.
PINNED_CONFIGS = (1, 2, 3, 4)

CASES = tuple(pytest.param(layout, *shape, id=f"{layout}-B{shape[0]}H{shape[1]}NQ{shape[2]}NKV{shape[3]}") for layout in LAYOUTS for shape in contract_shapes(HEAD_DIM))
SMALL_CASES = tuple(
    pytest.param(layout, batch, heads, n_q, n_kv, id=f"{layout}-B{batch}H{heads}NQ{n_q}NKV{n_kv}") for layout in LAYOUTS for batch, heads, n_q, n_kv in SMALL_SHAPES
)


def _operands(layout: str, batch: int, heads: int, n_q: int, n_kv: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (batch, heads, n_q, HEAD_DIM) if layout == "HND" else (batch, n_q, heads, HEAD_DIM)
    shape_kv = (batch, heads, n_kv, HEAD_DIM) if layout == "HND" else (batch, n_kv, heads, HEAD_DIM)
    return (
        torch.randn(shape_q, device="cuda", dtype=DTYPE, generator=generator),
        torch.randn(shape_kv, device="cuda", dtype=DTYPE, generator=generator),
        torch.randn(shape_kv, device="cuda", dtype=DTYPE, generator=generator),
    )


def _check(actual: torch.Tensor, expected: torch.Tensor, n_kv: int, label: str) -> None:
    assert torch.isfinite(actual).all(), f"{label}: output is not finite"
    worst = elementwise_ratio(actual, expected, RTOL)
    assert worst <= 1.0, f"{label}: worst element at {worst:.3g} of the atol=rtol={RTOL} envelope"
    error = rel_l2(actual, expected)
    assert error <= REL_L2, f"{label}: rel_l2={error:.3g} exceeds {REL_L2:.3g}"
    assert n_kv > 0


@pytest.mark.parametrize(("layout", "batch", "heads", "n_q", "n_kv", "head_dim"), CASES)
def test_bf16_d128_matches_aiter(layout, batch, heads, n_q, n_kv, head_dim) -> None:
    with torch.inference_mode():
        q, k, v = _operands(layout, batch, heads, n_q, n_kv, seed=0)
        actual, _ = feather_attn_bf16_d128(q, k, v, layout=layout)
        _check(actual, aiter_fwd(q, k, v, layout), n_kv, f"{layout} NQ{n_q} NKV{n_kv}")


@pytest.mark.parametrize(("layout", "batch", "heads", "n_q", "n_kv"), SMALL_CASES)
def test_bf16_d128_matches_fp32(layout, batch, heads, n_q, n_kv) -> None:
    """An anchor that does not depend on AITER, on the shapes cheap enough to reconstruct."""
    with torch.inference_mode():
        q, k, v = _operands(layout, batch, heads, n_q, n_kv, seed=1)
        actual, _ = feather_attn_bf16_d128(q, k, v, layout=layout)
        _check(actual, fp32_fwd(q, k, v, layout), n_kv, f"{layout} NQ{n_q} NKV{n_kv} vs fp32")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("config", CONFIGS)
def test_bf16_d128_every_config_launches(config, layout) -> None:
    """Every tile configuration the operator accepts must be an instantiated, accurate image."""
    with torch.inference_mode():
        q, k, v = _operands(layout, 1, 3, 300, 200, seed=2)
        actual, _ = feather_attn_bf16_d128(q, k, v, layout=layout, config=config)
        _check(actual, aiter_fwd(q, k, v, layout), 200, f"config={config} {layout}")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("softmax_scale", (0.15, 0.05))
def test_bf16_d128_honours_softmax_scale(softmax_scale, layout) -> None:
    """A non-default scale must reach the kernel. It is an operator argument, not a constant."""
    with torch.inference_mode():
        q, k, v = _operands(layout, 1, 3, 300, 200, seed=3)
        actual, _ = feather_attn_bf16_d128(q, k, v, layout=layout, softmax_scale=softmax_scale)
        expected = aiter_fwd(q, k, v, layout, softmax_scale=softmax_scale)
        _check(actual, expected, 200, f"scale={softmax_scale} {layout}")
        assert not torch.allclose(actual, feather_attn_bf16_d128(q, k, v, layout=layout)[0])


def test_bf16_d128_rejects_fp16_operands() -> None:
    with torch.inference_mode():
        q, k, v = _operands("HND", 1, 2, 128, 128, seed=4)
        with pytest.raises(RuntimeError):
            feather_attn_bf16_d128(q.half(), k.half(), v.half(), layout="HND")


def test_bf16_d128_rejects_d64() -> None:
    with torch.inference_mode():
        generator = torch.Generator(device="cuda").manual_seed(5)
        q = torch.randn((1, 2, 128, 64), device="cuda", dtype=DTYPE, generator=generator)
        with pytest.raises(RuntimeError):
            feather_attn_bf16_d128(q, q, q, layout="HND")


def test_bf16_d128_hnd_and_nhd_agree_on_the_same_data() -> None:
    """The two physical orders must produce the same numbers for the same tile configuration.

    The configuration is pinned because the heuristic picks a different tile per layout, and a
    different tile changes the online-softmax accumulation order. With it pinned the two paths
    differ only in the grid order and the operand strides, so the outputs are bit-identical.
    """
    with torch.inference_mode():
        q, k, v = _operands("HND", 1, 4, 300, 300, seed=6)
        for config in PINNED_CONFIGS:
            hnd, hnd_lse = feather_attn_bf16_d128(q, k, v, layout="HND", config=config)
            nhd, nhd_lse = feather_attn_bf16_d128(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                layout="NHD",
                config=config,
            )
            assert torch.equal(hnd, to_hnd(nhd, "NHD").contiguous()), f"config={config}"
            assert torch.equal(hnd_lse, nhd_lse), f"config={config} lse"


@pytest.mark.parametrize(("layout", "batch", "heads", "n_q", "n_kv"), SMALL_CASES)
def test_bf16_d128_lse_matches_fp64(layout, batch, heads, n_q, n_kv) -> None:
    """The emitted log-sum-exp, against the definition evaluated in fp64.

    The backward reads it, so it has to be the real thing rather than whatever the online
    recurrence happens to hold: the accumulator works in log2 (the hardware `v_log_f32` is a
    base-2 logarithm on this target) and is converted once, at the end.
    """
    with torch.inference_mode():
        q, k, v = _operands(layout, batch, heads, n_q, n_kv, seed=7)
        _, lse = feather_attn_bf16_d128(q, k, v, layout=layout)
        qd = to_hnd(q, layout).double()
        kd = to_hnd(k, layout).double()
        scores = (qd @ kd.transpose(-1, -2)) * (HEAD_DIM**-0.5)
        expected = torch.logsumexp(scores, dim=-1)
        assert torch.isfinite(lse).all(), f"{layout} NQ{n_q} NKV{n_kv}: lse is not finite"
        worst = float((lse.double() - expected).abs().max())
        assert worst <= LSE_ATOL, f"{layout} NQ{n_q} NKV{n_kv}: |lse - fp64| = {worst:.3g}"
