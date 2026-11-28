"""fp16 forward contract with the E5M2-quantized Q tile, D64 and D128.

The gate is deliberately looser than the bf16 kernel's and it depends on the key length: Q is
rounded to E5M2, which is worth about 5e-2 relative L2 by itself. Measured over the contract,
the worst element uses 100% of an `atol=rtol=0.10` envelope below 1024 keys and 79% of a
`0.05` envelope at or above, so the gate below carries a deliberate 20% margin on each rather
than sitting on the measured value. `rel_l2 <= 0.10` bounds catastrophic drift. The measured
worst is 0.062. Cosine is not part of the forward contract.

The scale check is differential instead of absolute: at a sharper-than-default scale the E5M2
error grows past the envelope above, so what is asserted there is that the output tracks AITER
at the requested scale much more closely than AITER at the default one.
"""

import pytest
import torch

from feather_attn.contract import LAYOUTS
from feather_attn.ops_fp16_q8 import feather_attn_fp16_q8
from feather_attn.reference import aiter_fwd
from tests.reference import elementwise_ratio, fp32_fwd, rel_l2
from tests.shapes import SMALL_SHAPES, contract_shapes

BATCH = 1
DTYPE = torch.float16
HEAD_DIMS = (64, 128)
RTOL_SMALL_KV = 0.12
RTOL = 0.06
KV_THRESHOLD = 1024
REL_L2 = 0.10

CASES = tuple(
    pytest.param(layout, *shape, id=f"{layout}-B{shape[0]}H{shape[1]}NQ{shape[2]}NKV{shape[3]}D{shape[4]}")
    for layout in LAYOUTS
    for head_dim in HEAD_DIMS
    for shape in contract_shapes(head_dim)
)
SMALL_CASES = tuple(
    pytest.param(layout, head_dim, batch, heads, n_q, n_kv, id=f"{layout}-D{head_dim}-B{batch}H{heads}NQ{n_q}NKV{n_kv}")
    for layout in LAYOUTS
    for head_dim in HEAD_DIMS
    for batch, heads, n_q, n_kv in SMALL_SHAPES
)


def _operands(layout: str, head_dim: int, batch: int, heads: int, n_q: int, n_kv: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (batch, heads, n_q, head_dim) if layout == "HND" else (batch, n_q, heads, head_dim)
    shape_kv = (batch, heads, n_kv, head_dim) if layout == "HND" else (batch, n_kv, heads, head_dim)
    return (
        torch.randn(shape_q, device="cuda", dtype=DTYPE, generator=generator),
        torch.randn(shape_kv, device="cuda", dtype=DTYPE, generator=generator),
        torch.randn(shape_kv, device="cuda", dtype=DTYPE, generator=generator),
    )


def _check(actual: torch.Tensor, expected: torch.Tensor, n_kv: int, label: str) -> None:
    factor = RTOL_SMALL_KV if n_kv < KV_THRESHOLD else RTOL
    assert torch.isfinite(actual).all(), f"{label}: output is not finite"
    worst = elementwise_ratio(actual, expected, factor)
    assert worst <= 1.0, f"{label}: worst element at {worst:.3g} of the atol=rtol={factor} envelope"
    error = rel_l2(actual, expected)
    assert error <= REL_L2, f"{label}: rel_l2={error:.3g} exceeds {REL_L2:.3g}"


@pytest.mark.parametrize(("layout", "batch", "heads", "n_q", "n_kv", "head_dim"), CASES)
def test_fp16_q8_matches_aiter(layout, batch, heads, n_q, n_kv, head_dim) -> None:
    with torch.inference_mode():
        q, k, v = _operands(layout, head_dim, batch, heads, n_q, n_kv, seed=0)
        actual = feather_attn_fp16_q8(q, k, v, layout=layout)
        _check(actual, aiter_fwd(q, k, v, layout), n_kv, f"{layout} D{head_dim} NQ{n_q} NKV{n_kv}")


@pytest.mark.parametrize(("layout", "head_dim", "batch", "heads", "n_q", "n_kv"), SMALL_CASES)
def test_fp16_q8_matches_fp32(layout, head_dim, batch, heads, n_q, n_kv) -> None:
    """An anchor that does not depend on AITER, on the shapes cheap enough to reconstruct."""
    with torch.inference_mode():
        q, k, v = _operands(layout, head_dim, batch, heads, n_q, n_kv, seed=1)
        actual = feather_attn_fp16_q8(q, k, v, layout=layout)
        _check(actual, fp32_fwd(q, k, v, layout), n_kv, f"{layout} D{head_dim} NQ{n_q} NKV{n_kv} vs fp32")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("softmax_scale", (0.15, 0.05))
def test_fp16_q8_honours_softmax_scale(softmax_scale, head_dim, layout) -> None:
    """The scale is an operator argument now, not a constant folded into the kernel."""
    with torch.inference_mode():
        q, k, v = _operands(layout, head_dim, 1, 3, 300, 200, seed=3)
        actual = feather_attn_fp16_q8(q, k, v, layout=layout, softmax_scale=softmax_scale)
        assert torch.isfinite(actual).all()
        near = rel_l2(actual, aiter_fwd(q, k, v, layout, softmax_scale=softmax_scale))
        far = rel_l2(actual, aiter_fwd(q, k, v, layout))
        assert near < far / 1.5, (
            f"D{head_dim} {layout}: rel_l2 at scale={softmax_scale} is {near:.3g} but {far:.3g} against the default-scale control, so the scale is not reaching the kernel"
        )


def test_fp16_q8_rejects_bf16_operands() -> None:
    with torch.inference_mode():
        q, k, v = _operands("HND", 128, 1, 2, 128, 128, seed=4)
        with pytest.raises(RuntimeError):
            feather_attn_fp16_q8(q.bfloat16(), k.bfloat16(), v.bfloat16(), layout="HND")


def test_fp16_q8_rejects_d256() -> None:
    with torch.inference_mode():
        generator = torch.Generator(device="cuda").manual_seed(5)
        q = torch.randn((1, 2, 128, 256), device="cuda", dtype=DTYPE, generator=generator)
        with pytest.raises(RuntimeError):
            feather_attn_fp16_q8(q, q, q, layout="HND")
