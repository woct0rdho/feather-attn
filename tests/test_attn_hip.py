from itertools import product

import pytest
import torch

from feather_attn import feather_attn

flash_attn_amd = pytest.importorskip(
    "aiter.ops.triton._triton_kernels.flash_attn_triton_amd",
    reason="AITER is required as the attention reference",
)
flash_attn_2 = flash_attn_amd.flash_attn_2
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU is required")

BATCH = 1
BENCHMARK_HEAD_COUNTS = (16, 32, 56)
GENERAL_HEAD_COUNTS = (1, 2, 3, 4, 7, 24, 30, 40, 48)
HEAD_DIMS = (64, 128)
LAYOUTS = ("HND", "NHD")
BENCHMARK_SEQ_LENS = (4096, 8192, 16384)
GENERAL_HEAD_SEQ_LEN = 257
GENERAL_SEQ_LENS = (
    1,
    16,
    17,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    1000,
    1023,
    1024,
    1025,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16383,
    16385,
)


def _contract_shapes() -> tuple[tuple[int, int, int, int, int], ...]:
    shapes = []
    for head_dim in HEAD_DIMS:
        shapes.extend((BATCH, heads, seq_len, seq_len, head_dim) for heads, seq_len in product(BENCHMARK_HEAD_COUNTS, BENCHMARK_SEQ_LENS))
        shapes.extend((BATCH, BENCHMARK_HEAD_COUNTS[0], seq_len, seq_len, head_dim) for seq_len in GENERAL_SEQ_LENS if seq_len not in BENCHMARK_SEQ_LENS)
        shapes.extend((BATCH, heads, GENERAL_HEAD_SEQ_LEN, GENERAL_HEAD_SEQ_LEN, head_dim) for heads in GENERAL_HEAD_COUNTS if heads not in BENCHMARK_HEAD_COUNTS)
        shapes.extend(
            (
                (1, 3, 1, 65, head_dim),
                (1, 3, 65, 64, head_dim),
                (1, 3, 128, 129, head_dim),
                (1, 3, 129, 128, head_dim),
                (1, 3, 150, 1024, head_dim),
                (1, 3, 256, 1025, head_dim),
                (2, 3, 129, 65, head_dim),
            )
        )
    return tuple(shapes)


def _contract_cases() -> tuple[object, ...]:
    return tuple(
        pytest.param(
            layout,
            batch,
            heads,
            n_q,
            n_kv,
            head_dim,
            id=f"{layout}-B{batch}-H{heads}-NQ{n_q}-NKV{n_kv}-D{head_dim}",
        )
        for layout, (batch, heads, n_q, n_kv, head_dim) in product(LAYOUTS, _contract_shapes())
    )


def _aiter_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "HND":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    out, _, _, _ = flash_attn_2.fwd(
        q,
        k,
        v,
        out=None,
        alibi_slopes=None,
        dropout_p=0.0,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        return_softmax=False,
    )
    return out.transpose(1, 2) if layout == "HND" else out


@pytest.mark.parametrize(("layout", "batch", "heads", "n_q", "n_kv", "head_dim"), _contract_cases())
def test_feather_attn_matches_aiter(
    layout: str,
    batch: int,
    heads: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> None:
    with torch.inference_mode():
        torch.manual_seed(0)
        q = torch.randn((batch, heads, n_q, head_dim), dtype=torch.float16, device="cuda")
        k = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.float16, device="cuda")
        v = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.float16, device="cuda")
        if layout == "NHD":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        expected = _aiter_attn(q, k, v, layout)
        actual = feather_attn(q, k, v, layout=layout)
        difference = actual.float() - expected.float()
        tolerance_factor = 0.10 if n_kv < 1024 else 0.05
        tolerance = tolerance_factor * expected.float().abs() + tolerance_factor
        tolerance_ratio = float((difference.abs() / tolerance).max())
        rel_l2 = float(difference.norm() / expected.float().norm().clamp_min(1.0e-6))
        max_abs = float(difference.abs().max())

        assert torch.isfinite(actual).all()
        assert torch.all(difference.abs() <= tolerance), (
            f"rel_l2={rel_l2:.3g} max_abs={max_abs:.3g} gate={tolerance_factor:.2f}/{tolerance_factor:.2f} tol_ratio={tolerance_ratio:.3g}"
        )
