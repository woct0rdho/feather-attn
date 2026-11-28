from itertools import product

import pytest
import torch
from aiter.ops.triton._triton_kernels.attention.dao_ai import interface_v2 as flash_attn_2

from feather_attn.ops_bf16 import feather_attn_bf16_d128

BATCH = 1
HEAD_DIM = 128
# Configurations the benchmark matrix selects, plus one fallback tile per axis.
CONFIGS = (0, 1, 3)
HEAD_COUNTS = (16, 32, 56)
SEQ_LENS = (4096, 8192, 16384)
LAYOUTS = ("HND", "NHD")
GENERAL_HEAD_COUNTS = (1, 2, 3, 7, 24, 30, 48)
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
    shapes.extend((BATCH, heads, seq_len, seq_len, HEAD_DIM) for heads, seq_len in product(HEAD_COUNTS, SEQ_LENS))
    shapes.extend((BATCH, HEAD_COUNTS[0], seq_len, seq_len, HEAD_DIM) for seq_len in GENERAL_SEQ_LENS if seq_len not in SEQ_LENS)
    shapes.extend((BATCH, heads, GENERAL_HEAD_SEQ_LEN, GENERAL_HEAD_SEQ_LEN, HEAD_DIM) for heads in GENERAL_HEAD_COUNTS if heads not in HEAD_COUNTS)
    shapes.extend(
        (
            (1, 3, 1, 65, HEAD_DIM),
            (1, 3, 65, 64, HEAD_DIM),
            (1, 3, 128, 129, HEAD_DIM),
            (1, 3, 129, 128, HEAD_DIM),
            (1, 3, 150, 1024, HEAD_DIM),
            (1, 3, 256, 1025, HEAD_DIM),
            (2, 3, 129, 65, HEAD_DIM),
        )
    )
    return tuple(shapes)


def _contract_cases() -> tuple[object, ...]:
    return tuple(
        pytest.param(
            config,
            layout,
            batch,
            heads,
            n_q,
            n_kv,
            head_dim,
            id=f"c{config}-{layout}-B{batch}-H{heads}-NQ{n_q}-NKV{n_kv}-D{head_dim}",
        )
        for config, layout, (batch, heads, n_q, n_kv, head_dim) in product(CONFIGS, LAYOUTS, _contract_shapes())
    )


def _small_cases() -> tuple[object, ...]:
    """A small independent anchor: fp32 is only used where it is cheap, AITER covers
    the rest of the contract."""
    shapes = (
        (1, 1, 128, 128),
        (1, 3, 257, 257),
        (1, 3, 1, 65),
        (1, 3, 65, 64),
        (1, 3, 129, 128),
        (1, 3, 150, 1024),
        (1, 7, 1000, 1024),
        (2, 3, 129, 65),
    )
    return tuple(
        pytest.param(config, layout, batch, heads, n_q, n_kv, HEAD_DIM, id=f"c{config}-{layout}-B{batch}-H{heads}-NQ{n_q}-NKV{n_kv}")
        for config, layout, (batch, heads, n_q, n_kv) in product((0, 1, 3), LAYOUTS, shapes)
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


@pytest.mark.parametrize(("config", "layout", "batch", "heads", "n_q", "n_kv", "head_dim"), _contract_cases())
def test_feather_attn_bf16_d128_matches_aiter(
    config: int,
    layout: str,
    batch: int,
    heads: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> None:
    with torch.inference_mode():
        torch.manual_seed(0)
        q = torch.randn((batch, heads, n_q, head_dim), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.bfloat16, device="cuda")
        if layout == "NHD":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        expected = _aiter_attn(q, k, v, layout)
        actual = feather_attn_bf16_d128(q, k, v, layout=layout, config=config)
        difference = actual.float() - expected.float()
        # bf16 probabilities with a truncated (unbiased) rounding step and a
        # different accumulation order than AITER. Measured rel-L2 against an fp32
        # reconstruction is 4.4e-3. The envelope below is the screening gate.
        tolerance_factor = 0.02
        tolerance = tolerance_factor * expected.float().abs() + tolerance_factor
        tolerance_ratio = float((difference.abs() / tolerance).max())
        rel_l2 = float(difference.norm() / expected.float().norm().clamp_min(1.0e-6))
        max_abs = float(difference.abs().max())

        assert torch.isfinite(actual).all()
        assert torch.all(difference.abs() <= tolerance), (
            f"rel_l2={rel_l2:.3g} max_abs={max_abs:.3g} gate={tolerance_factor:.3f}/{tolerance_factor:.3f} tol_ratio={tolerance_ratio:.3g}"
        )
        assert rel_l2 <= 1.0e-2, f"rel_l2={rel_l2:.3g}"
        assert tolerance_ratio <= 1.0


def _fp32_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference attention accumulated in fp32, blocked over queries and heads so the
    score tensor never materializes."""
    batch, heads, n_q, _ = q.shape
    out = torch.empty((batch, heads, n_q, q.shape[3]), device=q.device, dtype=torch.float32)
    kd = k.float()
    vd = v.float()
    block = 256
    for b in range(batch):
        for h in range(heads):
            for start in range(0, n_q, block):
                stop = min(start + block, n_q)
                scores = (q[b, h, start:stop].float() @ kd[b, h].transpose(-1, -2)) * scale
                torch.softmax(scores, dim=-1, out=scores)
                out[b, h, start:stop] = scores @ vd[b, h]
    return out


@pytest.mark.parametrize(("config", "layout", "batch", "heads", "n_q", "n_kv", "head_dim"), _small_cases())
def test_feather_attn_bf16_d128_matches_fp32(
    config: int,
    layout: str,
    batch: int,
    heads: int,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> None:
    with torch.inference_mode():
        torch.manual_seed(1)
        q = torch.randn((batch, heads, n_q, head_dim), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((batch, heads, n_kv, head_dim), dtype=torch.bfloat16, device="cuda")
        reference = _fp32_reference(q, k, v, head_dim**-0.5)
        if layout == "NHD":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        actual = feather_attn_bf16_d128(q, k, v, layout=layout, config=config)
        if layout == "NHD":
            actual = actual.transpose(1, 2)
        difference = actual.float() - reference
        rel_l2 = float(difference.norm() / reference.norm().clamp_min(1.0e-6))
        a64 = actual.double().flatten()
        r64 = reference.double().flatten()
        cosine = float(torch.dot(a64, r64) / (a64.norm() * r64.norm()).clamp_min(1.0e-30))

        assert torch.isfinite(actual).all()
        assert rel_l2 <= 1.0e-2, f"rel_l2={rel_l2:.3g}"
        assert cosine >= 0.9999, f"cosine={cosine:.8f}"
