#!/usr/bin/env python3
"""Benchmark the forward attention kernel against AITER's Triton flash attention.

`aiter` is the Triton implementation from `aiter.ops.triton`, `feather` is `feather_attn`. Both are called
with the same tensors and the same layout. The reported number is forward-pass flops over the median
launch time, and the ratio column is the paired speedup against AITER.

Usage:
    python -m bench.benchmark_attn_hip.py
    python -m bench.benchmark_attn_hip.py --shapes 16x4096,32x8192 --layouts NHD
    FEATHER_ATTN_BENCH_HEAD_DIMS=64,128 FEATHER_ATTN_BENCH_LAYOUTS=HND,NHD python bench/benchmark_attn_hip.py
"""

import argparse
import os

import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2

from bench.protocol import Case, add_protocol_args, run_cases
from feather_attn.ops import feather_attn

BATCH = 1
HEAD_DIMS = [int(value) for value in os.environ.get("FEATHER_ATTN_BENCH_HEAD_DIMS", "64,128").split(",")]
LAYOUTS = [value.strip().upper() for value in os.environ.get("FEATHER_ATTN_BENCH_LAYOUTS", "HND,NHD").split(",")]
if not LAYOUTS or any(layout not in {"HND", "NHD"} for layout in LAYOUTS):
    raise ValueError("FEATHER_ATTN_BENCH_LAYOUTS must contain only HND and/or NHD")
HEAD_COUNTS = [16, 32, 56]
SEQ_LENS = [4096, 8192, 16384]


def aiter_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "HND":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
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


def build_case(heads: int, seq_len: int, head_dim: int, layout: str) -> Case:
    shape = (BATCH, heads, seq_len, head_dim) if layout == "HND" else (BATCH, seq_len, heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    providers = {
        "aiter": lambda: aiter_attn(q, k, v, layout),
        "feather": lambda: feather_attn(q, k, v, layout=layout),
    }
    return Case(f"H={heads} N={seq_len} D={head_dim} {layout}", providers, scale=4 * BATCH * heads * seq_len**2 * head_dim / 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="", help="HxN;HxN;... Default is the whole matrix")
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--head-dims", default=",".join(str(value) for value in HEAD_DIMS))
    add_protocol_args(parser)
    args = parser.parse_args()

    shapes = (
        [(int(heads), int(seq_len)) for heads, seq_len in (token.split("x") for token in args.shapes.split(","))]
        if args.shapes
        else [(heads, seq_len) for heads in HEAD_COUNTS for seq_len in SEQ_LENS]
    )
    layouts = [value.strip().upper() for value in args.layouts.split(",")]
    head_dims = [int(value) for value in args.head_dims.split(",")]
    cases = [build_case(heads, seq_len, head_dim, layout) for layout in layouts for head_dim in head_dims for heads, seq_len in shapes]

    run_cases(
        cases,
        unit="TFLOPS",
        control="aiter",
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
