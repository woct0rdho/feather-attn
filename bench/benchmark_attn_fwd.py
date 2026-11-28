#!/usr/bin/env python3
"""Compare every forward attention implementation this repo can run, on one matrix.

One row per `(layout, head dim, sequence, dtype)`, every requested implementation
interleaved inside it so the ratio columns are paired. The shape grid and the FLOP model come
from `feather_attn.contract`, the providers from `bench/providers_attn.py`, and the control
is AITER.

Speed only: accuracy is `pytest tests/`, which is expected to pass before this file says
anything.

Run it as a module from the repository root, so that `bench` is this repository's package:

    python -m bench.benchmark_attn_fwd
    python -m bench.benchmark_attn_fwd --impls aiter,feather_bf16_d128 --shapes 16x4096
    python -m bench.benchmark_attn_fwd --impls feather_bf16_d128 --configs 1,2
    python -m bench.benchmark_attn_fwd --dtypes bf16 --layouts NHD --head-dims 128
"""

import argparse
import sys

import torch

from bench.protocol import Case, add_protocol_args, run_cases
from bench.providers_attn import (
    add_aiter_fwd_providers,
    add_bf16_d128_providers,
    add_fp16_q8_providers,
)
from feather_attn.contract import BENCH_HEAD_COUNTS, BENCH_SEQ_LENS, LAYOUTS, bench_shapes, flops_fwd

BATCH = 1
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
IMPLS = ("aiter", "feather_fp16_q8", "feather_bf16_d128")
BF16_CONFIGS = (1, 2)


def build_case(heads, seq_len, head_dim, layout, dtype, impls, configs) -> Case:
    shape = (BATCH, heads, seq_len, head_dim) if layout == "HND" else (BATCH, seq_len, heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    providers: dict = {}
    if "aiter" in impls:
        add_aiter_fwd_providers(providers, q, k, v, layout)
    if "feather_fp16_q8" in impls:
        add_fp16_q8_providers(providers, q, k, v, layout)
    if "feather_bf16_d128" in impls:
        add_bf16_d128_providers(providers, q, k, v, layout, configs)
    label = f"H={heads} N={seq_len} D={head_dim} {layout} {str(dtype).removeprefix('torch.')}"
    return Case(label, providers, scale=flops_fwd(BATCH, heads, seq_len, seq_len, head_dim) / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="", help="HxN;HxN;... Default is the whole matrix")
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--head-dims", default="64,128")
    parser.add_argument("--dtypes", default="fp16,bf16", help="comma separated: fp16, bf16")
    parser.add_argument("--impls", default=",".join(IMPLS), help=f"comma separated: {', '.join(IMPLS)}")
    parser.add_argument("--configs", default=",".join(str(value) for value in BF16_CONFIGS), help="bf16 tile configurations")
    parser.add_argument("--control", default="aiter", help="implementation the ratios are taken against")
    add_protocol_args(parser)
    args = parser.parse_args()

    impls = {value.strip() for value in args.impls.split(",") if value.strip()}
    unknown = impls - set(IMPLS)
    assert not unknown, f"unknown implementation(s) {sorted(unknown)}. Known: {sorted(IMPLS)}"
    configs = tuple(int(value) for value in args.configs.split(",") if value.strip())
    shapes = (
        [(int(heads), int(seq_len)) for heads, seq_len in (token.split("x") for token in args.shapes.split(","))]
        if args.shapes
        else list(bench_shapes(BENCH_HEAD_COUNTS, BENCH_SEQ_LENS))
    )
    cases = [
        build_case(heads, seq_len, head_dim, layout, DTYPES[dtype], impls, configs)
        for dtype in (value.strip() for value in args.dtypes.split(","))
        for layout in (value.strip().upper() for value in args.layouts.split(","))
        for head_dim in (int(value) for value in args.head_dims.split(","))
        for heads, seq_len in shapes
    ]
    cases = [case for case in cases if case.providers]
    control = args.control if any(args.control in case.providers for case in cases) else None
    if control is None and cases:
        print(f"control {args.control!r} is not in the matrix. Ratios use the first provider", file=sys.stderr)
    run_cases(
        cases,
        unit="TFLOPS",
        control=control,
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
