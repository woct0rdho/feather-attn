#!/usr/bin/env python3
"""The backward benchmark, with AITER's Triton backward as the control.

Speed only: accuracy is `pytest tests/`, which is expected to pass before this file says
anything.

Run it as a module from the repository root, so that the `bench` package is this
repository's:

    python -m bench.benchmark_attn_hip_bwd --output <path>
"""

import argparse
import gc
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from aiter.ops.triton._triton_kernels.attention.dao_ai.bwd import attention_backward_triton_impl

from bench.protocol import Case, measure, speedup
from feather_attn import feather_attn_bwd
from feather_attn.contract import BENCH_HEAD_COUNTS, BENCH_SEQ_LENS, LAYOUTS, bench_shapes, flops_bwd
from feather_attn.layout import to_bshd
from feather_attn.reference import aiter_fwd_lse

BATCH = 1
DTYPE = torch.float16
DTYPE_NAME = "fp16"
HEAD_DIM = 64
GEMM_COUNT = 7
BENCHMARK_SHAPES = bench_shapes(BENCH_HEAD_COUNTS, BENCH_SEQ_LENS)
PROVIDERS = ("aiter", "feather_bwd")


@dataclass
class SavedState:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    dout: torch.Tensor
    scale: float
    layout: str


def _bshd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """AITER's backward takes BSHD, so an HND call sees a transposed view."""
    return to_bshd(tensor, layout)


def _saved_state(heads: int, seq_len: int, seed: int, layout: str) -> SavedState:
    """A forward pass to differentiate. The forward is AITER for both providers, so the
    comparison isolates the backward."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (BATCH, heads, seq_len, HEAD_DIM) if layout == "HND" else (BATCH, seq_len, heads, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=DTYPE, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=DTYPE, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=DTYPE, generator=generator)
    dout = torch.randn(shape, device="cuda", dtype=DTYPE, generator=generator)
    scale = HEAD_DIM**-0.5
    out, lse = aiter_fwd_lse(q, k, v, layout, scale)
    return SavedState(
        q=q,
        k=k,
        v=v,
        out=out.contiguous(),
        lse=lse,
        dout=dout,
        scale=scale,
        layout=layout,
    )


class Provider:
    def __init__(self, name: str, state: SavedState):
        self.name = name
        self.state = state
        self.dq = torch.empty_like(state.q)
        self.dk = torch.empty_like(state.k)
        self.dv = torch.empty_like(state.v)
        self.delta = torch.empty_like(state.lse)

    def launch(self) -> None:
        x = self.state
        if self.name == "feather_bwd":
            feather_attn_bwd(
                x.dout,
                x.q,
                x.k,
                x.v,
                x.out,
                x.lse,
                softmax_scale=x.scale,
                layout=x.layout,
            )
        elif self.name == "aiter":
            seq_len = x.q.shape[2] if x.layout == "HND" else x.q.shape[1]
            attention_backward_triton_impl(
                do=_bshd(x.dout, x.layout),
                q=_bshd(x.q, x.layout),
                k=_bshd(x.k, x.layout),
                v=_bshd(x.v, x.layout),
                o=_bshd(x.out, x.layout),
                softmax_lse=x.lse,
                dq=_bshd(self.dq, x.layout),
                dk=_bshd(self.dk, x.layout),
                dv=_bshd(self.dv, x.layout),
                delta=self.delta,
                sm_scale=x.scale,
                alibi_slopes=None,
                causal=False,
                layout="bshd",
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                dropout_p=0.0,
                rng_state=None,
                use_exp2=True,
                window_size_left=-1,
                window_size_right=-1,
            )
        else:
            raise ValueError(f"unknown provider: {self.name}")


def _timing(measurements, flops_f7: float, flops_pipeline: float) -> dict[str, object]:
    return {
        name: {
            "median_ms": measurement.median_ms,
            "p20_ms": measurement.p20_ms,
            "p80_ms": measurement.p80_ms,
            "tflops_f7": flops_f7 / (measurement.median_ms * 1.0e9),
            "tflops_pipeline": flops_pipeline / (measurement.median_ms * 1.0e9),
            "samples_ms": list(measurement.samples_ms),
        }
        for name, measurement in measurements.items()
    }


def _run_shape(
    heads: int,
    seq_len: int,
    layout: str,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    state = _saved_state(heads, seq_len, seed, layout)
    providers = {name: Provider(name, state) for name in PROVIDERS}
    case = Case(f"H={heads} N={seq_len} D={HEAD_DIM} {layout}", {name: provider.launch for name, provider in providers.items()})
    measurements = measure(case.providers, warmup=warmups, rounds=repeats)
    flops_f7 = 14.0 * BATCH * heads * seq_len * seq_len * HEAD_DIM
    flops_pipeline = flops_bwd(BATCH, heads, seq_len, seq_len, HEAD_DIM, GEMM_COUNT)
    return {
        "shape": {
            "batch": BATCH,
            "heads": heads,
            "seq_len": seq_len,
            "head_dim": HEAD_DIM,
            "dtype": DTYPE_NAME,
            "layout": layout,
        },
        "seed": seed,
        "timings": _timing(measurements, flops_f7, flops_pipeline),
        "feather_bwd_over_aiter": speedup(measurements["feather_bwd"], measurements["aiter"]),
        "paired_ratio_p20": measurements["aiter"].p20_ms / measurements["feather_bwd"].p80_ms,
        "paired_ratio_p80": measurements["aiter"].p80_ms / measurements["feather_bwd"].p20_ms,
    }


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    if value == "default":
        return list(BENCHMARK_SHAPES)
    shapes = []
    for item in value.split(","):
        heads, seq_len = item.lower().split("x", maxsplit=1)
        shape = (int(heads), int(seq_len))
        if min(shape) <= 0:
            raise ValueError("head counts and sequence lengths must be positive")
        shapes.append(shape)
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _parse_layouts(value: str) -> list[str]:
    layouts = [layout.strip().upper() for layout in value.split(",")]
    if not layouts or any(layout not in LAYOUTS for layout in layouts):
        raise ValueError("layouts must contain only HND and/or NHD")
    return layouts


def _speedups(rows: list[dict[str, object]]) -> list[float]:
    ratios = []
    for row in rows:
        ratio = row["feather_bwd_over_aiter"]
        if not isinstance(ratio, float):
            raise TypeError("benchmark speedup must be a float")
        ratios.append(ratio)
    return ratios


def _write_result(
    path: Path,
    rows: list[dict[str, object]],
    *,
    warmups: int,
    repeats: int,
) -> None:
    ratios = _speedups(rows)
    result = {
        "matrix": {
            "batch": BATCH,
            "head_dim": HEAD_DIM,
            "dtype": DTYPE_NAME,
            "head_counts": list(BENCH_HEAD_COUNTS),
            "seq_lens": list(BENCH_SEQ_LENS),
            "layouts": list(LAYOUTS),
            "flops_f7": "14 * B * H * N^2 * D",
            "flops_pipeline": f"2 * {GEMM_COUNT} * B * H * N^2 * D",
            "gemm_count": GEMM_COUNT,
            "providers": list(PROVIDERS),
            "warmups": warmups,
            "repeats": repeats,
            "forward_in_timing": False,
            "outputs_preallocated": True,
        },
        "rows": rows,
        "geometric_mean_feather_bwd_over_aiter": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)),
        "feather_bwd_wins": sum(ratio > 1.0 for ratio in ratios),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    global HEAD_DIM, GEMM_COUNT, DTYPE, DTYPE_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="default", help="comma-separated HxN rows, or 'default' for the primary matrix")
    parser.add_argument("--layouts", default=",".join(LAYOUTS), help="comma-separated physical layouts (HND and/or NHD)")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default=DTYPE_NAME)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=HEAD_DIM)
    parser.add_argument("--gemm-count", type=int, choices=(5, 6, 7, 8), default=GEMM_COUNT)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    HEAD_DIM = args.head_dim
    GEMM_COUNT = args.gemm_count
    DTYPE_NAME = args.dtype
    DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    shapes = _parse_shapes(args.shapes)
    layouts = _parse_layouts(args.layouts)
    benchmark_rows = [(heads, seq_len, layout) for layout in layouts for heads, seq_len in shapes]
    rows = []
    for index, (heads, seq_len, layout) in enumerate(benchmark_rows):
        print(
            f"BEGIN H={heads} N={seq_len} D={HEAD_DIM} dtype={DTYPE_NAME} layout={layout}",
            flush=True,
        )
        row = _run_shape(
            heads,
            seq_len,
            layout,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed + index,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.output is not None:
            _write_result(
                args.output,
                rows,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        gc.collect()
        torch.cuda.empty_cache()

    ratios = _speedups(rows)
    geometric_mean = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    print(f"Summary: Feather/AITER={geometric_mean:.3f}x wins={sum(ratio > 1.0 for ratio in ratios)}/{len(rows)}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
