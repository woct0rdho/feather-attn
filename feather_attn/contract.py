"""The benchmark matrix and the FLOP model.

`docs/input_shapes_attn.md` is the source: the rows are the `(heads, sequence)` bands the
DiT/video families actually run. The forward reports `flops_fwd`, the backward `flops_bwd`,
and every benchmark row is also a correctness row in `tests/shapes.py`.
"""

LAYOUTS = ("HND", "NHD")

# The benchmark matrix: the head counts and sequence lengths of the fused-attention
# families in docs/input_shapes_attn.md at the two image sizes the workload runs.
BENCH_HEAD_COUNTS = (16, 32, 56)
BENCH_SEQ_LENS = (4096, 8192, 16384)


def bench_shapes(
    head_counts: tuple[int, ...] = BENCH_HEAD_COUNTS,
    seq_lens: tuple[int, ...] = BENCH_SEQ_LENS,
) -> tuple[tuple[int, int], ...]:
    """The `(heads, sequence)` grid of the throughput matrix."""
    return tuple((heads, seq_len) for heads in head_counts for seq_len in seq_lens)


def flops_fwd(batch: int, heads: int, n_q: int, n_kv: int, head_dim: int) -> float:
    """Two GEMMs per (query, key) pair: `Q @ K^T` and `P @ V`."""
    return 4.0 * batch * heads * n_q * n_kv * head_dim


def flops_bwd(batch: int, heads: int, n_q: int, n_kv: int, head_dim: int, gemm_count: int = 7) -> float:
    """The backward is `gemm_count` GEMMs of the same shape. 7 is the counted pipeline."""
    return 2.0 * gemm_count * batch * heads * n_q * n_kv * head_dim
