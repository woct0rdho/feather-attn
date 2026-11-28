"""The correctness contract: every `(batch, heads, n_q, n_kv, head_dim)` a kernel must handle.

`docs/input_shapes_attn.md` is the source. `contract_shapes` unions three families: every row
of the benchmark matrix (a benchmark row that is not also a correctness row is a bug), the
general rows - lengths and head counts that are not a tile multiple, both sequence tails
independently, odd head counts, cross-attention with `n_q > n_kv`, batch two - and
`SMALL_SHAPES`, the subset cheap enough to compare against the blocked fp32 reference.
"""

from feather_attn.contract import BENCH_HEAD_COUNTS, BENCH_SEQ_LENS, bench_shapes

BATCH = 1

# Lengths that are not tile multiples, plus the 1023/1024/1025 tolerance boundary.
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

# Head counts of the model rows that are not in the benchmark matrix.
GENERAL_HEAD_COUNTS = (1, 2, 3, 4, 7, 24, 30, 40, 48)
GENERAL_HEAD_SEQ_LEN = 257

# Independent query and key/value tails, cross-attention, and batch two.
TAIL_SHAPES = (
    (1, 3, 1, 65),
    (1, 3, 65, 64),
    (1, 3, 128, 129),
    (1, 3, 129, 128),
    (1, 3, 150, 1024),
    (1, 3, 256, 1025),
    (2, 3, 129, 65),
)

# Cheap enough for a blocked fp32 reference in a test.
SMALL_SHAPES = (
    (1, 1, 128, 128),
    (1, 3, 257, 257),
    (1, 3, 1, 65),
    (1, 3, 65, 64),
    (1, 3, 129, 128),
    (1, 3, 150, 1024),
    (1, 7, 1000, 1024),
    (2, 3, 129, 65),
)


def contract_shapes(head_dim: int) -> tuple[tuple[int, int, int, int, int], ...]:
    """Every `(batch, heads, n_q, n_kv, head_dim)` a kernel is expected to handle."""
    shapes = []
    shapes.extend((BATCH, heads, seq_len, seq_len, head_dim) for heads, seq_len in bench_shapes())
    shapes.extend((BATCH, BENCH_HEAD_COUNTS[0], seq_len, seq_len, head_dim) for seq_len in GENERAL_SEQ_LENS if seq_len not in BENCH_SEQ_LENS)
    shapes.extend((BATCH, heads, GENERAL_HEAD_SEQ_LEN, GENERAL_HEAD_SEQ_LEN, head_dim) for heads in GENERAL_HEAD_COUNTS if heads not in BENCH_HEAD_COUNTS)
    shapes.extend((*shape, head_dim) for shape in TAIL_SHAPES)
    return tuple(shapes)
