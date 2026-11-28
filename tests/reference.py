"""The independent fp32 reconstruction and the metrics the gates are stated in.

Test-only: the package keeps what a benchmark needs (the AITER control and the scale), and
the correctness machinery lives here. `fp32_fwd` shares no code with any kernel, so an error
in both implementations cannot cancel.

The metrics are used as: `elementwise_ratio` and `rel_l2` for both forwards (the forward
contract is the elementwise envelope plus the relative L2), and `cosine` first for the
backward, whose magnitude is normalized by the optimizer, so a direction error is the one
that matters.
"""

import torch

from feather_attn.layout import to_hnd
from feather_attn.reference import default_softmax_scale


def fp32_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    softmax_scale: float | None = None,
    query_block: int = 256,
) -> torch.Tensor:
    """Reference attention accumulated in fp32, blocked over queries and heads.

    The score block is `query_block x n_kv`, so memory stays at a few megabytes for any
    contract row. The full `[B, H, NQ, NKV]` fp32 score tensor would be 470 MB on the largest
    one and does not fit in fp64 at all.
    """
    q, k, v = to_hnd(q, layout), to_hnd(k, layout), to_hnd(v, layout)
    batch, heads, n_q, head_dim = q.shape
    scale = default_softmax_scale(head_dim, softmax_scale)
    kd = k.float()
    vd = v.float()
    out = torch.empty((batch, heads, n_q, head_dim), device=q.device, dtype=torch.float32)
    for b in range(batch):
        for h in range(heads):
            for start in range(0, n_q, query_block):
                stop = min(start + query_block, n_q)
                scores = (q[b, h, start:stop].float() @ kd[b, h].transpose(-1, -2)) * scale
                torch.softmax(scores, dim=-1, out=scores)
                out[b, h, start:stop] = scores @ vd[b, h]
    return out if layout == "HND" else out.transpose(1, 2)


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error: a magnitude-normalized squared-error measure."""
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1.0e-6))


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Cosine similarity of the flattened tensors."""
    a = actual.double().flatten()
    b = expected.double().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1.0e-30))


def elementwise_ratio(actual: torch.Tensor, expected: torch.Tensor, factor: float) -> float:
    """Worst element as a fraction of the `factor * (|expected| + 1)` envelope."""
    difference = (actual.float() - expected.float()).abs()
    return float((difference / (factor * expected.float().abs() + factor)).max())
