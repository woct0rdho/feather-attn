"""Self-contained attention reference used on Windows, where AITER is not available.

On Linux/ROCm with AITER installed the tests keep importing AITER
(flash_attn_2) as the reference and never touch this module; it is only
imported from the test files when running on win32 without AITER.
"""

import torch

_CHUNK = 1024


def _hnd(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    return tensor if layout == "HND" else tensor.transpose(1, 2)


def exact_out_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact fp32 attention forward with the flash-attention saved state.

    Returns (out, lse):
      * out is fp16, in the caller's tensor layout (HND or NHD);
      * lse is fp32 with shape (B, H, S_q) in the HND domain, equal to the
        log-sum-exp of the *scaled* scores, matching flash_attn_2 semantics.
    The scores are computed in fp32 in chunks of _CHUNK query rows so the
    reference stays memory-bounded on an iGPU for sequences up to 16384+.
    """
    qh, kh, vh = (_hnd(t, layout) for t in (q, k, v))
    d = qh.shape[-1]
    if scale is None:
        scale = d**-0.5
    batch, heads, n_q, _ = qh.shape
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for start in range(0, n_q, _CHUNK):
        q_block = qh[:, :, start : start + _CHUNK]
        scores = (
            torch.matmul(q_block.float(), kh.float().transpose(-1, -2)) * scale
        )
        row_max = scores.max(-1, keepdim=True).values
        unnormalized = torch.exp(scores - row_max)
        row_sum = unnormalized.sum(-1, keepdim=True)
        probabilities = unnormalized / row_sum
        lse = row_max.squeeze(-1) + torch.log(unnormalized.sum(-1))
        outputs.append(torch.matmul(probabilities, vh.float()))
        lses.append(lse.float())
    out = torch.cat(outputs, 2).half()
    lse = torch.cat(lses, 2)
    if layout == "NHD":
        out = out.transpose(1, 2)
    return out, lse


def exact_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str
) -> torch.Tensor:
    out, _ = exact_out_lse(q, k, v, layout)
    return out