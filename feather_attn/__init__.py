import torch  # noqa: F401 Import torch before loading the extension.

from . import _C  # noqa: F401 Register native operators before torch.ops use. # ty: ignore[unresolved-import]
from .ops import feather_attn, feather_attn_backward

__all__ = ["feather_attn", "feather_attn_backward"]
