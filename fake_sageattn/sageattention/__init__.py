"""
Fake ``sageattention`` package.

A drop-in replacement for the public API of `SageAttention` that redirects every
call to FeatherAttention (a ROCm-native attention kernel). Install this package
*instead of* the real ``sageattention`` so ComfyUI's ``from sageattention import
sageattn`` resolves here.

Only the entry points ComfyUI actually uses are provided:

* ``sageattn``        -- forwards to ``feather_attn`` / ``feather_attn_backward``.
* ``sageattn_varlen`` -- not supported by FeatherAttention; raises so callers can
  fall back.
"""

from .core import sageattn
from .core import sageattn_varlen

__all__ = ["sageattn", "sageattn_varlen"]
__version__ = "0.1.0"