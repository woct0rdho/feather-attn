# fake sageattn

A drop-in replacement for the `sageattention` package that redirects every
SageAttention call to [FeatherAttention](https://github.com/woctordho/feather-attn).

ComfyUI's official build cannot call FeatherAttention directly — it imports
`from sageattention import sageattn` and calls `sageattn(q, k, v, **kwargs)`.
This package installs itself under the name `sageattention`, exposes the exact
same `sageattn` entry point and signature, and forwards the computation to
`feather_attn` / `feather_attn_backward`.

## Why

* FeatherAttention is a **ROCm-native** attention kernel for RDNA3/3.5 (AMD).
* SageAttention's own kernels are NVIDIA/ROCm CUDA/triton kernels that this project
  replaces with FeatherAttention for better performance on AMD.
* ComfyUI does not call `feather_attn` directly, so this shim bridges the gap with
  zero changes to ComfyUI source.

## Install

FeatherAttention is only built for a ROCm PyTorch build. First build/install
`feather_attn` (see the root README), then install this shim **in place of** the
real `sageattention`:

```bash
pip install -e ./fake_sageattn
```

Then start ComfyUI with `--use-sage-attention` as usual.

## What is bridged

| SageAttention API            | FeatherAttention backend            |
|------------------------------|--------------------------------------|
| `sageattn(...)` (forward)    | `feather_attn` (FP16, HND/NHD)       |
| autograd                     | `feather_attn_backward`              |

Key behaviours / limitations:

* `tensor_layout` `"HND"` and `"NHD"` are supported (this is what ComfyUI uses).
* `sm_scale` is honored by absorbing the scale into `q`/`k` (defaults to
  `1 / sqrt(head_dim)`).
* `is_causal` is **not** supported and raises `NotImplementedError` (ComfyUI always
  passes `False`, so it is never hit in practice).
* `attn_mask` / `smooth_k` are not supported and raise `NotImplementedError`.
* head_dim must be 64 or 128 (or a value that pads to 64/128, e.g. 40 -> 64, 80 -> 128).
  Other head_dims raise `NotImplementedError`.
* bf16 inputs are round-tripped through FP16 (FeatherAttention is FP16-only).
* When `feather_attn` is not importable (e.g. non-ROCm build), `sageattn` raises a
  `RuntimeError` so ComfyUI's existing exception handler gracefully falls back to
  PyTorch attention.

## Layout of the real `sageattention` this mirrors

The real package exports several entry points; ComfyUI only ever imports and calls
`sageattn`. This shim therefore provides `sageattn` plus `sageattn_varlen` (which
raises `NotImplementedError`). The signature intentionally does **not** declare
`attn_mask` as a named parameter so that `inspect.signature(sageattn)` reports the
same mask-support capability as the real package.