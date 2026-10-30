import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import bwd, fwd_prefill

from feather_attn import aiter_tuning, feather_attn_func


def _signatures(configs):
    return [(config.kwargs, config.num_warps, config.num_stages) for config in configs]


def test_feather_attn_func_installs_aiter_tuned_configs() -> None:
    aiter_tuning._INSTALLED_DEVICES.clear()
    for autotuner in (
        fwd_prefill.attn_fwd,
        bwd.bwd_kernel_fused_causal,
        bwd.bwd_kernel_fused_noncausal,
    ):
        autotuner.configs = []
        autotuner.early_config_prune = None
        autotuner.cache.clear()

    q = torch.randn((1, 128, 2, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    feather_attn_func(q, k, v)

    assert _signatures(fwd_prefill.attn_fwd.configs) == _signatures(aiter_tuning._fwd_configs())
    assert _signatures(bwd.bwd_kernel_fused_causal.configs) == _signatures(aiter_tuning._bwd_configs(causal=True))
    assert _signatures(bwd.bwd_kernel_fused_noncausal.configs) == _signatures(aiter_tuning._bwd_configs(causal=False))
    assert fwd_prefill.attn_fwd.early_config_prune is not None
    assert bwd.bwd_kernel_fused_causal.early_config_prune is not None
    assert bwd.bwd_kernel_fused_noncausal.early_config_prune is not None
