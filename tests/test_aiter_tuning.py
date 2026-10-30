import subprocess
import sys

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import bwd, fwd_prefill

from feather_attn import aiter_tuning


def _signatures(configs):
    return [(config.kwargs, config.num_warps, config.num_stages) for config in configs]


def _clear_autotuners() -> None:
    for autotuner in (fwd_prefill.attn_fwd, bwd.bwd_kernel_fused_causal, bwd.bwd_kernel_fused_noncausal):
        autotuner.configs = []
        autotuner.early_config_prune = None
        autotuner.cache.clear()


def _assert_tuned() -> None:
    assert _signatures(fwd_prefill.attn_fwd.configs) == _signatures(aiter_tuning._fwd_configs())
    assert _signatures(bwd.bwd_kernel_fused_causal.configs) == _signatures(aiter_tuning._bwd_configs(causal=True))
    assert _signatures(bwd.bwd_kernel_fused_noncausal.configs) == _signatures(aiter_tuning._bwd_configs(causal=False))
    assert fwd_prefill.attn_fwd.early_config_prune is not None
    assert bwd.bwd_kernel_fused_causal.early_config_prune is not None
    assert bwd.bwd_kernel_fused_noncausal.early_config_prune is not None


def test_import_installs_aiter_tuned_configs() -> None:
    """Upstream AITER has no gfx1151 tiles, so importing feather_attn must install them."""
    script = """
import feather_attn
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import bwd, fwd_prefill
from feather_attn import aiter_tuning

sig = lambda cs: [(c.kwargs, c.num_warps, c.num_stages) for c in cs]
assert sig(fwd_prefill.attn_fwd.configs) == sig(aiter_tuning._fwd_configs())
assert sig(bwd.bwd_kernel_fused_causal.configs) == sig(aiter_tuning._bwd_configs(causal=True))
assert sig(bwd.bwd_kernel_fused_noncausal.configs) == sig(aiter_tuning._bwd_configs(causal=False))
assert fwd_prefill.attn_fwd.early_config_prune is not None
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_install_aiter_tuned_configs_is_idempotent() -> None:
    _clear_autotuners()
    aiter_tuning.install_aiter_tuned_configs()
    _assert_tuned()
    aiter_tuning.install_aiter_tuned_configs()
    _assert_tuned()
