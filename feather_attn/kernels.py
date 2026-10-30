import torch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import bwd as _bwd_mod
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import fwd_prefill as _fwd_mod
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import BWD_MODE, PHILOX_OFFSET, PHILOX_SEED, USE_EXP2, get_arch
from torch.library import triton_op, wrap_triton

from .aiter_tuning import install_aiter_tuned_configs

# AITER's launchers query the Triton driver, which Dynamo refuses to trace. Its
# lru_cache must be warm before the first trace, otherwise torch.compile fails.
get_arch()

install_aiter_tuned_configs()


class _wrap_triton_launches:
    """Route Triton launches of an AITER launcher through `wrap_triton`.

    AITER's `attention_*_triton_impl` functions launch their kernels as module
    globals, which `torch.compile` cannot see. Replacing them for the duration
    of the call makes the launches capture as Triton kernel HOPs. Under eager
    execution `wrap_triton` is disabled and this is a no-op.
    """

    def __init__(self, module, names):
        self.module = module
        self.names = names
        self.saved = {}

    def __enter__(self):
        for name in self.names:
            self.saved[name] = getattr(self.module, name)
            setattr(self.module, name, wrap_triton(self.saved[name]))
        return self

    def __exit__(self, *exc):
        for name, kernel in self.saved.items():
            setattr(self.module, name, kernel)
        self.saved.clear()
        return False


# Every BWD_MODE variant, so the swap above is mode independent.
_BWD_KERNEL_NAMES = (
    "_bwd_preprocess",
    "bwd_kernel_fused_causal",
    "bwd_kernel_fused_noncausal",
    "_bwd_kernel_fused_atomic_causal",
    "_bwd_kernel_fused_atomic_noncausal",
    "_bwd_kernel_split_dkdv_causal",
    "_bwd_kernel_split_dq_causal",
    "_bwd_kernel_split_dkdv_noncausal",
    "_bwd_kernel_split_dq_noncausal",
)


@triton_op("feather_attn::fwd_out", mutates_args={"out", "softmax_lse", "sd_mask"})
def fwd_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    sd_mask: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    dropout_p: float,
    return_scores: bool,
) -> None:
    out.zero_()
    with _wrap_triton_launches(_fwd_mod, ("attn_fwd",)):
        _fwd_mod.attention_forward_prefill_triton_impl(
            q,
            k,
            v,
            out,
            softmax_lse,
            sd_mask,
            softmax_scale,
            alibi_slopes,
            causal,
            window_left,
            window_right,
            None,  # bias
            "bshd",
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            q.shape[1],  # max_seqlens_q
            k.shape[1],  # max_seqlens_k
            dropout_p,
            PHILOX_SEED,
            PHILOX_OFFSET,
            return_scores,
            USE_EXP2,
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
        )


@triton_op("feather_attn::bwd_aiter_out", mutates_args={"dq", "dk", "dv", "delta"})
def bwd_aiter_out(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    delta: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    softmax_scale: float,
    causal: bool,
    window_left: int,
    window_right: int,
    dropout_p: float,
) -> None:
    # AITER's forward always seeds dropout with the module-level PHILOX_SEED/OFFSET,
    # so passing the same pair here reproduces the forward mask without threading the
    # rng_state tensor (and without its .item() host sync) through autograd.
    with _wrap_triton_launches(_bwd_mod, _BWD_KERNEL_NAMES):
        _bwd_mod.attention_backward_triton_impl(
            do=dout,
            q=q,
            k=k,
            v=v,
            o=out,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk,
            dv=dv,
            delta=delta,
            sm_scale=softmax_scale,
            alibi_slopes=alibi_slopes,
            causal=causal,
            layout="bshd",
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            dropout_p=dropout_p,
            philox_seed=PHILOX_SEED,
            philox_offset=PHILOX_OFFSET,
            use_exp2=USE_EXP2,
            mode=BWD_MODE,
            window_size_left=window_left,
            window_size_right=window_right,
        )
