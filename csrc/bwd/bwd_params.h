// Argument ABI for the backward kernels.

#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace feather_attn {

constexpr int32_t kBackwardDtypeFloat16  = 0;
constexpr int32_t kBackwardDtypeBFloat16 = 1;

struct BackwardLaunchParams
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* out_ptr;
    const void* lse_ptr;
    const void* dout_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* delta_ptr;
    void* dq_acc_ptr;
    int32_t head_count;
    int32_t n_q;
    int32_t n_kv;
    int32_t num_heads;
    int32_t layout;
    int32_t nhd_group_count;
    int32_t dtype;
    float softmax_scale;
    hipStream_t stream;
};

extern "C" bool feather_attn_bwd_d64_fused(const BackwardLaunchParams& params);
extern "C" bool feather_attn_bwd_d128_fused(const BackwardLaunchParams& params);
extern "C" bool feather_attn_bwd_d128_fused_dq(const BackwardLaunchParams& params);

} // namespace feather_attn
