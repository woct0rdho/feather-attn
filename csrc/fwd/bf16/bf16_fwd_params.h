// Argument ABI for the bf16 D128 forward kernel. Kept in its own header so the
// existing fp8 forward and the backward ABI stay untouched.

#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace feather_attn {

// Row-major element strides for one operand. `head_stride` is the stride between
// consecutive physical heads, `batch_stride` the stride between batches, so the
// same kernel covers HND ([B,H,N,D]) and NHD ([B,N,H,D]) storage and the mixed
// layout rows in docs/input_shapes_attn.md without a transpose.
struct Bf16Operand
{
    const void* ptr;
    int32_t row_stride;
    int32_t head_stride;
    int32_t batch_stride;
};

struct Bf16FwdParams
{
    Bf16Operand q;
    Bf16Operand k;
    Bf16Operand v;
    Bf16Operand o;
    float softmax_scale_log2; // log2(e) / sqrt(D)
    int32_t n_q;
    int32_t n_kv;
    int32_t head_start;
    int32_t launch_heads;
    uint32_t grid_size;
    hipStream_t stream;
};

extern "C" bool feather_attn_fwd_bf16_d128_hnd_br256_bc64_swz(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br256_bc64_swz(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br256_bc32_pad(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br256_bc32_pad(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br256_bc64_pad_buf1(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br256_bc64_pad_buf1(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br128_bc32_pad(
    const Bf16FwdParams& params);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br128_bc32_pad(
    const Bf16FwdParams& params);

} // namespace feather_attn
