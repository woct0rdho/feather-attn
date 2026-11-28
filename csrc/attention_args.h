// Argument ABI shared by every forward attention kernel.
//
// One struct for every forward image: the operands carry their own row/head/batch strides,
// so a kernel never derives the layout from a head count, and the same launch policy covers
// HND, NHD, and future mixed physical orders. The head group (`head_start`, `head_step`,
// `launch_heads`) expresses both a plain launch (step 1) and the partition-aware strided
// launch (`head = first + i * group_count`) without a second ABI or a second kernel image.
//
// Deliberately free of ck_tile and torch so a host translation unit can include it.

#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace feather_attn {

// Element strides of one operand. `row_stride` is the sequence axis, `head_stride` the
// physical head axis, `batch_stride` the batch axis.
struct AttentionOperand
{
    const void* ptr;
    int32_t row_stride;
    int32_t head_stride;
    int32_t batch_stride;
};

// What a kernel reads.
struct AttentionFwdArgs
{
    AttentionOperand q;
    AttentionOperand k;
    AttentionOperand v;
    AttentionOperand o;
    float softmax_scale_log2;
    int32_t n_q;
    int32_t n_kv;
    int32_t head_start;
    int32_t head_step;
    int32_t launch_heads;
};

// What a host launcher passes: the kernel arguments plus the launch geometry.
struct AttentionFwdParams
{
    AttentionOperand q;
    AttentionOperand k;
    AttentionOperand v;
    AttentionOperand o;
    float softmax_scale_log2;
    int32_t n_q;
    int32_t n_kv;
    int32_t head_start;
    int32_t head_step;
    int32_t launch_heads;
    uint32_t grid_size;
    hipStream_t stream;
};

// Every forward image has this signature.
using AttentionFwdLauncher = bool (*)(const AttentionFwdParams&);

namespace detail {

inline AttentionFwdArgs MakeAttentionFwdArgs(const AttentionFwdParams& params)
{
    return AttentionFwdArgs{params.q,
                            params.k,
                            params.v,
                            params.o,
                            params.softmax_scale_log2,
                            params.n_q,
                            params.n_kv,
                            params.head_start,
                            params.head_step,
                            params.launch_heads};
}

} // namespace detail

} // namespace feather_attn
