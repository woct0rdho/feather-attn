// Host-side validation shared by every forward attention operator.
//
// One place for the operand contract: device, dtype, rank, positivity, the size
// relations between Q/K/V/O, compactness, the signed-int32 offset contract, and the
// int32 range of every extent. The kernels are written against the returned strides, so a
// shape the checks accept is a shape the kernels address correctly.
//
// The stride derivation lives here rather than in each host: an operand carries its own
// sequence/head/batch strides, which is what removed the head count from the kernel ABI.

#pragma once

#include <hip/hip_runtime.h>

#include "attention_args.h"
#include "common.h"

#ifndef NO_PYTORCH
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#endif
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <climits>
#include <cstdint>

namespace feather_attn {

struct AttentionFwdView
{
    AttentionOperand q;
    AttentionOperand k;
    AttentionOperand v;
    AttentionOperand o;
    AttentionOperand lse;
    int64_t batch;
    int64_t heads;
    int64_t n_q;
    int64_t n_kv;
    int64_t head_dim;
    bool nhd;
};

// `nhd` says whether the two middle axes are stored [N, H] (true) or [H, N] (false). It is
// the only information the tensors themselves cannot express, because both orders put the
// head dimension innermost.
//
// `lse` is optional: nullptr skips the log-sum-exp store. When given it must be a contiguous
// float32 `[B, H, N]` tensor, the layout both backward kernels read.
inline AttentionFwdView ValidateAttentionFwd(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& o,
    torch::stable::Tensor* lse,
    bool nhd,
    torch::stable::ScalarType dtype,
    int64_t head_dim_min,
    int64_t head_dim_max)
{
    STD_TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda(),
                    "q, k, v, and o must be CUDA tensors");
    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k.get_device_index() == device_index &&
                        v.get_device_index() == device_index &&
                        o.get_device_index() == device_index,
                    "q, k, v, and o must be on the same device");
    STD_TORCH_CHECK(q.scalar_type() == dtype && k.scalar_type() == dtype &&
                        v.scalar_type() == dtype && o.scalar_type() == dtype,
                    "q, k, v, and o must all have the operator's dtype");
    STD_TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && o.dim() == 4,
                    "q, k, v, and o must be 4D");

    const int64_t head_axis = nhd ? 2 : 1;
    const int64_t seq_axis  = 3 - head_axis;
    const int64_t batch     = q.size(0);
    const int64_t heads     = q.size(head_axis);
    const int64_t n_q       = q.size(seq_axis);
    const int64_t n_kv      = k.size(seq_axis);
    const int64_t head_dim  = q.size(3);

    STD_TORCH_CHECK(head_dim >= head_dim_min && head_dim <= head_dim_max,
                    "head dimension is outside this operator's supported range");
    STD_TORCH_CHECK(batch > 0 && heads > 0 && n_q > 0 && n_kv > 0,
                    "batch, heads, N_Q, and N_KV must be positive");
    STD_TORCH_CHECK(k.size(0) == batch && v.size(0) == batch && o.size(0) == batch,
                    "batch size mismatch");
    STD_TORCH_CHECK(k.size(head_axis) == heads && v.size(head_axis) == heads &&
                        o.size(head_axis) == heads,
                    "head count mismatch");
    STD_TORCH_CHECK(k.size(seq_axis) == n_kv && v.size(seq_axis) == n_kv &&
                        o.size(seq_axis) == n_q,
                    "sequence length mismatch");
    STD_TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim &&
                        o.size(3) == head_dim,
                    "head dimension mismatch");
    STD_TORCH_CHECK(IsContiguousND(q, 4) && IsContiguousND(k, 4) &&
                        IsContiguousND(v, 4) && IsContiguousND(o, 4),
                    "q, k, v, and o must be contiguous");
    STD_TORCH_CHECK(IsInt32Addressable(q, sizeof(uint16_t)) &&
                        IsInt32Addressable(k, sizeof(uint16_t)) &&
                        IsInt32Addressable(v, sizeof(uint16_t)) &&
                        IsInt32Addressable(o, sizeof(uint16_t)),
                    "q, k, v, or o exceeds the signed-int32 byte-offset contract");
    STD_TORCH_CHECK(batch <= INT32_MAX && heads <= INT32_MAX && n_q <= INT32_MAX &&
                        n_kv <= INT32_MAX,
                    "batch, heads, N_Q, or N_KV exceeds signed int32");

    auto operand = [&](const void* ptr, const torch::stable::Tensor& t) {
        return AttentionOperand{ptr,
                                static_cast<int32_t>(t.stride(seq_axis)),
                                static_cast<int32_t>(t.stride(head_axis)),
                                static_cast<int32_t>(t.stride(0))};
    };
    AttentionOperand lse_operand{nullptr, 0, 0, 0};
    if(lse != nullptr)
    {
        STD_TORCH_CHECK(lse->is_cuda() && lse->get_device_index() == device_index,
                        "lse must be a CUDA tensor on q's device");
        STD_TORCH_CHECK(lse->scalar_type() == torch::stable::ScalarType::Float,
                        "lse must be float32");
        STD_TORCH_CHECK(lse->dim() == 3 && lse->size(0) == batch &&
                            lse->size(1) == heads && lse->size(2) == n_q,
                        "lse must be [batch, heads, N_Q]");
        STD_TORCH_CHECK(IsContiguousND(*lse, 3) && IsInt32Addressable(*lse, sizeof(float)),
                        "lse must be contiguous and within the signed-int32 byte-offset contract");
        lse_operand = AttentionOperand{lse->mutable_data_ptr(),
                                       static_cast<int32_t>(lse->stride(2)),
                                       static_cast<int32_t>(lse->stride(1)),
                                       static_cast<int32_t>(lse->stride(0))};
    }
    return AttentionFwdView{operand(q.const_data_ptr(), q),
                            operand(k.const_data_ptr(), k),
                            operand(v.const_data_ptr(), v),
                            operand(o.mutable_data_ptr(), o),
                            lse_operand,
                            batch,
                            heads,
                            n_q,
                            n_kv,
                            head_dim,
                            nhd};
}

// How one launch set is split over heads. A policy returns a plan. The loop over groups is
// then identical for a flat launch, an LLC-sized contiguous grouping, and the
// partition-aware strided mapping.
struct AttentionHeadPlan
{
    int64_t group_count = 1;
    int64_t head_step   = 1; // 1: contiguous block per group; >1: heads interleaved across groups
    int64_t group_size  = 0; // contiguous block size, used when head_step == 1
};

inline AttentionHeadPlan PlanHeadGroups(int64_t heads, int64_t group_count, int64_t head_step)
{
    if(group_count <= 1)
        return AttentionHeadPlan{1, 1, heads};
    if(head_step > 1)
        return AttentionHeadPlan{group_count, head_step, 0};
    return AttentionHeadPlan{group_count, 1, (heads + group_count - 1) / group_count};
}

inline void HeadGroupRange(const AttentionHeadPlan& plan, int64_t heads, int64_t group,
                           int64_t& head_start, int64_t& launch_heads)
{
    if(plan.head_step > 1)
    {
        head_start   = group;
        launch_heads = (heads - 1 - group) / plan.head_step + 1;
    }
    else
    {
        head_start   = group * plan.group_size;
        launch_heads = heads - head_start < plan.group_size ? heads - head_start
                                                            : plan.group_size;
    }
}

// Builds the launch arguments for one `[head_start, head_start + launch_heads)` head group.
// `head_step` is 1 for a contiguous group and the group count for the strided mapping.
// `q_tile_rows` is the query tile height of the image being launched, which fixes the grid.
inline AttentionFwdParams MakeAttentionFwdParams(const AttentionFwdView& view,
                                                 float softmax_scale_log2,
                                                 int64_t head_start,
                                                 int64_t head_step,
                                                 int64_t launch_heads,
                                                 int64_t q_tile_rows,
                                                 hipStream_t stream)
{
    const __int128 q_tiles = (view.n_q + q_tile_rows - 1) / q_tile_rows;
    const __int128 grid = static_cast<__int128>(view.batch) * launch_heads * q_tiles;
    STD_TORCH_CHECK(grid > 0 && grid <= UINT32_MAX, "launch grid exceeds unsigned int32");
    return AttentionFwdParams{view.q,
                              view.k,
                              view.v,
                              view.o,
                              view.lse,
                              softmax_scale_log2,
                              static_cast<int32_t>(view.n_q),
                              static_cast<int32_t>(view.n_kv),
                              static_cast<int32_t>(head_start),
                              static_cast<int32_t>(head_step),
                              static_cast<int32_t>(launch_heads),
                              static_cast<uint32_t>(grid),
                              stream};
}

} // namespace feather_attn

#endif // NO_PYTORCH
