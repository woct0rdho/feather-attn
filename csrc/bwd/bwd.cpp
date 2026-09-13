#include <hip/hip_runtime.h>

#include "common.h"
#include "launch_params.h"

#ifndef NO_PYTORCH
#if defined(__clang__)
// Thrust emits a gfx1100/gfx1101 clock warning for gfx1151. This code does not use timing APIs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#endif
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

#include <climits>
#include <cmath>
#include <cstdint>

#ifndef NO_PYTORCH

namespace feather_attn {

void AttnBwdFp16FeatherImpl(
    const torch::stable::Tensor& dout,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& out,
    const torch::stable::Tensor& softmax_lse,
    torch::stable::Tensor& dq,
    torch::stable::Tensor& dk,
    torch::stable::Tensor& dv,
    torch::stable::Tensor& delta,
    double softmax_scale,
    int64_t layout,
    torch::stable::Tensor* dq_acc,
    bool fused)
{
    STD_TORCH_CHECK(layout == 0 || layout == 1,
                    "layout must be 0 (HND) or 1 (NHD)");
    STD_TORCH_CHECK(q.is_cuda(), "all backward tensors must be CUDA tensors");
    const auto device_index = q.get_device_index();
    auto CheckDevice = [&](const torch::stable::Tensor& tensor) {
        STD_TORCH_CHECK(tensor.is_cuda(), "all backward tensors must be CUDA tensors");
        STD_TORCH_CHECK(tensor.get_device_index() == device_index,
                        "all backward tensors must be on the same device");
    };
    CheckDevice(k);
    CheckDevice(v);
    CheckDevice(out);
    CheckDevice(softmax_lse);
    CheckDevice(dout);
    CheckDevice(dq);
    CheckDevice(dk);
    CheckDevice(dv);
    CheckDevice(delta);
    if(dq_acc != nullptr)
        CheckDevice(*dq_acc);
    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Half &&
                        k.scalar_type() == torch::stable::ScalarType::Half &&
                        v.scalar_type() == torch::stable::ScalarType::Half &&
                        out.scalar_type() == torch::stable::ScalarType::Half &&
                        dout.scalar_type() == torch::stable::ScalarType::Half &&
                        dq.scalar_type() == torch::stable::ScalarType::Half &&
                        dk.scalar_type() == torch::stable::ScalarType::Half &&
                        dv.scalar_type() == torch::stable::ScalarType::Half,
                    "q, k, v, out, dout, dq, dk, and dv must be float16");
    STD_TORCH_CHECK(
        softmax_lse.scalar_type() == torch::stable::ScalarType::Float &&
            delta.scalar_type() == torch::stable::ScalarType::Float,
        "softmax_lse and delta must be float32");
    auto CheckAttentionDim = [&](const torch::stable::Tensor& tensor) {
        STD_TORCH_CHECK(tensor.dim() == 4, "attention tensors must be 4D");
    };
    CheckAttentionDim(q);
    CheckAttentionDim(k);
    CheckAttentionDim(v);
    CheckAttentionDim(out);
    CheckAttentionDim(dout);
    CheckAttentionDim(dq);
    CheckAttentionDim(dk);
    CheckAttentionDim(dv);

    const bool nhd = layout == 1;
    const int64_t head_axis = nhd ? 2 : 1;
    const int64_t seq_axis = nhd ? 1 : 2;
    const int64_t batch = q.size(0);
    const int64_t heads = q.size(head_axis);
    const int64_t n_q = q.size(seq_axis);
    const int64_t head_dim = q.size(3);
    const int64_t n_kv = k.size(seq_axis);
    STD_TORCH_CHECK(batch > 0 && heads > 0 && n_q > 0 && n_kv > 0,
                    "attention dimensions must be positive");
    STD_TORCH_CHECK(head_dim == 64 || head_dim == 128,
                    "backward supports D64 and D128 only");
    STD_TORCH_CHECK(softmax_lse.dim() == 3 && delta.dim() == 3,
                    "softmax_lse and delta must be [B, H, N]");
    STD_TORCH_CHECK(k.size(0) == batch && v.size(0) == batch &&
                        out.size(0) == batch && dout.size(0) == batch &&
                        dq.size(0) == batch && dk.size(0) == batch &&
                        dv.size(0) == batch,
                    "batch dimensions must match");
    STD_TORCH_CHECK(k.size(head_axis) == heads &&
                        v.size(head_axis) == heads &&
                        out.size(head_axis) == heads &&
                        dout.size(head_axis) == heads &&
                        dq.size(head_axis) == heads &&
                        dk.size(head_axis) == heads &&
                        dv.size(head_axis) == heads,
                    "head dimensions must match");
    STD_TORCH_CHECK(k.size(seq_axis) == n_kv &&
                        v.size(seq_axis) == n_kv &&
                        dk.size(seq_axis) == n_kv &&
                        dv.size(seq_axis) == n_kv,
                    "key/value sequence dimensions must match");
    STD_TORCH_CHECK(out.size(seq_axis) == n_q &&
                        dout.size(seq_axis) == n_q &&
                        dq.size(seq_axis) == n_q,
                    "query/output sequence dimensions must match");
    STD_TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim &&
                        out.size(3) == head_dim && dout.size(3) == head_dim &&
                        dq.size(3) == head_dim && dk.size(3) == head_dim &&
                        dv.size(3) == head_dim,
                    "head dimensions must match");
    STD_TORCH_CHECK(softmax_lse.size(0) == batch &&
                        softmax_lse.size(1) == heads &&
                        softmax_lse.size(2) == n_q &&
                        delta.size(0) == batch && delta.size(1) == heads &&
                        delta.size(2) == n_q,
                    "softmax_lse and delta shapes must be [B, H, NQ]");
    auto CheckAttentionLayout = [&](const torch::stable::Tensor& tensor) {
        STD_TORCH_CHECK(tensor.is_contiguous(),
                        "backward attention tensors must be contiguous");
        STD_TORCH_CHECK(IsInt32Addressable(tensor, sizeof(uint16_t)),
                        "an attention tensor exceeds signed-int32 addressing");
    };
    CheckAttentionLayout(q);
    CheckAttentionLayout(k);
    CheckAttentionLayout(v);
    CheckAttentionLayout(out);
    CheckAttentionLayout(dout);
    CheckAttentionLayout(dq);
    CheckAttentionLayout(dk);
    CheckAttentionLayout(dv);
    if(dq_acc != nullptr)
    {
        STD_TORCH_CHECK(dq_acc->scalar_type() == torch::stable::ScalarType::Float,
                        "dq_acc must be float32");
        STD_TORCH_CHECK(dq_acc->sizes() == q.sizes(),
                        "dq_acc must have the same shape as q");
        STD_TORCH_CHECK(dq_acc->is_contiguous(),
                        "dq_acc must be contiguous");
        STD_TORCH_CHECK(IsInt32Addressable(*dq_acc, sizeof(float)),
                        "dq_acc exceeds signed-int32 addressing");
    }
    STD_TORCH_CHECK(IsContiguousND(softmax_lse, 3) &&
                        IsContiguousND(delta, 3),
                    "softmax_lse and delta must be contiguous");
    STD_TORCH_CHECK(IsInt32Addressable(softmax_lse, sizeof(float)) &&
                        IsInt32Addressable(delta, sizeof(float)),
                    "softmax_lse or delta exceeds signed-int32 addressing");
    STD_TORCH_CHECK(batch <= INT32_MAX && heads <= INT32_MAX &&
                        n_q <= INT32_MAX && n_kv <= INT32_MAX,
                    "attention dimensions exceed signed int32");
    STD_TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                    "softmax_scale must be finite and positive");

    const __int128 head_count = static_cast<__int128>(batch) * heads;
    const __int128 q_rows = head_count * n_q;
    const __int128 kv_rows = head_count * n_kv;
    const __int128 q_elements = q_rows * head_dim;
    const __int128 kv_elements = kv_rows * head_dim;
    STD_TORCH_CHECK(head_count <= INT32_MAX && q_rows <= INT32_MAX &&
                        kv_rows <= INT32_MAX && q_elements <= INT32_MAX &&
                        kv_elements <= INT32_MAX,
                    "backward tensor size exceeds signed int32");
    int64_t nhd_group_count =
        nhd && head_dim == 64 ? ResolveNhdD64StridedGroupCount(heads, n_kv)
                              : 1;
    STD_TORCH_CHECK(nhd_group_count > 0 && nhd_group_count <= heads &&
                        nhd_group_count <= INT32_MAX,
                    "NHD backward group count exceeds head or int32 limits");

    STD_TORCH_CHECK(!fused || head_dim == 128,
                    "the fused dQ path supports D128 only");
    STD_TORCH_CHECK(!fused || dq_acc != nullptr,
                    "the fused dQ path requires dq_acc");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const BackwardLaunchParams params{
        q.const_data_ptr(),
        k.const_data_ptr(),
        v.const_data_ptr(),
        out.const_data_ptr(),
        softmax_lse.const_data_ptr(),
        dout.const_data_ptr(),
        dq.mutable_data_ptr(),
        dk.mutable_data_ptr(),
        dv.mutable_data_ptr(),
        delta.mutable_data_ptr(),
        dq_acc != nullptr ? dq_acc->mutable_data_ptr() : nullptr,
        static_cast<int32_t>(head_count),
        static_cast<int32_t>(n_q),
        static_cast<int32_t>(n_kv),
        static_cast<int32_t>(heads),
        static_cast<int32_t>(layout),
        static_cast<int32_t>(nhd_group_count),
        static_cast<float>(softmax_scale),
        reinterpret_cast<hipStream_t>(raw_stream)};
    (void)hipGetLastError();
    const bool launched =
        head_dim == 64
            ? feather_attn_bwd_d64_fused(params)
            : (fused ? feather_attn_bwd_d128_fused_dq(params)
                     : feather_attn_bwd_d128_fused(params));
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launched && launch_error == hipSuccess,
                    "FeatherAttn backward launch failed: ",
                    hipGetErrorString(launch_error));
}

void AttnBwdFp16Feather(
    const torch::stable::Tensor& dout,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& out,
    const torch::stable::Tensor& softmax_lse,
    torch::stable::Tensor& dq,
    torch::stable::Tensor& dk,
    torch::stable::Tensor& dv,
    torch::stable::Tensor& delta,
    double softmax_scale,
    int64_t layout)
{
    AttnBwdFp16FeatherImpl(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, delta,
        softmax_scale, layout, nullptr, false);
}

void AttnBwdFp16FeatherFused(
    const torch::stable::Tensor& dout,
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& out,
    const torch::stable::Tensor& softmax_lse,
    torch::stable::Tensor& dq,
    torch::stable::Tensor& dk,
    torch::stable::Tensor& dv,
    torch::stable::Tensor& delta,
    torch::stable::Tensor& dq_acc,
    double softmax_scale,
    int64_t layout)
{
    AttnBwdFp16FeatherImpl(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, delta,
        softmax_scale, layout, &dq_acc, true);
}

} // namespace feather_attn

STABLE_TORCH_LIBRARY_FRAGMENT(feather_attn_fp16, m)
{
    m.def(
        "attn_bwd_fp16_feather("
        "Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, "
        "Tensor softmax_lse, Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, "
        "Tensor(d!) delta, float softmax_scale, int layout"
        ") -> ()");
    m.def(
        "attn_bwd_fp16_feather_fused("
        "Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, "
        "Tensor softmax_lse, Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, "
        "Tensor(d!) delta, Tensor(e!) dq_acc, float softmax_scale, int layout"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_bwd_fp16_feather", TORCH_BOX(&feather_attn::AttnBwdFp16Feather));
    m.impl("attn_bwd_fp16_feather_fused", TORCH_BOX(&feather_attn::AttnBwdFp16FeatherFused));
}

#endif // NO_PYTORCH
