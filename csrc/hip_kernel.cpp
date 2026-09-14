#if defined(_MSC_VER)
#include <hip/hip_runtime_api.h>
#else
#include <hip/hip_runtime.h>
#endif

#include "hip_kernel.h"

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
namespace {

using half_bits_t = uint16_t;

bool IsContiguous4D(const torch::stable::Tensor& tensor)
{
    uint64_t expected_stride = 1;
    for(int64_t dim = 3; dim >= 0; --dim)
    {
        const uint64_t size = static_cast<uint64_t>(tensor.size(dim));
        if(size != 1 &&
           static_cast<uint64_t>(tensor.stride(dim)) != expected_stride)
            return false;
        if(size != 0 && expected_stride > UINT64_MAX / size)
            return false;
        expected_stride *= size > 0 ? size : 1;
    }
    return true;
}

bool IsInt32HalfAddressable(const torch::stable::Tensor& tensor)
{
    uint64_t max_element_offset = 0;
    for(int64_t dim = 0; dim < 4; ++dim)
    {
        const int64_t stride = tensor.stride(dim);
        if(stride < 0 || stride > INT32_MAX)
            return false;
        const uint64_t span =
            static_cast<uint64_t>(tensor.size(dim)) - 1;
        if(span != 0)
        {
            const uint64_t step = static_cast<uint64_t>(stride);
            if(step != 0 && span > UINT64_MAX / step)
                return false;
            const uint64_t add = step == 0 ? 0 : span * step;
            if(max_element_offset > UINT64_MAX - add)
                return false;
            max_element_offset += add;
        }
    }
    return max_element_offset <= INT32_MAX / static_cast<uint64_t>(sizeof(half_bits_t));
}

// Returns true when the product of the (non-negative) factors does not exceed
// `limit`, without relying on wide-integer types. MSVC does not provide
// __int128, so overflow is avoided via division checks.
bool ProductFits(uint64_t limit, uint64_t a, uint64_t b, uint64_t c)
{
    if(a > limit || b > limit || c > limit)
        return false;
    if(a != 0 && b > limit / a)
        return false;
    const uint64_t ab = a * b;
    if(ab != 0 && c > limit / ab)
        return false;
    return true;
}

using Launcher = bool (*)(const feather_attn::LaunchParams&);
using StridedLauncher =
    bool (*)(const feather_attn::StridedLaunchParams&);

Launcher SelectLauncher(int64_t head_dim, bool nhd, bool pad_q, bool pad_kv)
{
    if(head_dim == 64)
    {
        if(pad_q && pad_kv)
            return nhd ? feather_attn::feather_attn_nhd_d64_query_key_tail
                       : feather_attn::feather_attn_hnd_d64_query_key_tail;
        if(pad_q)
            return nhd ? feather_attn::feather_attn_nhd_d64_query_tail
                       : feather_attn::feather_attn_hnd_d64_query_tail;
        if(pad_kv)
            return nhd ? feather_attn::feather_attn_nhd_d64_key_tail
                       : feather_attn::feather_attn_hnd_d64_key_tail;
        return nhd ? feather_attn::feather_attn_nhd_d64_aligned
                   : feather_attn::feather_attn_hnd_d64_aligned;
    }
    if(pad_q && pad_kv)
        return nhd ? feather_attn::feather_attn_nhd_d128_query_key_tail
                   : feather_attn::feather_attn_hnd_d128_query_key_tail;
    if(pad_q)
        return nhd ? feather_attn::feather_attn_nhd_d128_query_tail
                   : feather_attn::feather_attn_hnd_d128_query_tail;
    if(pad_kv)
        return nhd ? feather_attn::feather_attn_nhd_d128_key_tail
                   : feather_attn::feather_attn_hnd_d128_key_tail;
    return nhd ? feather_attn::feather_attn_nhd_d128_aligned
               : feather_attn::feather_attn_hnd_d128_aligned;
}

int64_t ResolveNhdHeadGroupSize(int64_t heads, int64_t n_kv, int64_t head_dim)
{
    if(head_dim != 128)
        return heads;

    constexpr uint64_t kLlcBytes = 32ull * 1024 * 1024;
    const uint64_t kv_bytes_per_head =
        static_cast<uint64_t>(n_kv) * head_dim * 2 * sizeof(half_bits_t);
    if(kv_bytes_per_head == 0 || kv_bytes_per_head > kLlcBytes)
        return heads;

    constexpr uint64_t kTotalLimit = kLlcBytes * 3 / 2;
    const uint64_t heads_u = static_cast<uint64_t>(heads);
    if(heads_u <= (kTotalLimit - 1) / kv_bytes_per_head)
        return heads;

    const uint64_t group_size = kLlcBytes / kv_bytes_per_head;
    if(group_size < 4 || group_size >= heads_u)
        return heads;
    return static_cast<int64_t>(group_size);
}

int64_t ResolveNhdD64StridedGroupCount(int64_t heads, int64_t n_kv)
{
    constexpr uint64_t kLlcBytes = 32ull * 1024 * 1024;
    constexpr int64_t kHeadDim  = 64;
    const uint64_t kv_bytes_per_head =
        static_cast<uint64_t>(n_kv) * kHeadDim * 2 * sizeof(half_bits_t);
    if(kv_bytes_per_head == 0 || kv_bytes_per_head > kLlcBytes)
        return 1;

    constexpr uint64_t kTotalLimit = kLlcBytes * 3 / 2;
    const uint64_t heads_u = static_cast<uint64_t>(heads);
    if(heads_u <= (kTotalLimit - 1) / kv_bytes_per_head)
        return 1;

    const uint64_t group_size = kLlcBytes / kv_bytes_per_head;
    if(group_size < 4 || group_size >= heads_u)
        return 1;

    int64_t group_count = (heads + static_cast<int64_t>(group_size) - 1) /
                          static_cast<int64_t>(group_size);
    if(group_count < 3)
        return 1;
    // A physical-head stride divisible by eight recreates the NHD partition cliff.
    if(group_count % 8 == 0)
        ++group_count;
    return group_count;
}

StridedLauncher SelectNhdD64StridedLauncher(bool pad_q, bool pad_kv)
{
    if(pad_q && pad_kv)
        return feather_attn::feather_attn_nhd_d64_strided_query_key_tail;
    if(pad_q)
        return feather_attn::feather_attn_nhd_d64_strided_query_tail;
    if(pad_kv)
        return feather_attn::feather_attn_nhd_d64_strided_key_tail;
    return feather_attn::feather_attn_nhd_d64_strided_aligned;
}

} // namespace

void AttnFp16Feather(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    torch::stable::Tensor& o,
    int64_t layout)
{
    STD_TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda(),
                    "q, k, v, and o must be CUDA tensors");
    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k.get_device_index() == device_index &&
                        v.get_device_index() == device_index &&
                        o.get_device_index() == device_index,
                    "q, k, v, and o must be on the same device");
    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Half &&
                        k.scalar_type() == torch::stable::ScalarType::Half &&
                        v.scalar_type() == torch::stable::ScalarType::Half &&
                        o.scalar_type() == torch::stable::ScalarType::Half,
                    "q, k, v, and o must be float16");
    STD_TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && o.dim() == 4,
                    "q, k, v, and o must be 4D");

    STD_TORCH_CHECK(layout == 0 || layout == 1,
                    "layout must be 0 (HND) or 1 (NHD)");
    const bool nhd      = layout == 1;
    const int64_t batch = q.size(0);
    const int64_t heads = q.size(nhd ? 2 : 1);
    const int64_t n_q   = q.size(nhd ? 1 : 2);
    const int64_t d     = q.size(3);
    const int64_t n_kv  = k.size(nhd ? 1 : 2);
    STD_TORCH_CHECK(batch > 0 && heads > 0 && n_q > 0 && n_kv > 0,
                    "batch, heads, N_Q, and N_KV must be positive");
    STD_TORCH_CHECK(d == 64 || d == 128, "head dimension must be 64 or 128");
    STD_TORCH_CHECK(k.size(0) == batch && v.size(0) == batch && o.size(0) == batch,
                    "batch size mismatch");
    const int64_t head_axis = nhd ? 2 : 1;
    const int64_t seq_axis  = nhd ? 1 : 2;
    STD_TORCH_CHECK(k.size(head_axis) == heads &&
                        v.size(head_axis) == heads &&
                        o.size(head_axis) == heads,
                    "head count mismatch");
    STD_TORCH_CHECK(k.size(3) == d && v.size(3) == d && o.size(3) == d,
                    "head dimension mismatch");
    STD_TORCH_CHECK(v.size(seq_axis) == n_kv && o.size(seq_axis) == n_q,
                    "sequence length mismatch");
    STD_TORCH_CHECK(IsContiguous4D(q) && IsContiguous4D(k) &&
                        IsContiguous4D(v) && IsContiguous4D(o),
                    "q, k, v, and o must be contiguous");
    STD_TORCH_CHECK(IsInt32HalfAddressable(q) && IsInt32HalfAddressable(k) &&
                        IsInt32HalfAddressable(v) && IsInt32HalfAddressable(o),
                    "q, k, v, or o exceeds the signed-int32 byte-offset contract");
    STD_TORCH_CHECK(batch <= INT32_MAX && heads <= INT32_MAX &&
                        n_q <= INT32_MAX && n_kv <= INT32_MAX,
                    "batch, heads, N_Q, or N_KV exceeds signed int32");

    const int64_t q_tiles = n_q / 128 + (n_q % 128 != 0);
    STD_TORCH_CHECK(
        q_tiles > 0 && q_tiles <= UINT32_MAX &&
            ProductFits(UINT32_MAX, static_cast<uint64_t>(batch),
                        static_cast<uint64_t>(heads),
                        static_cast<uint64_t>(q_tiles)),
        "launch grid exceeds unsigned int32");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);
    const int64_t group_size = nhd ? ResolveNhdHeadGroupSize(heads, n_kv, d) : heads;
    const int64_t strided_group_count =
        nhd && d == 64 ? ResolveNhdD64StridedGroupCount(heads, n_kv) : 1;
    const bool use_strided_d64 = strided_group_count > 1;
    const Launcher launcher =
        SelectLauncher(d, nhd, n_q % 128 != 0, n_kv % 64 != 0);
    const StridedLauncher strided_launcher =
        use_strided_d64 ? SelectNhdD64StridedLauncher(n_q % 128 != 0, n_kv % 64 != 0)
                        : nullptr;

    (void)hipGetLastError();
    bool launched = true;
    if(use_strided_d64)
    {
        STD_TORCH_CHECK(strided_group_count <= INT32_MAX,
                        "strided head-group count exceeds signed int32");
        for(int64_t group_index = 0; group_index < strided_group_count;
            ++group_index)
        {
            const int64_t launch_heads =
                (heads - 1 - group_index) / strided_group_count + 1;
            STD_TORCH_CHECK(
                launch_heads > 0 && launch_heads <= INT32_MAX &&
                    ProductFits(UINT32_MAX, static_cast<uint64_t>(batch),
                                static_cast<uint64_t>(launch_heads),
                                static_cast<uint64_t>(q_tiles)),
                "strided head-group launch exceeds device limits");
            const int64_t grouped_grid_size = batch * launch_heads * q_tiles;
            const feather_attn::StridedLaunchParams params{
                q.const_data_ptr(),
                k.const_data_ptr(),
                v.const_data_ptr(),
                o.mutable_data_ptr(),
                static_cast<int32_t>(n_q),
                static_cast<int32_t>(n_kv),
                static_cast<int32_t>(heads),
                static_cast<int32_t>(group_index),
                static_cast<int32_t>(launch_heads),
                static_cast<int32_t>(strided_group_count),
                static_cast<uint32_t>(grouped_grid_size),
                stream};
            if(!strided_launcher(params))
            {
                launched = false;
                break;
            }
        }
    }
    else
    {
        for(int64_t head_start = 0; head_start < heads; head_start += group_size)
        {
            const int64_t launch_heads =
                group_size < heads - head_start ? group_size : heads - head_start;
            STD_TORCH_CHECK(
                ProductFits(UINT32_MAX, static_cast<uint64_t>(batch),
                            static_cast<uint64_t>(launch_heads),
                            static_cast<uint64_t>(q_tiles)),
                "head-group launch exceeds device limits");
            const int64_t grouped_grid_size = batch * launch_heads * q_tiles;
            const feather_attn::LaunchParams params{
                q.const_data_ptr(),
                k.const_data_ptr(),
                v.const_data_ptr(),
                o.mutable_data_ptr(),
                static_cast<int32_t>(n_q),
                static_cast<int32_t>(n_kv),
                static_cast<int32_t>(heads),
                static_cast<int32_t>(head_start),
                static_cast<int32_t>(launch_heads),
                static_cast<uint32_t>(grouped_grid_size),
                stream};
            if(!launcher(params))
            {
                launched = false;
                break;
            }
        }
    }
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launched && launch_error == hipSuccess,
                    "FeatherAttn launch failed: ",
                    hipGetErrorString(launch_error));
}

STABLE_TORCH_LIBRARY(feather_attn_fp16, m)
{
    m.def(
        "attn_fp16_feather("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) out, "
        "int layout"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_fp16_feather", TORCH_BOX(&AttnFp16Feather));
}

namespace feather_attn {
namespace {

bool IsContiguous3D(const torch::stable::Tensor& tensor)
{
    uint64_t expected_stride = 1;
    for(int64_t dim = 2; dim >= 0; --dim)
    {
        const uint64_t size = static_cast<uint64_t>(tensor.size(dim));
        if(size != 1 &&
           static_cast<uint64_t>(tensor.stride(dim)) != expected_stride)
            return false;
        if(size != 0 && expected_stride > UINT64_MAX / size)
            return false;
        expected_stride *= size > 0 ? size : 1;
    }
    return true;
}

bool IsInt32Addressable(
    const torch::stable::Tensor& tensor,
    int64_t element_size)
{
    uint64_t max_element_offset = 0;
    for(int64_t dim = 0; dim < tensor.dim(); ++dim)
    {
        const int64_t stride = tensor.stride(dim);
        if(stride < 0 || stride > INT32_MAX)
            return false;
        const uint64_t span =
            static_cast<uint64_t>(tensor.size(dim)) - 1;
        if(span != 0)
        {
            const uint64_t step = static_cast<uint64_t>(stride);
            if(step != 0 && span > UINT64_MAX / step)
                return false;
            const uint64_t add = step == 0 ? 0 : span * step;
            if(max_element_offset > UINT64_MAX - add)
                return false;
            max_element_offset += add;
        }
    }
    return static_cast<uint64_t>(element_size) != 0 &&
           max_element_offset <=
               INT32_MAX / static_cast<uint64_t>(element_size);
}

} // namespace

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
    STD_TORCH_CHECK(IsContiguous3D(softmax_lse) && IsContiguous3D(delta),
                    "softmax_lse and delta must be contiguous");
    STD_TORCH_CHECK(IsInt32Addressable(softmax_lse, sizeof(float)) &&
                        IsInt32Addressable(delta, sizeof(float)),
                    "softmax_lse or delta exceeds signed-int32 addressing");
    STD_TORCH_CHECK(batch <= INT32_MAX && heads <= INT32_MAX &&
                        n_q <= INT32_MAX && n_kv <= INT32_MAX,
                    "attention dimensions exceed signed int32");
    STD_TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                    "softmax_scale must be finite and positive");

    const int64_t head_count = batch * heads;
    STD_TORCH_CHECK(head_count > 0 && head_count <= INT32_MAX,
                    "backward tensor size exceeds signed int32");
    const int64_t q_rows = head_count * n_q;
    const int64_t kv_rows = head_count * n_kv;
    STD_TORCH_CHECK(q_rows > 0 && q_rows <= INT32_MAX &&
                        kv_rows > 0 && kv_rows <= INT32_MAX,
                    "backward tensor size exceeds signed int32");
    const int64_t q_elements = q_rows * head_dim;
    const int64_t kv_elements = kv_rows * head_dim;
    STD_TORCH_CHECK(q_elements <= INT32_MAX && kv_elements <= INT32_MAX,
                    "backward tensor size exceeds signed int32");
    int64_t nhd_group_count =
        nhd && head_dim == 64 ? ResolveNhdD64StridedGroupCount(heads, n_kv)
                              : 1;
    STD_TORCH_CHECK(nhd_group_count > 0 && nhd_group_count <= heads &&
                        nhd_group_count <= INT32_MAX,
                    "NHD backward group count exceeds head or int32 limits");

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
        static_cast<int32_t>(head_count),
        static_cast<int32_t>(n_q),
        static_cast<int32_t>(n_kv),
        static_cast<int32_t>(heads),
        static_cast<int32_t>(layout),
        static_cast<int32_t>(nhd_group_count),
        static_cast<float>(softmax_scale),
        reinterpret_cast<hipStream_t>(raw_stream)};
    (void)hipGetLastError();
    const bool launched = head_dim == 64
                              ? feather_attn_bwd_d64_fused(params)
                              : feather_attn_bwd_d128_fused(params);
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launched && launch_error == hipSuccess,
                    "FeatherAttn backward launch failed: ",
                    hipGetErrorString(launch_error));
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
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_bwd_fp16_feather", TORCH_BOX(&feather_attn::AttnBwdFp16Feather));
}
#endif
