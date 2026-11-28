// Host side of the fp16/E5M2-Q forward: operand validation, head-group policy, and the
// opaque torch op.

#include <hip/hip_runtime.h>

#include "attention_host.h"
#include "common.h"
#include "fwd/fwd_images.h"

#ifndef NO_PYTORCH
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#endif

#include <cmath>
#include <cstdint>

namespace feather_attn {

namespace {

constexpr int64_t kQueryTileRows = 128;
constexpr int64_t kKeyTileRows   = 64;

AttentionFwdLauncher SelectFp16Q8Image(int64_t head_dim, bool nhd, bool pad_q, bool pad_kv)
{
    if(head_dim == 64)
    {
        if(pad_q && pad_kv)
            return nhd ? feather_attn_fwd_fp16_q8_nhd_d64_query_key_tail
                       : feather_attn_fwd_fp16_q8_hnd_d64_query_key_tail;
        if(pad_q)
            return nhd ? feather_attn_fwd_fp16_q8_nhd_d64_query_tail
                       : feather_attn_fwd_fp16_q8_hnd_d64_query_tail;
        if(pad_kv)
            return nhd ? feather_attn_fwd_fp16_q8_nhd_d64_key_tail
                       : feather_attn_fwd_fp16_q8_hnd_d64_key_tail;
        return nhd ? feather_attn_fwd_fp16_q8_nhd_d64_aligned
                   : feather_attn_fwd_fp16_q8_hnd_d64_aligned;
    }
    if(pad_q && pad_kv)
        return nhd ? feather_attn_fwd_fp16_q8_nhd_d128_query_key_tail
                   : feather_attn_fwd_fp16_q8_hnd_d128_query_key_tail;
    if(pad_q)
        return nhd ? feather_attn_fwd_fp16_q8_nhd_d128_query_tail
                   : feather_attn_fwd_fp16_q8_hnd_d128_query_tail;
    if(pad_kv)
        return nhd ? feather_attn_fwd_fp16_q8_nhd_d128_key_tail
                   : feather_attn_fwd_fp16_q8_hnd_d128_key_tail;
    return nhd ? feather_attn_fwd_fp16_q8_nhd_d128_aligned
               : feather_attn_fwd_fp16_q8_hnd_d128_aligned;
}

// D128 token-major: cap the K/V working set of one sublaunch so a head group fits in the
// LLC. Contiguous blocks, because the L2 reuse this buys is per head.
int64_t ResolveNhdD128GroupCount(const AttentionFwdView& view)
{
    constexpr __int128 kLlcBytes = static_cast<__int128>(32) * 1024 * 1024;
    const __int128 kv_bytes_per_head =
        static_cast<__int128>(view.n_kv) * view.head_dim * 2 * sizeof(uint16_t);
    const __int128 total_kv_bytes = static_cast<__int128>(view.heads) * kv_bytes_per_head;
    if(total_kv_bytes * 2 < kLlcBytes * 3 || kv_bytes_per_head > kLlcBytes)
        return 1;
    const int64_t group_size = static_cast<int64_t>(kLlcBytes / kv_bytes_per_head);
    if(group_size < 4 || group_size >= view.heads)
        return 1;
    return (view.heads + group_size - 1) / group_size;
}

// Head-group policy for this family. D64 token-major reads K/V of one head at a large
// stride, so it interleaves heads across the sublaunches to reach the memory partitions.
// D128 token-major groups contiguous heads to fit the LLC. Everything else is one grid.
AttentionHeadPlan PlanFp16Q8HeadGroups(const AttentionFwdView& view)
{
    if(!view.nhd)
        return PlanHeadGroups(view.heads, 1, 1);
    if(view.head_dim == 64)
    {
        const int64_t group_count =
            ResolveNhdD64StridedGroupCount(view.heads, view.n_kv);
        return PlanHeadGroups(view.heads, group_count, group_count > 1 ? group_count : 1);
    }
    return PlanHeadGroups(view.heads, ResolveNhdD128GroupCount(view), 1);
}

bool LaunchFp16Q8(const AttentionFwdView& view, float softmax_scale_log2, hipStream_t stream)
{
    const AttentionFwdLauncher launcher = SelectFp16Q8Image(
        view.head_dim, view.nhd, view.n_q % kQueryTileRows != 0, view.n_kv % kKeyTileRows != 0);
    const AttentionHeadPlan plan = PlanFp16Q8HeadGroups(view);
    for(int64_t group = 0; group < plan.group_count; ++group)
    {
        int64_t head_start = 0;
        int64_t launch_heads = 0;
        HeadGroupRange(plan, view.heads, group, head_start, launch_heads);
        const AttentionFwdParams params = MakeAttentionFwdParams(
            view, softmax_scale_log2, head_start, plan.head_step, launch_heads,
            kQueryTileRows, stream);
        if(!launcher(params))
            return false;
    }
    return true;
}

} // namespace

void AttnFp16Q8Feather(const torch::stable::Tensor& q,
                       const torch::stable::Tensor& k,
                       const torch::stable::Tensor& v,
                       torch::stable::Tensor& o,
                       int64_t layout,
                       double softmax_scale)
{
    STD_TORCH_CHECK(layout == kLayoutHnd || layout == kLayoutNhd,
                    "layout must be 0 (HND) or 1 (NHD)");
    STD_TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                    "softmax_scale must be finite and positive");
    // This family has no log-sum-exp output. The shared ABI carries a null operand.
    const AttentionFwdView view = ValidateAttentionFwd(
        q, k, v, o, nullptr, IsNhdLayout(layout), torch::stable::ScalarType::Half, 64, 128);
    const float scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(q.get_device_index(), &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    (void)hipGetLastError();
    STD_TORCH_CHECK(LaunchFp16Q8(view, scale_log2, stream),
                    "fp16/E5M2-Q forward launch failed");
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launch_error == hipSuccess,
                    "fp16/E5M2-Q forward launch failed: ",
                    hipGetErrorString(launch_error));
}

} // namespace feather_attn

#ifndef NO_PYTORCH
// A fragment: the DEF registration of the feather_attn namespace lives in the backward.
STABLE_TORCH_LIBRARY_FRAGMENT(feather_attn, m)
{
    m.def(
        "attn_fwd_fp16_q8_feather("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) out, "
        "int layout, "
        "float softmax_scale"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn, CUDA, m)
{
    m.impl("attn_fwd_fp16_q8_feather", TORCH_BOX(&feather_attn::AttnFp16Q8Feather));
}
#endif
