// Host side of the bf16 D128 forward kernel: validation, stride derivation, grid
// policy, launch, and the opaque torch op.

#include <hip/hip_runtime.h>

#include "common.h"
#include "fwd/bf16/bf16_fwd_params.h"

#ifndef NO_PYTORCH
#if defined(__clang__)
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
#include <cstdint>

namespace feather_attn {

namespace {

using Launcher = bool (*)(const Bf16FwdParams&);

Launcher SelectLauncher(bool head_fast, int config)
{
    switch(config)
    {
    case 1: return head_fast ? feather_attn_fwd_bf16_d128_nhd_br256_bc64_swz
                             : feather_attn_fwd_bf16_d128_hnd_br256_bc64_swz;
    case 2: return head_fast ? feather_attn_fwd_bf16_d128_nhd_br256_bc32_pad
                             : feather_attn_fwd_bf16_d128_hnd_br256_bc32_pad;
    case 3: return head_fast ? feather_attn_fwd_bf16_d128_nhd_br256_bc64_pad_buf1
                             : feather_attn_fwd_bf16_d128_hnd_br256_bc64_pad_buf1;
    case 4: return head_fast ? feather_attn_fwd_bf16_d128_nhd_br128_bc32_pad
                             : feather_attn_fwd_bf16_d128_hnd_br128_bc32_pad;
    default: return nullptr;
    }
}

// Query rows per workgroup for each configuration. n_q below 256 cannot fill the
// 256-row tile, so the 128-row fallback keeps the grid from over-fetching Q.
int ConfigBlockM(int config)
{
    return config == 4 ? 128 : 256;
}

} // namespace

#ifdef NO_PYTORCH

bool AttnBf16D128(const Bf16Operand& q,
                  const Bf16Operand& k,
                  const Bf16Operand& v,
                  const Bf16Operand& o,
                  float softmax_scale_log2,
                  int32_t batch,
                  int32_t heads,
                  int32_t n_q,
                  int32_t n_kv,
                  bool head_fast,
                  int config,
                  hipStream_t stream)
{
    const int resolved = config == 0 ? (head_fast ? 2 : 1) : config;
    const int32_t q_tile_rows = ConfigBlockM(resolved);
    const int32_t q_tiles = static_cast<int32_t>((n_q + q_tile_rows - 1) / q_tile_rows);
    const Launcher launcher = SelectLauncher(head_fast, resolved);
    const uint32_t grid = static_cast<uint32_t>(batch) *
                          static_cast<uint32_t>(heads) *
                          static_cast<uint32_t>(q_tiles);
    const Bf16FwdParams params{q, k, v, o, softmax_scale_log2, n_q, n_kv,
                               0, heads, grid, stream};
    return launcher(params);
}

#else

void AttnBf16D128Feather(const torch::stable::Tensor& q,
                         const torch::stable::Tensor& k,
                         const torch::stable::Tensor& v,
                         torch::stable::Tensor& o,
                         int64_t layout,
                         int64_t config)
{
    STD_TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda(),
                    "q, k, v, and o must be CUDA tensors");
    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k.get_device_index() == device_index &&
                        v.get_device_index() == device_index &&
                        o.get_device_index() == device_index,
                    "q, k, v, and o must be on the same device");
    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::BFloat16 &&
                        k.scalar_type() == torch::stable::ScalarType::BFloat16 &&
                        v.scalar_type() == torch::stable::ScalarType::BFloat16 &&
                        o.scalar_type() == torch::stable::ScalarType::BFloat16,
                    "q, k, v, and o must be bfloat16");
    STD_TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && o.dim() == 4,
                    "q, k, v, and o must be 4D");
    STD_TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 (HND) or 1 (NHD)");

    const bool head_fast = layout == 1;
    // HND is [B,H,N,D], NHD is [B,N,H,D]. The sequence axis difference is what the
    // kernel needs from the layout, everything else comes from the strides.
    const int64_t head_axis = head_fast ? 2 : 1;
    const int64_t seq_axis  = 3 - head_axis;
    const int64_t batch     = q.size(0);
    const int64_t heads     = q.size(head_axis);
    const int64_t n_q       = q.size(seq_axis);
    const int64_t n_kv      = k.size(seq_axis);
    const int64_t d         = q.size(3);

    STD_TORCH_CHECK(d == 128, "head dimension must be 128");
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
    STD_TORCH_CHECK(k.size(3) == d && v.size(3) == d && o.size(3) == d,
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

    auto make_operand = [&](const void* ptr, const torch::stable::Tensor& t) {
        return Bf16Operand{ptr,
                           static_cast<int32_t>(t.stride(seq_axis)),
                           static_cast<int32_t>(t.stride(head_axis)),
                           static_cast<int32_t>(t.stride(0))};
    };
    const Bf16Operand q_op = make_operand(q.const_data_ptr(), q);
    const Bf16Operand k_op = make_operand(k.const_data_ptr(), k);
    const Bf16Operand v_op = make_operand(v.const_data_ptr(), v);
    const Bf16Operand o_op = make_operand(o.mutable_data_ptr(), o);

    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    // The 256-row tile amortises K/V staging over twice the queries. The 128-row
    // tile is used when the query extent cannot fill it.
    int resolved_config = static_cast<int>(config);
    if(resolved_config == 0)
        resolved_config = head_fast ? 2 : 1;
    STD_TORCH_CHECK(resolved_config >= 1 && resolved_config <= 4,
                    "config must be 0..4");

    const int64_t q_tile_rows = ConfigBlockM(resolved_config);
    const __int128 q_tiles   = (n_q + q_tile_rows - 1) / q_tile_rows;
    // No NHD head grouping: this kernel stages 256 query rows per workgroup, which
    // already halves the K/V traffic of the 128-row kernel the LLC gate was tuned
    // against, and the measured NHD rows are 3-9% faster without the sublaunches.
    const Launcher launcher = SelectLauncher(head_fast, resolved_config);
    STD_TORCH_CHECK(launcher != nullptr, "no bf16 D128 launcher for this config");

    const __int128 grid = static_cast<__int128>(batch) * heads * q_tiles;
    STD_TORCH_CHECK(grid > 0 && grid <= UINT32_MAX,
                    "launch grid exceeds unsigned int32");
    const Bf16FwdParams params{
        q_op,
        k_op,
        v_op,
        o_op,
        static_cast<float>(1.4426950408889634 / 11.313708498984761),
        static_cast<int32_t>(n_q),
        static_cast<int32_t>(n_kv),
        0,
        static_cast<int32_t>(heads),
        static_cast<uint32_t>(grid),
        stream};
    (void)hipGetLastError();
    STD_TORCH_CHECK(launcher(params), "bf16 D128 launch failed");
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launch_error == hipSuccess,
                    "bf16 D128 launch failed: ",
                    hipGetErrorString(launch_error));
}

#endif // NO_PYTORCH

} // namespace feather_attn

#ifndef NO_PYTORCH
// A fragment, because fwd_fp8.cpp already owns the DEF registration of the
// feather_attn namespace. A second DEF for the same namespace aborts at load time.
STABLE_TORCH_LIBRARY_FRAGMENT(feather_attn, m)
{
    m.def(
        "attn_bf16_d128_feather("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) out, "
        "int layout, "
        "int config"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn, CUDA, m)
{
    m.impl("attn_bf16_d128_feather", TORCH_BOX(&feather_attn::AttnBf16D128Feather));
}
#endif
