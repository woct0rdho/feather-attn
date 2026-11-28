// Host side of the bf16 D128 forward: operand validation, tile-config selection, and the
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

using AttentionFwdLauncher = bool (*)(const AttentionFwdParams&);

// Tile configurations, in the same order as `feather_attn/impls.py`. Id 0 is the automatic
// choic. ids 1..4 pin a tile so a benchmark can compare them inside one matrix.
struct TileConfig
{
    int id;
    int64_t q_tile_rows;
    bool head_fast; // grid order the image was instantiated for
};

constexpr TileConfig kTileConfigs[] = {
    {0, 0, false},  // auto
    {1, 256, false}, // 256x64 keys, double buffered, xor-swizzled LDS
    {2, 256, true},  // 256x32 keys, double buffered, row-padded LDS
    {3, 256, false}, // 256x64 keys, single buffer, row-padded LDS
    {4, 128, false}, // 128x32 keys, double buffered, row-padded LDS
};

const TileConfig* FindTileConfig(int64_t id, bool head_fast)
{
    for(const TileConfig& config : kTileConfigs)
        if(config.id == id)
            return &config;
    return nullptr;
}

AttentionFwdLauncher SelectBf16D128Image(bool head_fast, int64_t config)
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

// The tile a benchmark should pin when it does not name one: the 256x64 swizzled image for
// head-major storage, the 256x32 padded one for token-major, which is the pair the
// throughput matrix settled on (see docs/feather_attn_gfx1151_fwd_bf16_d128_plan.md).
int64_t DefaultBf16D128Config(bool head_fast)
{
    return head_fast ? 2 : 1;
}

bool LaunchBf16D128(const AttentionFwdView& view, float softmax_scale_log2, int64_t config,
                    hipStream_t stream)
{
    const AttentionFwdLauncher launcher = SelectBf16D128Image(view.nhd, config);
    const TileConfig* tile = FindTileConfig(config, view.nhd);
    STD_TORCH_CHECK(launcher != nullptr && tile != nullptr,
                    "no bf16 D128 image for this config");
    // One flat grid: this family does not group heads. A 256-row workgroup already halves
    // the K/V traffic of the 128-row kernel the LLC gate was tuned against, and the NHD rows
    // measure 3-9% faster without the sublaunches.
    const AttentionHeadPlan plan = PlanHeadGroups(view.heads, 1, 1);
    const AttentionFwdParams params = MakeAttentionFwdParams(
        view, softmax_scale_log2, 0, plan.head_step, view.heads, tile->q_tile_rows, stream);
    return launcher(params);
}

} // namespace

void AttnBf16D128Feather(const torch::stable::Tensor& q,
                         const torch::stable::Tensor& k,
                         const torch::stable::Tensor& v,
                         torch::stable::Tensor& o,
                         torch::stable::Tensor& lse,
                         int64_t layout,
                         double softmax_scale,
                         int64_t config)
{
    STD_TORCH_CHECK(layout == kLayoutHnd || layout == kLayoutNhd,
                    "layout must be 0 (HND) or 1 (NHD)");
    STD_TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                    "softmax_scale must be finite and positive");
    const AttentionFwdView view = ValidateAttentionFwd(
        q, k, v, o, &lse, IsNhdLayout(layout), torch::stable::ScalarType::BFloat16, 128, 128);
    const float scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);

    const int64_t resolved = config == 0 ? DefaultBf16D128Config(view.nhd) : config;
    STD_TORCH_CHECK(resolved >= 1 && resolved <= 4, "config must be 0..4");

    torch::stable::accelerator::DeviceGuard device_guard(q.get_device_index());
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(q.get_device_index(), &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    (void)hipGetLastError();
    STD_TORCH_CHECK(LaunchBf16D128(view, scale_log2, resolved, stream),
                    "bf16 D128 forward launch failed");
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launch_error == hipSuccess,
                    "bf16 D128 forward launch failed: ",
                    hipGetErrorString(launch_error));
}

} // namespace feather_attn

#ifndef NO_PYTORCH
// A fragment: the DEF registration of the feather_attn namespace lives in the backward.
STABLE_TORCH_LIBRARY_FRAGMENT(feather_attn, m)
{
    m.def(
        "attn_fwd_bf16_d128_feather("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) out, "
        "Tensor(b!) lse, "
        "int layout, "
        "float softmax_scale, "
        "int config"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn, CUDA, m)
{
    m.impl("attn_fwd_bf16_d128_feather", TORCH_BOX(&feather_attn::AttnBf16D128Feather));
}
#endif
