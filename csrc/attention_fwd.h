// Device-side pieces every forward attention kernel shares: the block decomposition, the
// WMMA wrapper, the log-domain online softmax, and the accumulator rescale.
//
// The two forward families differ in storage dtype, LDS layout, and how they prepare the
// probability tile, but the softmax recurrence and the accumulator update are the same
// arithmetic. Keeping them here means a numerical fix lands in both kernels at once, which
// is the class of bug that the previously duplicated code was exposed to.
//
// Host code must not include this header (it pulls in ck_tile). Include `attention_args.h`
// for the ABI instead.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>

#include "attention_args.h"

namespace feather_attn {

using ck_tile::index_t;

// One launcher for every forward image: a kernel family only supplies the type.
template <typename Kernel>
bool LaunchAttentionFwd(const AttentionFwdParams& params)
{
    const auto kernel = ck_tile::make_kernel<Kernel::kWaves, ck_tile::gfx115_t>(
        Kernel{}, dim3(params.grid_size), dim3(Kernel::kBlockSize), 0,
        detail::MakeAttentionFwdArgs(params));
    (void)hipGetLastError();
    ck_tile::launch_kernel(ck_tile::stream_config{params.stream}, kernel);
    return hipPeekAtLastError() == hipSuccess;
}

// Where this workgroup sits in the (batch, head, query tile) decomposition.
struct AttentionBlockIndex
{
    index_t q_tile;
    index_t head;
    index_t batch;
};

// `kHeadFast` picks the grid order only: head-fast for token-major storage, so adjacent
// workgroups touch adjacent heads, and head-slow for head-major storage, so adjacent
// workgroups walk the query axis of one head and share its K/V.
template <bool kHeadFast>
CK_TILE_DEVICE AttentionBlockIndex DecodeAttentionBlock(
    index_t block, index_t q_tiles, const AttentionFwdArgs& args)
{
    const index_t head = args.head_start + (block % args.launch_heads) * args.head_step;
    if constexpr(kHeadFast)
    {
        const index_t tile_batch = block / args.launch_heads;
        return AttentionBlockIndex{tile_batch % q_tiles, head, tile_batch / q_tiles};
    }
    else
    {
        const index_t linear = block / q_tiles;
        return AttentionBlockIndex{
            block % q_tiles,
            args.head_start + (linear % args.launch_heads) * args.head_step,
            linear / args.launch_heads};
    }
}

CK_TILE_DEVICE static float Exp2(float value)
{
    return __builtin_amdgcn_exp2f(value);
}

CK_TILE_DEVICE static uint32_t SwapLaneGroupBits(uint32_t bits)
{
    return __builtin_amdgcn_permlanex16(0u, bits, 0x76543210u, 0xfedcba98u, false, true);
}

// Swap the two 16-lane halves of a wave: joins the per-lane halves of a row reduction, and
// moves a packed pair to the other lane group for the probability transpose.
CK_TILE_DEVICE static float SwapLaneGroup(float value)
{
    return ck_tile::bit_cast<float>(SwapLaneGroupBits(ck_tile::bit_cast<uint32_t>(value)));
}

// 16x16x16 WMMA with an fp32 accumulator. The mnemonic is the only type-specific part, so
// each storage type gets its own specialization and the kernel bodies stay generic.
template <typename T>
struct WmmaOps;

template <>
struct WmmaOps<ck_tile::fp16_t>
{
    using AVecType = ck_tile::fp16x16_t;
    using BVecType = ck_tile::fp16x16_t;
    using CVecType = ck_tile::fp32x8_t;

    CK_TILE_DEVICE static void Init(CVecType& c, const AVecType& a, const BVecType& b)
    {
        asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, 0" : "=v"(c) : "v"(a), "v"(b));
    }
    CK_TILE_DEVICE static void InPlace(CVecType& c, const AVecType& a, const BVecType& b)
    {
        asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    }
};

template <>
struct WmmaOps<ck_tile::bf16_t>
{
    using AVecType = ck_tile::bf16x16_t;
    using BVecType = ck_tile::bf16x16_t;
    using CVecType = ck_tile::fp32x8_t;

    CK_TILE_DEVICE static void Init(CVecType& c, const AVecType& a, const BVecType& b)
    {
        asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, 0" : "=v"(c) : "v"(a), "v"(b));
    }
    CK_TILE_DEVICE static void InPlace(CVecType& c, const AVecType& a, const BVecType& b)
    {
        asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    }
};

// The log-domain online softmax state. `Update` is the whole recurrence: it returns the
// factor for the persistent accumulator (`alpha`) and the factor for the current probability
// tile (`beta`), then advances the running log-sum-exp. `value` is the log-sum-exp in the
// log2 domain the kernel works in. `Nat` converts it to the natural-log value the backward
// kernels and FlashAttention's `softmax_lse` use.
//
// `__builtin_amdgcn_logf` is the hardware `v_log_f32`, which returns log2 on this target -
// the same base `Exp2`/`__builtin_amdgcn_exp2f` exponentiates in, so the pair is used
// directly with no base conversion inside the loop. Multiplying by log2(e) here is wrong by
// exactly that factor. `tests/test_attn_fwd_bf16_d128.py` pins the emitted log-sum-exp
// against fp64.
struct OnlineLse
{
    float value = -__builtin_inff();

    CK_TILE_DEVICE void Update(float scaled_max, float tile_sum, float& alpha, float& beta)
    {
        const float previous = value;
        const float merged   = __builtin_fmaxf(previous, scaled_max);
        const float old_term = Exp2(previous - merged);
        const float new_term = Exp2(scaled_max - merged);
        const float combined = old_term + new_term * tile_sum;
        const float inv      = __builtin_amdgcn_rcpf(combined);
        alpha                = old_term * inv;
        beta                 = new_term * inv;
        value                = merged + __builtin_amdgcn_logf(combined);
    }

    CK_TILE_DEVICE float Nat() const
    {
        return value * 0.6931471805599453f;
    }
};

// Rescale the persistent output fragments by a per-query-row factor. `alpha` is uniform per
// row and lives in the lane that owns that row, so it is fanned out once per row.
template <index_t kDTiles>
CK_TILE_DEVICE void RescaleOutput(ck_tile::fp32x8_t (&output)[kDTiles], float alpha,
                                  index_t lane_group)
{
    const int alpha_bits = ck_tile::bit_cast<int>(alpha);
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
        const int even_alpha = __builtin_amdgcn_readlane(alpha_bits, i.value * 2);
        const int odd_alpha  = __builtin_amdgcn_readlane(alpha_bits, i.value * 2 + 1);
        const float row_alpha =
            ck_tile::bit_cast<float>(lane_group == 0 ? even_alpha : odd_alpha);
        ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
            output[d_tile.value][i.value] *= row_alpha;
        });
    });
}

} // namespace feather_attn
