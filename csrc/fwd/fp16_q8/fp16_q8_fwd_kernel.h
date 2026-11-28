// Forward attention with an E5M2-quantized Q tile, fp16 storage, fp32 accumulation.
//
// Row-owned FlashAttention-2 topology: one workgroup owns kBlockM query rows of one
// physical head, each wave owns 16 of those rows from score generation through softmax to
// the persistent output accumulator. Scores are computed transposed (K * Q^T) so a lane
// holds 8 keys of the single query lane%16 and the softmax needs no cross-lane reduction.
//
// Q is the only quantized operand: it is the operand that is read once per query tile,
// while K and V are re-read once per query tile for every query tile, so the quantization
// only pays for itself on Q. gfx1151 has no fp8 WMMA and no fp8 conversion instruction, so
// the E5M2 tile is decoded back to fp16 immediately before the WMMA.
//
// The ABI, the block decomposition, the WMMA wrapper, the log-domain softmax recurrence
// and the accumulator rescale are shared with the bf16 kernel (csrc/attention_fwd.h).

#pragma once

#include <hip/hip_runtime.h>

#include <ck_tile/core.hpp>

#include "attention_fwd.h"

#include <cstdint>

namespace feather_attn {

using ck_tile::fp16_t;
using ck_tile::index_t;
using ck_tile::number;

// E5M2 encode/decode helpers for the fp8-quantized Q tile.
struct Q8Ops
{
    static constexpr index_t kNumWaves = 8;
    static constexpr index_t kLaneRows = 16;
    static constexpr index_t kDPerWmma = 16;

    template <index_t kDTiles>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptor()
    {
        return ck_tile::make_naive_tensor_descriptor_packed(
            ck_tile::make_tuple(number<kNumWaves>{},
                                number<kDTiles>{},
                                number<kLaneRows>{},
                                number<kDPerWmma>{}));
    }

    CK_TILE_DEVICE static uint8_t EncodeE5M2(fp16_t value)
    {
        const uint16_t bits    = ck_tile::bit_cast<uint16_t>(value);
        const uint16_t rounded = bits + 0x007fu + ((bits >> 8) & 1u);
        return static_cast<uint8_t>(rounded >> 8);
    }

    CK_TILE_DEVICE static ck_tile::fp16x16_t DecodeE5M2x16(const uint8_t* src)
    {
        const auto packed = *reinterpret_cast<const ck_tile::uint32x4_t*>(src);
        ck_tile::uint32x8_t decoded;
        ck_tile::static_for<0, 4, 1>{}([&](auto i) {
            decoded[i.value * 2] =
                __builtin_amdgcn_perm(0u, packed[i.value], 0x010c000cu);
            decoded[i.value * 2 + 1] =
                __builtin_amdgcn_perm(0u, packed[i.value], 0x030c020cu);
        });
        return ck_tile::bit_cast<ck_tile::fp16x16_t>(decoded);
    }
};

// K tile layout: 8-fp16 chunks xor-swizzled by the key row.
struct QKOps
{
    static constexpr index_t kKPack = 8;

    template <index_t kHeadDim>
    CK_TILE_DEVICE static index_t LdsOffset(index_t row, index_t chunk)
    {
        constexpr index_t kKChunks = kHeadDim / kKPack;
        const index_t physical_chunk = chunk ^ (row % kKChunks);
        return (row * kHeadDim + physical_chunk * kKPack) * sizeof(fp16_t);
    }
};

// Transposed C fragment -> WMMA A fragment for the PV GEMM. Type-specific to fp16.
struct PTransposeOps
{
    using Wmma = WmmaOps<fp16_t>;

    CK_TILE_DEVICE static typename Wmma::AVecType TransposedCFragToA(
        const typename Wmma::CVecType& fragment,
        index_t lane)
    {
        ck_tile::uint32x8_t words;
        ck_tile::static_for<0, 4, 1>{}([&](auto i) {
            ck_tile::fp16x2_t pair;
            pair[0]          = ck_tile::type_convert<fp16_t>(fragment[i.value * 2]);
            pair[1]          = ck_tile::type_convert<fp16_t>(fragment[i.value * 2 + 1]);
            const uint32_t v = ck_tile::bit_cast<uint32_t>(pair);
            const uint32_t w = SwapLaneGroupBits(v);
            const uint32_t selector_0 = lane < 16 ? 0x05040100u : 0x01000504u;
            const uint32_t selector_1 = lane < 16 ? 0x07060302u : 0x03020706u;
            words[i.value * 2]     = __builtin_amdgcn_perm(w, v, selector_0);
            words[i.value * 2 + 1] = __builtin_amdgcn_perm(w, v, selector_1);
        });
        return ck_tile::bit_cast<typename Wmma::AVecType>(words);
    }
};

template <index_t kHeadDimValue, bool kNHD, bool kPadQ, bool kPadKV>
struct Fp16Q8FwdKernel
{
    using Wmma = WmmaOps<fp16_t>;
    using CVecType = typename Wmma::CVecType;
    using AVecType = typename Wmma::AVecType;
    using BVecType = typename Wmma::BVecType;

    static constexpr index_t kBlockSize  = Q8Ops::kNumWaves * 32;
    static constexpr index_t kWaves      = Q8Ops::kNumWaves;
    static constexpr int kMinBlockPerCu  = static_cast<int>(kWaves);
    static constexpr index_t kBlockM     = 128;
    static constexpr index_t kBlockN     = 64;
    static constexpr index_t kHeadDim    = kHeadDimValue;
    static constexpr index_t kWaveSize   = 32;
    static constexpr index_t kLaneRows   = 16;
    static constexpr index_t kDPerWmma   = 16;
    static constexpr index_t kDTiles     = kHeadDim / kDPerWmma;
    static constexpr index_t kNTiles     = kBlockN / kLaneRows;
    static constexpr index_t kQLdsBytes  = kBlockM * kHeadDim;
    static constexpr index_t kKVLdsBytes = kBlockN * kHeadDim * sizeof(fp16_t);
    static constexpr index_t kLdsBytes   = kQLdsBytes + kKVLdsBytes;
    static constexpr index_t kKPack      = 8;
    static constexpr index_t kKChunks    = kHeadDim / kKPack;
    static constexpr index_t kVPack      = 8;
    static constexpr index_t kVChunks    = kBlockN / kVPack;
    static constexpr index_t kCachedQTiles =
        kHeadDim == 64 && !kNHD ? (kPadKV && !kPadQ ? 3 : kDTiles) : 0;

    CK_TILE_DEVICE static index_t VLdsOffset(index_t d_row, index_t n_chunk)
    {
        const index_t d_phase = (d_row ^ (d_row / kVChunks)) % kVChunks;
        const index_t physical_chunk = n_chunk ^ d_phase;
        return (d_row * kBlockN + physical_chunk * kVPack) * sizeof(fp16_t);
    }

    CK_TILE_DEVICE static index_t VLdsElementOffset(index_t d_row, index_t n)
    {
        const index_t n_chunk = n / kVPack;
        const index_t d_phase = (d_row ^ (d_row / kVChunks)) % kVChunks;
        const index_t physical_chunk = n_chunk ^ d_phase;
        return (d_row * kBlockN + physical_chunk * kVPack + n % kVPack) *
               sizeof(fp16_t);
    }

    CK_TILE_DEVICE void operator()(AttentionFwdArgs kargs) const
    {
        constexpr auto q_lds_desc = Q8Ops::MakeLdsDescriptor<kDTiles>();
        alignas(16) __shared__ uint8_t lds[kLdsBytes];
        uint8_t* const q_lds  = lds;
        uint8_t* const kv_lds = lds + kQLdsBytes;

        const index_t tid        = ck_tile::get_thread_local_1d_id();
        const index_t wave       = tid / kWaveSize;
        const index_t lane       = tid % kWaveSize;
        const index_t lane_row   = lane % kLaneRows;
        const index_t lane_group = lane / kLaneRows;
        const index_t q_row      = wave * kLaneRows + lane_row;
        const index_t q_tiles    = (kargs.n_q + kBlockM - 1) / kBlockM;
        const index_t block      = ck_tile::get_block_1d_id();
        const AttentionBlockIndex at = DecodeAttentionBlock<kNHD>(block, q_tiles, kargs);

        const int32_t q_head_base =
            at.batch * kargs.q.batch_stride + at.head * kargs.q.head_stride;
        const int32_t k_head_base =
            at.batch * kargs.k.batch_stride + at.head * kargs.k.head_stride;
        const int32_t v_head_base =
            at.batch * kargs.v.batch_stride + at.head * kargs.v.head_stride;
        const int32_t o_head_base =
            at.batch * kargs.o.batch_stride + at.head * kargs.o.head_stride;

        const index_t q_row_stride  = kargs.q.row_stride;
        const index_t kv_row_stride = kargs.k.row_stride;
        const index_t q_start       = at.q_tile * kBlockM;

        const fp16_t* const q_ptr = reinterpret_cast<const fp16_t*>(kargs.q.ptr);
        const fp16_t* const k_ptr = reinterpret_cast<const fp16_t*>(kargs.k.ptr);
        const fp16_t* const v_ptr = reinterpret_cast<const fp16_t*>(kargs.v.ptr);
        fp16_t* const o_ptr = reinterpret_cast<fp16_t*>(const_cast<void*>(kargs.o.ptr));

        const fp16_t* const q_tile_ptr =
            q_ptr + q_head_base + q_start * q_row_stride;
        const fp16_t* const k_head_ptr = k_ptr + k_head_base;
        const fp16_t* const v_head_ptr = v_ptr + v_head_base;
        fp16_t* const o_tile_ptr = o_ptr + o_head_base + q_start * q_row_stride;

        ck_tile::static_for<0, kDTiles / 2, 1>{}([&](auto i) {
            constexpr index_t d_tile_in_group = i.value * 2;
            const index_t d_tile              = d_tile_in_group + lane_group;
            ck_tile::fp16x16_t q = {};
            if constexpr(kPadQ)
            {
                if(q_start + q_row < kargs.n_q)
                    q = *reinterpret_cast<const ck_tile::fp16x16_t*>(
                        q_tile_ptr + q_row * q_row_stride + d_tile * kDPerWmma);
            }
            else
            {
                q = *reinterpret_cast<const ck_tile::fp16x16_t*>(
                    q_tile_ptr + q_row * q_row_stride + d_tile * kDPerWmma);
            }

            ck_tile::uint8x16_t q8;
            ck_tile::static_for<0, kDPerWmma, 1>{}([&](auto j) {
                const float scaled =
                    ck_tile::type_convert<float>(q[j.value]) * kargs.softmax_scale_log2;
                q8[j.value] =
                    Q8Ops::EncodeE5M2(ck_tile::type_convert<fp16_t>(scaled));
            });

            const index_t q_lds_offset = q_lds_desc.calculate_offset(
                ck_tile::make_tuple(wave, d_tile, lane_row, index_t{0}));
            *reinterpret_cast<ck_tile::uint8x16_t*>(q_lds + q_lds_offset) = q8;
        });

        CVecType output[kDTiles] = {};
        uint32_t store_wave;
        asm volatile("v_readfirstlane_b32 %0, %1" : "=s"(store_wave) : "v"(wave));
        OnlineLse lse;
        AVecType cached_q[kCachedQTiles > 0 ? kCachedQTiles : 1] = {};

        #pragma unroll 1
        for(index_t key_start = 0; key_start < kargs.n_kv; key_start += kBlockN)
        {
            constexpr index_t kKLoadIssues = kBlockN * kKChunks / kBlockSize;
            ck_tile::static_for<0, kKLoadIssues, 1>{}([&](auto issue) {
                const index_t linear_chunk = tid + issue.value * kBlockSize;
                const index_t k_row        = linear_chunk / kKChunks;
                const index_t k_chunk      = linear_chunk % kKChunks;
                ck_tile::fp16x8_t k = {};
                if constexpr(kPadKV)
                {
                    if(key_start + k_row < kargs.n_kv)
                        k = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                            k_head_ptr + (key_start + k_row) * kv_row_stride +
                            k_chunk * kKPack);
                }
                else
                {
                    k = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        k_head_ptr + (key_start + k_row) * kv_row_stride +
                        k_chunk * kKPack);
                }
                *reinterpret_cast<ck_tile::fp16x8_t*>(
                    kv_lds + QKOps::LdsOffset<kHeadDim>(k_row, k_chunk)) = k;
            });

            ck_tile::block_sync_lds();

            CVecType scores[kNTiles];
            ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                const index_t q_lds_offset = q_lds_desc.calculate_offset(
                    ck_tile::make_tuple(wave, d_tile.value, lane_row, index_t{0}));
                AVecType q;
                if constexpr(d_tile.value < kCachedQTiles)
                {
                    if(key_start == 0)
                        cached_q[d_tile.value] =
                            Q8Ops::DecodeE5M2x16(q_lds + q_lds_offset);
                    q = cached_q[d_tile.value];
                }
                else
                {
                    q = Q8Ops::DecodeE5M2x16(q_lds + q_lds_offset);
                }

                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t k_row   = n_tile.value * kLaneRows + lane_row;
                    const index_t chunk_0 = d_tile.value * 2;
                    const auto k_lo = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + QKOps::LdsOffset<kHeadDim>(k_row, chunk_0));
                    const auto k_hi = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + QKOps::LdsOffset<kHeadDim>(k_row, chunk_0 + 1));
                    BVecType k;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        k[j.value]     = k_lo[j.value];
                        k[j.value + 8] = k_hi[j.value];
                    });
                    if constexpr(d_tile.value == 0)
                        Wmma::Init(scores[n_tile.value], k, q);
                    else
                        Wmma::InPlace(scores[n_tile.value], k, q);
                });
            });

            if constexpr(kPadKV)
            {
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                        const index_t key_row = key_start + n_tile.value * kLaneRows +
                                                i.value * 2 + lane_group;
                        if(key_row >= kargs.n_kv)
                            scores[n_tile.value][i.value] = -__builtin_inff();
                    });
                });
            }

            float row_max = -__builtin_inff();
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    row_max = __builtin_fmaxf(row_max, scores[n_tile.value][i.value]);
                });
            });
            row_max = __builtin_fmaxf(row_max, SwapLaneGroup(row_max));

            float local_sum = 0.0f;
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    const float p =
                        Exp2(scores[n_tile.value][i.value] - row_max);
                    scores[n_tile.value][i.value] = p;
                    local_sum += p;
                });
            });
            const float tile_sum = local_sum + SwapLaneGroup(local_sum);
            float alpha = 0.0f;
            float beta  = 0.0f;
            lse.Update(row_max, tile_sum, alpha, beta);

            RescaleOutput<kDTiles>(output, alpha, lane_group);
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    scores[n_tile.value][i.value] *= beta;
                });
            });

            ck_tile::block_sync_lds();

            index_t kv_lane_row;
            asm volatile("v_mov_b32 %0, %1" : "=v"(kv_lane_row) : "v"(lane_row));
            constexpr index_t kVDChunks = kHeadDim / kVPack;
            constexpr index_t kVDGroups = kLaneRows / kVDChunks;
            constexpr index_t kVRowsPerLane = 4 / kVDGroups;
            constexpr index_t kVColsPerLane = 8;
            const index_t v_n_base = wave * 8 + lane_group * (kVRowsPerLane * kVDGroups) +
                                     (lane_row / kVDChunks) * kVRowsPerLane;
            const index_t v_d_base = (lane_row % kVDChunks) * kVColsPerLane;
            ck_tile::fp16x8_t v_rows[kVRowsPerLane] = {};
            ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto row) {
                if constexpr(kPadKV)
                {
                    if(key_start + v_n_base + row.value < kargs.n_kv)
                        v_rows[row.value] = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                            v_head_ptr +
                            (key_start + v_n_base + row.value) * kv_row_stride +
                            v_d_base);
                }
                else
                {
                    v_rows[row.value] = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        v_head_ptr +
                        (key_start + v_n_base + row.value) * kv_row_stride +
                        v_d_base);
                }
            });
            ck_tile::static_for<0, kVColsPerLane, 1>{}([&](auto col) {
                if constexpr(kVRowsPerLane == 4)
                {
                    ck_tile::fp16x4_t v_column;
                    ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto row) {
                        v_column[row.value] = v_rows[row.value][col.value];
                    });
                    *reinterpret_cast<ck_tile::fp16x4_t*>(
                        kv_lds + VLdsElementOffset(v_d_base + col.value, v_n_base)) =
                        v_column;
                }
                else
                {
                    ck_tile::fp16x2_t v_column;
                    ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto row) {
                        v_column[row.value] = v_rows[row.value][col.value];
                    });
                    *reinterpret_cast<ck_tile::fp16x2_t*>(
                        kv_lds + VLdsElementOffset(v_d_base + col.value, v_n_base)) =
                        v_column;
                }
            });

            ck_tile::block_sync_lds();

            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                const AVecType p = PTransposeOps::TransposedCFragToA(scores[n_tile.value], lane);
                ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                    const index_t d_row = d_tile.value * kDPerWmma + kv_lane_row;
                    const auto v_lo = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + VLdsOffset(d_row, n_tile.value * 2));
                    const auto v_hi = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + VLdsOffset(d_row, n_tile.value * 2 + 1));
                    BVecType v;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        v[j.value]     = v_lo[j.value];
                        v[j.value + 8] = v_hi[j.value];
                    });
                    Wmma::InPlace(output[d_tile.value], p, v);
                });
            });

            ck_tile::block_sync_lds();
        }

        uint32_t store_lane;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(store_lane));
        ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const index_t row =
                    store_wave * kLaneRows + i.value * 2 + store_lane / kLaneRows;
                const index_t col =
                    d_tile.value * kDPerWmma + store_lane % kLaneRows;
                if constexpr(kPadQ)
                {
                    if(q_start + row < kargs.n_q)
                        o_tile_ptr[row * q_row_stride + col] =
                            ck_tile::type_convert<fp16_t>(output[d_tile.value][i.value]);
                }
                else
                {
                    o_tile_ptr[row * q_row_stride + col] =
                        ck_tile::type_convert<fp16_t>(output[d_tile.value][i.value]);
                }
            });
        });
    }
};

// One launcher for every image of this family. The image selects head dim and tail modes.
template <index_t kHeadDim, bool kNHD, bool kPadQ, bool kPadKV>
bool LaunchFp16Q8Fwd(const AttentionFwdParams& params)
{
    static_assert(kHeadDim == 64 || kHeadDim == 128);
    return LaunchAttentionFwd<Fp16Q8FwdKernel<kHeadDim, kNHD, kPadQ, kPadKV>>(params);
}

} // namespace feather_attn
