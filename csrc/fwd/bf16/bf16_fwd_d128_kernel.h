// bf16 D128 forward attention for gfx1151.
//
// Row-owned FlashAttention-2 topology: one workgroup owns kBr query rows of one
// physical head, each wave owns 16 of those rows from QK through softmax to the
// persistent output accumulator. Scores are computed transposed (K * Q^T) so a
// lane holds 8 keys of the single query lane%16 and the softmax needs no
// cross-lane reduction along the key axis.
//
// Design notes:
// - bf16 Q lives in registers as WMMA A-fragments. A wave only ever reads its
//   own 16 rows, and lanes 16-31 naturally re-read the address of lanes 0-15,
//   which is exactly the A-fragment replication the ISA requires. No Q LDS, no
//   encode, no decode, no per-iteration predicate.
// - K and V tiles are row-padded (kSwizzled = false) or xor-swizzled
//   (kSwizzled = true). Padding makes every LDS address affine in the tile
//   index (one base register plus compile-time immediates) and costs 512 bytes
//   per row of tile, which at kBc = 64 pushes the four-buffer footprint past
//   the 64 KiB workgroup limit. The swizzled form fits exactly but needs a
//   runtime xor, shift and add per load. Both are instantiated.
// - The keys of a 16-key score tile are permuted so that lane group 0 owns keys
//   0..7 and lane group 1 owns keys 8..15 (instead of even/odd). That makes the
//   two bf16 probability pairs of each lane contiguous, so the P -> A transpose
//   costs 3 ALU ops per fragment word instead of 5.
// - The KV tail zeroes the out-of-range K and V rows and subtracts their exact
//   softmax mass from the row sum, which keeps the per-tile mask out of the hot
//   loop.
// - The softmax scale is folded into the exponentiation as
//   exp2(fma(score, scale, -c)), so it costs no instruction relative to the
//   unscaled form and Q keeps its full bf16 precision.

#pragma once

#include <hip/hip_runtime.h>

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>

#include "fwd/bf16/bf16_fwd_params.h"

namespace feather_attn {

using ck_tile::bf16_t;
using ck_tile::index_t;

struct Bf16Wmma
{
    using AVecType = ck_tile::bf16x16_t;
    using BVecType = ck_tile::bf16x16_t;
    using CVecType = ck_tile::fp32x8_t;
};

// kBuffersPerTile = 2 publishes both K and V one tile ahead (one barrier per KV
// tile). = 1 reuses a single K and V buffer (two barriers) for the configurations
// whose double-buffered footprint would exceed the 64 KiB LDS cap.
// kSwizzled = false uses row-padded, affine LDS addressing. true uses the compact
// xor-swizzled layout.
template <bool kHeadFast, index_t kBr, index_t kBc, index_t kBuffersPerTile, bool kSwizzled>
struct Bf16FwdD128Kernel
{
    static constexpr index_t kD         = 128;
    static constexpr index_t kBlockM    = kBr;
    static constexpr index_t kBlockN    = kBc;
    static constexpr index_t kWaves     = kBr / 16;
    static constexpr index_t kBlockSize = kWaves * 32;
    static constexpr index_t kLaneRows  = 16;
    static constexpr index_t kDTiles    = kD / 16;
    static constexpr index_t kNTiles    = kBc / 16;
    static constexpr index_t kKPack     = 8;
    static constexpr index_t kPad       = kSwizzled ? 0 : 8;
    static constexpr index_t kKStride   = kD + kPad;
    static constexpr index_t kVStride   = kBc + kPad;
    static constexpr index_t kKBufBytes = kBc * kKStride * sizeof(bf16_t);
    static constexpr index_t kVBufBytes = kD * kVStride * sizeof(bf16_t);
    static constexpr index_t kKRegion   = kBuffersPerTile * kKBufBytes;
    static constexpr index_t kVRegion   = kBuffersPerTile * kVBufBytes;
    static constexpr index_t kLdsBytes  = kKRegion + kVRegion;
    static constexpr index_t kKStep     = 16 * kKStride * sizeof(bf16_t);
    static constexpr index_t kNStep     = 2 * kKPack * sizeof(bf16_t);
    static constexpr index_t kDPitch    = 16 * kVStride * sizeof(bf16_t);
    static constexpr index_t kKLoadIssues = kBc * (kD / kKPack) / kBlockSize;
    static constexpr index_t kKeysPerWave  = kBc / kWaves;
    static constexpr index_t kVRowsPerLane = kKeysPerWave / 2;
    static constexpr index_t kVColsPerLane = 8;
    static_assert(kKeysPerWave % 2 == 0 && kVRowsPerLane >= 1 && kVRowsPerLane <= 4);
    static_assert(kLdsBytes <= 65536, "per-workgroup LDS limit");

    using Wmma = Bf16Wmma;

    struct Kargs
    {
        Bf16Operand q;
        Bf16Operand k;
        Bf16Operand v;
        Bf16Operand o;
        float softmax_scale_log2;
        int32_t n_q;
        int32_t n_kv;
        int32_t head_start;
        int32_t launch_heads;
    };

    CK_TILE_DEVICE static void WmmaInit(
        typename Wmma::CVecType& c,
        const typename Wmma::AVecType& a,
        const typename Wmma::BVecType& b)
    {
        asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, 0" : "=v"(c) : "v"(a), "v"(b));
    }

    CK_TILE_DEVICE static void WmmaInPlace(
        typename Wmma::CVecType& c,
        const typename Wmma::AVecType& a,
        const typename Wmma::BVecType& b)
    {
        asm volatile("v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    }

    CK_TILE_DEVICE static float Exp2(float value)
    {
        return __builtin_amdgcn_exp2f(value);
    }

    // bf16 pair with the mantissa truncated. The (1 + 2^-9) folded into beta makes
    // the truncation unbiased to first order.
    CK_TILE_DEVICE static uint32_t PackBf16Pair(float lo, float hi)
    {
        return ck_tile::bit_cast<uint32_t>(lo) >> 16 |
               (ck_tile::bit_cast<uint32_t>(hi) & 0xffff0000u);
    }

    CK_TILE_DEVICE static float SwapLaneGroup(float value)
    {
        const uint32_t bits = ck_tile::bit_cast<uint32_t>(value);
        return ck_tile::bit_cast<float>(
            __builtin_amdgcn_permlanex16(0u, bits, 0x76543210u, 0xfedcba98u, false, true));
    }

    CK_TILE_DEVICE static uint32_t SwapLaneGroupBits(uint32_t bits)
    {
        return __builtin_amdgcn_permlanex16(
            0u, bits, 0x76543210u, 0xfedcba98u, false, true);
    }

    CK_TILE_DEVICE static index_t VPhase(index_t d)
    {
        return (d ^ (d / kKPack)) % (kBc / kKPack);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        alignas(16) __shared__ uint8_t lds[kLdsBytes];

        const index_t tid        = ck_tile::get_thread_local_1d_id();
        const index_t wave       = tid / 32;
        const index_t lane       = tid % 32;
        const index_t lane_row   = lane % kLaneRows;
        const index_t lane_group = lane / kLaneRows;
        const index_t q_row      = wave * kLaneRows + lane_row;

        const index_t q_tiles = (kargs.n_q + kBlockM - 1) / kBlockM;
        const index_t block   = ck_tile::get_block_1d_id();
        index_t q_tile;
        index_t head;
        index_t batch;
        if constexpr(kHeadFast)
        {
            head    = block % kargs.launch_heads + kargs.head_start;
            const index_t tile_batch = block / kargs.launch_heads;
            q_tile  = tile_batch % q_tiles;
            batch   = tile_batch / q_tiles;
        }
        else
        {
            q_tile = block % q_tiles;
            const index_t linear = block / q_tiles;
            head  = linear % kargs.launch_heads + kargs.head_start;
            batch = linear / kargs.launch_heads;
        }
        const index_t q_start = q_tile * kBlockM;

        const int32_t q_head_base =
            batch * kargs.q.batch_stride + head * kargs.q.head_stride;
        const int32_t k_head_base =
            batch * kargs.k.batch_stride + head * kargs.k.head_stride;
        const int32_t v_head_base =
            batch * kargs.v.batch_stride + head * kargs.v.head_stride;
        const int32_t o_head_base =
            batch * kargs.o.batch_stride + head * kargs.o.head_stride;

        const bf16_t* const q_ptr = reinterpret_cast<const bf16_t*>(kargs.q.ptr);
        const bf16_t* const k_ptr = reinterpret_cast<const bf16_t*>(kargs.k.ptr);
        const bf16_t* const v_ptr = reinterpret_cast<const bf16_t*>(kargs.v.ptr);
        bf16_t* const o_ptr =
            reinterpret_cast<bf16_t*>(const_cast<void*>(kargs.o.ptr));

        // Q into A-fragments, once per query tile.
        const index_t q_row_global = q_start + q_row;
        const index_t q_row_safe =
            q_row_global < kargs.n_q ? q_row_global : kargs.n_q - 1;
        const int32_t q_lane_off =
            q_head_base + static_cast<int32_t>(q_row_safe) * kargs.q.row_stride;
        typename Wmma::AVecType q_frag[kDTiles];
        ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
            q_frag[d_tile.value] = *reinterpret_cast<const ck_tile::bf16x16_t*>(
                q_ptr + q_lane_off + d_tile.value * 16);
        });

        typename Wmma::CVecType output[kDTiles] = {};
        float lane_lse = -__builtin_inff();

        const index_t k_lane_bytes = lane_row * kKStride * sizeof(bf16_t);
        // In the swizzled layout the second d-chunk of a pair sits one chunk away
        // in physical order, and its direction is bit 0 of the lane row.
        const index_t k_hi_delta =
            kSwizzled ? ((lane_row & 1) ? -static_cast<index_t>(kKPack * sizeof(bf16_t))
                                        : static_cast<index_t>(kKPack * sizeof(bf16_t)))
                      : static_cast<index_t>(kKPack * sizeof(bf16_t));
        const index_t v_lane_bytes = lane_row * kVStride * sizeof(bf16_t);
        const index_t v_n_base = wave * kKeysPerWave + lane_group * kVRowsPerLane;
        const index_t v_store_chunk = (v_n_base / kKPack) * kKPack * sizeof(bf16_t);
        const index_t v_store_off   = (v_n_base % kKPack) * sizeof(bf16_t);

        const index_t kv_tiles = (kargs.n_kv + kBlockN - 1) / kBlockN;
        const ck_tile::bf16x8_t zero8 = {};

        auto k_row_off = [&](index_t tile, index_t key_in_tile) {
            const index_t key = tile * kBlockN + key_in_tile;
            const index_t safe = key < kargs.n_kv ? key : kargs.n_kv - 1;
            return k_head_base + static_cast<int32_t>(safe) * kargs.k.row_stride;
        };
        auto v_row_off = [&](index_t tile, index_t key_in_tile) {
            const index_t key = tile * kBlockN + key_in_tile;
            const index_t safe = key < kargs.n_kv ? key : kargs.n_kv - 1;
            return v_head_base + static_cast<int32_t>(safe) * kargs.v.row_stride;
        };

        ck_tile::bf16x8_t k_pref[kKLoadIssues]  = {};
        ck_tile::bf16x8_t v_pref[kVRowsPerLane] = {};

        auto prefetch_k = [&](index_t tile) {
            ck_tile::static_for<0, kKLoadIssues, 1>{}([&](auto issue) {
                const index_t linear  = tid + issue.value * kBlockSize;
                const index_t lds_row = linear / (kD / kKPack);
                const index_t k_chunk = linear % (kD / kKPack);
                // The staged row order is permuted so lane group 0 owns the first
                // half of the keys. Invert it to find the tensor row to read.
                const index_t key = (lds_row & ~index_t{15}) +
                                    ((lds_row & 1) ? 8 + ((lds_row & 15) >> 1)
                                                   : ((lds_row & 15) >> 1));
                ck_tile::bf16x8_t value =
                    *reinterpret_cast<const ck_tile::bf16x8_t*>(
                        k_ptr + k_row_off(tile, key) + k_chunk * kKPack);
                if(tile * kBlockN + key >= kargs.n_kv)
                    value = zero8;
                k_pref[issue.value] = value;
            });
        };
        auto commit_k = [&](index_t buf) {
            ck_tile::static_for<0, kKLoadIssues, 1>{}([&](auto issue) {
                const index_t linear  = tid + issue.value * kBlockSize;
                const index_t lds_row = linear / (kD / kKPack);
                const index_t k_chunk = linear % (kD / kKPack);
                const index_t off =
                    kSwizzled
                        ? lds_row * (kD * sizeof(bf16_t)) +
                              ((k_chunk ^ (lds_row & (kD / kKPack - 1))) *
                               kKPack * sizeof(bf16_t))
                        : lds_row * (kKStride * sizeof(bf16_t)) +
                              k_chunk * (kKPack * sizeof(bf16_t));
                *reinterpret_cast<ck_tile::bf16x8_t*>(lds + buf + off) =
                    k_pref[issue.value];
            });
        };
        auto prefetch_v = [&](index_t tile) {
            ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto r) {
                const index_t key = v_n_base + r.value;
                ck_tile::bf16x8_t value =
                    *reinterpret_cast<const ck_tile::bf16x8_t*>(
                        v_ptr + v_row_off(tile, key) + lane_row * kVColsPerLane);
                if(tile * kBlockN + key >= kargs.n_kv)
                    value = zero8;
                v_pref[r.value] = value;
            });
        };
        auto commit_v = [&](index_t buf) {
            ck_tile::static_for<0, kVColsPerLane, 1>{}([&](auto col) {
                const index_t d = lane_row * kVColsPerLane + col.value;
                const index_t off =
                    kSwizzled
                        ? (d * kBc +
                           (((v_n_base / kKPack) ^ VPhase(d)) * kKPack)) *
                              sizeof(bf16_t)
                        : d * (kVStride * sizeof(bf16_t)) + v_store_chunk;
                uint8_t* const dst = lds + buf + off + v_store_off;
                if constexpr(kVRowsPerLane == 4)
                {
                    ck_tile::bf16x4_t packed;
                    ck_tile::static_for<0, 4, 1>{}([&](auto r) {
                        packed[r.value] = v_pref[r.value][col.value];
                    });
                    *reinterpret_cast<uint64_t*>(dst) =
                        ck_tile::bit_cast<uint64_t>(packed);
                }
                else if constexpr(kVRowsPerLane == 2)
                {
                    ck_tile::bf16x2_t packed;
                    packed[0] = v_pref[0][col.value];
                    packed[1] = v_pref[1][col.value];
                    *reinterpret_cast<uint32_t*>(dst) =
                        ck_tile::bit_cast<uint32_t>(packed);
                }
                else
                {
                    *reinterpret_cast<uint16_t*>(dst) =
                        ck_tile::bit_cast<uint16_t>(v_pref[0][col.value]);
                }
            });
        };

        prefetch_k(0);
        commit_k(0);
        prefetch_v(0);
        commit_v(kKRegion);
        ck_tile::block_sync_lds();
        if(kv_tiles > 1)
        {
            prefetch_k(1);
            prefetch_v(1);
        }

        index_t kcur = 0;
        index_t knxt = kKBufBytes;
        index_t vcur = kKRegion;
        index_t vnxt = kKRegion + kVBufBytes;

        #pragma unroll 1
        for(index_t tile = 0; tile < kv_tiles; ++tile)
        {
            const index_t key_start = tile * kBlockN;
            const bool tail_tile    = key_start + kBlockN > kargs.n_kv;

            typename Wmma::CVecType scores[kNTiles];
            ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                const index_t k_off =
                    kcur + k_lane_bytes +
                    (kSwizzled ? (((2 * d_tile.value) ^ lane_row) * kKPack *
                                  sizeof(bf16_t))
                               : d_tile.value * (2 * kKPack * sizeof(bf16_t)));
                const typename Wmma::AVecType& q = q_frag[d_tile.value];
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t base = k_off + n_tile.value * kKStep;
                    const ck_tile::bf16x8_t k_lo =
                        *reinterpret_cast<const ck_tile::bf16x8_t*>(lds + base);
                    const ck_tile::bf16x8_t k_hi =
                        *reinterpret_cast<const ck_tile::bf16x8_t*>(
                            lds + base + k_hi_delta);
                    typename Wmma::BVecType k;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        k[j.value]     = k_lo[j.value];
                        k[j.value + 8] = k_hi[j.value];
                    });
                    if constexpr(d_tile.value == 0)
                        WmmaInit(scores[n_tile.value], k, q);
                    else
                        WmmaInPlace(scores[n_tile.value], k, q);
                });
            });

            // online softmax in the log2 domain
            float row_max = -__builtin_inff();
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    row_max = __builtin_fmaxf(row_max, scores[n_tile.value][i.value]);
                });
            });
            row_max = __builtin_fmaxf(row_max, SwapLaneGroup(row_max));

            const float scale      = kargs.softmax_scale_log2;
            const float scaled_max = row_max * scale;
            const float bias       = -scaled_max;

            float local_sum = 0.0f;
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    const float p = Exp2(__builtin_fmaf(
                        scores[n_tile.value][i.value], scale, bias));
                    scores[n_tile.value][i.value] = p;
                    local_sum += p;
                });
            });

            if(tail_tile)
            {
                // Out-of-range K and V rows are zeroed while staging, so their
                // scores are exactly 0 and their probability is exp2(-scaled_max).
                index_t invalid = 0;
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t remaining =
                        kargs.n_kv - key_start - n_tile.value * 16 - 8 * lane_group;
                    invalid += remaining >= 8 ? 0 : (remaining > 0 ? 8 - remaining : 8);
                });
                local_sum -= static_cast<float>(invalid) * Exp2(-scaled_max);
            }

            const float tile_sum  = local_sum + SwapLaneGroup(local_sum);
            const float old_lse   = lane_lse;
            const float new_max   = __builtin_fmaxf(old_lse, scaled_max);
            const float old_term  = Exp2(old_lse - new_max);
            const float tile_term = Exp2(scaled_max - new_max);
            const float combined  = old_term + tile_term * tile_sum;
            const float inv       = __builtin_amdgcn_rcpf(combined);
            const float alpha     = old_term * inv;
            // The probability tile is rounded to bf16 by truncation, whose mean
            // error is half an ulp. Folding (1 + 2^-9) into beta cancels it.
            const float beta      = tile_term * inv * 1.001953125f;
            lane_lse              = new_max + __builtin_amdgcn_logf(combined);

            // Rescale the persistent accumulator: alpha is uniform per query row,
            // so it is fanned out from the owning lane once per row.
            const int alpha_bits = ck_tile::bit_cast<int>(alpha);
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const int even_alpha =
                    __builtin_amdgcn_readlane(alpha_bits, i.value * 2);
                const int odd_alpha =
                    __builtin_amdgcn_readlane(alpha_bits, i.value * 2 + 1);
                const float row_alpha = ck_tile::bit_cast<float>(
                    lane_group == 0 ? even_alpha : odd_alpha);
                ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                    output[d_tile.value][i.value] *= row_alpha;
                });
            });

            // P scaled by beta, then transposed into A-fragment layout.
            typename Wmma::AVecType p_frag[kNTiles];
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::uint32x8_t words;
                uint32_t own[4];
                ck_tile::static_for<0, 4, 1>{}([&](auto k) {
                    own[k.value] = PackBf16Pair(
                        scores[n_tile.value][2 * k.value] * beta,
                        scores[n_tile.value][2 * k.value + 1] * beta);
                });
                ck_tile::static_for<0, 4, 1>{}([&](auto k) {
                    const uint32_t partner = SwapLaneGroupBits(own[k.value]);
                    words[k.value]     = lane_group == 0 ? own[k.value] : partner;
                    words[k.value + 4] = lane_group == 0 ? partner : own[k.value];
                });
                p_frag[n_tile.value] =
                    ck_tile::bit_cast<typename Wmma::AVecType>(words);
            });

            // PV accumulate
            ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                const index_t v_row_bytes =
                    vcur +
                    (kSwizzled
                         ? (d_tile.value * 16 + lane_row) * kBc * sizeof(bf16_t)
                         : v_lane_bytes + d_tile.value * kDPitch);
                const index_t v_phase =
                    kSwizzled ? VPhase(d_tile.value * 16 + lane_row) : 0;
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t base =
                        kSwizzled
                            ? v_row_bytes +
                                  (((2 * n_tile.value) ^ v_phase) * kKPack *
                                   sizeof(bf16_t))
                            : v_row_bytes + n_tile.value * kNStep;
                    const index_t hi_delta =
                        kSwizzled
                            ? ((((2 * n_tile.value) ^ v_phase) & 1) ? -16 : 16)
                            : static_cast<index_t>(kKPack * sizeof(bf16_t));
                    const ck_tile::bf16x8_t v_lo =
                        *reinterpret_cast<const ck_tile::bf16x8_t*>(lds + base);
                    const ck_tile::bf16x8_t v_hi =
                        *reinterpret_cast<const ck_tile::bf16x8_t*>(
                            lds + base + hi_delta);
                    typename Wmma::BVecType v;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        v[j.value]     = v_lo[j.value];
                        v[j.value + 8] = v_hi[j.value];
                    });
                    WmmaInPlace(output[d_tile.value], p_frag[n_tile.value], v);
                });
            });

            const index_t next_tile = tile + 1;
            const index_t after =
                next_tile + 1 < kv_tiles ? next_tile + 1 : kv_tiles - 1;
            if constexpr(kBuffersPerTile == 2)
            {
                if(next_tile < kv_tiles)
                {
                    commit_k(knxt);
                    commit_v(vnxt);
                    prefetch_k(after);
                    prefetch_v(after);
                }
                ck_tile::block_sync_lds();
                const index_t sk = kcur;
                kcur = knxt;
                knxt = sk;
                const index_t sv = vcur;
                vcur = vnxt;
                vnxt = sv;
            }
            else
            {
                ck_tile::block_sync_lds();
                if(next_tile < kv_tiles)
                {
                    commit_k(kcur);
                    commit_v(vcur);
                    prefetch_k(after);
                    prefetch_v(after);
                }
                ck_tile::block_sync_lds();
            }
        }

        // epilogue
        const index_t out_row = q_start + wave * kLaneRows + lane_group;
        ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const index_t row = out_row + i.value * 2;
                if(row < kargs.n_q)
                {
                    const index_t col = d_tile.value * 16 + lane_row;
                    o_ptr[o_head_base +
                          static_cast<int32_t>(row) * kargs.o.row_stride + col] =
                        ck_tile::type_convert<bf16_t>(output[d_tile.value][i.value]);
                }
            });
        });
    }
};

template <bool kHeadFast, index_t kBr, index_t kBc, index_t kBuffers, bool kSwizzled>
bool LaunchBf16FwdD128(const Bf16FwdParams& params)
{
    using Kernel = Bf16FwdD128Kernel<kHeadFast, kBr, kBc, kBuffers, kSwizzled>;
    const typename Kernel::Kargs kargs{
        params.q,
        params.k,
        params.v,
        params.o,
        params.softmax_scale_log2,
        params.n_q,
        params.n_kv,
        params.head_start,
        params.launch_heads};
    const auto kernel = ck_tile::make_kernel<Kernel::kWaves, ck_tile::gfx115_t>(
        Kernel{},
        dim3(params.grid_size),
        dim3(Kernel::kBlockSize),
        0,
        kargs);
    (void)hipGetLastError();
    ck_tile::launch_kernel(ck_tile::stream_config{params.stream}, kernel);
    return hipPeekAtLastError() == hipSuccess;
}

} // namespace feather_attn
