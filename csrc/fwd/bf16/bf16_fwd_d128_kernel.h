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

#include "attention_fwd.h"

namespace feather_attn {

using ck_tile::bf16_t;
using ck_tile::index_t;

// kBuffersPerTile = 2 publishes both K and V one tile ahead (one barrier per KV
// tile). = 1 reuses a single K and V buffer (two barriers) for the configurations
// whose double-buffered footprint would exceed the 64 KiB LDS cap.
// kSwizzled = false uses row-padded, affine LDS addressing. true uses the compact
// xor-swizzled layout.
// kLazy = true keeps the exponential base fixed until a tile's row maximum exceeds it by
// more than kTheta, so the per-tile rescale, the per-tile normalisation of the probability
// tile and the online log-sum-exp update all leave the loop. The accumulator holds the
// unnormalised sum of p*v and is divided once, in the epilogue.
template <bool kHeadFast, index_t kBr, index_t kBc, index_t kBuffersPerTile, bool kSwizzled, bool kLazy = false>
struct Bf16FwdD128Kernel
{
    // Log2 units of headroom the lazy base keeps above the running row maximum. The
    // probabilities of a tile therefore reach at most 2^kTheta, which bf16 and the fp32
    // accumulator both carry without a relative-precision change.
    static constexpr float kTheta = 8.0f;
    static constexpr index_t kD         = 128;
    static constexpr index_t kBlockM    = kBr;
    static constexpr index_t kBlockN    = kBc;
    static constexpr index_t kWaves     = kBr / 16;
    static constexpr int kMinBlockPerCu = static_cast<int>(kWaves);
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

    using Wmma = WmmaOps<ck_tile::bf16_t>;
    using CVecType = typename Wmma::CVecType;
    using AVecType = typename Wmma::AVecType;
    using BVecType = typename Wmma::BVecType;

    // bf16 pair with the mantissa truncated. The (1 + 2^-9) folded into beta makes the
    // truncation unbiased to first order, which is why no rounding step is needed.
    CK_TILE_DEVICE static uint32_t PackBf16Pair(float lo, float hi)
    {
        return ck_tile::bit_cast<uint32_t>(lo) >> 16 |
               (ck_tile::bit_cast<uint32_t>(hi) & 0xffff0000u);
    }

    CK_TILE_DEVICE static index_t VPhase(index_t d)
    {
        return (d ^ (d / kKPack)) % (kBc / kKPack);
    }

    CK_TILE_DEVICE void operator()(AttentionFwdArgs kargs) const
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
        const AttentionBlockIndex at = DecodeAttentionBlock<kHeadFast>(block, q_tiles, kargs);
        const index_t q_start = at.q_tile * kBlockM;

        const bf16_t* const q_ptr = reinterpret_cast<const bf16_t*>(kargs.q.ptr);
        const bf16_t* const k_ptr = reinterpret_cast<const bf16_t*>(kargs.k.ptr);
        const bf16_t* const v_ptr = reinterpret_cast<const bf16_t*>(kargs.v.ptr);
        bf16_t* const o_ptr = reinterpret_cast<bf16_t*>(const_cast<void*>(kargs.o.ptr));

        const int32_t q_head_base =
            at.batch * kargs.q.batch_stride + at.head * kargs.q.head_stride;
        const int32_t k_head_base =
            at.batch * kargs.k.batch_stride + at.head * kargs.k.head_stride;
        const int32_t v_head_base =
            at.batch * kargs.v.batch_stride + at.head * kargs.v.head_stride;
        const int32_t o_head_base =
            at.batch * kargs.o.batch_stride + at.head * kargs.o.head_stride;

        // Q into A-fragments, once per query tile.
        const index_t q_row_global = q_start + q_row;
        const index_t q_row_safe =
            q_row_global < kargs.n_q ? q_row_global : kargs.n_q - 1;
        const int32_t q_lane_off =
            q_head_base + static_cast<int32_t>(q_row_safe) * kargs.q.row_stride;
        AVecType q_frag[kDTiles];
        ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
            q_frag[d_tile.value] = *reinterpret_cast<const ck_tile::bf16x16_t*>(
                q_ptr + q_lane_off + d_tile.value * 16);
        });

        CVecType output[kDTiles] = {};
        OnlineLse lse;
        // The lazy-base state: the running exponential base and the running row sum, both per
        // lane, both in the `q_row` ownership the rescale fan-out already uses.
        float lazy_base = -__builtin_inff();
        float lazy_sum  = 0.0f;

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

            CVecType scores[kNTiles];
            ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                const index_t k_off =
                    kcur + k_lane_bytes +
                    (kSwizzled ? (((2 * d_tile.value) ^ lane_row) * kKPack *
                                  sizeof(bf16_t))
                               : d_tile.value * (2 * kKPack * sizeof(bf16_t)));
                const AVecType& q = q_frag[d_tile.value];
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t base = k_off + n_tile.value * kKStep;
                    const ck_tile::bf16x8_t k_lo =
                        *reinterpret_cast<const ck_tile::bf16x8_t*>(lds + base);
                    const ck_tile::bf16x8_t k_hi =
                        *reinterpret_cast<const ck_tile::bf16x8_t*>(
                            lds + base + k_hi_delta);
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

            // online softmax in the log2 domain
            const float scale = kargs.softmax_scale_log2;
            // Four accumulators instead of one 16-link chain: the row maximum and the row sum
            // are both serial dependencies that the WMMA schedule cannot hide past a few links.
            float max_acc[4] = {-__builtin_inff(), -__builtin_inff(), -__builtin_inff(),
                                -__builtin_inff()};
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    max_acc[(2 * n_tile.value + i.value / 4) & 3] =
                        __builtin_fmaxf(max_acc[(2 * n_tile.value + i.value / 4) & 3],
                                        scores[n_tile.value][i.value]);
                });
            });
            float row_max = __builtin_fmaxf(
                __builtin_fmaxf(max_acc[0], max_acc[1]),
                __builtin_fmaxf(max_acc[2], max_acc[3]));
            row_max = __builtin_fmaxf(row_max, SwapLaneGroup(row_max));

            const float scaled_max = row_max * scale;

            // Both paths produce the same three things for this tile: the exponent bias the
            // probability tile is computed against, a factor that rescales the accumulator
            // that is already there, and a factor applied to the probability tile itself.
            float bias    = -scaled_max;
            float rescale = 1.0f;
            float p_scale = 1.0f;
            if constexpr(kLazy)
            {
                const float target = __builtin_fmaxf(lazy_base, scaled_max - kTheta);
                rescale            = Exp2(lazy_base - target);
                bias               = -target;
                // The rescale fans a per-row factor out with `readlane`, so it must run
                // wave-uniformly: one row growing past its base makes every row rescale, and
                // the rows that did not grow pass a factor of exactly 1.
                if(__any_sync(0xffffffffffffffffull, target > lazy_base))
                    RescaleOutput<kDTiles>(output, rescale, lane_group);
                lazy_base = target;
            }

            float sum_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    const float p = Exp2(__builtin_fmaf(
                        scores[n_tile.value][i.value], scale, bias));
                    scores[n_tile.value][i.value] = p;
                    sum_acc[(2 * n_tile.value + i.value / 4) & 3] += p;
                });
            });
            float local_sum = (sum_acc[0] + sum_acc[1]) + (sum_acc[2] + sum_acc[3]);

            if(tail_tile)
            {
                // Out-of-range K and V rows are zeroed while staging, so their
                // scores are exactly 0 and their probability is exp2(bias).
                index_t invalid = 0;
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t remaining =
                        kargs.n_kv - key_start - n_tile.value * 16 - 8 * lane_group;
                    invalid += remaining >= 8 ? 0 : (remaining > 0 ? 8 - remaining : 8);
                });
                local_sum -= static_cast<float>(invalid) * Exp2(bias);
            }

            const float tile_sum = local_sum + SwapLaneGroup(local_sum);
            if constexpr(kLazy)
            {
                lazy_sum = lazy_sum * rescale + tile_sum;
            }
            else
            {
                float alpha = 0.0f;
                float beta  = 0.0f;
                lse.Update(scaled_max, tile_sum, alpha, beta);
                // The probability tile is rounded to bf16 by truncation, whose mean error is
                // half an ulp. Folding (1 + 2^-9) into beta cancels it to first order. The lazy
                // path needs no correction: the same relative error sits in the numerator and
                // the denominator of the final division, so it cancels there instead.
                p_scale = beta * 1.001953125f;
                RescaleOutput<kDTiles>(output, alpha, lane_group);
            }

            // P scaled into the A-fragment layout. `p_scale` is 1 on the lazy path, so the
            // multiply folds away.
            AVecType p_frag[kNTiles];
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::uint32x8_t words;
                uint32_t own[4];
                ck_tile::static_for<0, 4, 1>{}([&](auto k) {
                    own[k.value] = PackBf16Pair(
                        scores[n_tile.value][2 * k.value] * p_scale,
                        scores[n_tile.value][2 * k.value + 1] * p_scale);
                });
                ck_tile::static_for<0, 4, 1>{}([&](auto k) {
                    const uint32_t partner = SwapLaneGroupBits(own[k.value]);
                    words[k.value]     = lane_group == 0 ? own[k.value] : partner;
                    words[k.value + 4] = lane_group == 0 ? partner : own[k.value];
                });
                p_frag[n_tile.value] =
                    ck_tile::bit_cast<AVecType>(words);
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
                    BVecType v;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        v[j.value]     = v_lo[j.value];
                        v[j.value + 8] = v_hi[j.value];
                    });
                    Wmma::InPlace(output[d_tile.value], p_frag[n_tile.value], v);
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

        // The lazy path holds the unnormalised sum of p*v, so the division by the row sum
        // happens here, once, with the same per-row lane fan-out the in-loop rescale uses.
        if constexpr(kLazy)
        {
            const int inv_bits = ck_tile::bit_cast<int>(__builtin_amdgcn_rcpf(lazy_sum));
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const int even_inv = __builtin_amdgcn_readlane(inv_bits, i.value * 2);
                const int odd_inv  = __builtin_amdgcn_readlane(inv_bits, i.value * 2 + 1);
                const float row_inv =
                    ck_tile::bit_cast<float>(lane_group == 0 ? even_inv : odd_inv);
                ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                    output[d_tile.value][i.value] *= row_inv;
                });
            });
        }

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

        // The scores are computed as `K @ Q^T`, so a lane owns one query row: the column of
        // its score fragment is `lane_row`, and the two lane groups hold the same row's log
        // sum because the row max and row sum are merged across them. One store per lane, from
        // the low group so the two groups do not write the same address.
        if(kargs.lse.ptr != nullptr && lane_group == 0)
        {
            const index_t row = q_start + q_row;
            if(row < kargs.n_q)
            {
                float* const lse_ptr =
                    reinterpret_cast<float*>(const_cast<void*>(kargs.lse.ptr));
                const int32_t lse_head_base =
                    at.batch * kargs.lse.batch_stride + at.head * kargs.lse.head_stride;
                // `__builtin_amdgcn_logf` is log2 on this target, so the lazy path adds the
                // running base in the log2 domain and converts once, exactly like OnlineLse.
                const float lse_bits = kLazy ? (lazy_base + __builtin_amdgcn_logf(lazy_sum))
                                             : lse.value;
                lse_ptr[lse_head_base +
                        static_cast<int32_t>(row) * kargs.lse.row_stride] =
                    lse_bits * 0.6931471805599453f;
            }
        }
    }
};

template <bool kHeadFast, index_t kBr, index_t kBc, index_t kBuffers, bool kSwizzled, bool kLazy = false>
bool LaunchBf16FwdD128(const AttentionFwdParams& params)
{
    return LaunchAttentionFwd<Bf16FwdD128Kernel<kHeadFast, kBr, kBc, kBuffers, kSwizzled, kLazy>>(params);
}

} // namespace feather_attn
