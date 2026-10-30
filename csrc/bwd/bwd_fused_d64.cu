#include <hip/hip_runtime.h>

#include "launch_params.h"

#include <cstdint>
#include <type_traits>

namespace feather_attn {

namespace {

constexpr int kRowsBlock = 256;
constexpr int kThreads = 128;
constexpr int kKVThreads = 256;
constexpr int kQThreads = 256;
constexpr int kWaveSize = 32;
constexpr int kOwnerBlock = 64;
constexpr int kKVOwnerBlock = 128;
constexpr int kQOwnerBlock = 128;
constexpr int kInnerBlock = 16;
constexpr int kHeadDim = 64;
constexpr int kPacked = 8;
constexpr int kTransposeStride = 20;
constexpr float kLog2E = 0x1.715476p+0f;

template<typename Element, int N>
using Vec = Element __attribute__((ext_vector_type(N)));

using Float8 = float __attribute__((ext_vector_type(8)));
using UInt8 = uint32_t __attribute__((ext_vector_type(8)));

// gfx1151 lacks a native f32-to-bf16 convert/pack instruction. These helpers
// apply round-to-nearest-even in integer arithmetic and use v_perm_b32 to pack
// two bf16 values for a WMMA A fragment.
template<typename Element>
__device__ inline Element FromFloat(float value)
{
    if constexpr(std::is_same_v<Element, _Float16>)
    {
        return static_cast<Element>(value);
    }
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __builtin_bit_cast(
        Element, static_cast<uint16_t>(bits >> 16));
}

template<typename Element>
__device__ inline uint32_t PackPair(float x, float y)
{
    if constexpr(std::is_same_v<Element, _Float16>)
    {
        Vec<Element, 2> pair;
        pair[0] = static_cast<Element>(x);
        pair[1] = static_cast<Element>(y);
        return __builtin_bit_cast(uint32_t, pair);
    }
    uint32_t x_bits = __builtin_bit_cast(uint32_t, x);
    uint32_t y_bits = __builtin_bit_cast(uint32_t, y);
    x_bits += 0x7fffu + ((x_bits >> 16) & 1u);
    y_bits += 0x7fffu + ((y_bits >> 16) & 1u);
    return __builtin_amdgcn_perm(x_bits, y_bits, 0x03020706u);
}

__device__ inline int KVLdsOffset(int row, int chunk)
{
    const int physical_chunk = chunk ^ (row % (kHeadDim / kPacked));
    return row * kHeadDim + physical_chunk * kPacked;
}

template<typename Element>
__device__ inline Vec<Element, 16> LoadKVRow16(
    const Element* ptr,
    int row,
    int d_tile)
{
    const int chunk = d_tile * 2;
    const Vec<Element, 8> lo = *reinterpret_cast<const Vec<Element, 8>*>(
        ptr + KVLdsOffset(row, chunk));
    const Vec<Element, 8> hi = *reinterpret_cast<const Vec<Element, 8>*>(
        ptr + KVLdsOffset(row, chunk + 1));
    Vec<Element, 16> result;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        result[i] = lo[i];
        result[i + 8] = hi[i];
    }
    return result;
}

template<typename Element>
__device__ inline Vec<Element, 16> LoadTransposeRow16(
    const Element* ptr,
    int row)
{
    Vec<Element, 16> result;
#pragma unroll
    for(int chunk = 0; chunk < kInnerBlock / 4; ++chunk)
    {
        const Vec<Element, 4> value = *reinterpret_cast<const Vec<Element, 4>*>(
            ptr + row * kTransposeStride + chunk * 4);
#pragma unroll
        for(int i = 0; i < 4; ++i)
            result[chunk * 4 + i] = value[i];
    }
    return result;
}

template<typename Element>
__device__ inline void WmmaInPlace(
    Float8& c,
    const Vec<Element, 16>& a,
    const Vec<Element, 16>& b)
{
    if constexpr(std::is_same_v<Element, __bf16>)
    {
        asm volatile(
            "v_wmma_f32_16x16x16_bf16 %0, %1, %2, %0"
            : "+v"(c)
            : "v"(a), "v"(b));
    }
    else
    {
        asm volatile(
            "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0"
            : "+v"(c)
            : "v"(a), "v"(b));
    }
}

template<typename Element>
__device__ inline Vec<Element, 16> CFragmentToA(
    const Float8& fragment,
    int lane)
{
    UInt8 words;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        const uint32_t value =
            PackPair<Element>(fragment[i * 2], fragment[i * 2 + 1]);
        const uint32_t peer = __builtin_amdgcn_permlanex16(
            0u, value, 0x76543210u, 0xfedcba98u, false, true);
        const uint32_t selector_0 =
            lane < 16 ? 0x05040100u : 0x01000504u;
        const uint32_t selector_1 =
            lane < 16 ? 0x07060302u : 0x03020706u;
        words[i * 2] = __builtin_amdgcn_perm(
            peer, value, selector_0);
        words[i * 2 + 1] = __builtin_amdgcn_perm(
            peer, value, selector_1);
    }
    return __builtin_bit_cast(Vec<Element, 16>, words);
}

template<bool kNhd>
__device__ inline int HeadOffset(
    int head_linear,
    int rows,
    int num_heads)
{
    if constexpr(kNhd)
    {
        const int batch = head_linear / num_heads;
        const int head = head_linear - batch * num_heads;
        return (batch * rows * num_heads + head) * kHeadDim;
    }
    return head_linear * rows * kHeadDim;
}

template<typename Element, bool kNhd>
__global__ void DeltaKernel(
    const Element* __restrict__ out,
    const Element* __restrict__ dout,
    float* __restrict__ delta,
    int rows,
    int n_q,
    int num_heads)
{
    const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;
    int delta_row = linear;
    if constexpr(kNhd)
    {
        const int batch = linear / (n_q * num_heads);
        const int batch_row = linear - batch * n_q * num_heads;
        const int q_row = batch_row / num_heads;
        const int head = batch_row - q_row * num_heads;
        delta_row = (batch * num_heads + head) * n_q + q_row;
    }
    const int offset = linear * kHeadDim;
    const Element* out_row = out + offset;
    const Element* dout_row = dout + offset;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < kHeadDim; ++d)
        value += static_cast<float>(out_row[d]) *
                 static_cast<float>(dout_row[d]);
    delta[delta_row] = value;
}

template<typename Element, int kPhase, bool kNhd, bool kAlignedQ, bool kAlignedKV>
__global__ __launch_bounds__(
    kPhase == 1 && !kNhd
        ? kKVThreads
        : (kPhase == 2 ? kQThreads : kThreads)) void
SevenGemmD64Kernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ out,
    const Element* __restrict__ dout,
    const float* __restrict__ lse,
    float* __restrict__ delta,
    Element* __restrict__ dq,
    Element* __restrict__ dk,
    Element* __restrict__ dv,
    int n_q,
    int n_kv,
    int num_heads,
    int head_count,
    int nhd_group_count,
    float scale)
{
    using Vec8 = Vec<Element, 8>;
    using Vec16 = Vec<Element, 16>;
    constexpr bool wide_kv = kPhase == 1 && !kNhd;
    constexpr int phase_threads =
        wide_kv ? kKVThreads : (kPhase == 2 ? kQThreads : kThreads);
    constexpr int kv_owner_block =
        wide_kv ? kKVOwnerBlock : kOwnerBlock;
    constexpr int q_owner_block =
        kPhase == 2 ? kQOwnerBlock : kOwnerBlock;
    constexpr int kv_bytes =
        kv_owner_block * kHeadDim * sizeof(Element);
    constexpr int transpose_bytes =
        2 * kTransposeStride * kHeadDim * sizeof(Element);
    constexpr int q_kv_tile_bytes =
        2 * kInnerBlock * kHeadDim * sizeof(Element);
    constexpr int do_row_stride = 72;
    constexpr int do_row_bytes =
        kInnerBlock * do_row_stride * sizeof(Element);
    constexpr int stage_buffers = wide_kv ? 2 : 1;
    constexpr int q_stage_buffers = kPhase == 2 ? 2 : 1;
    constexpr int transpose_buffers =
        wide_kv ? stage_buffers : q_stage_buffers;
    constexpr int q_kv_storage =
        kPhase == 2 ? 2 * q_stage_buffers * q_kv_tile_bytes : kv_bytes;
    constexpr int lds_bytes =
        q_kv_storage + transpose_buffers * transpose_bytes +
        (kPhase == 1 ? stage_buffers * do_row_bytes : 0);
    static_assert(kv_bytes == (wide_kv ? 16384 : 8192));
    static_assert(q_kv_tile_bytes == 4096);
    static_assert(do_row_bytes == 2304);
    static_assert(transpose_bytes == 5120);
    static_assert(
        lds_bytes ==
        (kPhase == 2 ? 26624
                     : (kPhase == 1 ? (wide_kv ? 31232 : 15616) : 13312)));

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* kv_v_lds = reinterpret_cast<Element*>(lds);
    auto* q_k_lds = reinterpret_cast<Element*>(lds);
    auto* q_v_lds = reinterpret_cast<Element*>(
        lds + (kPhase == 2 ? 2 * q_kv_tile_bytes : q_kv_tile_bytes));
    auto* transpose_lds = lds + q_kv_storage;
    auto* k_transpose_lds = reinterpret_cast<Element*>(transpose_lds);
    auto* q_transpose_lds = reinterpret_cast<Element*>(transpose_lds);
    auto* do_transpose_lds = q_transpose_lds + kTransposeStride * kHeadDim;
    auto* do_row_lds =
        reinterpret_cast<Element*>(transpose_lds + transpose_buffers * transpose_bytes);
    static_assert(
        transpose_buffers * transpose_bytes +
            (kPhase == 1 ? stage_buffers * do_row_bytes : 0) <=
        lds_bytes - q_kv_storage);

    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_row = lane % kInnerBlock;
    const int lane_group = lane / kInnerBlock;
    const int q_tiles = (n_q + q_owner_block - 1) / q_owner_block;
    const int kv_tiles = (n_kv + kv_owner_block - 1) / kv_owner_block;
    const int owner_tiles =
        kPhase == 1 ? kv_tiles
                    : (kPhase == 2 ? q_tiles
                                   : (q_tiles > kv_tiles ? q_tiles : kv_tiles));
    int owner_tile;
    int head_linear;
    if constexpr(kNhd)
    {
        if(nhd_group_count > 1 && num_heads % 16 == 0)
        {
            const int head_slot = static_cast<int>(blockIdx.x) % head_count;
            const int batch = head_slot / num_heads;
            const int slot = head_slot - batch * num_heads;
            const int small_group_size = num_heads / nhd_group_count;
            const int large_groups = num_heads % nhd_group_count;
            const int large_group_size = small_group_size + 1;
            const int large_span = large_groups * large_group_size;
            int group;
            int local_head;
            if(slot < large_span)
            {
                group = slot / large_group_size;
                local_head = slot - group * large_group_size;
            }
            else
            {
                const int small_slot = slot - large_span;
                group = large_groups + small_slot / small_group_size;
                local_head = small_slot % small_group_size;
            }
            head_linear = batch * num_heads + group +
                          local_head * nhd_group_count;
            owner_tile = static_cast<int>(blockIdx.x) / head_count;
        }
        else if(num_heads % 16 == 0)
        {
            head_linear = static_cast<int>(blockIdx.x) % head_count;
            owner_tile = static_cast<int>(blockIdx.x) / head_count;
        }
        else
        {
            owner_tile = static_cast<int>(blockIdx.x) % owner_tiles;
            head_linear = static_cast<int>(blockIdx.x) / owner_tiles;
        }
    }
    else
    {
        owner_tile = static_cast<int>(blockIdx.x) % owner_tiles;
        head_linear = static_cast<int>(blockIdx.x) / owner_tiles;
    }
    const int q_offset = HeadOffset<kNhd>(head_linear, n_q, num_heads);
    const int kv_offset = HeadOffset<kNhd>(head_linear, n_kv, num_heads);
    const int q_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const int kv_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;

    if(kPhase != 2 && owner_tile < kv_tiles)
    {
        const int kv_start = owner_tile * kv_owner_block;
        const float scale_log2 = scale * kLog2E;
#pragma unroll
        for(int issue = 0;
            issue < kv_owner_block * kHeadDim / kPacked / phase_threads;
            ++issue)
        {
            const int linear_chunk = tid + issue * phase_threads;
            const int row = linear_chunk / (kHeadDim / kPacked);
            const int chunk = linear_chunk % (kHeadDim / kPacked);
            Vec8 v_value = {};
            if(kAlignedKV || kv_start + row < n_kv)
            {
                const int global_offset =
                    kv_offset + (kv_start + row) * kv_row_stride +
                    chunk * kPacked;
                v_value = *reinterpret_cast<const Vec8*>(v + global_offset);
            }
            *reinterpret_cast<Vec8*>(kv_v_lds + KVLdsOffset(row, chunk)) =
                v_value;
        }
        __syncthreads();

        Float8 dk_accum[4] = {};
        Float8 dv_accum[4] = {};
        Vec16 k_rows[kHeadDim / kInnerBlock];
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            const int kv_row = wave * kInnerBlock + lane_row;
            k_rows[d_tile] = {};
            if(kAlignedKV || kv_start + kv_row < n_kv)
            {
                const int global_offset =
                    kv_offset + (kv_start + kv_row) * kv_row_stride +
                    d_tile * kInnerBlock;
                k_rows[d_tile] = *reinterpret_cast<const Vec16*>(
                    k + global_offset);
            }
        }

        for(int q_start = 0; q_start < n_q; q_start += kInnerBlock)
        {
            if constexpr(!wide_kv)
                __syncthreads();
            const int stage_phase =
                wide_kv ? ((q_start / kInnerBlock) & 1) : 0;
            Element* const q_stage_lds =
                q_transpose_lds +
                stage_phase * 2 * kTransposeStride * kHeadDim;
            Element* const do_stage_lds =
                do_transpose_lds +
                stage_phase * 2 * kTransposeStride * kHeadDim;
            Element* const do_row_stage_lds =
                do_row_lds + stage_phase * do_row_stride * kInnerBlock;
            if(tid < kInnerBlock * kHeadDim / kPacked)
            {
                const int stage_row = tid / (kHeadDim / kPacked);
                const int stage_chunk = tid % (kHeadDim / kPacked);
                Vec8 q_stage = {};
                Vec8 do_stage = {};
                if(kAlignedQ || q_start + stage_row < n_q)
                {
                    const int global_offset =
                        q_offset + (q_start + stage_row) * q_row_stride +
                        stage_chunk * kPacked;
                    q_stage = *reinterpret_cast<const Vec8*>(q + global_offset);
                    do_stage = *reinterpret_cast<const Vec8*>(
                        dout + global_offset);
                }
#pragma unroll
                for(int i = 0; i < kPacked; ++i)
                {
                    const int d = stage_chunk * kPacked + i;
                    q_stage_lds[d * kTransposeStride + stage_row] =
                        q_stage[i];
                    do_stage_lds[d * kTransposeStride + stage_row] =
                        do_stage[i];
                }
                if constexpr(kPhase == 1)
                {
                    *reinterpret_cast<Vec8*>(
                        do_row_stage_lds + stage_row * do_row_stride +
                        stage_chunk * kPacked) = do_stage;
                }
            }
            __syncthreads();

            Float8 score = {};
#pragma unroll
            for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
            {
                Vec16 q_row = {};
                if(kAlignedQ || q_start + lane_row < n_q)
                {
                    const int offset =
                        q_offset + (q_start + lane_row) * q_row_stride +
                        d_tile * 16;
                    q_row = *reinterpret_cast<const Vec16*>(q + offset);
                }
                WmmaInPlace(score, q_row, k_rows[d_tile]);
            }

            const bool lane_q_valid = kAlignedQ || q_start + lane_row < n_q;
            const float lane_lse = lane_q_valid
                                       ? lse[head_linear * n_q + q_start + lane_row]
                                       : 0.0f;
            const int lane_lse_bits =
                __builtin_bit_cast(int, lane_lse * kLog2E);
            const int kv_row = wave * kInnerBlock + lane_row;
            const bool kv_valid = kAlignedKV || kv_start + kv_row < n_kv;
            Float8 probability;
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int q_column = i * 2 + lane_group;
                const int even_lse_bits =
                    __builtin_amdgcn_readlane(lane_lse_bits, i * 2);
                const int odd_lse_bits =
                    __builtin_amdgcn_readlane(lane_lse_bits, i * 2 + 1);
                const float column_lse_log2 = __builtin_bit_cast(
                    float, lane_group == 0 ? even_lse_bits : odd_lse_bits);
                const bool valid =
                    kv_valid && (kAlignedQ || q_start + q_column < n_q);
                probability[i] =
                    valid
                        ? __builtin_amdgcn_exp2f(
                              score[i] * scale_log2 - column_lse_log2)
                        : 0.0f;
            }
            const Vec16 p_fragment = CFragmentToA<Element>(probability, lane);

#pragma unroll
            for(int d_pair = 0;
                d_pair < kHeadDim / (2 * kInnerBlock);
                ++d_pair)
            {
                const int d0 = d_pair * 2 * kInnerBlock + lane_row;
                const int d1 = d0 + kInnerBlock;
                const Vec16 do_column0 =
                    LoadTransposeRow16(do_stage_lds, d0);
                const Vec16 do_column1 =
                    LoadTransposeRow16(do_stage_lds, d1);
                WmmaInPlace(dv_accum[d_pair * 2], p_fragment, do_column0);
                WmmaInPlace(dv_accum[d_pair * 2 + 1], p_fragment, do_column1);
            }

            Float8 d_probability = {};
#pragma unroll
            for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
            {
                Vec16 do_row = {};
                if constexpr(kPhase == 1)
                {
                    do_row = *reinterpret_cast<const Vec16*>(
                        do_row_stage_lds + lane_row * do_row_stride +
                        d_tile * kInnerBlock);
                }
                else if(kAlignedQ || q_start + lane_row < n_q)
                {
                    const int offset =
                        q_offset + (q_start + lane_row) * q_row_stride +
                        d_tile * 16;
                    do_row = *reinterpret_cast<const Vec16*>(dout + offset);
                }
                WmmaInPlace(
                    d_probability,
                    do_row,
                    LoadKVRow16(kv_v_lds, kv_row, d_tile));
            }

            const float lane_delta = lane_q_valid
                                         ? delta[head_linear * n_q + q_start + lane_row]
                                         : 0.0f;
            const int lane_delta_bits = __builtin_bit_cast(int, lane_delta);
            Float8 d_score;
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int q_column = i * 2 + lane_group;
                const int even_delta_bits =
                    __builtin_amdgcn_readlane(lane_delta_bits, i * 2);
                const int odd_delta_bits =
                    __builtin_amdgcn_readlane(lane_delta_bits, i * 2 + 1);
                const float column_delta = __builtin_bit_cast(
                    float, lane_group == 0 ? even_delta_bits : odd_delta_bits);
                const bool valid =
                    kv_valid && (kAlignedQ || q_start + q_column < n_q);
                const float p = probability[i];
                d_score[i] =
                    valid ? p * (d_probability[i] - column_delta) : 0.0f;
            }
            const Vec16 ds_fragment = CFragmentToA<Element>(d_score, lane);

#pragma unroll
            for(int d_pair = 0;
                d_pair < kHeadDim / (2 * kInnerBlock);
                ++d_pair)
            {
                const int d0 = d_pair * 2 * kInnerBlock + lane_row;
                const int d1 = d0 + kInnerBlock;
                const Vec16 q_column0 =
                    LoadTransposeRow16(q_stage_lds, d0);
                const Vec16 q_column1 =
                    LoadTransposeRow16(q_stage_lds, d1);
                WmmaInPlace(dk_accum[d_pair * 2], ds_fragment, q_column0);
                WmmaInPlace(dk_accum[d_pair * 2 + 1], ds_fragment, q_column1);
            }
            __builtin_amdgcn_wave_barrier();
        }

#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = wave * kInnerBlock + i * 2 + lane_group;
                if(kAlignedKV || kv_start + row < n_kv)
                {
                    const int d = d_tile * kInnerBlock + lane_row;
                    const int offset =
                        kv_offset + (kv_start + row) * kv_row_stride + d;
                    dk[offset] = FromFloat<Element>(dk_accum[d_tile][i] * scale);
                    dv[offset] = FromFloat<Element>(dv_accum[d_tile][i]);
                }
            }
        }
    }

    if(kPhase != 1 && owner_tile < q_tiles)
    {
        const int q_block_start = owner_tile * q_owner_block;
        const int q_start = q_block_start + wave * kInnerBlock;
        const float scale_log2 = scale * kLog2E;
        Float8 dq_accum[4] = {};
        Vec16 q_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            if(kAlignedQ || q_start + lane_row < n_q)
            {
                const int offset =
                    q_offset + (q_start + lane_row) * q_row_stride +
                    d_tile * kInnerBlock;
                q_rows[d_tile] = *reinterpret_cast<const Vec16*>(q + offset);
            }
        }
        Vec16 do_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            if(kAlignedQ || q_start + lane_row < n_q)
            {
                const int offset =
                    q_offset + (q_start + lane_row) * q_row_stride +
                    d_tile * kInnerBlock;
                do_rows[d_tile] = *reinterpret_cast<const Vec16*>(
                    dout + offset);
            }
        }
        const bool q_valid = kAlignedQ || q_start + lane_row < n_q;
        const float row_lse_log2 =
            q_valid
                ? lse[head_linear * n_q + q_start + lane_row] * kLog2E
                : 0.0f;
        float row_delta = 0.0f;
        if constexpr(kPhase == 2)
        {
            if(q_valid)
            {
                float delta_value = 0.0f;
#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const int offset =
                        q_offset + (q_start + lane_row) * q_row_stride +
                        d_tile * kInnerBlock;
                    const Vec16 out_row =
                        *reinterpret_cast<const Vec16*>(out + offset);
#pragma unroll
                    for(int i = 0; i < kInnerBlock; ++i)
                    {
                        delta_value += static_cast<float>(out_row[i]) *
                                       static_cast<float>(do_rows[d_tile][i]);
                    }
                }
                delta[head_linear * n_q + q_start + lane_row] =
                    delta_value;
                row_delta = delta_value;
            }
        }
        else
        {
            row_delta = q_valid
                            ? delta[head_linear * n_q + q_start + lane_row]
                            : 0.0f;
        }

        for(int kv_start = 0; kv_start < n_kv;
            kv_start += 2 * kInnerBlock)
        {
            if constexpr(kPhase != 2)
                __syncthreads();
            const int q_phase = kPhase == 2
                                    ? ((kv_start / (2 * kInnerBlock)) & 1)
                                    : 0;
            Element* const q_k_stage_lds =
                q_k_lds + q_phase * (q_kv_tile_bytes / 2);
            Element* const q_v_stage_lds =
                q_v_lds + q_phase * (q_kv_tile_bytes / 2);
            Element* const k_transpose_stage_lds =
                k_transpose_lds + q_phase * 2 * kHeadDim * kTransposeStride;
#pragma unroll
            for(int issue = 0;
                issue < 2 * kInnerBlock * kHeadDim / kPacked / phase_threads;
                ++issue)
            {
                const int linear_chunk = tid + issue * phase_threads;
                const int stage_row = linear_chunk / (kHeadDim / kPacked);
                const int stage_chunk = linear_chunk % (kHeadDim / kPacked);
                Vec8 k_stage = {};
                Vec8 v_stage = {};
                if(kAlignedKV || kv_start + stage_row < n_kv)
                {
                    const int global_offset =
                        kv_offset + (kv_start + stage_row) * kv_row_stride +
                        stage_chunk * kPacked;
                    k_stage = *reinterpret_cast<const Vec8*>(k + global_offset);
                    v_stage = *reinterpret_cast<const Vec8*>(v + global_offset);
                }
                *reinterpret_cast<Vec8*>(
                    q_k_stage_lds + KVLdsOffset(stage_row, stage_chunk)) = k_stage;
                *reinterpret_cast<Vec8*>(
                    q_v_stage_lds + KVLdsOffset(stage_row, stage_chunk)) = v_stage;
#pragma unroll
                for(int i = 0; i < kPacked; ++i)
                {
                    const int d = stage_chunk * kPacked + i;
                    const int tile = stage_row / kInnerBlock;
                    const int tile_row = stage_row % kInnerBlock;
                    k_transpose_stage_lds[
                        tile * kHeadDim * kTransposeStride +
                        d * kTransposeStride + tile_row] = k_stage[i];
                }
            }
            __syncthreads();

#pragma unroll
            for(int tile = 0; tile < 2; ++tile)
            {
                const int tile_kv_start = kv_start + tile * kInnerBlock;
                const int lds_row = tile * kInnerBlock + lane_row;
                Float8 score = {};
#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const Vec16 k_row =
                        LoadKVRow16(q_k_stage_lds, lds_row, d_tile);
                    WmmaInPlace(score, k_row, q_rows[d_tile]);
                }

                Float8 probability;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int kv_column = i * 2 + lane_group;
                    const bool valid =
                        q_valid && (kAlignedKV || tile_kv_start + kv_column < n_kv);
                    probability[i] =
                        valid
                            ? __builtin_amdgcn_exp2f(
                                  score[i] * scale_log2 - row_lse_log2)
                            : 0.0f;
                }
                Float8 d_probability = {};
#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const Vec16 v_row =
                        LoadKVRow16(q_v_stage_lds, lds_row, d_tile);
                    WmmaInPlace(d_probability, v_row, do_rows[d_tile]);
                }

                Float8 d_score;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int kv_column = i * 2 + lane_group;
                    const bool valid =
                        q_valid && (kAlignedKV || tile_kv_start + kv_column < n_kv);
                    const float p = probability[i];
                    d_score[i] = valid
                                     ? p * (d_probability[i] - row_delta)
                                     : 0.0f;
                }
                const Vec16 ds_fragment = CFragmentToA<Element>(d_score, lane);

#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const int d = d_tile * kInnerBlock + lane_row;
                    const Vec16 k_column = LoadTransposeRow16(
                        k_transpose_stage_lds +
                            tile * kHeadDim * kTransposeStride,
                        d);
                    WmmaInPlace(dq_accum[d_tile], ds_fragment, k_column);
                }
            }
        }

#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = i * 2 + lane_group;
                if(kAlignedQ || q_start + row < n_q)
                {
                    const int d = d_tile * kInnerBlock + lane_row;
                    dq[q_offset + (q_start + row) * q_row_stride + d] =
                        FromFloat<Element>(dq_accum[d_tile][i] * scale);
                }
            }
        }
    }
}

template<bool kNhd, typename Element>
bool LaunchSevenGemmBackward(const BackwardLaunchParams& params)
{
    const auto* q = reinterpret_cast<const Element*>(params.q_ptr);
    const auto* k = reinterpret_cast<const Element*>(params.k_ptr);
    const auto* v = reinterpret_cast<const Element*>(params.v_ptr);
    const auto* out = reinterpret_cast<const Element*>(params.out_ptr);
    const auto* lse = reinterpret_cast<const float*>(params.lse_ptr);
    const auto* dout = reinterpret_cast<const Element*>(params.dout_ptr);
    auto* dq = reinterpret_cast<Element*>(params.dq_ptr);
    auto* dk = reinterpret_cast<Element*>(params.dk_ptr);
    auto* dv = reinterpret_cast<Element*>(params.dv_ptr);
    auto* delta = reinterpret_cast<float*>(params.delta_ptr);
    const int32_t q_rows = params.head_count * params.n_q;
    const int32_t q_tiles =
        (params.n_q + kOwnerBlock - 1) / kOwnerBlock;
    const int32_t q_phase_tiles =
        (params.n_q + kQOwnerBlock - 1) / kQOwnerBlock;
    const int32_t kv_tiles =
        (params.n_kv + kOwnerBlock - 1) / kOwnerBlock;
    const int32_t kv_phase_tiles =
        (params.n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;
    const int32_t owner_tiles = q_tiles > kv_tiles ? q_tiles : kv_tiles;

    constexpr int long_kv_owner = kNhd ? kOwnerBlock : kKVOwnerBlock;
    const bool aligned_q = params.n_q % kInnerBlock == 0;
    const bool aligned_q_owner = params.n_q % kQOwnerBlock == 0;
    const bool aligned_kv_short = params.n_kv % kOwnerBlock == 0;
    const bool aligned_kv_long = params.n_kv % long_kv_owner == 0;
    const bool aligned_kv_q = params.n_kv % (2 * kInnerBlock) == 0;
    auto launch = [&]<int kPhase, bool AQ, bool AKV>(
                      const dim3& grid, const dim3& block) {
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<Element, kPhase, kNhd, AQ, AKV>),
            grid,
            block,
            0,
            params.stream,
            q,
            k,
            v,
            out,
            dout,
            lse,
            delta,
            dq,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.num_heads,
            params.head_count,
            params.nhd_group_count,
            params.softmax_scale);
    };

    const dim3 main_block(kThreads);
    if(params.n_q >= 4096 && params.n_kv >= 4096)
    {
        const dim3 q_grid(
            static_cast<uint32_t>(params.head_count * q_phase_tiles));
        const dim3 q_block(kQThreads);
        if(aligned_q_owner && aligned_kv_q)
            launch.template operator()<2, true, true>(q_grid, q_block);
        else if(aligned_q_owner)
            launch.template operator()<2, true, false>(q_grid, q_block);
        else if(aligned_kv_q)
            launch.template operator()<2, false, true>(q_grid, q_block);
        else
            launch.template operator()<2, false, false>(q_grid, q_block);

        const dim3 kv_grid(
            static_cast<uint32_t>(
                params.head_count * (kNhd ? kv_tiles : kv_phase_tiles)));
        const dim3 kv_block(kNhd ? kThreads : kKVThreads);
        if(aligned_q && aligned_kv_long)
            launch.template operator()<1, true, true>(kv_grid, kv_block);
        else if(aligned_q)
            launch.template operator()<1, true, false>(kv_grid, kv_block);
        else if(aligned_kv_long)
            launch.template operator()<1, false, true>(kv_grid, kv_block);
        else
            launch.template operator()<1, false, false>(kv_grid, kv_block);

    }
    else
    {
        const dim3 delta_block(kRowsBlock);
        const dim3 delta_grid(
            (static_cast<uint32_t>(q_rows) + kRowsBlock - 1) /
            kRowsBlock);
        hipLaunchKernelGGL(
            (DeltaKernel<Element, kNhd>),
            delta_grid,
            delta_block,
            0,
            params.stream,
            out,
            dout,
            delta,
            q_rows,
            params.n_q,
            params.num_heads);

        const dim3 main_grid(
            static_cast<uint32_t>(params.head_count * owner_tiles));
        if(aligned_q && aligned_kv_short)
            launch.template operator()<0, true, true>(main_grid, main_block);
        else if(aligned_q)
            launch.template operator()<0, true, false>(main_grid, main_block);
        else if(aligned_kv_short)
            launch.template operator()<0, false, true>(main_grid, main_block);
        else
            launch.template operator()<0, false, false>(main_grid, main_block);
    }
    return hipPeekAtLastError() == hipSuccess;
}

template<typename Element>
bool DispatchSevenGemm(const BackwardLaunchParams& params)
{
    return params.layout == 1 ? LaunchSevenGemmBackward<true, Element>(params)
                              : LaunchSevenGemmBackward<false, Element>(params);
}

} // namespace

extern "C" bool feather_attn_bwd_d64_fused(
    const BackwardLaunchParams& params)
{
    return params.dtype == kBackwardDtypeBFloat16
               ? DispatchSevenGemm<__bf16>(params)
               : DispatchSevenGemm<_Float16>(params);
}

} // namespace feather_attn
