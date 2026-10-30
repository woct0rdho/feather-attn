#include <hip/hip_runtime.h>

#include "launch_params.h"

#include <cstdint>
#include <type_traits>

namespace feather_attn {

namespace {

constexpr int kRowsBlock = 256;
constexpr int kKVThreads = 256;
constexpr int kQThreads = 512;
constexpr int kWaveSize = 32;
constexpr int kInnerBlock = 16;
constexpr int kHeadDim = 128;
constexpr int kPacked = 8;
constexpr int kTransposeStride = 20;
constexpr int kQOwnerBlock = 256;
constexpr int kKVOwnerBlock = 64;
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
        return __builtin_bit_cast(
            uint32_t, __builtin_amdgcn_cvt_pkrtz(x, y));
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

__device__ inline void NativeAtomicAdd(float* address, float value)
{
#if defined(__gfx1151__) && \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
    (void)__builtin_amdgcn_global_atomic_fadd_f32(address, value);
#else
    atomicAdd(address, value);
#endif
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
        words[i * 2] = __builtin_amdgcn_perm(peer, value, selector_0);
        words[i * 2 + 1] = __builtin_amdgcn_perm(
            peer, value, selector_1);
    }
    return __builtin_bit_cast(Vec<Element, 16>, words);
}

template<typename Element>
__device__ inline void StoreAFragmentSoA(
    Element* base,
    int group,
    int lane,
    const Vec<Element, 16>& fragment)
{
    Element* destination =
        base + group * kInnerBlock * kWaveSize;
    const Vec<Element, 2>* source =
        reinterpret_cast<const Vec<Element, 2>*>(&fragment);
#pragma unroll
    for(int j = 0; j < kInnerBlock / 2; ++j)
    {
        *reinterpret_cast<Vec<Element, 2>*>(
            destination + (j * kWaveSize + lane) * 2) = source[j];
    }
}

template<typename Element>
__device__ inline Vec<Element, 16> LoadAFragmentSoA(
    const Element* base,
    int group,
    int lane)
{
    const Element* source =
        base + group * kInnerBlock * kWaveSize;
    Vec<Element, 16> fragment;
    Vec<Element, 2>* destination =
        reinterpret_cast<Vec<Element, 2>*>(&fragment);
#pragma unroll
    for(int j = 0; j < kInnerBlock / 2; ++j)
    {
        destination[j] = *reinterpret_cast<const Vec<Element, 2>*>(
            source + (j * kWaveSize + lane) * 2);
    }
    return fragment;
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

template<bool kNhd>
__device__ inline void DecodeOwner(
    int block,
    int owner_tiles,
    int head_count,
    int& owner_tile,
    int& head_linear)
{
    if constexpr(kNhd)
    {
        head_linear = block % head_count;
        owner_tile = block / head_count;
    }
    else
    {
        owner_tile = block % owner_tiles;
        head_linear = block / owner_tiles;
    }
}

template<typename Element, bool kNhd>
__global__ void DeltaD128Kernel(
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
    const Element* out_row = out + linear * kHeadDim;
    const Element* dout_row = dout + linear * kHeadDim;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < kHeadDim; ++d)
        value += static_cast<float>(out_row[d]) *
                 static_cast<float>(dout_row[d]);
    delta[delta_row] = value;
}

template<typename Element, bool kNhd, bool kComputeDq, bool kAlignedQ, bool kAlignedKV>
__global__ __launch_bounds__(kKVThreads) void D128TwoPhaseKVKernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    float* __restrict__ dq_acc,
    Element* __restrict__ dk,
    Element* __restrict__ dv,
    int n_q,
    int n_kv,
    int num_heads,
    int head_count,
    float scale)
{
    using Vec8 = Vec<Element, 8>;
    using Vec16 = Vec<Element, 16>;
    constexpr int transpose_bytes =
        kHeadDim * kTransposeStride * sizeof(Element);
    constexpr int transpose_elements = kHeadDim * kTransposeStride;
    constexpr int transpose_buffers = kComputeDq ? 2 : 1;
    constexpr int exchange_floats = 32 * 8;
    constexpr int exchange_bytes =
        2 * (kKVOwnerBlock / kInnerBlock) * exchange_floats * sizeof(float);
    constexpr int fragment_bytes =
        (kKVOwnerBlock / kInnerBlock) * 32 * kInnerBlock * sizeof(Element);
    constexpr int dst_bytes =
        kComputeDq ? (kKVOwnerBlock / kInnerBlock) * kInnerBlock * 20 * sizeof(float) : 0;
    constexpr int lse_bytes = 2 * kInnerBlock * sizeof(float);
    constexpr int lds_bytes =
        2 * transpose_buffers * transpose_bytes + exchange_bytes +
        2 * fragment_bytes + dst_bytes + lse_bytes;
    static_assert(transpose_bytes == 5120);
    static_assert(exchange_bytes == 8192);
    static_assert(fragment_bytes == 4096);
    static_assert(lse_bytes == 128);
    static_assert(lds_bytes == (kComputeDq ? 42112 : 26752));

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* q_transpose_lds = reinterpret_cast<Element*>(lds);
    auto* do_transpose_lds =
        q_transpose_lds + transpose_buffers * transpose_elements;
    uint8_t* const exchange_ptr =
        lds + 2 * transpose_buffers * transpose_bytes;
    auto* exchange = reinterpret_cast<float*>(exchange_ptr);
    auto* p_a_lds = reinterpret_cast<Element*>(
        exchange_ptr + exchange_bytes);
    auto* ds_a_lds = p_a_lds +
                     (kKVOwnerBlock / kInnerBlock) * 32 * kInnerBlock;
    auto* ds_t_c_lds = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(ds_a_lds) + fragment_bytes);
    auto* lse_lds = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(ds_t_c_lds) + dst_bytes);
    auto* delta_lds = lse_lds + kInnerBlock;

    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_row = lane % kInnerBlock;
    const int lane_group = lane / kInnerBlock;
    const int row_group = wave / 2;
    const bool is_producer = (wave % 2) == 0;
    const int kv_tiles = (n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;
    int owner_tile;
    int head_linear;
    DecodeOwner<kNhd>(
        static_cast<int>(blockIdx.x),
        kv_tiles,
        head_count,
        owner_tile,
        head_linear);
    const int kv_start = owner_tile * kKVOwnerBlock;
    const int kv_row_local = row_group * kInnerBlock + lane_row;
    const int q_offset = HeadOffset<kNhd>(head_linear, n_q, num_heads);
    const int kv_offset = HeadOffset<kNhd>(head_linear, n_kv, num_heads);
    const int q_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const int kv_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const float scale_log2 = scale * kLog2E;
    const bool kv_valid = kAlignedKV || kv_start + kv_row_local < n_kv;

    const Element* const operand_base = is_producer ? k : v;
    Vec16 operand_rows[kHeadDim / kInnerBlock] = {};
    if(kv_valid)
    {
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            const int offset = kv_offset +
                               (kv_start + kv_row_local) * kv_row_stride +
                               d_tile * kInnerBlock;
            operand_rows[d_tile] =
                *reinterpret_cast<const Vec16*>(operand_base + offset);
        }
    }

    Float8 dk_accum[kKVOwnerBlock / kInnerBlock] = {};
    Float8 dv_accum[kKVOwnerBlock / kInnerBlock] = {};
    Vec16 k_b_fragments[kComputeDq ? (kKVOwnerBlock / kInnerBlock) : 1] = {};
    if constexpr(kComputeDq)
    {
        const int d_column = wave * kInnerBlock + lane_row;
#pragma unroll
        for(int r = 0; r < kKVOwnerBlock / kInnerBlock; ++r)
        {
            Vec16 fragment = {};
#pragma unroll
            for(int j = 0; j < kInnerBlock; ++j)
            {
                const int kv_row = kv_start + r * kInnerBlock + j;
                if(kAlignedKV || kv_row < n_kv)
                    fragment[j] = k[kv_offset + kv_row * kv_row_stride + d_column];
            }
            k_b_fragments[r] = fragment;
        }
    }

    for(int q_start = 0; q_start < n_q; q_start += kInnerBlock)
    {
        if constexpr(!kComputeDq)
            __syncthreads();
        const int q_phase =
            kComputeDq ? ((q_start / kInnerBlock) & 1) : 0;
        Element* const q_transpose =
            q_transpose_lds + q_phase * transpose_elements;
        Element* const do_transpose =
            do_transpose_lds + q_phase * transpose_elements;

        Float8 dq_accum = {};
        Float8 own_fragment = {};
        const bool lane_q_valid = kAlignedQ || q_start + lane_row < n_q;
        const int lane_q_row = q_start + (kAlignedQ || lane_q_valid ? lane_row : 0);
        const Element* const q_row_ptr =
            q + q_offset + lane_q_row * q_row_stride;
        const Element* const do_row_ptr =
            dout + q_offset + lane_q_row * q_row_stride;
        if(is_producer)
        {
#pragma unroll
            for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
            {
                Vec16 q_row = {};
                if(lane_q_valid)
                    q_row = *reinterpret_cast<const Vec16*>(
                        q_row_ptr + d_tile * kInnerBlock);
                WmmaInPlace(own_fragment, q_row, operand_rows[d_tile]);
            }
        }
        else
        {
#pragma unroll
            for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
            {
                Vec16 do_row = {};
                if(lane_q_valid)
                    do_row = *reinterpret_cast<const Vec16*>(
                        do_row_ptr + d_tile * kInnerBlock);
                WmmaInPlace(
                    own_fragment,
                    do_row,
                    operand_rows[d_tile]);
            }
        }

        const int own_matrix = is_producer ? 0 : 1;
        float* own_exchange =
            exchange + ((row_group * 2 + own_matrix) * 8) * 32;
#pragma unroll
        for(int i = 0; i < 8; ++i)
            own_exchange[i * 32 + lane] = own_fragment[i];

        {
            const int q_base = (wave / 4) * 8;
            const int d_chunk = (wave % 4) * 4 + (lane % 4);
            const int stage_row = q_base + lane / 4;
            const int stage_chunk = d_chunk;
            Vec8 q_stage = {};
            Vec8 do_stage = {};
            if(kAlignedQ || q_start + stage_row < n_q)
            {
                const int offset = q_offset +
                                   (q_start + stage_row) * q_row_stride +
                                   stage_chunk * kPacked;
                q_stage = *reinterpret_cast<const Vec8*>(q + offset);
                do_stage = *reinterpret_cast<const Vec8*>(dout + offset);
            }
#pragma unroll
            for(int i = 0; i < kPacked; ++i)
            {
                const int d = stage_chunk * kPacked + i;
                q_transpose[d * kTransposeStride + stage_row] = q_stage[i];
                do_transpose[d * kTransposeStride + stage_row] = do_stage[i];
            }
            if(tid < kInnerBlock)
            {
                const int row = tid;
                float lse_value = 0.0f;
                float delta_value = 0.0f;
                if(kAlignedQ || q_start + row < n_q)
                {
                    lse_value = lse[head_linear * n_q + q_start + row];
                    delta_value = delta[head_linear * n_q + q_start + row];
                }
                lse_lds[row] = lse_value;
                delta_lds[row] = delta_value;
            }
        }
        __syncthreads();

        const float* score_exchange =
            exchange + ((row_group * 2 + 0) * 8) * 32;
        const float* dp_exchange =
            exchange + ((row_group * 2 + 1) * 8) * 32;
        Float8 score;
        Float8 d_probability;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            score[i] = score_exchange[i * 32 + lane];
            d_probability[i] = dp_exchange[i * 32 + lane];
        }

        const float lane_lse = lse_lds[lane_row];
        const int lane_lse_bits =
            __builtin_bit_cast(int, lane_lse * kLog2E);
        const float lane_delta = delta_lds[lane_row];
        const int lane_delta_bits =
            __builtin_bit_cast(int, lane_delta);
        Float8 probability;
        Float8 d_score;
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
            const int even_delta_bits =
                __builtin_amdgcn_readlane(lane_delta_bits, i * 2);
            const int odd_delta_bits =
                __builtin_amdgcn_readlane(lane_delta_bits, i * 2 + 1);
            const float column_delta = __builtin_bit_cast(
                float, lane_group == 0 ? even_delta_bits : odd_delta_bits);
            const bool valid =
                kv_valid && (kAlignedQ || q_start + q_column < n_q);
            probability[i] =
                valid
                    ? __builtin_amdgcn_exp2f(
                          score[i] * scale_log2 - column_lse_log2)
                    : 0.0f;
            d_score[i] = valid
                             ? probability[i] *
                                   (d_probability[i] - column_delta)
                             : 0.0f;
        }

        if(is_producer)
        {
            const Vec16 p_fragment = CFragmentToA<Element>(probability, lane);
            StoreAFragmentSoA(p_a_lds, row_group, lane, p_fragment);
        }
        else
        {
            const Vec16 ds_fragment = CFragmentToA<Element>(d_score, lane);
            StoreAFragmentSoA(ds_a_lds, row_group, lane, ds_fragment);
            if constexpr(kComputeDq)
            {
                float* transposed =
                    ds_t_c_lds + row_group * kInnerBlock * 20;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int q_column = i * 2 + lane_group;
                    transposed[q_column * 20 +
                               (lane_row & 1) * 8 + (lane_row >> 1)] =
                        d_score[i];
                }
            }
        }
        __syncthreads();

        const int d_row = wave * kInnerBlock + lane_row;
        const Vec16 q_b_fragment =
            LoadTransposeRow16(q_transpose, d_row);
        const Vec16 do_b_fragment =
            LoadTransposeRow16(do_transpose, d_row);
#pragma unroll
        for(int r = 0; r < kKVOwnerBlock / kInnerBlock; ++r)
        {
            const Vec16 ds_fragment = LoadAFragmentSoA<Element>(ds_a_lds, r, lane);
            const Vec16 p_fragment = LoadAFragmentSoA<Element>(p_a_lds, r, lane);
            WmmaInPlace(dk_accum[r], ds_fragment, q_b_fragment);
            WmmaInPlace(dv_accum[r], p_fragment, do_b_fragment);
            if constexpr(kComputeDq)
            {
                Float8 ds_t_fragment;
                const float* source =
                    ds_t_c_lds + r * kInnerBlock * 20 +
                    lane_row * 20 + lane_group * 8;
                const float4 lo = *reinterpret_cast<const float4*>(source);
                const float4 hi =
                    *reinterpret_cast<const float4*>(source + 4);
                ds_t_fragment[0] = lo.x;
                ds_t_fragment[1] = lo.y;
                ds_t_fragment[2] = lo.z;
                ds_t_fragment[3] = lo.w;
                ds_t_fragment[4] = hi.x;
                ds_t_fragment[5] = hi.y;
                ds_t_fragment[6] = hi.z;
                ds_t_fragment[7] = hi.w;
                const Vec16 ds_t_a = CFragmentToA<Element>(ds_t_fragment, lane);
                WmmaInPlace(dq_accum, ds_t_a, k_b_fragments[r]);
            }
        }
        if constexpr(kComputeDq)
        {
            float* const dq_lane_base =
                dq_acc + q_offset + q_start * q_row_stride +
                wave * kInnerBlock + lane_row;
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int q_row = i * 2 + lane_group;
                if(kAlignedQ || q_start + q_row < n_q)
                {
                    NativeAtomicAdd(
                        dq_lane_base + q_row * q_row_stride,
                        dq_accum[i] * scale);
                }
            }
        }
    }

#pragma unroll
    for(int r = 0; r < kKVOwnerBlock / kInnerBlock; ++r)
    {
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int row = r * kInnerBlock + i * 2 + lane_group;
            if(kAlignedKV || kv_start + row < n_kv)
            {
                const int d = wave * kInnerBlock + lane_row;
                const int offset = kv_offset +
                                   (kv_start + row) * kv_row_stride + d;
                dk[offset] = FromFloat<Element>(dk_accum[r][i] * scale);
                dv[offset] = FromFloat<Element>(dv_accum[r][i]);
            }
        }
    }
}

template<typename Element, bool kNhd>
__global__ __launch_bounds__(kQThreads) void D128QKernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    Element* __restrict__ dq,
    int n_q,
    int n_kv,
    int num_heads,
    int head_count,
    float scale)
{
    using Vec8 = Vec<Element, 8>;
    using Vec16 = Vec<Element, 16>;
    constexpr int tile_bytes =
        kInnerBlock * kHeadDim * sizeof(Element);
    constexpr int transpose_bytes =
        kHeadDim * kTransposeStride * sizeof(Element);
    constexpr int lds_bytes = 2 * tile_bytes + transpose_bytes;
    static_assert(tile_bytes == 4096);
    static_assert(transpose_bytes == 5120);
    static_assert(lds_bytes == 13312);

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* k_lds = reinterpret_cast<Element*>(lds);
    auto* v_lds = reinterpret_cast<Element*>(lds + tile_bytes);
    auto* k_transpose_lds = reinterpret_cast<Element*>(
        lds + 2 * tile_bytes);

    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_row = lane % kInnerBlock;
    const int lane_group = lane / kInnerBlock;
    const int q_tiles = (n_q + kQOwnerBlock - 1) / kQOwnerBlock;
    int owner_tile;
    int head_linear;
    DecodeOwner<kNhd>(
        static_cast<int>(blockIdx.x),
        q_tiles,
        head_count,
        owner_tile,
        head_linear);
    const int q_block_start = owner_tile * kQOwnerBlock;
    const int q_start = q_block_start + wave * kInnerBlock;
    const int q_offset = HeadOffset<kNhd>(head_linear, n_q, num_heads);
    const int kv_offset = HeadOffset<kNhd>(head_linear, n_kv, num_heads);
    const int q_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const int kv_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const float scale_log2 = scale * kLog2E;
    const bool q_valid = q_start + lane_row < n_q;
    const float row_lse_log2 =
        q_valid
            ? lse[head_linear * n_q + q_start + lane_row] * kLog2E
            : 0.0f;
    const float row_delta = q_valid
                                ? delta[head_linear * n_q + q_start + lane_row]
                                : 0.0f;
    Vec16 q_rows[kHeadDim / kInnerBlock] = {};
    Vec16 do_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
    for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
    {
        if(q_valid)
        {
            const int offset = q_offset +
                               (q_start + lane_row) * q_row_stride +
                               d_tile * kInnerBlock;
            q_rows[d_tile] = *reinterpret_cast<const Vec16*>(q + offset);
            do_rows[d_tile] = *reinterpret_cast<const Vec16*>(dout + offset);
        }
    }
    Float8 dq_accum[kHeadDim / kInnerBlock] = {};

    for(int kv_start = 0; kv_start < n_kv; kv_start += kInnerBlock)
    {
        __syncthreads();
        const int linear_chunk = tid;
        if(linear_chunk < kInnerBlock * kHeadDim / kPacked)
        {
            const int stage_row =
                linear_chunk / (kHeadDim / kPacked);
            const int stage_chunk =
                linear_chunk % (kHeadDim / kPacked);
            Vec8 k_stage = {};
            Vec8 v_stage = {};
            if(kv_start + stage_row < n_kv)
            {
                const int offset = kv_offset +
                                   (kv_start + stage_row) * kv_row_stride +
                                   stage_chunk * kPacked;
                k_stage = *reinterpret_cast<const Vec8*>(k + offset);
                v_stage = *reinterpret_cast<const Vec8*>(v + offset);
            }
            *reinterpret_cast<Vec8*>(
                k_lds + KVLdsOffset(stage_row, stage_chunk)) = k_stage;
            *reinterpret_cast<Vec8*>(
                v_lds + KVLdsOffset(stage_row, stage_chunk)) = v_stage;
#pragma unroll
            for(int i = 0; i < kPacked; ++i)
            {
                const int d = stage_chunk * kPacked + i;
                k_transpose_lds[d * kTransposeStride + stage_row] =
                    k_stage[i];
            }
        }
        __syncthreads();

        Float8 score = {};
        Float8 d_probability = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            WmmaInPlace(
                score,
                LoadKVRow16(k_lds, lane_row, d_tile),
                q_rows[d_tile]);
            WmmaInPlace(
                d_probability,
                LoadKVRow16(v_lds, lane_row, d_tile),
                do_rows[d_tile]);
        }

        Float8 d_score;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int kv_column = i * 2 + lane_group;
            const bool valid = q_valid &&
                               kv_start + kv_column < n_kv;
            const float probability =
                valid
                    ? __builtin_amdgcn_exp2f(
                          score[i] * scale_log2 - row_lse_log2)
                    : 0.0f;
            d_score[i] = valid
                             ? probability *
                                   (d_probability[i] - row_delta)
                             : 0.0f;
        }
        const Vec16 ds_fragment = CFragmentToA<Element>(d_score, lane);
#pragma unroll
        for(int d_pair = 0;
            d_pair < kHeadDim / (2 * kInnerBlock);
            ++d_pair)
        {
            const int d0 = d_pair * 2 * kInnerBlock + lane_row;
            const int d1 = d0 + kInnerBlock;
            WmmaInPlace(
                dq_accum[d_pair * 2],
                ds_fragment,
                LoadTransposeRow16(k_transpose_lds, d0));
            WmmaInPlace(
                dq_accum[d_pair * 2 + 1],
                ds_fragment,
                LoadTransposeRow16(k_transpose_lds, d1));
        }
    }

#pragma unroll
    for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
    {
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int row = i * 2 + lane_group;
            if(q_start + row < n_q)
            {
                const int d = d_tile * kInnerBlock + lane_row;
                dq[q_offset + (q_start + row) * q_row_stride + d] =
                    FromFloat<Element>(dq_accum[d_tile][i] * scale);
            }
        }
    }
}

template<typename Element, bool kNhd>
__global__ void DeltaD128ClearKernel(
    const Element* __restrict__ out,
    const Element* __restrict__ dout,
    float* __restrict__ delta,
    float* __restrict__ dq_acc,
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
    const Element* out_row = out + linear * kHeadDim;
    const Element* dout_row = dout + linear * kHeadDim;
    float* dq_row = dq_acc + linear * kHeadDim;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < kHeadDim; ++d)
    {
        value += static_cast<float>(out_row[d]) *
                 static_cast<float>(dout_row[d]);
        dq_row[d] = 0.0f;
    }
    delta[delta_row] = value;
}

template<typename Element>
__global__ void D128DqConvertKernel(
    const float* __restrict__ dq_acc,
    Element* __restrict__ dq,
    int elements)
{
    const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear < elements)
        dq[linear] = FromFloat<Element>(dq_acc[linear]);
}

template<bool kNhd, typename Element>
bool LaunchD128Backward(const BackwardLaunchParams& params)
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
        (params.n_q + kQOwnerBlock - 1) / kQOwnerBlock;
    const int32_t kv_tiles =
        (params.n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;

    const dim3 delta_block(kRowsBlock);
    const dim3 delta_grid(
        (static_cast<uint32_t>(q_rows) + kRowsBlock - 1) /
        kRowsBlock);
    hipLaunchKernelGGL(
        (DeltaD128Kernel<Element, kNhd>),
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

    const dim3 kv_block(kKVThreads);
    const dim3 kv_grid(
        static_cast<uint32_t>(params.head_count * kv_tiles));
    const bool aligned_q = params.n_q % kInnerBlock == 0;
    const bool aligned_kv = params.n_kv % kKVOwnerBlock == 0;
    auto launch_fallback = [&]<bool kAlignedQ, bool kAlignedKV>() {
        hipLaunchKernelGGL(
            (D128TwoPhaseKVKernel<Element, kNhd, false, kAlignedQ, kAlignedKV>),
            kv_grid,
            kv_block,
            0,
            params.stream,
            q,
            k,
            v,
            dout,
            lse,
            delta,
            static_cast<float*>(nullptr),
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.num_heads,
            params.head_count,
            params.softmax_scale);
    };
    if(aligned_q && aligned_kv)
        launch_fallback.template operator()<true, true>();
    else if(aligned_q)
        launch_fallback.template operator()<true, false>();
    else if(aligned_kv)
        launch_fallback.template operator()<false, true>();
    else
        launch_fallback.template operator()<false, false>();

    const dim3 q_grid(
        static_cast<uint32_t>(params.head_count * q_tiles));
    const dim3 q_block(kQThreads);
    hipLaunchKernelGGL(
        (D128QKernel<Element, kNhd>),
        q_grid,
        q_block,
        0,
        params.stream,
        q,
        k,
        v,
        dout,
        lse,
        delta,
        dq,
        params.n_q,
        params.n_kv,
        params.num_heads,
        params.head_count,
        params.softmax_scale);
    return hipPeekAtLastError() == hipSuccess;
}

template<bool kNhd, typename Element>
bool LaunchD128FusedDq(const BackwardLaunchParams& params)
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
    auto* dq_acc = reinterpret_cast<float*>(params.dq_acc_ptr);
    const int32_t q_rows = params.head_count * params.n_q;
    const int32_t kv_tiles =
        (params.n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;

    const dim3 row_block(kRowsBlock);
    const dim3 delta_grid(
        (static_cast<uint32_t>(q_rows) + kRowsBlock - 1) /
        kRowsBlock);
    hipLaunchKernelGGL(
        (DeltaD128ClearKernel<Element, kNhd>),
        delta_grid,
        row_block,
        0,
        params.stream,
        out,
        dout,
        delta,
        dq_acc,
        q_rows,
        params.n_q,
        params.num_heads);

    const dim3 kv_grid(
        static_cast<uint32_t>(params.head_count * kv_tiles));
    const bool aligned_q = params.n_q % kInnerBlock == 0;
    const bool aligned_kv = params.n_kv % kKVOwnerBlock == 0;
    auto launch_fused = [&]<bool kAlignedQ, bool kAlignedKV>() {
        hipLaunchKernelGGL(
            (D128TwoPhaseKVKernel<Element, kNhd, true, kAlignedQ, kAlignedKV>),
            kv_grid,
            dim3(kKVThreads),
            0,
            params.stream,
            q,
            k,
            v,
            dout,
            lse,
            delta,
            dq_acc,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.num_heads,
            params.head_count,
            params.softmax_scale);
    };
    if(aligned_q && aligned_kv)
        launch_fused.template operator()<true, true>();
    else if(aligned_q)
        launch_fused.template operator()<true, false>();
    else if(aligned_kv)
        launch_fused.template operator()<false, true>();
    else
        launch_fused.template operator()<false, false>();

    const int32_t elements = params.head_count * params.n_q * kHeadDim;
    const dim3 convert_grid(
        (static_cast<uint32_t>(elements) + kRowsBlock - 1) /
        kRowsBlock);
    hipLaunchKernelGGL(
        (D128DqConvertKernel<Element>),
        convert_grid,
        row_block,
        0,
        params.stream,
        dq_acc,
        dq,
        elements);
    return hipPeekAtLastError() == hipSuccess;
}

template<typename Element>
bool DispatchD128Backward(const BackwardLaunchParams& params)
{
    return params.layout == 1
               ? LaunchD128Backward<true, Element>(params)
               : LaunchD128Backward<false, Element>(params);
}

template<typename Element>
bool DispatchD128FusedDq(const BackwardLaunchParams& params)
{
    return params.layout == 1
               ? LaunchD128FusedDq<true, Element>(params)
               : LaunchD128FusedDq<false, Element>(params);
}

} // namespace

extern "C" bool feather_attn_bwd_d128_fused(
    const BackwardLaunchParams& params)
{
    return params.dtype == kBackwardDtypeBFloat16
               ? DispatchD128Backward<__bf16>(params)
               : DispatchD128Backward<_Float16>(params);
}

extern "C" bool feather_attn_bwd_d128_fused_dq(
    const BackwardLaunchParams& params)
{
    return params.dtype == kBackwardDtypeBFloat16
               ? DispatchD128FusedDq<__bf16>(params)
               : DispatchD128FusedDq<_Float16>(params);
}

} // namespace feather_attn
