#pragma once

// Minimal, self-contained implementation of the small subset of the
// Composable Kernel (CK-Tile) API surface that featherattn_fwd_kernel.h
// relies on. CK itself does not support Windows, so this shim lets the
// forward kernels compile with the HIP/clang toolchain shipped by the
// ROCm Windows distribution ("The Rock") without any CK dependency.
//
// Only used when building on Windows (see featherattn_fwd_kernel.h).
// The Linux build keeps using the real CK-Tile headers unchanged.

#include <hip/hip_runtime.h>

#include <cstdint>

#define CK_TILE_HOST __host__
#define CK_TILE_DEVICE __device__
#define CK_TILE_HOST_DEVICE __host__ __device__

namespace ck_tile {

using index_t = int32_t;

using fp16_t = _Float16;
using fp16x2_t = _Float16 __attribute__((ext_vector_type(2)));
using fp16x4_t = _Float16 __attribute__((ext_vector_type(4)));
using fp16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));

using uint32x4_t = uint32_t __attribute__((ext_vector_type(4)));
using uint32x8_t = uint32_t __attribute__((ext_vector_type(8)));
using uint8x16_t = uint8_t __attribute__((ext_vector_type(16)));

struct gfx11_t
{
};
struct gfx115_t
{
};

template <index_t N>
struct number
{
    static constexpr index_t value = N;
};

template <typename To, typename From>
CK_TILE_HOST_DEVICE constexpr To bit_cast(const From& src)
{
    return __builtin_bit_cast(To, src);
}

template <typename To, typename From>
CK_TILE_HOST_DEVICE To type_convert(const From& src)
{
    return static_cast<To>(src);
}

template <index_t Begin, index_t N, index_t Step>
struct static_for
{
    template <typename F>
    CK_TILE_DEVICE void operator()(const F& fn) const
    {
        static_for_run<Begin>(fn);
    }

private:
    template <index_t I, typename F>
    CK_TILE_DEVICE static void static_for_run(const F& fn)
    {
        if constexpr(I < N)
        {
            fn(number<I>{});
            static_for_run<I + Step>(fn);
        }
    }
};

template <typename T0, typename T1, typename T2, typename T3>
struct make_tuple_t
{
    T0 v0;
    T1 v1;
    T2 v2;
    T3 v3;
};

template <typename T0, typename T1, typename T2, typename T3>
CK_TILE_HOST_DEVICE constexpr auto make_tuple(T0 v0, T1 v1, T2 v2, T3 v3)
{
    return make_tuple_t<T0, T1, T2, T3>{v0, v1, v2, v3};
}

template <index_t N0, index_t N1, index_t N2, index_t N3>
struct PackedDescriptor4
{
    CK_TILE_DEVICE index_t calculate_offset(
        const make_tuple_t<index_t, index_t, index_t, index_t>& idx) const
    {
        return (((idx.v0 * N1) + idx.v1) * N2 + idx.v2) * N3 + idx.v3;
    }
};

template <index_t N0, index_t N1, index_t N2, index_t N3>
CK_TILE_HOST_DEVICE constexpr auto make_naive_tensor_descriptor_packed(
    const make_tuple_t<number<N0>, number<N1>, number<N2>, number<N3>>&)
{
    return PackedDescriptor4<N0, N1, N2, N3>{};
}

template <typename Arch, typename A, typename B, typename C,
          index_t M, index_t N, index_t K>
struct WmmaTraits
{
};

template <typename Traits>
struct WarpGemmAttributeWmmaImpl
{
    // RDNA3 v_wmma_f32_16x16x16_f16 layout: A/B are 16x16 fp16 (16 per lane =
    // 8 VGPRs), C is 16x16 f32 (8 floats per lane = 8 VGPRs).
    using AVecType = fp16x16_t;
    using BVecType = fp16x16_t;
    using CVecType = float __attribute__((ext_vector_type(8)));
};

CK_TILE_DEVICE index_t get_thread_local_1d_id()
{
    return static_cast<index_t>(threadIdx.x);
}

CK_TILE_DEVICE index_t get_block_1d_id()
{
    return static_cast<index_t>(blockIdx.x);
}

CK_TILE_DEVICE void block_sync_lds()
{
    __syncthreads();
}

struct stream_config
{
    hipStream_t stream = nullptr;
};

template <typename Kernel, typename Arg>
__global__ void KernelEntry(Kernel kernel, Arg args)
{
    kernel(args);
}

template <typename Kernel, typename Arg>
struct KernelLaunch
{
    Kernel kernel;
    dim3 grid;
    dim3 block;
    unsigned dynamic_shared_bytes;
    Arg args;

    CK_TILE_HOST void run(hipStream_t stream) const
    {
        hipLaunchKernelGGL(
            (KernelEntry<Kernel, Arg>),
            grid,
            block,
            dynamic_shared_bytes,
            stream,
            kernel,
            args);
    }
};

template <unsigned kWaves, typename Arch, typename Kernel, typename Arg>
CK_TILE_HOST auto make_kernel(
    Kernel kernel,
    dim3 grid,
    dim3 block,
    unsigned dynamic_shared_bytes,
    Arg args)
{
    return KernelLaunch<Kernel, Arg>{
        kernel, grid, block, dynamic_shared_bytes, args};
}

template <typename Kernel>
CK_TILE_HOST void launch_kernel(
    const stream_config& config,
    const Kernel& kernel)
{
    kernel.run(config.stream);
}

} // namespace ck_tile