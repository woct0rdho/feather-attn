// Instantiations of the bf16 D128 forward kernel. The tile shape, the LDS
// buffering depth, and the LDS layout are part of the symbol so the host can pick
// a configuration without a rebuild.

#include "fwd/bf16/bf16_fwd_d128_kernel.h"

namespace feather_attn {

#define FEATHER_ATTN_BF16_D128_IMAGE(name, head_fast, br, bc, buf, swz)              \
    extern "C" bool feather_attn_fwd_bf16_d128_##name(const Bf16FwdParams& params)   \
    {                                                                                \
        return LaunchBf16FwdD128<head_fast, br, bc, buf, swz>(params);               \
    }

// 256-row tiles with 64 keys and double-buffered swizzled LDS: the HND default.
FEATHER_ATTN_BF16_D128_IMAGE(hnd_br256_bc64_swz, false, 256, 64, 2, true)
FEATHER_ATTN_BF16_D128_IMAGE(nhd_br256_bc64_swz, true, 256, 64, 2, true)
// 256-row tiles with 32 keys and padded affine LDS: reaches two workgroups per
// processor, which is what the NHD rows need.
FEATHER_ATTN_BF16_D128_IMAGE(hnd_br256_bc32_pad, false, 256, 32, 2, false)
FEATHER_ATTN_BF16_D128_IMAGE(nhd_br256_bc32_pad, true, 256, 32, 2, false)
// 64-key tiles with a single K/V buffer and a second barrier per tile.
FEATHER_ATTN_BF16_D128_IMAGE(hnd_br256_bc64_pad_buf1, false, 256, 64, 1, false)
FEATHER_ATTN_BF16_D128_IMAGE(nhd_br256_bc64_pad_buf1, true, 256, 64, 1, false)
// 128-row fallback for short query extents.
FEATHER_ATTN_BF16_D128_IMAGE(hnd_br128_bc32_pad, false, 128, 32, 2, false)
FEATHER_ATTN_BF16_D128_IMAGE(nhd_br128_bc32_pad, true, 128, 32, 2, false)

} // namespace feather_attn
