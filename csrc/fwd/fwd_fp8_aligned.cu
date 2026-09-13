#include "fwd_fp8_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp8_hnd_d64_aligned(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<64, false, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp8_hnd_d128_aligned(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<128, false, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d64_aligned(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<64, true, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d128_aligned(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<128, true, false, false>(params);
}

} // namespace feather_attn
