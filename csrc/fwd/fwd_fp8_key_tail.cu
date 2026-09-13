#include "fwd_fp8_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp8_hnd_d64_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<64, false, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_hnd_d128_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<128, false, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d64_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<64, true, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d128_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<128, true, false, true>(params);
}

} // namespace feather_attn
