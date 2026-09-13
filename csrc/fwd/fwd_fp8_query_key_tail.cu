#include "fwd_fp8_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp8_hnd_d64_query_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<64, false, true, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_hnd_d128_query_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<128, false, true, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d64_query_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<64, true, true, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d128_query_key_tail(
    const LaunchParams& params)
{
    return LaunchFwdFp8Variant<128, true, true, true>(params);
}

} // namespace feather_attn
