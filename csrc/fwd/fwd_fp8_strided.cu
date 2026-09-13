#define FEATHER_ATTN_STRIDED_NHD 1
#define FwdFp8Kernel StridedNhdFwdFp8Kernel
#define LaunchFwdFp8Variant LaunchStridedNhdFwdFp8Variant
#include "fwd_fp8_kernel.h"
#undef LaunchFwdFp8Variant
#undef FwdFp8Kernel
#undef FEATHER_ATTN_STRIDED_NHD

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp8_nhd_d64_strided_aligned(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdFwdFp8Variant<64, true, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d64_strided_query_tail(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdFwdFp8Variant<64, true, true, false>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d64_strided_key_tail(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdFwdFp8Variant<64, true, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp8_nhd_d64_strided_query_key_tail(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdFwdFp8Variant<64, true, true, true>(params);
}

} // namespace feather_attn
