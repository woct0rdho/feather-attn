// Ragged query extent: the Q tile and the output store are guarded.

#include "fwd/fp16_q8/fp16_q8_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_query_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, false, true, false>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_query_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, false, true, false>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_query_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, true, true, false>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_query_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, true, true, false>(params);
}

} // namespace feather_attn
