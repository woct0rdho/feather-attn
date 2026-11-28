// Both extents ragged.

#include "fwd/fp16_q8/fp16_q8_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_query_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, false, true, true>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_query_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, false, true, true>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_query_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, true, true, true>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_query_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, true, true, true>(params);
}

} // namespace feather_attn
