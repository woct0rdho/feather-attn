// Ragged key/value extent: the staging loads are guarded and out-of-range scores are masked.

#include "fwd/fp16_q8/fp16_q8_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, false, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, false, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, true, false, true>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_key_tail(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, true, false, true>(params);
}

} // namespace feather_attn
