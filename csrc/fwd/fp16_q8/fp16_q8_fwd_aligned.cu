// Aligned Q and K/V tiles: no tail guards, the fastest image of this family.

#include "fwd/fp16_q8/fp16_q8_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_aligned(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, false, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_aligned(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, false, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_aligned(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<64, true, false, false>(params);
}

extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_aligned(const AttentionFwdParams& params)
{
    return LaunchFp16Q8Fwd<128, true, false, false>(params);
}

} // namespace feather_attn
