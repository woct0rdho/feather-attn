// gfx1103-tuned forward launch: 256 query rows per block (512 threads).
// Each block reads each K/V tile once for 256 rows instead of 128, halving
// redundant DRAM K/V traffic on the 12-CU / 2MB-L2 variant. Dispatched at
// run time only for gfx1103 (see gfx1103 dispatch in hip_kernel.cpp); the
// gfx115x host path and all other variants are untouched.
#include "featherattn_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_nhd_d128_aligned_bm256(const LaunchParams& params)
{
    return LaunchVariant<128, true, false, false, 512, 256>(params);
}

} // namespace feather_attn