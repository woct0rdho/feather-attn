// The forward attention images. Every symbol has the same signature, so one dispatcher
// table can cover kernel families that differ in dtype, head dim, and tail handling.

#pragma once

#include "attention_args.h"

namespace feather_attn {

// fp16 storage with an E5M2-quantized Q tile, selected by head dim, physical order, and the
// two tail modes (csrc/fwd/fp16_q8).
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_aligned(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_aligned(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_aligned(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_aligned(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_query_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_query_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_query_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_query_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d64_query_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d64_query_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_hnd_d128_query_key_tail(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_fp16_q8_nhd_d128_query_key_tail(const AttentionFwdParams&);

// bf16 storage, D128 only, one image per tile configuration (csrc/fwd/bf16).
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br256_bc64_swz(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br256_bc64_swz(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br256_bc32_pad(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br256_bc32_pad(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br256_bc64_pad_buf1(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br256_bc64_pad_buf1(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_hnd_br128_bc32_pad(const AttentionFwdParams&);
extern "C" bool feather_attn_fwd_bf16_d128_nhd_br128_bc32_pad(const AttentionFwdParams&);

} // namespace feather_attn
