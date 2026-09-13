# FeatherAttn gfx1151 Backward FP8 Kernel (Inactive)

These experiments came from the historical D128 backward precision campaign. They are retained as negative evidence and must not be read as production plans.

| Candidate | Decisive result |
| --- | --- |
| FP8 P-only truncation | Directionally valid but added 68 static instructions and 10 permutes, stayed in the 192-VGPR tier, and measured `0.979607x` over six rows. |
| FP8 dS-only truncation | Directionally valid but added 21 instructions and 18 permutes, changed no useful tier, and measured `0.996503x` with no row wins. |
| FP8 P+dS truncation | Failed the direction gate: minimum flattened cosine `0.996022` and per-head cosine `0.994891`. Still allocated 192 VGPRs. |
| Fixed, power-of-two, and linear FP8 scaling | Conversion and register cost did not create a profitable resource transition on gfx1151, which lacks native E5M2 conversion/WMMA support. |
| Raw high-byte E5M2 truncation | All 65,536 encodings match `half_bits >> 8`. Finite magnitude, sign, zero, and infinity behavior verified. This is a primitive, not a profitable backward representation. |
| Two-permute E5M2 pack | Bit-exact and faster than scalar pack/decode in its isolated fixture, with 15 versus 16 logical VGPRs. Complete P/dS kernels still lost. |

Artifact: `~/tmp/feather_attn/e5m2_truncation_campaign/`.

## Conclusion

The archived backward experiments do not justify an FP8/INT8 production path on gfx1151:
- Native E5M2 conversion/WMMA support is absent.
- Packing, scaling, and decoding costs outweighed any memory or register benefit.
- P/dS truncation failed accuracy and did not change the resource tier.

Do not reopen this campaign without a new hardware primitive or a materially different complete-path model. Current backward optimization focuses on the FP16 D64 and D128 plans.
