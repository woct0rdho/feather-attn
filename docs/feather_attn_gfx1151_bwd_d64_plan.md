# FeatherAttn gfx1151 Backward D64 Kernel

## Public Contract

```python3
feather_attn_backward(dout, q, k, v, out, softmax_lse, dq=None, dk=None, dv=None, softmax_scale=None, delta=None, layout="NHD")
```

- Inputs are contiguous HND `[B,H,N,D]` or NHD `[B,N,H,D]` FP16 tensors.
- LSE is contiguous FP32 natural-log state in logical `[B,H,NQ]` order.
- Outputs are FP16 dQ/dK/dV in the input layout and FP32 Delta in logical `[B,H,NQ]` order.
- Arbitrary positive query and KV lengths are supported. Runtime guards handle tails. The benchmark shapes are not a support whitelist.
- D64 is workspace-free and uses direct FP32 WMMA accumulators stored as FP16.
- D64 direct-NHD and selected-HND results are bit-exact in the dispatch contract.
- NHD self-attention may use bounded HND transpose routing. All copy and allocation cost is inside the measured call.

## Current Design

The complete backward path computes `Delta = rowsum(dO * O)` and executes seven GEMM-equivalents:

| Owner | GEMMs |
| --- | --- |
| KV owner | QK score, dO/V dP, dS/Q dK, P/dO dV |
| Q owner | QK score, dO/V dP, dS/K dQ |

The long path is:

```text
Q kernel:
  compute Delta from out and dO
  QK score and dO/V dP
  dS/K dQ
  direct dQ store

KV kernel:
  read Delta
  QK score and dO/V dP
  dS/Q dK and P/dO dV
  direct dK/dV stores
```

Delta is computed inside the Q kernel from the already-persistent dO rows. The separate Delta launch is retained only for the short path.

### Launch and Layout Policy

| Selected path | Launch topology |
| --- | --- |
| Short or irregular direct path | Delta plus one 128-thread fused owner kernel |
| Long HND path | Q with fused Delta and dQ, then 256-thread/128-row KV |
| Long direct-NHD path | Q with fused Delta and dQ, then 128-thread/64-row KV |
| Selected NHD self-attention | Bounded five-input/two-output HND transpose path, then the selected HND backward path |

D64 phase fission requires both `NQ` and `NKV >= 4096`. NHD uses the validated grouped/head-fast owner policy where applicable. The global tile-fast order was rejected.

### Kernel Shape

| Property | Fused/Q | HND KV long | Direct-NHD KV long |
| --- | ---: | ---: | ---: |
| Workgroup | 128 fused, 256 Q | 256 | 128 |
| Owner rows | 64 fused, 128 Q | 128 | 64 |
| Main LDS | 13,312 B fused, 26,624 B Q | 31,232 B | 15,616 B |
| Logical / allocated VGPR | 178/192 fused, 192/192 Q | 176/192 | 176/192 |

D64 stages FP16 K/V in XOR-swizzled LDS rows, uses a 20-half transpose stride, and uses a 72-half row stride for the long KV dO copy. The wide HND KV phase and the Q phase use two-buffer staging and omit the loop-start barrier. Aligned-Q/aligned-KV variants remove inner-loop tail predicates for the benchmark shapes.

### Linked Resources

| Symbol | Logical / allocated VGPR | LDS | Private / spills / scratch |
| --- | ---: | ---: | ---: |
| D64 HND fused | 178 / 192 | 13,312 B | 0 / 0 / 0 |
| D64 NHD fused | 178 / 192 | 13,312 B | 0 / 0 / 0 |
| D64 HND KV | 176 / 192 | 31,232 B | 0 / 0 / 0 |
| D64 NHD KV | 176 / 192 | 15,616 B | 0 / 0 / 0 |
| D64 HND/NHD Q | 192 / 192 | 26,624 B | 0 / 0 / 0 |
| D64 HND/NHD Delta | 66 / 72 | 0 B | 0 / 0 / 0 |

## Benchmark and Correctness

`bench/benchmark_attn_hip_backward.py` reports `F7`/`F_actual` for D64. D64 executes seven GEMMs, so F7 and F_actual are the same:

```text
F7 = F_actual = 7 * 2 * B * H * NQ * NKV * D
F5 = 5/7 * F7  # reporting cross-check only
```

The production matrix uses five warmups and ten repeats, complete provider timing, and the same paired AITER control.

### Production Matrix

Artifact: `~/tmp/feather_attn/d64_plan/final_d64_matrix_delta.json`.

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 24.383 | 31.860 | 1.304x |
| HND | 16 | 8192 | 23.367 | 30.691 | 1.297x |
| HND | 16 | 16384 | 22.775 | 30.247 | 1.326x |
| HND | 32 | 4096 | 21.451 | 30.897 | 1.442x |
| HND | 32 | 8192 | 22.583 | 30.141 | 1.335x |
| HND | 32 | 16384 | 22.254 | 30.116 | 1.353x |
| HND | 56 | 4096 | 21.166 | 29.373 | 1.364x |
| HND | 56 | 8192 | 21.927 | 29.907 | 1.362x |
| HND | 56 | 16384 | 21.591 | 29.321 | 1.363x |
| NHD | 16 | 4096 | 21.250 | 27.761 | 1.304x |
| NHD | 16 | 8192 | 22.172 | 28.107 | 1.253x |
| NHD | 16 | 16384 | 22.223 | 26.913 | 1.208x |
| NHD | 32 | 4096 | 19.666 | 26.974 | 1.382x |
| NHD | 32 | 8192 | 20.850 | 24.999 | 1.201x |
| NHD | 32 | 16384 | 20.298 | 25.390 | 1.229x |
| NHD | 56 | 4096 | 20.585 | 27.363 | 1.314x |
| NHD | 56 | 8192 | 21.147 | 26.430 | 1.256x |
| NHD | 56 | 16384 | 19.891 | 27.208 | 1.363x |

### Correctness Coverage

The public backward screen is `46/46`, including H `{16,32,56}`, batch two, selector boundaries, and asymmetric lengths through 16,385. D64 direct-NHD and selected-HND remain bit-exact. The forward regression contract is `188/188`.

## Profile and Roofline

D64 current counter evidence at H16/N4096:

| Metric | KV | Q |
| --- | ---: | ---: |
| LDS bank conflicts | 14.81% | 10.34% |
| LDS latency | 106.6 cycles | 77.8 cycles |
| Occupancy | about 45% | about 50% |

Current H16/N4096 HND diagnostic dispatch times after Q-first Delta fusion:

| Shape | Dimension | KV | Q (Delta + dQ) |
| --- | ---: | ---: | ---: |
| H16/N4096 HND | 64 | 4.07-4.29 ms | 2.64-2.71 ms |

The Delta work is inside the Q timing for long rows. The separate Delta kernel remains only for short rows.

D64 arithmetic intensity is approximately `0.875 N` FLOP/byte assuming about 16 B of minimum traffic per element. At N=4096 this is about `3,584 FLOP/byte`, far above the approximately `232 FLOP/byte` ridge. The current `29.3-31.9 TFLOPS` is about `49-54%` of the documented `59.4 TFLOPS` peak. D64 is compute/issue-bound, not DRAM-bound.

## Optimization Campaign

| Experiment | Change | Result | Decision |
| --- | --- | --- | --- |
| D64-A1 | Force-HND screen for NHD H16/H24/H32/H56 | Lost `0.807x-0.982x`, H24/N8192 tied | No threshold changes |
| D64-A2 | Aligned-Q / aligned-KV variants | Paired A/B `1.045x-1.064x` | Promoted |
| D64-A3 | Double-buffered wide-HND KV staging | Paired A/B `1.013x-1.017x`, LDS 31,232 B | Promoted |
| D64-A4 | Double-buffered Q-phase K/V/transpose staging | Paired A/B `1.011x-1.016x`, LDS 26,624 B | Promoted |
| D64-B1 | Counter baseline | KV 14.81% conflicts / 106.6 latency, Q 10.34% / 77.8 | Stop generic bank rewrite |
| D64-B2 | Tile-fast NHD owner order | Lost `0.560x-0.856x` | Rejected |
| D64-B3 | NHD group counts 2/3/4 | At most about 1.2% on one row | Rejected |
| D64-B4 | Phase-0 fused kernel for long rows | Lost `0.789x-0.971x` | Rejected |
| D64-D1 | Q-first Delta fusion | Paired A/B `1.024x-1.071x` | Promoted |
| D64-D2 | 512-thread/256-row Q owner | Near parity, sometimes worse | Not promoted |
| D64-D3 | Concurrent KV/Q after Delta | `0.964x-0.973x` | Rejected |
| D64-C1 | Five-GEMM fused KV, two topologies | `0.690x-0.744x` and `0.848x-0.883x` | Rejected |

D64-C1 detail: the 4-wave register-P/dS form linked at 224 VGPR / 21,504 B LDS and lost `0.690x-0.744x`. The 8-wave D128-style form linked at 176 VGPR / 31,872 B LDS with 48.7% occupancy but raised LDS instructions to about 5.1M from about 2.2M and lost `0.848x-0.883x`. The fixed 16*16 P/dS exchange is not amortized by K=64. D64 keeps P/dS in registers.

## Accepted Mechanisms

| Mechanism | Evidence |
| --- | --- |
| Seven-GEMM phase fission | Q falls into a narrower register tier. Confirmed at `1.01264x` with 48/48 block wins. Short rows stay fused. |
| Q-first Delta fusion | Q computes Delta from its persistent dO rows. Long Delta launch removed. `1.024x-1.071x`. |
| Aligned Q/KV variants | Tail predicates removed for aligned benchmark shapes. `1.045x-1.064x`. |
| Wide-HND KV double buffer | Removes the loop-start barrier while preserving residency. `1.013x-1.017x`. |
| Q-phase double buffer | Removes the Q-phase loop-start barrier. `1.011x-1.016x`. |
| Padded dO row stage | Stride 72 avoids stride-64 conflicts and preserves aligned vector loads. `1.015714x` over the frozen fission image with 48/48 block wins. |
| Wide HND KV owner | 256-thread/128-row owner retained for HND and transpose-routed NHD. `1.010580x` complete matrix. |
| Direct-output ownership | dK/dV/dQ stores are direct. No workspace or atomics. D64 128-row Q owner was `1.011283x` over its control. |
| Base-2 reconstruction | Native `exp2` with one-time LSE scaling. D64 gain `1.008737x` over the frozen topology. |
| NHD owner policy | Grouped/head-fast ordering. Quotient/remainder grouping prevents invalid heads. Global tile-fast ordering rejected. |

## Rejected Work

| Candidate | Result |
| --- | --- |
| D64 five-GEMM | Both tested topologies lost. Fixed P/dS LDS exchange dominates at K=64. |
| Force-HND for unforced NHD rows | Lost `0.807x-0.982x`. |
| Global tile-fast owner order | Lost `0.560x-0.856x`. |
| Long-row phase-0 fusion | Lost `0.789x-0.971x`. |
| 256-row Q owner | Near parity. Not reliably faster. |
| Concurrent KV/Q | Compute-bound overlap lost `0.964x-0.973x`. |
| KV probability recomputation | Regressed H16/N4096 by 2.98%. |
| Full K-row reload | Regressed by 3.34%. |
| Partial K cache | `0.9766x` geometric. |
| Q-row stage | Higher VGPR/LDS and worse than dO-only staging. |
| Linear 64-half dO stage | Severe LDS conflicts. |
| dO strides beyond 72 | Neutral or negative. |
| Direct global-to-LDS | Unsupported on gfx1151. |
| D64 ping-pong Q/dO staging | LDS rose to 23,040 B and lost `0.958193x` in the old test. |
| D64 combined Q-row and dO-row staging | Only `1.0014x` versus fission and slower than dO-only. |
| Precomputed KV tile bases | Removed 76 static bytes but changed no tier (`0.999524x`). |
| NHD long-phase fusion | `0.956310x` against AITER with long-row regressions. |
| Blanket NHD LLC grouping | Non-periodic H56 strides regressed. |
| Alternate NHD group counts | H32/N16384 counts 5/3 reached `0.904842x`/`0.911481x`. |
| Initial equal-size uneven grouping | Produced an invalid head. Quotient/remainder replacement retained. |
| Nine-wave launch bound | LLVM final eight-wave occupancy could not satisfy `__launch_bounds__(128,9)`. |
| Forced 168-VGPR attribute | Changed logical counts but not the 192-VGPR class. Naturally allocated dO-stage timing was better. |
| `owned` / `paired` / automatic selectors | Removed after losing frozen rows or becoming redundant. |
| Stock AITER CK backward | Default build receipt omits gfx1151. Retained only as external evidence. |
| D64 KV dV/dP interleave | Same allocation tier, larger code, `0.984856x` geometric, 0/36 block wins. |
| D64 trailing wave-barrier removal | `0.999392x`, 17/36 block wins. No repeatable gain. |
| Six-GEMM N^2 workspace | Excess storage and traffic. Rejected. |
| Old atomic FP32 D64 kernel | Removed. |

## Roofline Verdict

D64 is near its practical compute roof. The remaining large candidates are exhausted. The open work is targeted:
- Phase-specific NHD owner order and wide-KV screening for the weak NHD H32/N8192 row.
- Re-run the matrix after any policy or topology change.
- A future five-GEMM family must keep P/dS in registers and avoid the 16*16 LDS exchange. Copying D128 is not viable.

## Measurement and Profiling Methodology

- Paired same-process A/B, 20 samples where feasible, log medians and confidence intervals.
- Complete provider path: allocation, Delta, Q, KV, transposes/copies, and conversion.
- Do not promote from counter-only wins.
- Reject spills, private memory, or new scratch unless a large complete-path margin justifies review.
- Run the full 18-row matrix after policy changes.
- Keep generic/tail variants for arbitrary lengths.

- Promotion requires production-equivalent linked gfx1151 code, exact-symbol metadata/disassembly, zero private memory/spills/scratch, correct owner/tail/layout behavior, and a positive complete-path 18-row screen. D64 also keeps the `46/46` backward and `188/188` forward contracts.
- Timing uses alternating baseline/candidate blocks, raw samples, per-block ratios, log medians, and paired confidence intervals. Include Delta, Q, KV, allocation, conversion, and copies.
- Accuracy tiers: directional cosine at least `0.997`. Preferred Rel-L2 at most `0.10`, exploratory below `0.20`. Exact scheduling/layout candidates preserve the elementwise contract.
- Evidence hierarchy: runtime symbol from `rocprofv3`, linked metadata, exact-symbol disassembly, optimized MIR/pressure, then architecture/byte models. `llvm-readelf` is required for AMDGPU metadata. Virtual, allocated, and simultaneously live registers are distinct quantities.
- gfx1151 cannot collect all profiler counters in one pass. Separate legal passes are required. Wait counters overlap and must not be added. `FETCH_SIZE` is not a write counter, and profiler timestamps do not admit a speedup.
- Cross-dimension compiler findings: a phase-local source split matched the compile-time phase image. MIR pressure must distinguish virtual, allocated, and simultaneously live registers. Forced caps alone are not credible.
- A serious candidate keeps provenance, code-object hashes, metadata, disassembly, IR/MIR, pressure reports, correctness output, and paired timing together.

## Artifact Index

- Current D64 matrix: `~/tmp/feather_attn/d64_plan/final_d64_matrix_delta.json`.
- Aligned A/B: `~/tmp/feather_attn/d64_plan/ab_aligned.json`.
- KV/Q double-buffer A/B: `~/tmp/feather_attn/d64_plan/ab_stage.json`, `ab_qdouble.json`.
- Delta-fusion A/B: `~/tmp/feather_attn/d64_plan/ab_delta.json`.
- Concurrency A/B: `~/tmp/feather_attn/d64_plan/ab_conc.json`.
- Force/order A/B: `~/tmp/feather_attn/d64_plan/ab_policy.json`, `ab_order.json`.
- Counters: `~/tmp/feather_attn/d64_plan/counters2/`.
- Five-GEMM flagships: `~/tmp/feather_attn/d64_plan/ab_five.json`, `meta_five4/`, `five_counters2/`, `meta_five7/`, `five_counters3/`.
- Linked metadata: `~/tmp/feather_attn/d64_plan/meta_nhd/`.
- Phase fission: `~/tmp/feather_attn/phase_fission_campaign/`.
- Padded dO stage: `~/tmp/feather_attn/kv_dorow_stride72_campaign/`.
- D64 Q owner: `~/tmp/feather_attn/postprod_bwd_campaign/exp2_d64_q_owner128_wg256/`.
- D64 HND KV owner: `~/tmp/feather_attn/postprod_bwd_campaign/exp3b_d64_kv_hnd_owner128_wg256/`.
- Isolated base-2: `~/tmp/feather_attn/postprod_bwd_campaign/exp5d_d64_kv_exp2/`, `exp5e_d64_q_exp2/`.
- Rank-5 qualification: `~/tmp/feather_attn/postprod_bwd_campaign/exp5f_d64_kv_q_exp2/`.
