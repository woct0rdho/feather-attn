# FeatherAttn gfx1151 Backward D128 Kernel

## Public Contract

```python3
feather_attn_backward(dout, q, k, v, out, softmax_lse, dq=None, dk=None, dv=None, softmax_scale=None, delta=None, layout="NHD")
```

- Inputs are contiguous HND `[B,H,N,D]` or NHD `[B,N,H,D]` tensors. Q/K/V/out/dO/dQ/dK/dV must all be FP16 or all BF16.
- LSE is contiguous FP32 natural-log state in logical `[B,H,NQ]` order.
- Outputs are dQ/dK/dV in the input dtype and layout, plus FP32 Delta in logical `[B,H,NQ]` order.
- Arbitrary positive query and KV lengths are supported. Tile counts and guards cover tails.
- D128 allocates a temporary FP32 dQ accumulation tensor in Python. It remains internal and is passed through the private fused operator.
- The fused path is tolerance-equivalent on dQ because FP32 atomic accumulation changes rounding order. dK/dV/Delta are direct writes and remain bit-exact in paired dispatch checks.
- The fused D128 path uses five GEMM-equivalents. The fallback uses seven. Determinism is not a requirement for dQ.

## Current Design

The five unique backward GEMMs are executed once:

```text
S  = Q K^T
dP = dO V^T
dV = P^T dO
dK = dS^T Q
dQ = dS K
```

Per `64 KV * 16 Q`, the fused kernel executes 160 WMMA tiles, or five GEMM-equivalents.

### Two-Phase KV Ownership

Phase A:
- Four wave pairs handle four 16-row KV groups.
- The even wave computes score. The odd wave computes dP.
- Score and dP are exchanged through FP32 LDS.
- P and dS are published as 16-bit SoA A-fragment tiles (`half2` for FP16, `bf16x2` for BF16).

Phase B:
- Eight waves each own one D16 slice.
- Each wave computes dK and dV for the four KV groups and accumulates dQ over them.
- dK/dV remain in registers across the Q loop and are stored once.
- dQ is atomically added to the FP32 workspace once per Q tile.
- A final kernel converts the workspace to public 16-bit dQ.

### Launch and Layout Policy

| Selected path | Launch topology |
| --- | --- |
| Fused HND | Delta-clear, 256-thread/8-wave two-phase KV, dQ conversion |
| Fused NHD, `heads >= 8` and `NKV >= 4096` | Five-input/two-output transpose policy, fused HND path, gradient copies |
| NHD `heads < 8` or `NKV < 4096` | Delta, aligned seven-GEMM two-phase KV, old D128 Q |
| Selected NHD self-attention | Exact HND transposes. All copy/allocation cost included in timing |

The HND-copy path is limited to contiguous B1 self-attention for the production shapes. It is selected for `heads >= 8` at `N >= 4096`. Direct NHD fusion remains slower.

### Kernel Shape

| Property | Five-GEMM fused KV | Seven-GEMM fallback KV | Fallback Q |
| --- | ---: | ---: | ---: |
| Workgroup | 256 threads / 8 waves | 256 threads / 8 waves | 512 threads / 16 waves |
| Owner rows | 64 KV rows | 64 KV rows | 256 query rows |
| Ownership | Four score/dP wave pairs. Eight D16 slices for dK/dV/dQ | Four waves each for dK/dV | Persistent 16-bit Q/dO, FP32 dQ |
| Main LDS | 42,112 B | 26,752 B direct / 30,976 B cached-dO | 13,312 B |
| Logical / allocated VGPR | 232 / 240 | 184 / 192 | 256 / 256 |
| Effective residency | Six waves/SIMD, 36.6% occupancy | Six waves/SIMD | One workgroup/WGP, four waves/SIMD |

The fused kernel keeps K/V row fragments in registers per wave pair, double-buffers Q/dO transpose views, exchanges score/dP through an 8 KiB FP32 SoA buffer, and publishes P/dS as 16-bit SoA fragments (`half2`/`bf16x2`). The dSᵀ tile uses stride 20 and two 16 B loads. There is no K/V LDS tile. Aligned-Q/aligned-KV variants remove tail predicates for aligned shapes.

### Linked Resources

| Symbol | Logical / allocated VGPR | LDS | Private / spills / scratch |
| --- | ---: | ---: | ---: |
| D128 Delta-clear | 80 / 96 | 0 B | 0 / 0 / 0 |
| D128 five-GEMM fused KV | 232 / 240 | 42,112 B | 0 / 0 / 0 |
| D128 seven-GEMM fallback KV | 184 / 192 | 26,752 B | 0 / 0 / 0 |
| D128 fallback Q | 256 / 256 | 13,312 B | 0 / 0 / 0 |

Resource counts are for the FP16 instantiation. BF16 uses the same LDS layout.

## Benchmark and Correctness

D128 reports both `F5` (five-GEMM work) and `F_actual`:

```text
F7 = 7 * 2 * B * H * NQ * NKV * D
F5 = 5 * 2 * B * H * NQ * NKV * D
F_actual = G * 2 * B * H * NQ * NKV * D
```

In the current production matrix every row executes five GEMMs, so `TFLOPS_F5 == TFLOPS_actual`. The matrix uses five warmups and twelve repeats with the paired AITER control.

Both dtype instantiations use the same kernel topology. A same-process A/B screen measured BF16/FP16 between `1.02x` and `1.05x` across D128 HND/NHD. No separate BF16 matrix is retained. Use `--dtype bf16` to select BF16 in the benchmark.

### Production Matrix

Artifact: `~/tmp/feather_attn/d128_plan/final_double_stage_matrix.json`.

| Layout | H | N | Executed GEMMs | TFLOPS_F7 | TFLOPS_F5 | TFLOPS_actual | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 5 | 34.150 | 24.393 | 24.393 | 4.281x |
| HND | 16 | 8192 | 5 | 33.461 | 23.900 | 23.900 | 4.456x |
| HND | 16 | 16384 | 5 | 33.177 | 23.698 | 23.698 | 4.386x |
| HND | 32 | 4096 | 5 | 32.275 | 23.053 | 23.053 | 4.372x |
| HND | 32 | 8192 | 5 | 31.288 | 22.349 | 22.349 | 4.182x |
| HND | 32 | 16384 | 5 | 31.817 | 22.726 | 22.726 | 4.267x |
| HND | 56 | 4096 | 5 | 29.364 | 20.974 | 20.974 | 4.020x |
| HND | 56 | 8192 | 5 | 30.103 | 21.502 | 21.502 | 4.066x |
| HND | 56 | 16384 | 5 | 31.292 | 22.352 | 22.352 | 4.218x |
| NHD | 16 | 4096 | 5 | 22.846 | 16.319 | 16.319 | 3.433x |
| NHD | 16 | 8192 | 5 | 24.953 | 17.824 | 17.824 | 3.745x |
| NHD | 16 | 16384 | 5 | 26.845 | 19.175 | 19.175 | 4.331x |
| NHD | 32 | 4096 | 5 | 22.561 | 16.115 | 16.115 | 3.733x |
| NHD | 32 | 8192 | 5 | 24.154 | 17.253 | 17.253 | 3.963x |
| NHD | 32 | 16384 | 5 | 27.310 | 19.507 | 19.507 | 4.596x |
| NHD | 56 | 4096 | 5 | 24.293 | 17.352 | 17.352 | 3.537x |
| NHD | 56 | 8192 | 5 | 26.095 | 18.639 | 18.639 | 3.762x |
| NHD | 56 | 16384 | 5 | 28.769 | 20.549 | 20.549 | 4.125x |

The HND geometric is `4.247419x`, NHD is `3.898182x`, and the combined matrix is `4.069056x` with `18/18` rows won.

### Correctness Coverage

The public backward screen is `93/93` across FP16 and BF16, and the forward regression contract is `188/188` (FP16 forward only). The D128 fused dispatch keeps dK, dV, and Delta bit-exact and requires tolerance equality for dQ for both dtypes. The previous rank-5 per-head screen is historical evidence for the old D128 image. No new rank-5 campaign has been run for the five-GEMM path. The FP16 H16/N4096 saved-state reference maximum absolute errors are approximately `dq 2.02e-4`, `dk 2.22e-4`, `dv 1.36e-4`, and `delta 5.96e-7`.

## Profile and Roofline

Current H16/N4096 counter evidence:

| Metric | Value |
| --- | ---: |
| VGPR | 232 / 240 |
| LDS | 42,112 B |
| Occupancy | 36.6% |
| LDS bank conflicts | 24.27% |
| VALU instructions | 61,629 |
| Mean occupancy per active CU | 23.76 |
| Counter-replay kernel time | 12.8-13.3 ms |
| Complete path | about 14.1 ms |

The last valid stochastic PC sample, taken before double-buffered staging, reported barrier wait `25.0%` and ALU dependency `32.7%`. No post-change PC capture was retained. The double-buffer removed one of the three per-tile barriers.

D128 arithmetic intensity is approximately `0.385 N` FLOP/byte including the dQ workspace, or about `1,575 FLOP/byte` at N=4096. That is far above the approximately `232 FLOP/byte` ridge. D128 is compute/issue/LDS-bound, not DRAM-bound. At `20.9-24.4 TFLOPS` it reaches about `35-41%` of the documented `59.4 TFLOPS` peak.

### Occupancy Finding

Four workgroups require both `VGPR <= 192` and `LDS <= 32 KiB`. The combined experiment used a single-buffer Q/dO transpose (`31,872 B` LDS) and reloaded Phase-A K/V operand rows from L2 to reduce VGPR to `176`. It was resource-valid but measured only `0.72x-0.81x`: the extra K/V reload and loss of double-buffer overlap dominate. The straightforward occupancy path is closed.

## Bank-Conflict and Staging Campaign

Initial evidence suggested LDS layout and barrier serialization, not dQ atomics:
- `LDSBankConflict` was initially `59.4%`.
- `s_barrier` was the top sampled instruction.
- `global_atomic_add_f32` was only about `2.2%` of samples.

Completed changes:

| Experiment | Change | Result |
| --- | --- | --- |
| D128-B1 | Publish exchange and staging with one barrier. LSE/Delta from 128 B LDS | Removed one barrier and duplicate loads |
| D128-B2 | SoA score/dP exchange | Conflict degree 8 -> 1 |
| D128-B3 | half2 SoA P/dS, stride-20 dSᵀ with two 16 B loads | P/dS 8 -> 1, dSᵀ 8 -> 4 |
| D128-B4 | Q/dO staging remap to 8 Q rows * 32 D | Staging 16-way -> 4-way |
| D128-B5 | Lane-base hoisting and packed `cvt_pkrtz` | VALU 97,100 -> 83,508 |
| D128-B6 | Aligned-Q/aligned-KV variants | VGPR 240 -> 232, H16 15.75 -> 14.78 ms |
| D128-B7 | Aligned fallback KV | NHD fallback 21.33 -> 19.86 ms |
| D128-B8 | Double-buffer fused Q/dO staging. Fallback stays single-buffer | Paired A/B 1.031x-1.089x |
| D128-B9 | Force-HND threshold `N >= 4096` for `heads >= 8`. Force path selects fused | NHD H16/N4096 19.77 ms, H24/N4096 31.85 -> 26.02 ms |

Remaining LDS conflicts are near the transaction minimum for 16 B dSᵀ/staging accesses. Further layout-only work has low expected value.

## Accepted Mechanisms

| Mechanism | Evidence |
| --- | --- |
| Five-GEMM two-phase KV | 256 -> 160 WMMA tiles per 64 KV * 16 Q, matrix 4.069056x |
| Wave-pair score/dP split | Score/dP computed once and exchanged in FP32 LDS |
| FP32 dQ atomic workspace | Caller-owned, cleared by Delta, converted after fused KV |
| Force-HND policy | NHD `heads >= 8`, `N >= 4096` selected after paired screens |
| Conflict-free SoA LDS | Score/dP, P/dS, dSᵀ, and staging layouts |
| Double-buffered staging | Two-barrier fused schedule, A/B 1.031x-1.089x |
| Packed conversion and lane hoisting | `cvt_pkrtz`, VALU reduction |
| Aligned-Q/KV variants | Removes tail predicates, VGPR 232 |
| Exact C-to-A fragment handoff | FP32 P/dS/accumulators remain exact until the FP16 WMMA input handoff |
| Direct-output fallback | Seven-GEMM fallback and Q kernel remain workspace-free. 64-row KV and 512-thread/256-row persistent Q/dO owners retained. The 64-row KV owner measured `1.101567x` over its control. |
| Base-2 reconstruction | Native `exp2` with one-time LSE scaling. D128 gain `1.005163x` over the frozen topology |

## Rejected Work

| Candidate | Result |
| --- | --- |
| D128 seven/eight-GEMM KV/Q | Loses to five-GEMM on HND. Retained only for small NHD fallback |
| D128 Q stage of 32 K/V rows | Spilled or only `1.000558x`. Closed at current LDS |
| D128 persistent dO without Q | Slower than persistent Q in the old topology |
| Fold Delta into D128 Q | Spills and private memory. Rejected |
| Share P from dK to dV waves | Added LDS/barrier/residency cost. Five-GEMM replaced it |
| D128 direct NHD H16/N4096 | Direct fusion slower. Transpose route selected |
| Universal D128 direct dO reload | Improved HND but collapsed tile-fast H56 NHD to 330.47 ms versus AITER 244.87 ms. Layout policy remains required |
| Mechanical D64-to-D128 lift | Spilled heavily (KV 5-13 values, Q 265-268 values, 24-564 B private memory). Rejected before timing |
| Share D128 P from dK to dV waves | Added LDS/barrier/residency cost. Both barrier placements regressed to 26.22-26.76/51.23-52.31 ms |
| D128 independent KV/Q old topology | Historical alternate ownership. Replaced by the five-GEMM wave-pair/D16 layout |
| Older paired-ownership/five-GEMM variants | Extra launch/reduction cost or local-only gains. Current five-GEMM topology retained |
| Historical direct-D128 row cache | Applied only to old wider direct NHD layouts. Fused path uses the transpose route |
| D128 persistent Q/dO without Q | Same 216-VGPR tier as persistent Q but slower. Replaced by the fused path |
| Native FP16-output WMMA | FP16 and FP32 output fragments both use eight VGPRs. No accumulator saving |
| Precomputed log2 LSE workspace | Bit-exact but per-call allocation/workspace traffic lost |
| Transient P/dS pack then decode | Removed no traffic class and added conversion work |
| D128 staging before score/dP | Regressed 4-14%. Reverted |
| Half dSᵀ staging | Lower conflicts but VALU 61,629 -> 80,191, 0.94x-0.96x |
| Vector/Half4/interleaved A fragments | Multiple 4-12% regressions. half2 SoA retained |
| Occupancy-first single-buffer + K/V reload | 176 VGPR / 31,872 B LDS but 0.72x-0.81x |
| Bkv=128 | Accumulator/VGPR cost dominates unless a new LDS accumulation design appears |
| dQ head-group L2 | No-atomic ceiling about 3.6% at the largest shape. Low priority |
| Six-GEMM N^2 workspace | Excess storage/traffic. Rejected |

## Remaining Work

- Keep the two-barrier fused schedule. The next large D128 gain requires a different register/ownership design, not another layout pass.
- The low-priority dQ swizzled-workspace idea from CUTLASS/SageAttention could make atomic slots contiguous, but the no-atomic ceiling is small.
- Re-run the full matrix and resource contract after any compiler or policy change.
- Do not reopen FP8/INT8 quantization here. Archived FP8 evidence is in `docs/feather_attn_gfx1151_bwd_fp8_plan.md`.

## Measurement and Profiling Methodology

- Paired same-process A/B, log medians and confidence intervals.
- Complete provider path includes Delta-clear, fused KV, dQ conversion, allocation, and NHD transposes/copies.
- No counter-only promotion.
- Reject spills/private memory or new scratch unless a large complete-path margin justifies review.
- Full 18-row matrix after policy changes. Keep generic tails.

- Promotion requires production-equivalent linked gfx1151 code, exact-symbol metadata/disassembly, zero private memory/spills/scratch, correct owner/tail/layout behavior, and a positive complete-path 18-row screen. The full D128 screen keeps the `93/93` backward and `188/188` forward contracts.
- Timing uses alternating baseline/candidate blocks, raw samples, per-block ratios, log medians, and paired confidence intervals. Include Delta, Q, KV, allocation, conversion, and copies.
- Accuracy tiers: directional cosine at least `0.997`. Preferred Rel-L2 at most `0.10`, exploratory below `0.20`. Exact scheduling/layout candidates preserve the elementwise contract.
- Evidence hierarchy: runtime symbol from `rocprofv3`, linked metadata, exact-symbol disassembly, optimized MIR/pressure, then architecture/byte models. `llvm-readelf` is required for AMDGPU metadata. Virtual, allocated, and simultaneously live registers are distinct quantities.
- gfx1151 cannot collect all profiler counters in one pass. Separate legal passes are required. Wait counters overlap and must not be added. `FETCH_SIZE` is not a write counter, and profiler timestamps do not admit a speedup.
- Cross-dimension compiler findings: a phase-local source split matched the compile-time phase image. MIR pressure must distinguish virtual, allocated, and simultaneously live registers. Forced caps alone are not credible.
- A serious candidate keeps provenance, code-object hashes, metadata, disassembly, IR/MIR, pressure reports, correctness output, and paired timing together.

## Artifact Index

- Current D128 matrix: `~/tmp/feather_attn/d128_plan/final_double_stage_matrix.json`.
- Current counters: `~/tmp/feather_attn/d128_plan/double_counters2/`.
- Double/single staging A/B: `~/tmp/feather_attn/d128_plan/ab_stage.json`.
- NHD force-HND screens: `~/tmp/feather_attn/d128_plan/policy_nhd_fused.json`, `policy_nhd.json`.
- Bank/barrier campaign: `~/tmp/feather_attn/d128_plan/b4_d128.json`, `cvt_addr_d128.json`, `aligned_d128.json`.
- Historical matrix: `~/tmp/feather_attn/d128_plan/final_d128_matrix_5g.json`, `final_d128_matrix_bank.json`.
- Historical fallback: `~/tmp/feather_attn/d128_plan/nhd_h16_n4096_old_path.json`, `fallback_aligned.json`.
- Historical counters/PC: `~/tmp/feather_attn/d128_plan/final_counters/`, `final_pcs/`.
- Fused trace/resources: `~/tmp/feather_attn/d128_plan/profile_stage_c/`, `counters/`.
- Fallback trace/resources: `~/tmp/feather_attn/d128_plan/profile_fallback/`.
- Rank-5 qualification: `~/tmp/feather_attn/postprod_bwd_campaign/exp5c_d128_kv_q_exp2/`.
