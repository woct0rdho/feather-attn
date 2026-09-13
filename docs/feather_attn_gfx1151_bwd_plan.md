# FeatherAttn gfx1151 Backward Kernel

## Current Status

The gfx1151 backward implementation is in production for D64 and D128. The post-production campaign is closed at the qualified base-2 reconstruction image. There is no active kernel experiment waiting for promotion.

| Head dimension | Production source | Main topology | AITER matrix |
| ---: | --- | --- | ---: |
| 64 | `csrc/featherattn_bwd_fused_d64.cu` | Fused short path. Separate Delta, KV, and Q long path | `18/18`, `1.278536x` |
| 128 | `csrc/featherattn_bwd_fused_d128.cu` | Separate Delta, KV, and Q kernels | `18/18`, `3.034410x` |

Production qualification is complete:

| Check | D64 | D128 |
| --- | ---: | ---: |
| Public backward contract | Part of `44/44` combined pass | Part of `44/44` combined pass |
| NHD-to-HND dispatch equivalence | Bit-exact `dQ/dK/dV/Delta` | Bit-exact `dQ/dK/dV/Delta` |
| Forward regression contract | `188/188` combined pass | `188/188` combined pass |
| Minimum per-head cosine in rank-5 screen | `0.999999762` | `0.999999821` |
| Maximum Rel-L2 in rank-5 screen | `0.000297250` | `0.000297942` |

The retained production image includes the qualified base-2 reconstruction for both D64 and D128.

## Public Contract

```text
feather_attn_backward(dout, q, k, v, out, softmax_lse, dq=None, dk=None, dv=None, softmax_scale=None, delta=None, layout="NHD")
```

- Inputs are contiguous HND `[B,H,N,D]` or NHD `[B,N,H,D]` FP16 tensors.
- LSE is contiguous FP32 natural-log state in logical `[B,H,NQ]` order.
- The operation returns FP16 `dQ`, `dK`, and `dV` in the input layout and FP32 `Delta` in logical `[B,H,NQ]` order.
- D64 and D128 support independent arbitrary positive query and KV lengths within the existing signed-int32 dimension and addressability checks.
- Runtime tile counts and row guards handle tails. Matrix shapes are benchmark targets, not a support whitelist.
- There is no caller-visible workspace. Internal NHD transpose storage remains on the caller's current stream and allocation failure falls back to direct NHD execution.

### Source Map

| File | Role |
| --- | --- |
| `setup.py` | Packaged extension source list, CK build include, gfx1151 target, and ccache integration |
| `feather_attn/ops.py` | Public wrapper and bounded NHD-to-HND dispatch |
| `csrc/extension.cpp` | Limited-API Python module initializer |
| `csrc/hip_kernel.cpp` | Validation, layout policy, current-stream lookup, launch dispatch, and Torch registration |
| `csrc/hip_kernel.h` | Raw forward/backward launch ABI |
| `csrc/featherattn_bwd_fused_d64.cu` | D64 Delta helper and seven-GEMM kernels |
| `csrc/featherattn_bwd_fused_d128.cu` | D128 Delta, KV, and Q kernels |
| `bench/benchmark_attn_hip_backward.py` | Paired AITER/Feather benchmark for D64 and D128 |
| `tests/test_attn_hip_backward.py` | Saved-state correctness, layout, tail, and arbitrary-length coverage |

## Current Design

### Shared Algorithm

The complete backward path first computes `Delta = rowsum(dO * O)`. The main work is seven GEMM-equivalent operations: four in the KV-owned phase and three in the Q-owned phase.

| Owner | Work |
| --- | --- |
| KV | QK score, dO/V dP, dS/Q dK, and P/dO dV |
| Q | QK score, dO/V dP, and dS/K dQ |

Each output element has one owner. Gradients are accumulated in FP32 WMMA fragments and stored directly as FP16, so production has no gradient atomics, FP32 accumulation workspace, clear kernel, reduction kernel, or output conversion kernel. The FP32 C-to-FP16 A-fragment handoff uses lane permutes.

Both dimensions reconstruct probability with native base-2 exponentials:

```text
P = exp2(score * scale * log2(e) - LSE * log2(e))
```

Scale and lane-local LSE are converted once before repeated fragment reconstruction. The public saved LSE remains natural-log FP32.

### Launch and Layout Policy

| Selected path | Launch topology |
| --- | --- |
| D64 short or irregular direct path | Delta plus one 128-thread fused owner kernel |
| D64 long HND path | Delta, 256-thread/128-row KV, then 256-thread/128-row Q |
| D64 long direct-NHD path | Delta, 128-thread/64-row KV, then 256-thread/128-row Q |
| D128 direct path | Delta, 256-thread/64-row KV, then 512-thread/256-row Q |
| Selected NHD self-attention | Five FP16 input copies to HND, unchanged HND kernels, then three FP16 gradient copies to NHD |

D64 phase fission requires both `NQ` and `NKV` to be at least 4096. General NHD shapes use direct strided D-vector addressing. D64 direct NHD uses the validated grouped/head-fast/tile-fast owner-order policy. D128 direct NHD uses head-fast owner order and an NHD KV row-cache policy.

The exact HND-copy path is restricted to contiguous B1 self-attention:

| Dimension | Heads | Sequence length |
| ---: | --- | --- |
| D64 | `{32,56}` | `[8192,16384]` |
| D128 | `{16,32,56}` | `[4096,16384]` |

The eight temporary FP16 tensors require 128 MiB at D128 H16/N4096, 256 MiB at D64 H32/N8192, and up to 1,792 MiB at D128 H56/N16384. The copies and allocation are included in production timing.

### D64 Kernel Shape

| Property | Fused/Q | HND KV long | Direct-NHD KV long |
| --- | ---: | ---: | ---: |
| Workgroup | 128 fused, 256 Q | 256 | 128 |
| Owner rows | 64 fused, 128 Q | 128 | 64 |
| Main LDS | 13,312 B | 23,808 B | 15,616 B |
| Logical / allocated VGPRs | 178/192 fused, 166/168 Q | 169/192 | 169/192 |

D64 stages FP16 K/V in XOR-swizzled LDS rows, uses a 20-half transpose stride for K/Q/dO views, and uses a 72-half row stride for the long KV dO copy. The padded dO view preserves aligned vector loads while rotating rows across the 128-byte LDS bank cycle.

### D128 Kernel Shape

| Property | KV | Q |
| --- | ---: | ---: |
| Workgroup | 256 threads / 8 waves | 512 threads / 16 waves |
| Owner rows | 64 KV rows | 256 query rows |
| Ownership | Four waves each for dK and dV | Persistent FP16 Q/dO, FP32 dQ |
| Main LDS | 26,624 B direct/direct, 30,976 B direct-Q/cached-dO | 13,312 B |
| Logical / allocated VGPRs | 214/216 direct, 209/216 cached-dO | 246/256 |
| Effective residency | Six waves/SIMD | One workgroup/WGP, four waves/SIMD |

HND and NHD with at most 16 heads use direct Q/dO row reads in KV. Wider direct NHD uses direct Q and cached dO rows. The Q owner keeps exact FP16 Q and dO fragments in registers and streams one 16-row K/V tile at a time. D128 does not reuse D64's fused persistent-fragment topology because the mechanical lift spilled heavily.

### Linked Resource Contract

| Symbol | Logical / allocated VGPRs | SGPRs | LDS | Private/spills/scratch |
| --- | ---: | ---: | ---: | ---: |
| D64 HND fused | 178 / 192 | 48 | 13,312 B | 0 / 0 / 0 |
| D64 NHD fused | 178 / 192 | 52 | 13,312 B | 0 / 0 / 0 |
| D64 HND KV | 169 / 192 | 36 | 23,808 B | 0 / 0 / 0 |
| D64 NHD KV | 169 / 192 | 38 | 15,616 B | 0 / 0 / 0 |
| D64 HND/NHD Q | 166 / 168 | 33 / 38 | 13,312 B | 0 / 0 / 0 |
| D64 HND/NHD Delta | 66 / 72 | 10 / 14 | 0 B | 0 / 0 / 0 |
| D128 HND/NHD Delta | 81 / 96 | 10 / 14 | 0 B | 0 / 0 / 0 |
| D128 HND/NHD KV direct | 214 / 216 | 41 / 43 | 26,624 B | 0 / 0 / 0 |
| D128 NHD KV cached-dO | 209 / 216 | 43 | 30,976 B | 0 / 0 / 0 |
| D128 HND/NHD Q | 246 / 256 | 32 / 34 | 13,312 B | 0 / 0 / 0 |

Every production image has zero private bytes, zero spills, zero dynamic stack, and zero linked scratch load/store instructions.

## Benchmark and Validation Results

### Measurement Convention

Backward throughput uses the seven-GEMM convention:

```text
F7 = 7 * 2 * B * H * NQ * NKV * D FLOPs
TFLOPS = F7 / elapsed_seconds / 1e12
Feather / AITER = AITER elapsed time / Feather elapsed time
```

A ratio above `1.000x` means Feather is faster. The production matrix is B1, contiguous HND and NHD, equal Q/KV lengths, H `{16,32,56}`, and N `{4096,8192,16384}`. `bench/benchmark_attn_hip_backward.py` builds saved forward state outside timing, preallocates provider outputs and Delta, performs five warmups, and collects ten samples per provider in alternating order. AITER FlashAttention Triton backward is the control. All helper launches, internal temporary allocation, eight NHD transpose copies, synchronization, and provider work are inside the measured call.

The AITER column was re-measured with Triton `3.8.0+git669b31ac.rocm10.2.0a20261003` (LLVM 23 fork) and the tuned gfx1151 backward tiles (`~/aiter` `e483dc8b3`: non-causal D64 `16/64/16` ns1, D128 `16/16/16` ns1 below `HQ*N = 2e5` and `16/64/16` ns2 above it, causal `16/64/16`), changing launch knobs only. Raw rows: `~/tmp/feather_attn/attn_triton_bwd_tune/bwd_matrix_triton38.csv`, protocol file `bench_bwd_matrix.py`.

The same shape and tile drifts by up to about `20%` between sessions, and the largest D128 rows take about `4 s` per call. Re-measuring Feather against this AITER column is a separate task.

A Triton or LLVM update changes the generated code for these kernels, and therefore the winning tile, so the gfx1151 backward tiles must be re-tuned after a Triton upgrade. The previous AITER column came from the August stack, and the tuned tiles land at `0.9667x` of it (`0.9478x` at D64, `0.9859x` at D128): the June gfx1151 backward tile only existed on the `anima` branch, so the tree that produced the previous column and today's tree differ in the shipped tile as well as in the compiler.

| Dimension/layout | Rows won | Geometric Feather / AITER | Current Feather TFLOPS range |
| --- | ---: | ---: | ---: |
| D64 HND | 9/9 | `1.273882x` | 29.666-31.490 |
| D64 NHD | 9/9 | `1.283207x` | 24.612-29.627 |
| D64 combined | 18/18 | `1.278536x` | 24.612-31.490 |
| D128 HND | 9/9 | `3.136055x` | 21.695-23.423 |
| D128 NHD | 9/9 | `2.936060x` | 16.828-20.673 |
| D128 combined | 18/18 | `3.034410x` | 16.828-23.423 |

Against the frozen topology immediately before base-2 reconstruction, rank 5 is `1.008737x` for D64 and `1.005163x` for D128 over the complete 18-row candidate/control matrices.

### D64 Production Matrix

Artifacts: `~/tmp/feather_attn/postprod_bwd_campaign/exp5f_d64_kv_q_exp2/aiter_matrix.json` (previous AITER column) and `~/tmp/feather_attn/attn_triton_bwd_tune/bwd_matrix_triton38.csv` (AITER column re-measured with Triton 3.8).

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 26.143 | 29.666 | 1.135x |
| HND | 16 | 8192 | 25.230 | 30.800 | 1.221x |
| HND | 16 | 16384 | 24.349 | 31.032 | 1.274x |
| HND | 32 | 4096 | 24.575 | 30.426 | 1.238x |
| HND | 32 | 8192 | 24.053 | 30.927 | 1.286x |
| HND | 32 | 16384 | 23.356 | 30.758 | 1.317x |
| HND | 56 | 4096 | 23.387 | 31.490 | 1.346x |
| HND | 56 | 8192 | 23.249 | 30.734 | 1.322x |
| HND | 56 | 16384 | 22.843 | 30.630 | 1.341x |
| NHD | 16 | 4096 | 22.863 | 28.792 | 1.259x |
| NHD | 16 | 8192 | 22.984 | 29.138 | 1.268x |
| NHD | 16 | 16384 | 22.795 | 28.383 | 1.245x |
| NHD | 32 | 4096 | 21.034 | 27.706 | 1.317x |
| NHD | 32 | 8192 | 21.589 | 24.612 | 1.140x |
| NHD | 32 | 16384 | 20.779 | 26.985 | 1.299x |
| NHD | 56 | 4096 | 22.113 | 29.627 | 1.340x |
| NHD | 56 | 8192 | 21.807 | 27.737 | 1.272x |
| NHD | 56 | 16384 | 20.225 | 28.880 | 1.428x |

### D128 Production Matrix

Artifacts: `~/tmp/feather_attn/postprod_bwd_campaign/exp5c_d128_kv_q_exp2/aiter_matrix.json` (previous AITER column) and `~/tmp/feather_attn/attn_triton_bwd_tune/bwd_matrix_triton38.csv` (AITER column re-measured with Triton 3.8).

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 8.380 | 22.793 | 2.720x |
| HND | 16 | 8192 | 7.536 | 23.423 | 3.108x |
| HND | 16 | 16384 | 7.001 | 22.821 | 3.260x |
| HND | 32 | 4096 | 7.673 | 22.683 | 2.956x |
| HND | 32 | 8192 | 7.224 | 22.579 | 3.126x |
| HND | 32 | 16384 | 6.851 | 22.310 | 3.256x |
| HND | 56 | 4096 | 7.056 | 21.695 | 3.075x |
| HND | 56 | 8192 | 6.121 | 21.964 | 3.589x |
| HND | 56 | 16384 | 6.839 | 21.934 | 3.207x |
| NHD | 16 | 4096 | 6.918 | 17.010 | 2.459x |
| NHD | 16 | 8192 | 6.893 | 18.337 | 2.660x |
| NHD | 16 | 16384 | 6.303 | 19.892 | 3.156x |
| NHD | 32 | 4096 | 6.353 | 16.828 | 2.649x |
| NHD | 32 | 8192 | 6.327 | 18.403 | 2.909x |
| NHD | 32 | 16384 | 5.820 | 20.074 | 3.449x |
| NHD | 56 | 4096 | 6.712 | 18.139 | 2.703x |
| NHD | 56 | 8192 | 5.551 | 19.633 | 3.537x |
| NHD | 56 | 16384 | 6.687 | 20.673 | 3.091x |

### Correctness Coverage

The `44/44` public backward screen covers both dimensions, H `{16,32,56}`, batch two, long-path selector boundaries, and asymmetric lengths through 16,385, including `(4095,4097)`, `(4097,4099)`, `(8191,67)`, `(65,8193)`, `(16383,67)`, and `(65,16385)`. Dedicated H32/N8192 checks require bit-exact selected-HND and direct-NHD outputs. Private candidate screens additionally cover small owner boundaries, independent tails, cancellation, and lengths such as `(65,64)`, `(129,65)`, `(257,257)`, and `(512,513)`.

## Profile Results and Bottlenecks

The latest phase traces make KV the primary optimization target in both dimensions. Diagnostic dispatch times are approximately:

| Shape | Dimension | Delta | KV | Q |
| --- | ---: | ---: | ---: | ---: |
| H16/N4096 HND | 64 | small relative to main phases | 5.11 ms | 3.39 ms |
| H16/N4096 HND | 128 | 0.30 ms class | 17.32 ms | 6.77 ms |

Profiler timestamps are phase-attribution evidence, not promotion timing. The counter passes add instrumentation and are not substituted for complete paired provider timing.

The serialized D64 counter passes report:

| Capture | Phase TFLOPS | Occupancy | SIMD use | Wait-any / wave cycles | FLOP / normalized byte | Read + modeled output GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H16/N4096 KV | 28.040 | 47.46% | 99.25% | 23.58% | 2,520 | 11.13 |
| H16/N4096 Q | 32.809 | 52.28% | 97.20% | 16.57% | 2,038 | 16.10 |
| H32/N8192 KV | 27.648 | 49.14% | 99.24% | 22.69% | 4,372 | 6.32 |
| H32/N8192 Q | 32.834 | 55.63% | 99.68% | 16.78% | 2,752 | 11.93 |

The byte normalization adds profiler `FETCH_SIZE * 1024` to minimum logical FP16 output stores. It is not a measured bidirectional DRAM rate. Intensity is `2,038-4,372` FLOP/byte, far above the approximately 232 FLOP/byte nominal ridge point, while normalized traffic is only `6.3-16.1 GB/s`. Nominal DRAM bandwidth is therefore not the active limit.

Current profile interpretation:
- D64 KV exposes about 23.6% wait-any, 7.5% barrier wait, and 133-135-cycle LDS latency. Q is faster per GEMM-equivalent and has lower wait exposure.
- D128 KV shows about 63.6% LDS conflicts, 276-cycle LDS latency, and 24.5% barrier wait. D128 Q has low LDS stalls but 15-22.5% barrier wait.
- SIMD utilization is near full while occupancy is roughly half maximum. Instruction issue, LDS dependencies, and barrier serialization dominate. Generic bandwidth reduction is not a sufficient optimization thesis.
- Direct D128 NHD has pathological strided traffic and is a fallback path. Promotion measurements must use the complete selected provider path, including HND transposes where policy selects them.
- The post-allocation D64 snapshot peaks at 163 simultaneously live physical VGPRs while naming `VGPR168`. Allocation holes and contiguous tuple placement, not 169 simultaneous values, explain the 192-VGPR KV tier. Forced register caps are therefore not a credible optimization by themselves.

## Optimization Outcomes

### Accepted in Production

| Mechanism | Evidence and retained policy |
| --- | --- |
| Direct-output seven-GEMM ownership | Replaced FP32 workspaces, atomics, clears, reductions, and conversion launches. Unique owners store final FP16 gradients directly. |
| Exact C-to-A fragment handoff | FP32 P/dS and accumulators remain exact until the FP16 WMMA input handoff. Lane permutes avoid persistent P/dS storage. |
| D64 long-path phase fission | Q falls from the 192- to 168-VGPR tier. A 120-pair confirmation across H16/H32 and N4096/N8192 is `1.01264x` with 48/48 block wins. Short rows remain fused. |
| D64 padded dO row stage | A 72-half stride avoids the conflicts of stride 64. Linked complete timing is `1.015714x` geometric over the frozen fission image with 48/48 block wins. |
| Native HND and NHD contracts | Layout-aware Delta and seven-GEMM images preserve arbitrary lengths and pass the full dual-layout contract without layout-visible workspace. |
| NHD owner ordering | D64 retains only validated grouped/head-fast policies. The uneven-group quotient/remainder fix prevents invalid heads. D128 uses head-fast order, which reduced H56/N4096 direct NHD from 278.20 to 157.90 ms in the development image. |
| Bounded NHD-to-HND execution | Selected B1 self-attention shapes use exact transposes. D64 moved from 16/18 to 18/18 AITER row wins. D128 also wins every matrix row with all copy/allocation cost included. |
| D64 128-row Q owner | The 256-thread Q image remains 166/168 VGPRs and zero-spill. Its candidate/control matrix is `1.011283x`. |
| D64 wide HND KV owner | The 256-thread/128-row image is retained only for HND and transpose-routed NHD. The complete matrix improved `1.010580x`. Direct NHD keeps the narrower owner because the universal form regressed. |
| D128 independent KV/Q topology | Separate dK and dV wave groups avoid carrying both eight-fragment FP32 output banks. This replaced the mechanically lifted D64 topology that spilled. |
| D128 row-cache policy | Direct Q rows are used everywhere. dO rows are cached only for wider direct NHD. The selected images remain in the 216-VGPR tier with zero scratch. |
| D128 persistent Q and dO | Keeping exact FP16 Q/dO fragments reduced repeated VMEM traffic enough to win despite moving Q to the 256-VGPR tier. H16/H32 N4096 improved from 25.01/49.75 to 24.54/48.55 ms when dO persistence was added. |
| D128 512-thread/256-row Q owner | Halves block count and K/V staging relative to the 256-thread form. It remains zero-spill and reaches about 20.1-20.7 normalized TFLOPS in the topology campaign. |
| D128 256-thread/64-row KV owner | Four waves each own dK and dV. The complete candidate/control matrix is `1.101567x`, with 18/18 rows and 54/54 timing blocks won. |
| Base-2 reconstruction | Preserves all resource tiers and exact public behavior. D64 gains `1.008737x`. D128 gains `1.005163x` over their frozen topology controls. |

### Useful Supporting Results, Not Production Features

| Finding | Disposition |
| --- | --- |
| Raw high-byte E5M2 truncation | All 65,536 encodings match `half_bits >> 8`. Finite magnitude, sign, zero, and infinity behavior were verified. This establishes a primitive, not a profitable backward representation. |
| Two-permute E5M2 pack | Bit-exact and faster than scalar pack/decode in its isolated fixture, with 15 versus 16 logical VGPRs. Complete P/dS kernels still lost. |
| Phase-local kernel extraction | A dedicated KV source generated the same image as compile-time `kPhase == 1`, confirming that phase specialization already removes dead state. |
| MIR pressure analysis | Distinguished optimized virtual pressure, linked allocation, and physical simultaneous liveness. This invalidated source-count and forced-cap arguments that did not change final scheduling. |

### Rejected Topology and Scheduling Work

| Candidate | Decisive result |
| --- | --- |
| Mechanical D64-to-D128 lift | KV spilled 5-13 values and Q spilled 265-268 values with 24-564 B private memory. Rejected before timing. |
| D128 Q stage of 32 K/V rows | The unrolled form spilled 32 VGPRs/68 B. The rolled zero-spill form reached only `1.000558x` at H32/N8192 with CI `[0.999812,1.001292]`, while Q phase latency was worse. |
| D128 persistent dO without Q | Same 216-VGPR tier as persistent Q but consistently slower: 25.19/50.01 ms versus 25.01/49.75 ms at H16/H32 N4096. |
| Fold Delta into D128 Q | Removed a launch but produced 16 spills, 36 B private memory, and 12 scratch instructions. Rejected before timing. |
| Share D128 P from dK to dV waves | Saved score WMMA/exponentials but added LDS, a barrier, branch serialization, and reduced residency. Both barrier placements regressed complete timing to 26.22-26.76/51.23-52.31 ms. |
| D64 KV dV/dP interleave | Same allocation tier, larger code, and `0.984856x` geometric with 0/36 block wins. |
| D64 trailing wave-barrier removal | `0.999392x`, 17/36 block wins, and every row interval crossed one. No repeatable gain. |
| D64 ping-pong Q/dO staging | LDS rose to 23,040 B and the six-row result was `0.958193x` with 0/36 block wins. |
| D64 combined Q-row and dO-row staging | Bit-exact but only `1.0014x` versus fission and slower than dO-only at `1.0205x` in the focused comparison. |
| NHD long-phase fusion | Focused NHD result was `0.956310x` geometric against AITER, including severe H32 long-row regressions. |
| Blanket NHD LLC grouping | Non-periodic H56 strides regressed. Grouping remains restricted to the periodic cases that qualified. |
| Alternate NHD group counts | H32/N16384 counts 5 and 3 reached only `0.904842x` and `0.911481x` against AITER. Forward-derived count 4 remains selected. |
| Initial equal-size uneven grouping | Generated an illegal head when group count did not divide H. No timing result was admitted. Quotient/remainder grouping replaced it. |
| Paired-ownership and older five-GEMM variants | Either small local gains or extra launch/reduction cost failed complete-path gates and were superseded by direct seven-GEMM ownership. |

### Rejected Memory, LDS, and Address Work

| Candidate | Decisive result |
| --- | --- |
| D64 KV probability recomputation | Raised KV from 175 to 184 logical VGPRs and regressed H16/N4096 by 2.98%. |
| D64 full K-row reload | Reached the 168-VGPR tier but repeated all K loads per Q tile and regressed by 3.34%. |
| D64 partial K cache | Cache-two reached 162/168 VGPRs but lost every H16/H32 x N4096/N8192 block. Geometric result `0.9766x`. |
| Universal D128 direct dO reload | Improved HND but collapsed tile-fast H56 NHD to 330.47 ms versus AITER's 244.87 ms. Retained policy is layout/head-count specific. |
| D64 Q-row stage | A focused sample won, but the image rose to 192 logical VGPRs and 15,360 B LDS and was inferior to dO-only staging. |
| Linear 64-half dO stage | Severe LDS conflicts regressed H16/N4096 from 8.324604 to 11.090252 ms. |
| dO strides beyond 72 | Strides 80-120 increased LDS and all focused estimates were neutral or negative. Stride 88 confirmed at `0.999945x` with all intervals crossing one. |
| Precomputed KV tile bases | Removed 76 static bytes but changed no resource tier. Six-row result `0.999524x` with 13/36 block wins. |
| Direct gfx1151 global-to-LDS | gfx1151 disables the read-only `vmem-to-lds-load-insts` feature and rejects both direct encodings. The same fixture emits `global_load_dword ... lds` for gfx1030. |
| Six-GEMM N-squared workspace | Deferred/rejected for current topology because the reference shape needs about 256 MiB intermediate storage and 512 MiB extra traffic. |

### Rejected Precision and Arithmetic Work

| Candidate | Decisive result |
| --- | --- |
| FP8 P-only truncation | Directionally valid but added 68 static instructions and 10 permutes, stayed in the 192-VGPR tier, and measured `0.979607x` over six rows. |
| FP8 dS-only truncation | Directionally valid but added 21 instructions and 18 permutes, changed no useful tier, and measured `0.996503x` with no row wins. |
| FP8 P+dS truncation | Failed the direction gate: minimum flattened cosine `0.996022` and per-head cosine `0.994891`. Still allocated 192 VGPRs. |
| Transient P/dS pack then decode | Removed no traffic or allocation class and added software conversion work. |
| Fixed, power-of-two, and linear FP8 scaling | Conversion and register cost did not create a profitable resource transition on gfx1151, which lacks native E5M2 conversion/WMMA support. |
| Native FP16-output WMMA | FP16 and FP32 output fragments both occupy eight VGPRs on gfx1151, so it cannot halve dK/dV accumulator storage. |
| Precomputed log2 LSE workspace | Bit-exact and shortened KV/Q, but per-call allocation and workspace traffic produced `0.796521x` and `0.952446x`. All 12 focused blocks lost. |

### Rejected Resource Controls and Legacy Paths

| Candidate | Decisive result |
| --- | --- |
| Nine-wave launch bound | LLVM's final eight-wave occupancy could not satisfy `__launch_bounds__(128,9)`. |
| Forced 168-VGPR attribute | Changed reported logical counts but not the 192-VGPR campaign class. Naturally allocated dO-stage timing remained better. |
| Old atomic FP32 D64 kernel | Removed completely, including selector, workspace, and linked image. |
| `owned`, `paired`, automatic selection, and duplicate references | Removed after losing frozen complete rows or becoming redundant. |
| Stock AITER CK backward | Blocked because the default build receipt omits gfx1151. Retained only as external evidence, not a Feather kernel candidate. |

## Remaining Work

There is no active optimization required for the current gfx1151 topology. The ranked campaign is closed at the qualified D64/D128 base-2 image. Routine work is limited to requalification when the kernel, compiler, runtime, dispatch policy, or target matrix changes.

Potential work is conditional on new evidence:

| Trigger | Work that becomes justified |
| --- | --- |
| A new profile shows materially different issue/LDS/barrier exposure | Reopen scheduling or barrier-overlap work with a mechanism not equivalent to the failed interleave, barrier-removal, or ping-pong trials. |
| A new topology naturally crosses a VGPR/LDS occupancy boundary | Revisit persistent K/V compression or caching. Require the byte reduction to repay conversion and to improve complete timing. |
| A new architecture exposes legal VMEM-to-LDS instructions | Re-run the ISA microfixture, then test one row-major V stage. gfx1151 itself is closed. |
| NHD transpose memory becomes a product constraint | Investigate a lower-footprint exact path only if it preserves arbitrary-shape fallback and improves complete selected-provider behavior. |
| AITER CK adds a valid gfx1151 receipt | Re-run the external control matrix. This does not by itself require a Feather kernel change. |
| ROCm code generation changes | Rebuild exact linked images, inspect resource tiers/ISA, and rerun full qualification before claiming either a regression or a free gain. |

Do not reopen generic bandwidth work, forced VGPR controls, direct P/dS compression, precomputed-LSE allocation, unsupported global-to-LDS encodings, K reload/cache variants, or Q-row staging without a changed assumption that invalidates the recorded rejection.

## Optimization and Profiling Methodology

### Promotion Standard

A production candidate must preserve the public HND/NHD contract, FP16 inputs/outputs, natural-log FP32 LSE, raw ABI, current-device checks, and current-stream semantics. Experiments remain private and compile-time until qualified. No public selector is added for a tuning trial.

The evidence required for promotion is:
- Production-equivalent linked gfx1151 code from `clang -O3 -fno-gpu-rdc --offload-arch=gfx1151`.
- Exact-symbol metadata and disassembly showing the intended code change.
- No regression in the relevant VGPR/LDS allocation tier and zero private memory, spills, dynamic stack, and scratch instructions.
- Correct owner-boundary, tail, asymmetric, layout, and numerical behavior.
- Positive repeated paired timing for the complete selected provider path, not just a phase, fixture, static instruction count, or isolated sample.
- A complete H `{16,32,56}` x N `{4096,8192,16384}` dual-layout screen with no material row regression.
- The combined `44/44` backward contract, exact dispatch equivalence, and `188/188` forward regression contract before integration.

Stop at the first decisive resource, correctness, ISA, or focused-timing failure. Preserve the artifact and restore production rather than carrying a failed selector or workspace path.

### Timing Practice

- Freeze and hash both baseline and candidate code objects.
- Alternate baseline and candidate in repeated blocks to control drift.
- Retain raw samples, per-block ratios, geometric row summaries, and paired bootstrap 95% intervals. Small gains are admissible when repeated evidence is credible. There is no arbitrary minimum percentage gate.
- Include Delta, all main/helper launches, synchronization, allocation, conversions, copies, and fallback policy in complete provider timing.
- Use phase-local H16/N4096 and H32/N8192 traces only to attribute cost and reject candidates early.
- Normalize full backward timing with `F7`. Use `4/7 * F7` for KV and `3/7 * F7` only for diagnostic phase throughput.

### Accuracy Policy

| Tier | Required evidence |
| --- | --- |
| Directional candidate | Finite outputs and minimum flattened cosine at least `0.997` for dQ/dK/dV |
| Preferred numerical candidate | Directional tier plus maximum Rel-L2 at most `0.10` and reported norm-ratio error |
| Exploratory numerical candidate | Directional tier plus maximum Rel-L2 below `0.20`. Never a silent API change |
| Exact scheduling/layout candidate | Preserve the production approximately `0.0003` Rel-L2 behavior and public elementwise contract |

Per-head cosine is reported in addition to flattened cosine because a strong aggregate can conceal a weak head. Approximate precision work must first clear the direction gate, then justify its error with a complete speed/resource win.

### Evidence Hierarchy

| Question | Primary evidence | Why |
| --- | --- | --- |
| Which kernel ran? | `rocprofv3` trace/counter CSV | Provides exact demangled name, dispatch ID, grid, workgroup, and phase |
| What image shipped? | Linked gfx1151 code-object metadata | Establishes ABI, wave/workgroup limits, logical registers, LDS/private bytes, spills, and stack use |
| What did the compiler emit? | Exact-symbol linked disassembly | Establishes instruction sequence, memory widths, WMMA, waits, barriers, lane operations, branches, calls, and scratch |
| Why is allocation high? | Optimized MIR plus LLVM liveness/pressure analysis | Locates virtual lifetimes and pressure at specific blocks and instructions |
| How are physical registers used? | Post-allocation MIR correlated to linked ISA | Reveals simultaneous liveness, holes, tuple placement, and highest-register use |
| What limits occupancy or traffic? | Metadata plus explicit architecture/algorithm models | Separates rounded resources, theoretical occupancy, modeled bytes, and phase FLOPs |
| Did it help users? | Complete alternating paired benchmark | Measures the selected provider rather than a proxy |

Source filenames and template stems are not kernel identities. Match the exact mangled symbol observed at runtime to one extracted gfx1151 bundle, and compare candidate/control at the same symbol boundary. `llvm-readelf`, not GNU `readelf`, decodes the AMDGPU MessagePack metadata in this toolchain.

### Compiler and Resource Analysis

Record the repository state, generated translation-unit hash, compiler command, tool versions, target ID, extension hash, and shape. Obtain the actual command with `ninja -t commands`. Do not reconstruct defines and include paths by hand. The linked metadata and linked ISA are authoritative for final resources and code generation.

The installed ROCm LLVM 23 package has no standalone `llc`. Optimized MIR is emitted through clang from saved optimized gfx1151 device IR. The `phi-node-elimination` snapshot is the useful virtual-pressure input here. Later pre-RA MIR is a schedule cross-check and post-allocation MIR is a physical-placement diagnostic.

Never compare virtual-register numbers across separately generated MIR files. Correlate by exact function, machine block, slot neighborhood, opcode/operand shape, and final ISA. Request every physical register unit explicitly and group gfx1151's low/high 16-bit units into one VGPR. Keep these quantities distinct:
- Source-level or virtual pressure.
- Logical VGPR/SGPR counts in metadata.
- Allocation-rounded hardware tiers.
- Simultaneously live physical registers.
- Theoretical and profiler-observed occupancy.

Static models remain hypotheses until validated. Derive lane-level LDS addresses and bank mapping, then confirm transaction widths/offsets in `ds_*` ISA and dynamic conflict/latency counters. Derive compulsory bytes and algorithmic FLOPs independently of profiler traffic. A source reorder counts only when linked instructions change and dynamic exposure or complete timing improves.

### Profiler Limitations

gfx1151 cannot schedule the complete counter set in one `rocprofv3` pass. Core, traffic, cache, LDS, barrier, and stall groups are collected in separate legal passes against the same symbol and shape. Wait counters overlap and must not be added. `rocprof-compute` is unavailable in the current environment.

`FETCH_SIZE` includes cache/memory effects but is not a write counter. Any reported read-plus-output rate must label output stores as modeled. Profiler timestamps and static waits establish attribution and placement, not an end-to-end speedup. Only the paired provider benchmark admits a change.

### Reproducible Artifact Bundle

Retain one directory per serious candidate containing the provenance manifest, runtime identity row, baseline/candidate hashes, extracted code object, metadata, symbol table, exact-symbol disassembly, saved device IR, relevant MIR, pressure reports, static-model inputs, correctness output, and paired timing. This is the minimum evidence needed to reproduce a compiler, ISA, numerical, or performance claim after the build tree changes.

## Artifact Index

### Production and Qualification

- D64 rank-5 qualification: `~/tmp/feather_attn/postprod_bwd_campaign/exp5f_d64_kv_q_exp2/`.
- D128 rank-5 qualification: `~/tmp/feather_attn/postprod_bwd_campaign/exp5c_d128_kv_q_exp2/`.
- Frozen post-rank-4 baseline: `~/tmp/feather_attn/postprod_bwd_campaign/baseline_after_exp5_d64/`.
- Final D128 production matrix: `/tmp/feather_bwd_d128_final_matrix_production.json`.
- Post-D128 D64 regression matrix: `/tmp/feather_bwd_d64_regression_matrix_after_d128.json`.
- Linked D128 inspection: `/tmp/feather_bwd_d128_final_inspect/`.
- Linked D64/NHD inspection: `/tmp/feather_bwd_nhd_final_inspect_20260818/`.
- Forward matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.

### Retained Mechanisms

- D64 phase fission: `~/tmp/feather_attn/phase_fission_campaign/`.
- D64 padded dO stage: `~/tmp/feather_attn/kv_dorow_stride72_campaign/`.
- D64 Q owner: `~/tmp/feather_attn/postprod_bwd_campaign/exp2_d64_q_owner128_wg256/`.
- D64 HND KV owner: `~/tmp/feather_attn/postprod_bwd_campaign/exp3b_d64_kv_hnd_owner128_wg256/`.
- D128 KV owner: `~/tmp/feather_attn/postprod_bwd_campaign/exp4_d128_kv_owner64_wg256/`.
- Isolated base-2 screens: `~/tmp/feather_attn/postprod_bwd_campaign/exp5a_d128_kv_exp2/`, `exp5b_d128_q_exp2/`, `exp5d_d64_kv_exp2/`, and `exp5e_d64_q_exp2/` under the same campaign root.

### Rejected Campaigns

- D128 Q stage 32: `~/tmp/feather_attn/postprod_bwd_campaign/exp1b_d128_q_stage32_rolled/`.
- Universal D64 wide KV: `~/tmp/feather_attn/postprod_bwd_campaign/exp3_d64_kv_owner128_wg256/`.
- Precomputed log2 LSE: `~/tmp/feather_attn/postprod_bwd_campaign/exp6a_d128_lse_log2/`.
- gfx1151 global-to-LDS probe: `~/tmp/feather_attn/postprod_bwd_campaign/exp7_gfx1151_global_to_lds_probe/`.
- FP8/E5M2 work: `~/tmp/feather_attn/e5m2_truncation_campaign/`.
- KV recomputation/reload: `~/tmp/feather_attn/kv_recompute_campaign/`, `kv_reload_campaign/`, and `kv_partial_cache_campaign/`.
- Row staging: `~/tmp/feather_attn/kv_row_stage_campaign/` and `kv_dorow_linear_campaign/`.
- Schedule/address trials: `/tmp/feather_bwd_kv_interleave_campaign/`, `/tmp/feather_bwd_kv_stride_sweep_campaign/`, `/tmp/feather_bwd_barrier_simple_campaign/`, `/tmp/feather_bwd_barrier_campaign/`, and `/tmp/feather_bwd_address_campaign/`.
- Launch/VGPR diagnostics: `~/tmp/feather_attn/kv_dorow_stride72_lb9_campaign/` and `kv_dorow_stride72_vgpr168_campaign/`.

### Profiling and Compiler Analysis

- Roofline normalization: `~/tmp/feather_attn/current_roofline.json`.
- Serialized D64 counters: `~/tmp/feather_attn/d64_kv_profiles/summary.pretty.json`.
- Saved compiler outputs: `~/tmp/feather_attn/compiler_inspect/current/` and `current_device.ll`.
- Virtual pressure: `~/tmp/feather_attn/compiler_inspect/regalloc/pressure_O3_phi_phase1_v4.txt` and `pressure_O3_prera_phase1_v4.txt`.
- Physical pressure: `~/tmp/feather_attn/compiler_inspect/regalloc/pressure_O3_regallocfast_phase1_grouped.txt`.
- MIR/ISA tooling: `~/tmp/feather_attn/extract_mir_function.py`, `analyze_feather_pressure.cpp`, and `analyze_final_isa.py`.
