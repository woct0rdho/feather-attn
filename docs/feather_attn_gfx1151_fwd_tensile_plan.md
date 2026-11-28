# FeatherAttn gfx1151 bf16 D128 forward: assembly-generator plan

## Goal

Take the bf16 D128 forward from the shipped HIP kernel's 31-35 TFLOPS to 45-47 TFLOPS on the HND rows by generating gfx1151 assembly from Python, in the shape of `~/torch-ggml-ops/docs/ggtensile_plan.md` (GGTensile) rather than Tensile or TensileLite, whose writers are GEMM shapes with no place for a softmax.

The generator is not the point. The two things that are out of reach of HIP source are: a schedule the arbiter will actually issue the way we want, and a register count we choose rather than discover. Everything else is reachable from HIP first, which is why this plan starts with two HIP rounds.

## What bounds the shipped kernel

Measured on this machine, gfx1151 at 2.87 GHz, random bf16, probes in `~/tmp/feather_attn/asm_fwd/`:

| Measurement | Result |
| --- | ---: |
| WMMA ceiling, register operands, no loads | 54.0-54.4 TFLOPS (91% of the 59.4 spec) |
| `v_add_f32` alongside WMMA | 32 per WMMA cost 6.7 cycles (VOPD, partly hidden). Beyond ~64 per WMMA, ~0.9 cycle each |
| `v_exp_f32` alongside WMMA | 4.25 cycles each, quarter rate, zero overlap |
| `v_exp_f16` | ~2 cycles per element (plus ~1 for an f16->f32 conversion) |
| degree-2 polynomial exp2, 8 VALU, 8 chains | 9.2 cycles per element: 2x worse than `v_exp_f32` |
| `v_perm_b32` / `v_permlanex16` alongside WMMA | 1.1 / 1.0 cycle each |
| dependent add chain (the row sum) | 5.7 cycles per link standalone. Four 32-link chains cost +68 cycles per 4 waves in the loop |
| LDS fragment read, 16 lanes x 32 B, lanes 16-31 duplicating, 4-deep | 234 GB/s per WGP at 144 B / 272 B row pitch. 177-209 GB/s at a contiguous 32 B row. 46 GB/s at 136/148/276 B (16 B-misaligned) or >= 256 B power-of-two pitches |
| the same read with only lanes 0-15 active | exactly half the bytes: duplicate lanes are not merged, they pay full bandwidth |
| 1 LDS fragment per WMMA, prefetch depth 1 / 2 / 4 | 19.9 / 26.7 / 23.8 TFLOPS at 4 waves/SIMD. 29.6 / 40.3 / 34.6 at 8 waves/SIMD |
| 2 WMMAs per LDS fragment, depth 2 | 44.9 (4 waves/SIMD) / 49.4 (8 waves/SIMD) |
| 3 WMMAs per fragment, depth 2 | 49.8 / 51.2 |

Two ceilings stack on the shipped dataflow:
- LDS fragment bandwidth. A 16x16 fragment is read as two `ds_load_b128` whose lanes 16-31 duplicate lanes 0-15, so 1 KB is read per WMMA for 512 B of data. At the register-only WMMA rate that is 330 GB/s per WGP. The LDS delivers 234 GB/s for this access shape, which caps the dataflow at ~40 TFLOPS with the best schedule tried and 20-30 with the depth-1 schedule the compiler emits. The shipped kernel at 31-35 is at 85-90% of the depth-2 ceiling, which is why removing the mask or the E5M2 Q tile bought almost nothing.
- Non-WMMA issue slots, which share the WMMA resource. Per KV tile the loop carries 590-740 non-WMMA instructions, of which the 32 `v_exp_f32` alone are 136 cycles.

The compiler's schedule, from the disassembly of config 1 (`k1_hnd_bc64_swz.s`): `ds_load` x2 -> `s_waitcnt lgkmcnt(0)` -> `v_wmma`, 64 times per tile. A serial 32-link `v_add_f32` sum chain with `s_delay_alu` between links. 16 `v_readlane` + 64 `v_mul` for the accumulator rescale. Four 64-bit `v_mad_u64_u32` per tile for a row clamp that only matters on the last tile. Three `v_xor`/`v_add` per V fragment for the swizzle. 327 `s_delay_alu` per tile, and a full LDS round trip exposed at both phase boundaries, with all waves of a SIMD in lockstep.

## What the hardware gives, and what gfx1151 does not have

Probed with the target assembler and from HIP (`~/tmp/feather_attn/tensile_probe/`):

| Feature | gfx1151 |
| --- | --- |
| `global_load_lds_b32/b64/b128`, `buffer_load ... lds:` | absent in every spelling: global to LDS always goes through VGPRs |
| `v_mfma_*` | absent (no matrix cores) |
| `v_pk_add_bf16`, `v_pk_fma_bf16`, `v_pk_mul_bf16` | absent (no packed bf16 arithmetic) |
| `v_cvt_pk_bf16_f32` | absent |
| `v_cvt_pk_f16_f32` (round to nearest) | absent |
| `v_cvt_pk_rtz_f16_f32` | present, and reachable from HIP as `__builtin_amdgcn_cvt_pkrtz` |
| `v_pk_fma_f16`, `v_pk_mul_f16`, `v_pk_max_f16`, `v_cvt_f16_f32` | present (`__hfma2` and friends) |
| `v_exp_f16` | present (`h2exp2` lowers to two `v_exp_f16_e32`). `v_pk_exp_f16` does not exist |
| `v_dot2_f32_bf16`, `v_dot2_f32_f16` | present |
| VOPD `v_dual_fmac_f32`, `v_dual_mul_f32 :: v_dual_add_f32` | present. TensileLite exploits this as `UseDualFMAC` for RDNA3/3.5 `f32` source kernels |
| `v_max3_f32`, `v_min3_f32`, `v_med3_f32` | present (already used) |
| `ds_read2_b128` | absent. `ds_read_b64_tr_b16` (LDS transpose) absent |
| `s_setprio`, `s_sleep`, `s_wakeup`, `s_clause`, `s_waitcnt vmcnt(n) lgkmcnt(n)` | present. `s_setprio` and `s_waitcnt` are callable from HIP (`__builtin_amdgcn_s_setprio`, `__builtin_amdgcn_s_waitcnt`) |
| `s_barrier_signal` / `s_barrier_wait` | absent (gfx12 only) |
| `s_memtime` / `s_memrealtime` | absent: no wave-level timestamps |

ISA facts that drive the design:
- The arbiter issues one instruction per cycle per SIMD from the highest-priority ready wave and does not check operand readiness. A dependent instruction issued too early stalls the ALU for every wave, which is why `s_delay_alu` exists. Readiness is the program's job.
- `s_setprio`: arbiter key is `{min(3, SysPrio + UserPrio), WaveAge}`. User priority 0-3 is settable per wave at any time. LLVM has a pass for it (`AMDGPUSetWavePriority.cpp`, off by default, `-mllvm -amdgpu-set-wave-priority`).
- `s_sleep` / `s_wakeup`: a wave sleeps roughly 64 x N clocks and any wave of its workgroup can wake it. The ISA text describes exactly an LDS-flag producer/consumer ring without `s_barrier`.
- `s_clause` locks one instruction type to one wave for 2-63 instructions "even if that leaves the execution hardware idle". LLVM emits `s_clause 0x1` on every LDS pair and global-load pair of the shipped kernel.
- `s_waitcnt`: VMcnt and LGKMcnt are 6-bit instruction counts, and LDS instructions complete in order within a wave, so `lgkmcnt(N)` is an exact "all but the newest N" wait. This is the knob that buys the depth-2 pipeline.
- WMMA ignores EXEC and requires lanes 16-31 to replicate lanes 0-15 for A and B: a 16-row fragment read is 1 KB raw for 512 B of data.
- `v_permlanex16` always reads the alternate 16-lane row (`altrow`), so the P exchange is one VALU op per word and never touches LDS, at a measured 1.0 cycle per instruction next to WMMA. `v_perm_b32` follows the documented byte convention (LLVM emits `v_perm_b32 ... 0x5040100` in our own loop).
- LDS: 64 banks of 512 four-byte entries per WGP, 128 KB, two halves that run in parallel in CU mode. One workgroup may use 64 KB.
- Cache bits: GLC=1 forces an L1 miss, SLC=1 is stream/non-temporal in L2 and hit-evict in L1, DLC=1 is non-temporal for the Infinity Cache. L1 is 32 KB per CU, L2 2 MB with 128 B lines.

## Architecture

GGTensile's discipline, scoped to one operation and one problem type:

```text
ProblemContract   gfx1151, wave32, WMMA v1, code object v5, bf16 in / fp32 acc / bf16 out, D=128, dense
Problem           exact batch, heads, n_q, n_kv, layout (HND | NHD). Tails derived
FwdSpec           every active mechanism choice. Complete, immutable, the candidate identity
DerivedState      q_tiles, kv_tiles, tail flags, occupancy target, grid map
PhysicalPlan      register roles and lifetimes, LDS layout and offsets, barrier plan, wait plan, issue budget
Lowerer           one per mechanism family. The schedule lives here
Emission          .s + symbol + inspected resources (VGPR, LDS, private, spills) + predicted cycles
```

A field is a knob only when it is serialized, part of the candidate identity, validated for the selected problem, reflected in resource inspection, and consumed by a different lowering. Inert fields are rejected. This is the contract from `~/torch-ggml-ops/docs/ggtensile_tuning_knobs_plan.md`, and it is what keeps an asm generator from turning into a pile of dead flags.

Knobs, each with a distinct lowering:

| Group | Knobs |
| --- | --- |
| geometry | rows per wave {16, 32}. Waves per workgroup {4, 8, 16}. `Bc` {16, 32, 64}. CU mode or WGP mode |
| operand source | Q {registers, LDS}. K/V LDS slots {2, 3}. LDS layout {padded pitch 272/144, swizzled}. Staging owner {all waves, 1-2 producer waves}. Sync {`s_barrier`, LDS flags + `s_sleep`/`s_wakeup`} |
| softmax | base update threshold theta {0, 4, 8, 16}. Exp {`v_exp_f32`, `v_exp_f16`}. P dtype {bf16, fp16} with the PV WMMA dtype following it. Sum {chain, tree, VOPD pairs}. Pack {shift/and/or, `v_perm_b32`, `v_cvt_pk_rtz_f16_f32`}. Exchange {cndmask, per-lane-group V order}. Truncation bias {on, off} |
| schedule | prefetch depth per operand {1, 2, 4}. Waitcnt policy {per fragment, per group, counted}. Phase-boundary prefetch {off, V, V+K}. Per-wave rotation {none, d_tile, n_tile}. KV loop unroll {1, 2}. Clause policy {none, compiler-like, long}. `s_delay_alu` policy {compiler-like, none, explicit}. Setprio policy {none, loop/epilogue, producer/consumer}. Cache bits per access |
| group | head-fast or head-slow. Tail specialization {none, aligned, `n_kv % Bc == 0`}. Persistent or not, and the tile order when persistent. Per-workgroup head subset (LLC-bounded grouping) |

The generator scores its own schedule. The asm space is too large for compile-and-bench: every point costs assembly, linking, the 143-case correctness grid and timed rows. So the emitter walks its planned instruction list and produces a static cost model - cycles per pipe (VALU slots including VOPD pairs, LDS issue, VMEM issue, SFU), barrier and wait cost with a modelled multi-wave inflation, LDS bank conflicts from the chosen layout, register pressure from lifetimes - and prints predicted cycles per tile next to the emitted assembly. Candidates are ranked offline against the measured probe table above, and only the top few per shape class see silicon. This is the `evotensile` cost-model idea (`~/evotensile/docs/search_cost_model.md`) applied before measurement, and it is what makes a manual search over ~10^3-10^4 points tractable.

## Schedule mechanisms this exists for

Each is unreachable from HIP source without fighting the scheduler and the register allocator, which is the whole argument for generating assembly:
- Prefetch depth 2 on both fragment streams, one counted `s_waitcnt lgkmcnt(N)` per group instead of `lgkmcnt(0)` per fragment. Measured +35% in the LDS-fed regime. Depth 4 regresses (more live registers, no further latency to hide).
- Prefetch across phase boundaries. The V fragments for the PV of tile i do not depend on its softmax, so their loads belong before the max/exp block. The first K fragments of tile i+1 belong before the barrier that publishes its buffer. The shipped code exposes a full LDS round trip at both boundaries per tile.
- Rotate the fragment order per wave (wave w starts at `d_tile = w mod 8`) so the waves of a SIMD do not hit the same banks and the same LDS queue slot in the same cycle.
- Exact VGPR accounting. The 32-row design lands at 196-236 VGPR depending on `Bc` and depth, and whether it gets one or two workgroups per processor is decided by a dozen registers the allocator may or may not find. Config 1 ships at 256 VGPR with one spill. The generator pins `.amdhsa_next_free_vgpr` and derives occupancy exactly, so pipeline depth against waves/SIMD becomes a gene instead of a compile-and-see.
- Wave priority. `s_setprio 3` inside the KV loop, `s_setprio 0` in the prologue (Q staging) and epilogue (64 scattered 2-byte stores per lane), so a wave that is only feeding VMEM never wins the issue slot against a wave with a ready WMMA.
- Producer/consumer ring without `s_barrier`. One or two producer waves stream K/V into 3-4 slots and signal with an LDS flag plus `s_wakeup`. Consumers `s_sleep` between polls and signal "slot drained" with a `ds_add_rtn`. Loom measured a loss with this pattern, but its producers did dequant VALU that queued behind the WMMAs of the same SIMD and it had no `s_wakeup`, so it spun. Both differences go away here.
- Clause policy. The compiler's `s_clause 0x1` per LDS pair forces the LDS pipe to serve one wave for both instruction. Longer clauses, or none, is a one-knob measurement. VALU clauses (2-63 VALU instructions from one wave, no interleaving) could keep a wave's `exp2` block from interleaving with other waves' WMMAs.
- `s_sendmsg` dealloc-VGPR before `s_endpgm` so the next workgroup's waves can be created while the last stores drain. It matters more at 64-row workgroups, which have 4x as many of them.
- CU mode vs WGP mode (`.amdhsa_workgroup_processor_mode`): in CU mode the two LDS halves run in parallel, each serving two SIMDs. Free to set in a generated code object, impossible to ask for from HIP.
- `s_delay_alu` policy. The shipped loop carries 327 of them per tile. They exist to keep dependent issues from stalling the ALU for the whole SIMD. Whether explicit placement beats the compiler's is a measurement, and it is only expressible in generated assembly.
- Persistent workgroups with an L2-aware tile order. One workgroup per processor looping over `(head, q_tile)` so a resident K/V slice stays in L2, instead of one launch per tile. Two problems go away at once: tail quantisation on the small-grid rows (NHD H16/N4096 is 512 workgroups over 20 processors, the weakest row of the matrix at 1.018x) and the long-N re-fetch of K/V once per query tile. The backward already ships a persistent owner worth 1.1016x, and the fp8 forward measured the same mechanism as an LLC-bounded head grouping at 1.135x focused geomean with fetch dropping from 31.4 to 14.8 GiB. A generated code object can also select the mode (CU/WGP), which is part of why this belongs here rather than in HIP.

## Evidence from the forward fp8 and backward plans

Several mechanisms already paid for in this repository transfer directly, and are cheaper to believe than the probes:

| Mechanism | Where it already ships | Measured |
| --- | --- | --- |
| bf16 pair packing with `__builtin_amdgcn_perm(x, y, 0x03020706)` | `PackPair` in `csrc/bwd/bwd_fused_d128.cu`, `bwd_fused_d64.cu` | one instruction per pair instead of shift/and/or |
| fp16 P path with `__builtin_amdgcn_cvt_pkrtz` and `v_wmma_f32_16x16x16_f16` | same files (both WMMA dtypes are instantiated) | one instruction per pair, and the fp16 WMMA path already validated |
| Lane-base hoisting plus packed convert | backward D128-B5 | VALU 97,100 -> 83,508 (-14%) |
| Tail-free aligned images | backward D128-B6/B7 | VGPR 240 -> 232, 15.75 -> 14.78 ms (+6.6%) |
| Persistent owner (64-row KV) | backward, accepted mechanisms | 1.1016x over its control |
| LLC-bounded head grouping | fp8 forward | 1.135x focused geomean, fetch 31.4 -> 14.8 GiB (D128). 15.8 -> 13.5 GiB with L2 hits 2.08% -> 12.93% (D64 NHD) |
| Native base-2 softmax with a one-time log-sum-exp scale | backward D128 | 1.005x |

Three of their negative results matter as much:
- Do not buy occupancy with the pipeline. Four workgroups per processor was resource-valid at 176 VGPR in the backward only by reloading K/V and dropping the double-buffered staging, and it measured 0.72x-0.81x. The registers for two workgroups must come out of the tile representation, never out of the prefetch depth.
- Bank conflicts are not this kernel's LDS problem. The D128 forward measures 2.7% conflicts and `ALUStalledByLDS` of 0.026-0.029%. What limits it is LDS instruction bandwidth (1 KB per WMMA), which only fewer fragments per WMMA fixes.
- Do not chase VGPRs with lifetime barriers inside the current fragment representation. The fp8 plan closed that direction after direct source attempts preserved the same 191 VGPRs or introduced spills.

The fp8 plan also supplies the ranking rule this document follows: work paid once per query tile is amortized by the KV loop, work paid per KV tile is not. Only the latter is worth a lever.

## What the HIP campaign settled (input for this plan)

The HIP lever campaign ran before this generator, and its results change what the generator should attack:
- The 32-rows-per-wave dataflow does not fit the register file. Two WMMAs per LDS fragment is the only structural way off the 1 KB-per-WMMA ceiling, but it needs 128 VGPRs of fp32 accumulator (32 rows x 128 dims) plus 64 for Q before a single transient. Measured: 256 VGPRs and 404 B of scratch at `Bc` 32, 524 B of scratch at `Bc` 16 (where the compiler targets 192 VGPRs and spills Q). A wave cannot exceed 256 VGPRs on this ISA, so the generator cannot buy this with better scheduling. It would need a different accumulator representation (fp32 in LDS, or a D-split two-pass) and both cost more than they save. The LDS ceiling at 16 rows is therefore a hard wall, and the reachable gap is the 24% between the shipped 30-33 TFLOPS and the ~40 that the depth-2 schedule reaches in the probe.
- VALU-only levers are worth 2-4%, not the 20% the static accounting suggested. The lazy exponential base (which deletes 64 `v_mul`, 16 `v_readlane`, the `beta` multiply and the per-tile `exp2`/`rcp`/`log` of the log-sum-exp update) measured +2.4% head-major and +3.8% token-major. The single-instruction `v_perm_b32` pair pack, which removes two thirds of the pack instructions, measured +0.2% (below the indifference zone). Both are consistent with a loop that is LDS-bound: the arithmetic hides behind the LDS waits.
- Tail-free images are worth nothing here (-0.2% to +0.04%), unlike the backward's +6.6%: the forward's tail handling is a single `cndmask` and a clamp, not a predicate on every access.
- The measurement protocol to reuse. `~/tmp/feather_attn/hip_opt/paired.py`: two variants alternate inside each round, statistic is the median paired log ratio with a 95% interval from a robust sigma (max(stdev, 1.4826 MAD, IQR/1.349)), promotion gate is the pooled interval's lower bound above a 0.5% indifference zone. Its per-row sigma is 0.2-0.7% and its null effect -0.03% [-0.13, +0.05] over 12 rounds, which is what a 1% effect needs. Every generated candidate should be judged the same way, against the HIP kernel rather than against AITER, so the schedule's contribution is isolated.

Consequence for the campaign: the generator's expected value is concentrated in the schedule (measured candidates: counted-wait depth 2 at +35% in the dataflow probe, cross-boundary prefetch, per-wave rotation, priority) and in the launch policy (persistent grid, LLC-bounded grouping), not in further arithmetic removal. Its target row is the token-major long-sequence band, where the shipped kernel is 1.10-1.31x AITER and the LDS ceiling still leaves room.

## Expected results

| Step | HND rows | Confidence |
| --- | ---: | --- |
| shipped HIP kernel after the lever campaign (lazy base) | 32-35 TFLOPS, 1.165x AITER over the 18-row matrix | measured |
| 32 rows per wave (two WMMAs per LDS fragment) | would lift the LDS ceiling from ~40 to ~80, but does not fit 256 VGPRs in any variant tried | rejected by measurement |
| remaining VALU levers (fp16 P tail, pack, tree sums beyond the 4-way ones shipped) | +1-3% total | medium |
| counted-wait depth 2 plus cross-boundary prefetch, from generated assembly | the probe prices the depth-2 schedule at 40.3 TFLOPS against the 30-33 shipped in that dataflow, so 38-40 is the target | medium-high: measured in isolation, not yet composed |
| priority, clauses, rotation, sleep/wakeup ring, CU mode | +0-5% each, unknown | low until measured |
| persistent grid and LLC-bounded head grouping (launch policy, HIP-able) | +10-25% on the token-major long rows if the fetch reduction transfers from the fp8 kernel | medium |

Two caveats keep this honest. GGTensile reached only parity with well-tuned HIP on GEMM, so the generator by itself buys little. The gain is the dataflow and the schedule, and both can be prototyped in HIP first. And the LDS-fed ceiling is a hard property of 16-row fragments on this ISA: no schedule trick moves it, only feeding more WMMAs per fragment does.

## Shapes, correctness and dispatch

- One kernel family, many shapes: every generated image is shape-generic (a padded, mask-free fast path included). Tile logic is static. `n_q`, `n_kv`, head count, batch and per-operand strides are kernel arguments, exactly as the current ABI carries them. Tails are cold blocks. The host heuristic picks (rows/wave, `Bc`, grid mapping) from (layout, H, `n_q`, `n_kv`) the way hipBLASLt picks from a small exact catalog. No per-shape kernels.
- Every candidate runs the whole correctness contract, not one shape: `tests/test_attn_fwd_bf16_d128.py` is 143 cases over both layouts, both tails, odd head counts, `n_q > n_kv` and batch two, gated on `atol=rtol=0.02` plus `rel-L2 <= 1e-2` against AITER and against the blocked-fp32 reference. A tail specialization or a padded-K/V fast path does not ship until the generic grid passes with it selected.
- Semantics are not a knob. The online softmax recurrence, the fp32 accumulator, the natural-log log-sum-exp (or none) and the analytic tail correction are fixed. Knobs may change layout, order within the same arithmetic, and which dtype P is carried in. Spec-equivalence tests assert that all specs agree with each other within the elementwise envelope on a cheap shape set.
- Dispatch. Generated images register as `config >= 100` in the existing `feather_attn::attn_fwd_bf16_d128_feather` ABI, so `feather_attn/ops_bf16.py`, the public path and the benchmark matrix need no changes. The shape-to-config table is generated, in the pattern of `feather_ops/tiles.py` plus `tools/gen_tiles.py` (`--check` in tests).
- Bench. `bench/benchmark_attn_fwd.py` gains `feather_hip` (the current kernel as the baseline) and `feather_asm@<spec>` providers beside `aiter`. Bench stays speed-only. Accuracy lives in `tests/`.

## Campaign

- Phase 0 - hardware model. Already done for the probes above. The remaining item is the cycle model itself: a small set of microbenchmarks (WMMA interleaved with N VALU ops, with VOPD pairs, with `v_exp_f32`/`v_exp_f16`, and with LDS reads at 1-8 waves) turned into a predictor with error bars, plus model-versus-measured on the shipped kernel (should predict 31-35 from its own disassembly). Kill criterion for the whole plan: if total cost is not additive in WMMA + VALU + LDS, the ceilings are wrong and we stop.
- Phase 0.5 - HIP round 1. Levers 1, 2, 3, 9, 10, 11, 17, 18 of the HIP plan (`docs/feather_attn_gfx1151_fwd_bf16_d128_plan.md`). Gate: >= 38 TFLOPS on the HND rows. This is also the baseline the generator must beat, and the oracle it is tested against.
- Phase 1 - HIP round 2. The VALU and launch-policy levers (4-8, 12-13, 15-16), gated at >= 41 TFLOPS. The persistent grid and the LLC-bounded grouping are evaluated here because they are launch policy and do not depend on the tile rewrite.
- Phase 2 - generator skeleton. One spec, bf16, D128, no tail specialization: assemble, link a code-object-v5 blob with the pinned clang, load with `hipModuleLoadData`, inspect resources, pass the 143-case grid, then reproduce the HIP round-2 numbers within noise. Ablation ladder afterwards: delete one mechanism at a time and check the model's prediction.
- Phase 3 - sweep. Knobs in order of expected value: rows/wave, depth, lazy base, boundary prefetch, waves per workgroup, producer ring, priority, clauses, rotation, CU mode. Static model first, top-k on silicon with the paired protocol, then the shape-to-spec table.
- Phase 4 - only if Phase 3 lands. fp16 P path variants, persistent grid, `n_q < 128` shapes, and the same machinery for the backward (out of scope here).

Stop rules. A lane ends when either measured throughput is within 5% of the register-only ceiling on its target rows, or two consecutive rounds of new levers each report a measured stop reason. Every rejected lever keeps its measurement. The campaign ends if Phase 0 or Phase 0.5 misses its gate.

## Probing the arbiter: what to measure before trusting a schedule

The HIP campaign ended on one conclusion: the shipped loop is LDS-fed and the remaining gap is overlap, not arithmetic. Overlap is decided by mechanisms that HIP source cannot express and that no counter on this part reports directly, so the generator has to be built on a measured model of them. Three sources already give a partial model. The rest is a probe program.

### What is already established

| Fact | Value / effect | Source |
| --- | --- | --- |
| A wave issues from its own instruction stream. The arbiter picks the highest-priority ready wave and does not check operand readiness | `s_delay_alu` exists because an unready dependent issue stalls the ALU for every wave | ISA, and the shipped loop's 327 delay slots per tile |
| Priority is `{min(3, SysPrio + UserPrio), WaveAge}` | user priority 0-3 settable per wave | ISA |
| A 32-cycle WMMA does not consume all 32 issue slots of its wave | 32 independent `v_add_f32` alongside WMMAs cost 6.7 cycles total, i.e. ~0.2 cycles each. The same experiment with 32 dependent adds costs 5.7 cycles per link | `~/tmp/feather_attn/asm_fwd/IDEAS.md` |
| `v_exp_f32` is quarter rate and does not overlap with WMMA | 4.25 cycles per instruction | same |
| A single wave issues one instruction per cycle, not one per four: the ISA's own `s_delay_alu` worked example issues one instruction per cycle from one wave, with explicit delay cycles where the wave has nothing legal to issue | so MI300X-style "one instruction per SIMD every 4 cycles" rotation is not an ISA property here. Only the VMEM port may be CU-shared | `~/rdna35-isa-markdown/rdna35_instruction_set_architecture.md`, section 5.7 |
| A dependent issue stalls the ALU itself, so those cycles are lost to every wave, not just to the issuing one | "IB may issue dependent instructions that stall in the ALU, preventing those cycles from being utilized by other wavefronts" | same, section 5.7 and the `S_DELAY_ALU` description |
| `S_DELAY_ALU` is optional and packs two dependency descriptors plus a skip count. `INSTID` counts back issued VALU instructions. It may execute in zero cycles | the hazard model the generator must reproduce rather than guess | same, section 5.7 and Table 19 |
| VOPD (`v_dual_*`) is one issue slot for two independent VALU ops, and its legality is a register-bank condition: SRCX0/SRCY0 must be in different VGPR banks, VSRCX1/VSRCY1 in different banks, destinations one even and one odd, at most one literal, no DPP, wave32 only | pairing is a register-allocation problem, not a scheduling afterthought: this is why LLVM pairs only a quarter of a pairable stream | same, section 7.6 |
| `S_CLAUSE` locks the instruction arbiter onto one wave for 2-63 instructions | a clause takes issue bandwidth from every peer wave for its duration | same, section 8.3 |
| `S_WAITCNT` counters are 6-bit and count instructions. `VMcnt` covers loads only, `LGKMcnt` covers LDS/scalar-memory/GDS/messages, `FLAT` uses both, ordering is in-order within a type and out-of-order across types | the counted-wait depth-2 scheme the plan wants is expressible exactly | same, section 5.6 |
| LDS fragment read cost is set by issue shape, not by conflicts | 234 GB/s per WGP for the 16-lane 32 B duplicate shape. 46 GB/s for a 16 B-misaligned or >= 256 B power-of-two pitch. Duplicate lanes are not merged | same |
| The CU accepts one VMEM load per quad-cycle (4 cycles) regardless of SIMD (gfx942, see the transfer table) | "At most one such load can issue per compute unit per issue slot". In ideal lock-step, 16 loads take 64 cycles | AMD ROCm blog, gfx942 VMEM-load scheduling study |
| Each wave has a finite VMEM tracker. Crossing it stalls issue before the load is accepted (gfx942, see the transfer table) | MI300X knee at 13 in-flight `dwordx4` per wave. `SQ_WAIT_INST_ANY` jumps 2.6x from 13 to 14. ~5 cycles per load at 14, ~15 at 16 | same |
| Lock-step (barriers) is what creates the contention. De-phasing removes it (gfx942, see the transfer table) | 856 cycles (lock-step, 16 loads) -> 80-152 cycles (no barrier). 2-load/2-compute interleave: 1954 -> 1264 cycles | same |
| On this part, a WG with more than 4 active load-issuing waves enters a slow regime that is not explained by traffic or placement | cycles scale 1.56x for 1.25x work from `mask33` to `mask37`. Same-SIMD pairing matters much less than the count | `~/mgpusim/docs/gfx1151/profiler-tooling-issues.md` |
| `s_clause` locks one instruction type to one wave for 2-63 instructions, "even if that leaves the execution hardware idle" | LLVM emits `s_clause 0x1` on every LDS and global-load pair of the shipped loop | ISA |
| `s_sleep`/`s_wakeup` is the RDNA3.5 substitute for an mbarrier wait | a wave sleeps ~64*N clocks and any wave of its WG can wake it early | ISA |
| No `global_load_lds`, no `cp.async`, no `s_barrier_signal` | global -> LDS always goes through VGPRs. The barrier is the only WG-wide rendezvous | assembler probes in `~/tmp/feather_attn/tensile_probe/` |

### The model those facts imply

The current cost model is additive slots per pipe (`1024 MMA cycles + 361 other`), which predicted a 26% tax and failed to explain the measured 2-4% gains. The overlap model has to be state-based instead:

```text
per tile:  T = max( issue_slots, valu_cycles, lds_issue_cycles,
                    vmem_issue_cycles / quad_cycle_width,
                    inflight_loads / per_wave_tracker,
                    barriers * barrier_cost(waves, skew) )
           + wave_lifetime_backpressure(active_load_waves)
```

Each term is separately measurable, and three of them are new: the VMEM issue port (1 load per 4 cycles per CU), the per-wave in-flight tracker, and the backpressure term that turns on above four load-issuing waves. A schedule that shortens the critical path but adds concurrent load-issuing waves can lose. A schedule that de-phases the waves can win without touching the instruction mix. That is exactly what the HIP campaign could not test and what the generator exists to search.

### Probe program, ordered by expected value for this kernel

Each probe is a microkernel in the shape of `~/tmp/feather_attn/asm_fwd/probe_sched.cu` (a loop of N instructions with a counted `s_waitcnt`, launched at several occupancies), timed with the paired harness, and where useful profiled with `rocprofv3` counters plus stochastic PC sampling for `stall_reason`.

| # | Question | Probe | Number to extract | Consumed by |
| --- | --- | --- | --- | --- |
| 1 | How wide is the VMEM issue port (the ISA says nothing about a per-SIMD issue rotation, so only the memory port may be shared), and does a `b128` load with 32 active lanes cost 4 or 8 cycles of port time? | Load-only loop, 1-16 waves, 1-32 loads per wave, `buffer_load_b128` vs `global_load_b128`, plus the same loop with one VALU between loads | loads per cycle per CU, per WGP and per SIMD. Whether VALU between loads changes it | `GlobalReadPerWaveStep`, `InFlightBudget`. Whether K/V staging must be spread across SIMDs (probe also decides whether MGPUSim's `IssueRotation` is enabled for the gfx1151 preset, which the ISA suggests it should not be) |
| 2 | Per-wave in-flight tracker depth on gfx1151 | Single wave, N outstanding loads then a dependent use. Sweep N with `s_waitcnt vmcnt(N-1)` | the knee (MI300X: 13) and the cost curve past it | `InFlightBudget`, `WaitDepth` |
| 3 | Does wave count above four really create backpressure, and is it count or placement? | The mgpusim `load_wmma_dualwave_scheduler` kernel: modes overlap/wmma-only/load-only, masks 0x55/0xAA/0x0F, counts 1-8 | the threshold and the slope. Whether masking to 4 waves restores the fast regime | `LoadWaveCount` knob. Whether to use 4 load-issuing waves and a producer ring |
| 4 | How much VALU can a wave hide inside one 32-cycle WMMA, and does it depend on VOPD pairing or on wave count? | WMMA loop with 0/8/16/32/64 interleaved independent adds, at 1/2/4/8 waves | the per-WMMA free-issue count (probe suggests ~26 free of 32) | how much of the softmax can be interleaved without cost: the central knob of the schedule |
| 5 | Are WMMAs from different waves serialized on one SIMD, and does a WMMA block the peer SIMD? | WMMA-only at 1-8 waves, plus a peer-SIMD variant with different wave counts | WMMA issue interval vs wave count | occupancy target and `Waves` knob |
| 6 | LDS: latency, per-wave outstanding reads, and whether two waves on one SIMD interleave | pointer-chase and batched `ds_load_b128` variants. CU mode vs WGP mode | latency in cycles (external anchor: ~19.5 ns), reads in flight, interleave factor | `PrefetchLocalRead`, buffer count |
| 7 | Barrier cost as a function of wave count and skew | `s_barrier` in a loop, all waves lock-step vs deliberately de-phased | barrier cost per wave and the skew penalty | `BarrierMode`, `DePhase` |
| 8 | Does `s_setprio` actually buy issue slots, and at what granularity? | The same overlap kernel with loop waves at priority 3 and prologue/epilogue at 0, vs uniform | throughput delta for 4-8 wave mixes | `PrioritySchedule` |
| 9 | Does `s_sleep`/`s_wakeup` beat a barrier for a producer ring? | Ring of 3-4 LDS slots, 1-2 producer waves, consumers sleeping vs spinning vs barrier | cost per slot handoff | `SyncMode` |
| 10 | `s_clause` policy: does locking the LDS pipe to one wave help or hurt at 4-8 waves? | The shipped loop with clause lengths 0/1/4/none | throughput and LDS queue stall counts | `ClausePolicy` |
| 11 | Instruction fetch: do 96-bit literal instructions and long unrolled loops stall the front end? | The softmax loop with `v_perm_b32` literals vs register-valued selectors, at 4-8 waves | fetch stalls per loop | whether the pack may use immediates. Unroll depth |
| 12 | Cache policy: does marking K/V streaming (`slc`) protect the Q/LSE working set, and does it change issue behaviour? | The kernel with `.slc`/`.glc` variants on the staging loads, measuring L1/L2 hits and time | hit-rate delta and time delta | `CachePolicy` per access |
| 13 | Where does arbiter loss actually land? | ATT on one WGP of the load/WMMA overlap kernel, 1-8 load waves, `Idle` and `Stall` tabulated on `global_load_b128`, `s_waitcnt` and `v_wmma` | the `Idle`/`Stall` split per instruction as the load-wave count rises, and how it moves under `s_setprio` and de-phasing | validates the whole overlap model. Decides `LoadWaveCount`, `PrioritySchedule`, `DePhase` |

Probes 1-4 are the ones that change the desig. 5-12 tune it. Every probe should be run at the occupancy the real kernel uses (4 or 8 waves per SIMD, 256 VGPRs) because the mechanisms are occupancy-dependent.

### Tooling: what to use, and what not to trust

| Tool | Status on this machine | Use |
| --- | --- | --- |
| Hardware microkernels + paired timing | our own harness works, sigma 0.2-0.7% per row | the ground truth for every probe above |
| `rocprofv3` dispatch counters | works. `SQ_WAVES`, `Wavefronts`, `VALUInsts`, `SALUInsts`, `SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES`, `GL2C_MISS`, `FETCH_SIZE`, `SQ_INSTS_FLAT` were all usable | resource attribution per dispatch |
| `rocprofv3` stochastic PC sampling | works. Gives `stall_reason` per sampled instruction | finding where waves stall, not how much |
| `rocprofv3` ATT / SQTT | works when the selected CU/SIMD catches a wave. Zero-hit output is now diagnosed instead of silently wrong. See the ATT section below for what the hardware actually does on gfx1151 and which columns are trustworthy | per-instruction `Idle` vs `Stall` attribution, i.e. arbiter loss versus pipe backpressure |
| rocprofiler-compute | supports gfx115x panels (SPI, CPC, WGP, GRBM, GL0/GL1/GL2, GCEA), but "limited support for WMMA on gfx115x" | occupancy and memory panels. Not the WMMA roofline |
| rocJITsu (in `~/rocm-systems/emulation/rocjitsu`, config `configs/gfx1151.json`) | ships a gfx1151 `rdna3_5` config. `exec_mode: "clocked"` selects clocked execution. The CU has a pluggable `WavefrontScheduler` with the hardware GTO policy (`OldestFirstScheduler`) and per-wave `ready_cycle()` | the pre-silicon oracle: it is AMD's own ISA model for our exact target, and because the scheduler is a class we can also ask "what would a different arbiter policy buy" without touching hardware |
| MGPUSim gfx1151 fork (`~/mgpusim`) | functional path matches hardware bit-exactly for the Feather FP16/scaled_mm matrices. The timing path has per-section policies (`FrontendTimingSpec`, `SchedulerTimingSpec`, `ScoreboardTimingSpec`, `VALUTimingSpec`, `LDSTimingSpec`, `VectorMemoryTimingSpec`), per-wave scoreboards, VMEM result-ready latency, grouped-transaction penalties, WMMA/load peer penalty, `s_delay_alu`/`s_clause`/`s_setprio`/`s_waitcnt_depctr` modelling, and LDS bank conflicts | the calibration vehicle: its knobs are the model's unknowns, and its microkernel suite (placement, WMMA dual-wave, load-use, load+WMMA, overload-latch) is already written for gfx1151 |
| gem5 | AMD GPU model is gfx9/CDNA3 only (Vega, MI300X). No gfx11 | not usable for this target |
| Accel-Sim / GPGPU-Sim | NVIDIA | not usable |

### ATT on gfx1151: what it does, what it costs, and which number to read

The thread-trace path was read in-tree (`~/rocm-systems/projects/rocprofiler-sdk/source/lib/aqlprofile`: `pm4/sqtt_builder.h`, `gfxip/gfx11/gfx11_primitives.h`, `core/threadtrace.cpp`, `docs/how-to/using-thread-trace.rst`), because every overlap conclusion we draw from a trace depends on knowing how the trace is produced.

How it is produced:
- Userspace (aqlprofile) programs the trace through PM4: `SQ_THREAD_TRACE_MASK` selects one WGP per shader engine (`WGP_SEL`, `SA_SEL`) and one SIMD (`SIMD_SEL`), `SQ_THREAD_TRACE_TOKEN_MASK` selects which token classes are logged, `SQ_THREAD_TRACE_BUF0_SIZE/BASE` gives the buffer, `SQ_THREAD_TRACE_CTRL` starts the session, and `SQ_THREAD_TRACE_STATUS2` reports `BUF0_FULL` / `BUF1_FULL` / `WRITE_BUF_FULL` / `PACKET_LOST_BUF_NO_LOCKDOWN`.
- gfx115x has its own primitives (`gfx115x_sqtt_prim`) that set `TTRACE_EXEC=1` and `REG_DETAIL_ALL=1`, which is what makes per-instruction timing available at all. The plain gfx11 variant instead excludes the `VMEMEXEC` and `ALUEXEC` token classes. The token mask also takes `(exclude_wait, trace_exec)`, so a run can deliberately drop `s_waitcnt` tokens.
- `sqtt_stalling_enabled()` returns false on gfx11 and gfx12, so aqlprofile does not put the SQ into issue-stalling mode on this part. The high-token-rate failure mode here is a full buffer and lost packets, after which tracing stops, not a slowed kernel. The perturbation that remains is different but real: traced kernels are serialized (no concurrent kernels), only one WGP of one SE is sampled, and its L2/DRAM contention is therefore not production's.
- Perfmon streaming into the trace (`att-perfcounter-ctrl`, `att-perfcounters`, `att-activity`) is gfx9-only, so on gfx1151 the trace carries no counters, only tokens. `att-no-detail` drops instruction timing and keeps occupancy. The default `att-buffer-size` is 96 MB shared across SEs.

What that means for measurement:
- The stats CSV splits each instruction's cycles into `Latency` (= `Stall` + `Execute` on gfx10+), `Stall` (the pipe could not issue: TCP, LDS backpressure) and `Idle` (gap between the previous instruction retiring and this one beginning, caused by arbiter loss, register dependency or instruction-cache miss), all summed over `Hitcount`. `Idle` is the only hardware attribution of arbiter loss available on this part, which makes it the column to read for every overlap question in this plan.
- Never take throughput, utilization or absolute cycle counts from a trace: with lost packets the trace's own span is not the kernel's span, and with one WGP sampled the contention picture is not production's. Throughput comes from the paired harness in a profiling-free run. ATT contributes relative attribution shape and ordering.
- Protocol for any ATT-based conclusion: (1) throughput from the paired harness, no profiler attached. (2) counters in a separate run (`SQ_WAVE_CYCLES`, `SQ_BUSY_CYCLES`, `SQ_INSTS_FLAT`, `SALUInsts`, `VALUTime`, `GL2C_MISS`, `FETCH_SIZE`). (3) ATT with `att-buffer-size` raised, a deliberately short loop, and a grid large enough that the target CU catches a wave, checking the run for `Data Lost`. (4) a perturbation check: repeat with `att-no-detail` and with a different buffer size and confirm the `Idle`/`Stall` ratio on the key instructions is stable. If it moves, the attribution is trace pressure, not the kernel.
- The concrete use: tabulate `Idle` and `Stall` on `global_load_b128`, `s_waitcnt` and `v_wmma` for 1-8 load-issuing waves. If the AMD VMEM issue-port mechanism holds on this part, `Idle` on the loads grows with the number of load waves while `Stall` stays flat, and a `s_setprio` or de-phasing change should move `Idle` from the loader waves to the WMMA waves. That is probe 13 in the table above.

Recommended division of labour: hardware microkernels decide every number in the model. rocJITsu's clocked mode and the MGPUSim timing path are used tointerpolate between measured points and to rank schedules before silicon, and both must be re-anchored to hardware whenever a new mechanism is added. Neither is trusted for a claim that hardware has not confirmed.

### Where the schedule model comes from: MGPUSim, not rocJITsu

The two candidate simulators were checked against their current trees and upstream state (`~/rocm-systems/emulation/rocjitsu`, `~/mgpusim` on `rdna35`, `gh pr list` on both upstreams). They do not attack the same thing, and only one of them can become the schedule oracle this plan needs.

| | rocJITsu (ROCm, in `~/rocm-systems/emulation/rocjitsu`) | MGPUSim (`~/mgpusim`, branch `rdna35` against `sarchlab/mgpusim`) |
| --- | --- | --- |
| What it is | AMD's ISA emulator: functional execution, DBT, DBI, race detection, with `configs/gfx1151.json` present | a cycle-stepped microarchitecture simulator whose timing model is now being calibrated against hardware, plus a bit-exact functional path |
| Scheduling model | a pluggable `WavefrontScheduler` (`OldestFirstScheduler` = GTO) with per-wave `ready_cycle()` and a `step()`/`issue_from_slots()` hook per cycle | fetch arbiter, issue arbiter, wavefront arbiter, scoreboards, VMEM unit, LDS banks, front end, all with per-section spec fields |
| The clocked path | the clocked pipeline cycle is unimplemented: `ExecComputeUnit<CLOCKED>::execute_quantum()` is a `/// @todo: Support CLOCKED pipeline cycle.` The hooks exist for debugger/checkpoint use, not for performance modelling | the mechanism being extended: upstream just landed measured issue rotation, VALU issue overlap, taken-branch gap, LDS bank conflicts and instruction fetch |
| RDNA 3.5 status | arithmetic bit-exactness work in flight (upstream PR "Share RDNA 3 bit-exact entry points and qualify RDNA 3.5 for mode 2"), and its own qualification note says "no Strix gfx115x silicon capture yet" | `rdna35` branch: gfx1151 platform profile, wave32 launch, gfx1151 decode and emulation, FP16/mixed Feather kernels matching hardware bit-for-bit, plus five gfx1151 microkernel probes and their calibration commits |
| What investing in it buys us | nothing for scheduling: we would be writing the microarchitecture model inside a functional emulator, with no reference | the model is the thing we need, and upstream is already calibrating its mechanisms for gfx942 - the port is to enable and calibrate them for gfx1151 |
| Config fidelity | `gfx1151.json` describes 16 CUs / 32 SIMDs / 6 MB L2 / 2.7 GHz, against the real 40 CUs / 80 SIMDs / 20 WGPs / 2 MB L2 | the `rdna35` profile carries gfx1151 numbers already, and its probe harness on this machine is the reference |

Decision: MGPUSim is the simulator to finish. rocJITsu stays an ISA-semantics reference at zero cost. The decisive points are that rocJITsu's clocked mode is a `@todo` rather than a model, and that MGPUSim's timing terms are exactly the three this plan says are missing (issue rotation, LDS, fetch).

#### What "finishing" means concretely

Upstream MGPUSim has merged, in the last weeks, the mechanisms this plan needs, all as spec options that are off by default and enabled per preset builder:
- `IssueRotation`: the issue arbiter serves one wavefront pool per cycle in rotation, so a wave issues at most once every 4 cycles. It requires `SIMDCount == 4` (`cubuilder.go` panics otherwise) - our gfx1151 timing CU is modelled as a WGP with four SIMD32 units, so the shape matches.
- `VALUIssueOverlap`: a VALU instruction keeps one operand stage and four execute cycles, and its wave may issue again during the last execute cycle.
- `UniformVALUCycles`, `TakenBranchIssueGap` (MI300X: 32).
- gfx942 LDS bank-conflict modelling, `code_fetch_latency`, 32-byte instruction blocks with compute-unit prefetch, fetch-ahead, and branch-target-first issue.

Our `rdna35` branch predates that refactor: it grew its own `amd/timing/cu/timingconfig.go`, `valu_timing.go`, `vmem_timing.go`, `lds_timing.go`, the state trackers and a wavefront-level `scoreboard.go`, while upstream split the CU into `fetcharbiter.go`, `issuearbiter.go`, `wfarbiter.go` and moved the spec options into `comp.go`/`cubuilder.go` with per-preset builders. A 21-commit rebase over that drift conflicts from its first commit (9 files) and replays semantic conflicts in every later one.

The ISA settles one of the model's questions in passing: general instruction issue is one per wave per cycle (the `s_delay_alu` worked example issues an instruction every cycle from a single wave), so upstream's `IssueRotation` - one instruction per SIMD every four cycles - should stay off in the gfx1151 preset unless probe 1 measures otherwise. What remains genuinely unknown is the width of the memory issue port and the per-wave in-flight limit, and those are exactly the fields our branch already parametrizes (`VectorMemory.RequestIssueWidth`, `VectorMemory.InFlightTransactionLimit`) while upstream does not have them at all.

The port therefore goes as a five-commit series on current `main`, not as a rebase:
- RDNA3.5 architecture, wave32 launch and the gfx1151 platform profile (`amd/arch/rdna35.go`, driver/insts plumbing), rebased by hand onto the new structure.
- gfx1151 decode and emulation plus the FP16/mixed kernel HSACO execution.
- The gfx1151 microkernel suite and `docs/gfx1151/` (additive. Only one of our 63 new files exists upstream, so nothing is duplicated).
- A `gfx1151` timing preset that enables and sets the upstream mechanisms (issue rotation, VALU issue overlap, LDS banks, fetch) with our probe numbers as the values.
- One calibration commit per mechanism, in upstream's own style ("calibrate X with a dedicated probe"): each lands a probe, a measured value, and the field it sets.

The mapping between what our branch added and what upstream now has decides what is a merge and what is a contribution:

| Our branch | Upstream after the timing refactor | Action |
| --- | --- | --- |
| `cu/timingconfig.go` with per-section specs | spec options moved into `comp.go`/`cubuilder.go` plus per-preset `WithX` builders | re-express as a `timingconfig/gfx1151` preset |
| `VectorMemory.RequestIssueWidth`, `InFlightTransactionLimit` | not modelled upstream | contribute: the VMEM port width and per-wave tracker are the two mechanisms the AMD study is about |
| `cu/lds_timing.go` (banks, max latency, conflict degree) | upstream added gfx942 LDS bank-conflict modelling | merge upstream's, keep our gfx1151 numbers |
| `cu/frontendstatetracker.go` | upstream added 32-byte instruction blocks, prefetch, fetch-ahead, branch-target-first | merge upstream's, keep the gfx1151 latency |
| `cu/valu_timing.go`, `vmem_timing.go`, `schedulerpopulationtracker.go`, `wavefrontstatetracker.go`, `wavefront/scoreboard.go` | the CU was split into `fetcharbiter.go`, `issuearbiter.go`, `wfarbiter.go` with its own scoreboard plumbing | port the gfx1151 calibration values into the new structure. Drop the duplicate scaffolding |
| `docs/gfx1151/*` and the five microkernel probes | nothing equivalent upstream | contribute as-is (58 of our 63 new files do not exist upstream) |
| `IssueRotation`, `VALUIssueOverlap`, `TakenBranchIssueGap` | added upstream, MI300X-only | adopt the mechanism, decide per ISA/probe for gfx1151 (`IssueRotation` off, `VALUIssueOverlap` on to test, branch gap measured) |

Repository rules that apply when we do it (from `AGENTS.md` -> `CLAUDE.md` on `origin/main`): `go build ./...`. `ginkgo -r --skip-package=mccl,amd/tests/pagemigration`. `golangci-lint run ./... --timeout=10m` pinned at `v2.13.2`. Lines under 120 columns. `GOTOOLCHAIN=local go test ./internal/architecture` for component coverage. Per-preset `WithX` builder options rather than global defaults, and checkpoints carry no compatibility guarantee, so breaking them is allowed.

#### Measured first: the three probes, and the model terms they fix

Run before touching any simulator, with `~/tmp/feather_attn/hip_opt/probes/probe_sched2.hip.cpp` (four kernels, per-window sclk sampling, marginal differencing where the loop overhead matters). Two methodological facts had to be established first, because both silently ruin a probe: `volatile` global loads bypass L1 on this target (90 ns per load against 5.7 ns for a plain load, same address), and the DPM clock follows the workload's power draw (0.8 GHz for a pure-VALU loop, 2.4-2.6 GHz for a WMMA/LDS kernel), with `hwmon freq1_input` reporting a decayed average rather than an instantaneous level. Every comparison below is therefore same-window and same-workload-class.

The VMEM issue port is a fixed per-CU resource and does not scale with waves. A stream of L1-resident `global_load_b128`, eight in flight per wave, 4096 loads per wave, measured at 1/2/4/8 waves per CU:

| waves per CU | ns per load | ns per load with 2 VALU interleaved |
| --- | ---: | ---: |
| 1 | 5.72 | 5.88 |
| 2 | 5.72 | 5.88 |
| 4 | 5.99 | 6.20 |
| 8 | 6.18 | 6.57 |

Flat within 8% across an eightfold change in wave count, and two interleaved VALU ops cost 0.16 ns. At the clock this loop sustains (0.7-0.8 GHz) that is ~4-5 cycles per load per CU, which is the same "one VMEM load per CU per quad-cycle" behaviour AMD documents for gfx942, and VALU issue does not compete with it.
Consequence for the kernel: staging is port-limited, not bandwidth-limited. More waves cannot raise the aggregate load rate, so the levers are fewer and wider loads (b128 is already the maximum) and staging less often (a larger `Bc`), not more parallelism. It also means the K/V staging of a 16 KiB tile needs at least 16 KiB / 512 B = 32 load instructions, i.e. ~128-160 cycles of port time per tile per CU - 12-15% of a 1024-cycle tile, and it cannot be hidden by adding waves, only by overlapping it with compute within a wave.

A wave can co-issue VALU inside its own WMMA window, but only while few waves share the SIMD. Eight WMMAs per iteration with 0/32/64 independent adds:

| waves per CU (per SIMD) | cycles per WMMA | marginal cycles per add (64-add variant) |
| --- | ---: | ---: |
| 1 (0.5) | 29.3 | 0.34 |
| 2 (1) | 33.1 | 0.39 |
| 4 (2) | 51.9 | 0.96 |
| 8 (4) | 112.6 | 2.23 |

At one wave per SIMD a WMMA costs 29 cycles and an interleaved add 0.34 cycles, i.e. roughly 85 adds fit inside one WMMA window for a third of a cycle each. At the 4 waves per SIMD the shipped kernel runs at, the marginal add costs 2.2 cycles - six times more. The WMMA itself also stretches (29 -> 113 cycles) as waves share the port.
Consequence: the softmax VALU is nearly free at low occupancy and expensive at high occupancy, which reverses the usual "more waves hides latency" intuition for this kernel and argues for the generator to lower waves/SIMD and interleave inside the window rather than chase occupancy.

The per-wave in-flight VMEM capacity is about 6-8 loads. K independent loads followed by `s_waitcnt vmcnt(K-1)`, one wave per CU, K from 1 to 24: the marginal cost per load falls from ~28 ns (K = 4 -> 8) to ~3.6 ns (K = 8 -> 24), so beyond roughly eight in flight the loads overlap and below it they serialize at one load latency (~28 ns, i.e. 20-25 cycles at this clock). MI300X's documented knee is 13. This part is shallower.
Consequence: `InFlightBudget`/`WaitDepth` should keep 6-8 loads in flight per wave, and a depth-2 pipeline is enough - depth 4 regressing in the earlier probe is consistent with the tracker, not with register pressure.

In an attention-shaped loop, four load-issuing waves is the optimum and there is no cliff beyond it. The staging kernel (eight waves stage a 16 KiB K tile into LDS, barrier, then every wave runs four WMMAs from LDS):

| load waves | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ns per iteration | 97.8 | 56.3 | 46.8 | 37.5 | 44.9 | 40.5 | 36.2 | 36.2 |

Spreading the staging over four waves is 2.6x faster than one wave doing all of it, and beyond four there is nothing to gain. The MGPUSim "more than four active load waves is a slow regime" result does not reproduce in this shape: it came from a load-only microkernel, and in a loop that also computes, the load waves only need enough parallelism to cover the port.

What these four measurements change in the plan:

| Model term | Before | Measured on gfx1151 |
| --- | --- | --- |
| VMEM issue port | "1 load per CU per quad-cycle" borrowed from gfx942 | confirmed: ~4-5 cycles per `b128` per CU, independent of waves, VALU-free |
| Per-wave in-flight limit | 13 (gfx942) | 6-8 |
| Load latency to hide | emulator placeholder 11 / 179 cycles | ~28 ns ≈ 20-25 cycles for an L1 hit at the probe's clock |
| Issue rotation | MI300X enables it, ISA suggests it should not apply | VALU-only issue confirms one instruction per wave per cycle. `IssueRotation` stays off |
| Load-wave count | unknown, MGPUSim suggested < 4 | 4 is optimal, no cliff above it |
| VALU inside the WMMA window | assumed roughly free | free at 1 wave/SIMD (0.34 cycles per add), 2.2 cycles per add at 4 waves/SIMD |

Two of these are the reason the HIP campaign fought the ALU tax at all: at four waves per SIMD every softmax instruction costs two cycles, so the 590-740 non-WMMA instructions per tile are ~1200-1600 cycles of issue, not 600-750. That is the generator's opening: a schedule at two waves per SIMD with the softmax interleaved into the WMMA windows has a materially cheaper VALU stream than any further reduction of the instruction count at the current occupancy.

### Cross-architecture evidence: what transfers, what must be re-measured here

Most of the mechanisms in this plan were first measured somewhere else. The rule is: mechanisms that come from the RDNA family transfer as hypotheses worth testing. Magnitudes never transfer, and CDNA-specific mechanisms must be re-derived because the matrix pipe is different in kind. Concretely, `v_mfma` is a separate matrix core on CDNA, while `v_wmma` issues on the SIMD's VALU on RDNA - that single difference decides whether "the matrix op leaves issue slots for other work" is a mechanism at all.

| Borrowed result | Measured on | Why it may not transfer | Check on this machine |
| --- | --- | --- | --- |
| At most one VMEM load per CU per issue slot. 16 loads in the ideal 64 cycles | gfx942 (CDNA3), AMD VMEM scheduling study | CDNA CU is 4 SIMD64. gfx1151 CU is 2 SIMD32 inside a 4-SIMD WGP, and the memory path is an APU (GCEA over LPDDR5X), not HBM | probe 1: load-only loop, count loads per cycle per CU, per WGP and per SIMD |
| A per-wave VMEM tracker knee at 13 in-flight `dwordx4`, then ~5 to ~15 cycles per load | gfx942 | the tracker is a per-wave queue sized by the architecture. wave32 vs wave64 changes the bytes per entry, and `vmcnt` is 6-bit on both | probe 2: knee sweep with counted `s_waitcnt vmcnt(N-1)` |
| De-phasing beats lock-step by 5-10x on load issue (856 -> 80-152 cycles) | gfx942 | it needs the same port contention to exist, and gfx1151 has no `s_barrier_signal`, so de-phasing must come from `s_sleep`/LDS flags or from static skew | probes 3 and 9, plus ATT `Idle` attribution |
| Interleaving 2 loads with compute: -35% cycles. A 32-cycle matrix op leaves seven issue slots | gfx942 | MFMA is a separate pipe. RDNA WMMA runs on the VALU. The "free slots inside the matrix op" story may not exist at all here | probe 4. The only local evidence is 32 independent adds next to WMMAs costing 6.7 cycles, which must be re-measured per wave count and with/without VOPD |
| Above four active load-issuing waves a workgroup enters a slow regime (1.56x cycles for 1.25x work) | gfx1151, but on a synthetic microkernel with an unknown occupancy | it is local hardware evidence, yet not yet reproduced in the shape of a real attention tile | probe 3, inside the shipped and generated kernels rather than the synthetic one |
| `LdsPad` / `PrefetchGlobalRead` / store-batching gains (+8%, +15%) | gfx942/gfx950 GEMM campaigns | different pipe (MFMA), different LDS budget and pitch behaviour, and quantized decode paths that do not exist here | the local LDS probe already separates 234 GB/s from 46 GB/s by pitch. Re-derive the pad and prefetch optima locally (probes 6 and the HIP plan's LDS audit) |
| StreamK and persistent-CTA gains | Instinct | 100+ CUs make tail and wave-balance arguments that do not scale down to 20 WGPs. The local tail argument is quantisation of 512-1024 workgroups, not of thousands | local persistent-grid experiment (HIP-able, already a plan item) |
| LDS latency around 19.5 ns | Navi31 (gfx1100) | same LDS family, but gfx1151 sustains a different clock (2.29-2.87 GHz against ~2.5) and CU-mode halves may route differently | probe 6: latency in cycles at the sustained clock |
| VOPD pairing gives 2x fp32 issue rate (`UseDualFMAC`) | documented by TensileLite for RDNA3/3.5/4 | this one is squarely in-family and should apply | still measured locally: how many of our ALU stream's ops the compiler or the generator actually pair (probe 4) |
| WMMA = 32 cycles, 256 FLOP/cycle/SIMD, register-only ceiling 54 TFLOPS at 2.87 GHz | gfx1151, local probes | nothing to transfer | keep as the anchor |
| Timer-trace (ATT) semantics: `Idle` = arbiter loss | documented for gfx10+ | applies to gfx1151, but the token masks, the one-WGP-per-SE limit and the absence of perfmon streaming are Navi-specific | the ATT section above. Validate the `Idle`/`Stall` split against a deliberately arbiter-bound microkernel |
| rocJITsu `gfx1151.json`: 16 CUs, 32 SIMDs, 6 MB L2, 2.7 GHz | a stylized config, not this part | the real device is 40 CUs / 80 SIMDs / 20 WGPs with 2 MB of L2 and a 2.29-2.87 GHz sustained clock. Any occupancy or timing number out of that config is scaled by the wrong machine | patch the topology, L2 and clock in a local copy, then reproduce the register-only WMMA ceiling and the LDS bandwidth probe before trusting anything else |
| MGPUSim timing constants (VMEM result-ready 11, WMMA/load peer 112, VALU 4, VOPD 2, WMMA 32, LDS conflict cap 64) | a GCN3-derived model, seeded from documentation rather than measurement | they are the model's unknowns, not evidence | probes 1-7 are exactly the calibration targets for these fields |
| rocprofiler-compute RDNA3 panels | documented for gfx115x | its own limitation: "limited support for WMMA on gfx115x" | use occupancy and memory panels only, and cross-check every number against `rocprofv3` counters |

Two consequences for the campaign order. First, no borrowed magnitude enters the cost model: every number carries the probe that anchored it locally, and an unanchored number is marked as a guess with its source arch. Second, the two CDNA results this plan leans on hardest - the VMEM issue-port width and the "issue slots inside the matrix op" - are also the two that differ most in kind, so probes 1, 2 and 4 are prerequisites rather than refinements: without them the overlap model would be a translation of an Instinct result onto a chip with a different matrix pipe.

### What this buys the generator

- A cost model with the three missing terms, which is what makes the offline ranking trustworthy enough to skip most silicon measurement.
- Two new knob families that only exist because of these behaviours: load-issue shaping (`LoadWaveCount`, `GlobalReadPerWaveStep`, `InFlightBudget`, `DePhase`) and arbiter policy (`PrioritySchedule`, `ClausePolicy`, `SyncMode`).
- A concrete expectation to test first: the shipped kernel issues all its staging loads in a burst at one barrier, exactly the lock-step pattern the AMD study shows stalling. De-phasing the ten waves of a workgroup (wave `w` rotates its fragment order, and the KV loop starts one tile apart for odd waves) is a HIP-able experiment, and probe 3 is what would justify it.

## New ideas since the HIP campaign

- Fund the two-WMMA-per-fragment dataflow by moving the accumulator out of VGPRs. The 32-row geometry fails on one number: 128 fp32 VGPRs of output accumulator. Two representations are worth generating and measuring: (a) the accumulator in LDS with `ds_add` per WMMA epilogue (LDS traffic returns, but the LDS is at 25% in that geometry, so it may fit). (b) a D-split two-pass where each pass accumulates 64 of the 128 dims, which halves the accumulator to 64 VGPRs but doubles the score WMMAs (+50% MMA work, so probably not). The generator can emit both and the cost model can rank them before silicon.
- Spend the LDS budget on double-buffered fragments, not tiles. With the LDS at 54% of its ceiling in the 32-row geometry, extra LDS can buy a fourth K/V buffer or a per-wave fragment cache. The 16-row geometry instead wants buffers and the depth-2 read-ahead that HIP cannot express.
- Feed two waves from one LDS read via CU mode. In CU mode the two LDS halves serve two SIMDs in parallel, so a 4-wave workgroup in one CU can read the same fragment from both halves. That halves the LDS bytes per WMMA for pairs of waves without any register cost, which is a cheaper form of the 32-row idea. Gated on being able to place two 4-wave workgroups in one CU, which a generated code object can request and HIP cannot.
- Stage K/V once per WGP in LDS and let both workgroups read it. Requires the two workgroups to be in the same CU (CU mode) and to cooperate on the barrier, which means a persistent kernel with a workgroup pair per CU. The LDS cost per wave halves again. The risk is the cross-workgroup barrier.
- The L2-aware persistent order is worth more than any remaining arithmetic change. The fp8 kernel measured fetch dropping from 31.4 to 14.8 GiB with LLC-bounded grouping, and the backward's persistent owner measured 1.10x. Both are launch policy, both are HIP-able, and neither has been tried on the bf16 forward.

## Risks

- The ceiling is close. 40 is the depth-2 LDS ceiling for the current dataflow and 54 is the WMMA ceiling. 45-47 is 83-87% of the latter. Anyone promising 2x is measuring against the 59.4 spec peak rather than the 54 that register-only WMMA actually reaches at 2.87 GHz.
- Register budget against workgroup count. Two workgroups per processor needs <= 192 VGPR, which the 32-row design only reaches with `Bc = 16` or Q in LDS. This is the main geometric risk and the reason both variants are in the sweep.
- Tail and specialization combinatorics are where correctness bugs live. The rule is that every spec passes the whole contract, and the padded-K/V fast path is analytic rather than masked.
- Maintenance. This is the first part of the repository that is not readable C++. It is mitigated by generating from a table, by `--check` regeneration in tests, by the emitted cost model, and by keeping the HIP kernel as the reference implementation the generator is measured and validated against.

## Artifacts

- Hardware study and probes: `~/tmp/feather_attn/asm_fwd/` (`IDEAS.md`, `probe_sched.cu`, `probe_lds.cu`, run logs, `k1_hnd_bc64_swz.s`, `bf16_image_gfx1151.s`).
- ISA capability probes from this plan: `~/tmp/feather_attn/tensile_probe/` (`probe*.s`, `hip_intrinsics.hip`, `hip_cvt.hip`).
- Baseline kernel and its remaining levers: `docs/feather_attn_gfx1151_fwd_bf16_d128_plan.md`.
- In-repo mechanisms that transfer (packing, packed convert, aligned images, persistent owner, LLC grouping, base-2 softmax): `docs/feather_attn_gfx1151_bwd_d128_plan.md`, `docs/feather_attn_gfx1151_bwd_d64_plan.md`, `docs/feather_attn_gfx1151_fwd_fp8_plan.md`, and the sources they describe in `csrc/bwd/`.
- Scheduler/overlap prior art: `~/mgpusim/docs/gfx1151/` (plan, timing-infrastructure, wmma-dualwave-scheduler, load-wmma-dualwave-scheduler, profiler-tooling-issues, placement-probe, load-use-delay-probe). AMD's VMEM issue-port study (`https://rocm.blogs.amd.com/software-tools-optimization/scheduling_VMEM_loads_gfx942/README.html`). rocJITsu's gfx1151 config and pluggable GTO scheduler (`~/rocm-systems/emulation/rocjitsu/configs/gfx1151.json`, `lib/rocjitsu/src/rocjitsu/vm/amdgpu/wf_scheduler.h`). the ATT implementation read for the section above (`~/rocm-systems/projects/rocprofiler-sdk/source/lib/aqlprofile`, `docs/how-to/using-thread-trace.rst`). rocprofiler-compute's RDNA3 performance-model page.
- Prior art: `~/torch-ggml-ops/docs/ggtensile_plan.md` and `ggtensile_tuning_knobs_plan.md`, `~/evotensile/docs/` (cost model, noisy measurements, search space), `~/rocm-libraries/projects/hipblaslt/tensilelite/` (kernel writer, `UseDualFMAC`), `~/loom-skills/skills/loom-kb/device/rdna3/`.
