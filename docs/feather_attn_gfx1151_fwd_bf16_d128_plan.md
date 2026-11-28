# FeatherAttn gfx1151 bf16 D128 forward kernel

## Goal

A new forward attention kernel for gfx1151 that is faster than AITER's Triton `attn_fwd` on the shapes in `bench/benchmark_attn_fwd.py`, and that stays correct and reasonably fast on the D128 rows of `docs/input_shapes_attn.md`.

## Scope

| Property | Value |
| --- | --- |
| Dtype | bf16 input and output, fp32 score and output accumulation |
| Head dimension | 128 only |
| Attention | dense, non-causal, no bias, no dropout, no window, equal Q/KV heads |
| Layouts | contiguous 4D `[B,H,N,D]` or `[B,N,H,D]`, one layout per call for all four operands. The row/head/batch strides are derived in C++ from that layout, so the kernel itself no longer depends on a head count |
| Shapes | any `batch >= 1`, `heads >= 1`, `n_q >= 1`, `n_kv >= 1` |
| Grid policy | head-slow (HND) and head-fast (NHD) specializations |
| Out of scope | D64, fp8/int8 quantization, masks, causal, GQA, backward, public API dispatch |

The kernel is exposed as one opaque op, `feather_attn::attn_fwd_bf16_d128_feather`, reached from Python by `feather_attn/ops_bf16.py`. It writes the natural-log log-sum-exp as a second output, which is what makes it usable as the forward of `feather_attn.ops.feather_attn_func` for the dense non-causal bf16 D128 case. Every other request is delegated to AITER's Triton launcher. The operator takes `out` and `lse` as arguments rather than allocating them, so under `torch.compile` the allocations stay in the traced graph.

## Result

`python -m bench.benchmark_attn_fwd --impls aiter,feather_bf16_d128 --dtypes bf16 --head-dims 128` through `bench/protocol.py`: warm, providers interleaved, median of the round medians, paired ratio against AITER measured in the same round. B = 1, D = 128.

| case | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: |
| HND H16 N4096 | 30.67 | 34.72 | 1.136x |
| HND H16 N8192 | 29.88 | 34.39 | 1.161x |
| HND H16 N16384 | 29.03 | 33.38 | 1.145x |
| HND H32 N4096 | 28.49 | 32.34 | 1.148x |
| HND H32 N8192 | 28.41 | 33.04 | 1.164x |
| HND H32 N16384 | 27.91 | 32.65 | 1.171x |
| HND H56 N4096 | 27.38 | 32.70 | 1.206x |
| HND H56 N8192 | 27.96 | 32.55 | 1.176x |
| HND H56 N16384 | 27.54 | 32.31 | 1.173x |
| NHD H16 N4096 | 27.48 | 28.03 | 1.018x |
| NHD H16 N8192 | 27.21 | 30.04 | 1.089x |
| NHD H16 N16384 | 27.03 | 29.70 | 1.097x |
| NHD H32 N4096 | 25.31 | 29.04 | 1.145x |
| NHD H32 N8192 | 25.54 | 29.91 | 1.160x |
| NHD H32 N16384 | 21.95 | 28.73 | 1.309x |
| NHD H56 N4096 | 23.24 | 29.07 | 1.249x |
| NHD H56 N8192 | 23.44 | 29.05 | 1.239x |
| NHD H56 N16384 | 23.77 | 28.85 | 1.213x |

Measured with `python -m bench.benchmark_attn_fwd --impls aiter,feather_bf16_d128 --dtypes bf16 --head-dims 128`, i.e. both tile configurations beside the control, with the shipped per-layout choice applied: `HND -> config 1`, `NHD -> config 2`. Raw data: `~/tmp/feather_attn/hip_opt/final_lazy.json`:
- Geometric mean over the 18 rows: 1.165x against AITER. HND alone 1.164x, NHD alone 1.166x.
- The two shipped configurations are the lazy-base images (configs 5 and 6, see the optimization log below). This matrix is 1.132x on the previous non-lazy images, measured in `~/tmp/feather_attn/bf16_fwd/bench/final2.json`.
- All 18 rows beat AITER. The weakest is NHD H16 N4096 at 1.018x, which is 512 workgroups over 20 processors and therefore the most tail-quantised cell in the matrix.
- The same matrix measured 1.142x in an earlier session (raw data `post_refactor.json`), so the absolute TFLOPS and the ratio both move by ~1% between sessions. This machine's sustained clock drifts by more than that. The per-row ordering is stable.
- The log-sum-exp the operator emits is the natural-log value FlashAttention and the backward kernels use, converted once from the log2 accumulator. Against the definition evaluated in fp64 it measures 5.3e-6 at 4096 keys (`tests/test_attn_fwd_bf16_d128.py::test_bf16_d128_lse_matches_fp64`). Note for anyone touching the recurrence: `__builtin_amdgcn_logf` is the hardware `v_log_f32`, which returns log2 on this target, so the accumulator needs no base conversion inside the loop. Scaling it breaks the value by exactly that factor.
- Cost of the emitted log-sum-exp: one store per query row per block. H16/N4096/D128 measures 3.957 ms against 3.935 ms without it (`~0.6%`, inside session noise).
- Accuracy against AITER, gated in `tests/test_attn_fwd_bf16_d128.py`: `atol=rtol=0.02` plus `rel-L2 <= 1e-2`. The measured worst case over the contract is a rel-L2 of 5.0e-3 and an elementwise ratio of 0.26 of the envelope. Against a blocked fp32 reconstruction, rel-L2 is 4.4e-3. Cosine is deliberately not part of any forward gate. It is the backward's primary metric because the optimizer normalizes the gradient magnitude.

## Why the previous kernel left performance on the table

Measured from the released D128 image on this machine (see `~/tmp/feather_attn/q16_analysis/REPORT.md` for the full campaign):
- 2048 ALU cycles of WMMA against 984 non-WMMA issue slots per KV tile, i.e. a 67.5% WMMA-density ceiling, and only 49.3% of the 59.4 TFLOPS peak achieved, so the ALU was 73% busy. The other 27% was stall: four `s_barrier` per KV tile, the V `global_load` fully exposed between two of them, and 17 `scratch_load_b32` spill reloads in the loop, all targeting the same register.
- The E5M2 Q tile is a pure cost on this hardware. gfx1151 has no fp8 WMMA and no fp8 conversion instruction, VGPRs (not LDS) bind the occupancy at 8 waves/SIMD, and Q is 0.4% of both the DRAM and the LDS traffic. It bought at most `+0.79%` in the one place its decode could be removed, and it cost 5.4% output error.

## Design

### Tile and ownership

| Parameter | Value |
| --- | ---: |
| `Br` queries per workgroup | 256 |
| `Bc` keys per KV tile | 64 (HND default) or 32 (NHD default) |
| Threads per workgroup | 512 |
| Waves per workgroup | 16 wave32 waves |
| Query ownership | one wave owns 16 query rows |
| Q storage | bf16 WMMA A-fragments in registers, 8 tiles x 8 VGPR |
| K/V storage | bf16 in LDS, double buffered |
| Score and output state | fp32 in registers |

A 256-row workgroup is what makes double buffering affordable: LDS is allocated per workgroup, so 16 waves of ownership double the per-wave LDS budget relative to the existing 128-row, 8-wave workgroup, and halving the number of workgroups halves the K/V staging traffic per query row.

### Configurations

| id | tile | LDS | VGPR | waves/SIMD by VGPR | by LDS | default for |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 256x64, double buffered, swizzled | 65,536 B | 256 | 5 | 8 | HND |
| 2 | 256x32, double buffered, padded | 37,888 B | 188 | 8 | 12 | NHD |
| 3 | 256x64, single buffer, padded | 35,840 B | 213 | 7 | 12 | alternate |
| 4 | 128x32, double buffered, padded | 37,888 B | 198 | 7 | 6 | `n_q < 256` |
| 5 | 1 with the lazy exponential base | 65,536 B | 256 | 5 | 8 | HND, shipped |
| 6 | 2 with the lazy exponential base | 37,888 B | 187 | 8 | 12 | NHD, shipped |

Configs 5 and 6 are the shipped pair: the same two geometries with the lazy exponential base, measured +2.4% head-major and +3.8% token-major against 1 and 2 (see the optimization log below), and `DefaultBf16D128Config` returns them. Config 1 has the highest instruction efficiency and one barrier per KV tile but occupies only one workgroup per processor (a 16-wave workgroup needs 4 waves per SIMD and 5-wave VGPR headroom cannot hold two). Config 2 reaches two workgroups per processor at the cost of twice the per-key tile overhead. HND rows prefer 1 by 5-8%, NHD rows prefer 2 by up to 13%, so the host picks per layout.

Config 1 carries 8 bytes of private storage and one spill. The other three images are clean. It is the only image that exceeds zero private segment, and it is also the largest tile, so the trade is accepted and recorded here rather than hidden.

### Q in registers, no LDS for Q

A wave only ever reads its own 16 query rows, so Q never needs to be shared. The WMMA A-fragment layout (ISA 7.9) is exactly "lane `l` holds `A[l%16][0..15]`, lanes 16-31 replicate lanes 0-15", which is what a per-lane contiguous 32-byte load produces when lanes 16-31 simply re-read the address of lanes 0-15. The duplicate load is merged by the coalescer and costs 0.2% of the tile's traffic.

This removes the E5M2 encode pass, the Q LDS region, the per-KV-tile decode, and the `key_start == 0` predicated block that the old compiler had sunk into the loop body. It also makes Q exact: bf16 in, bf16 into the WMMA.

Q cannot move into LDS at any useful occupancy: a bf16 Q tile is `kBr * 128 * 2` bytes, i.e. 64 KiB at `Br = 256` and 32 KiB at `Br = 128`, which is the entire per-workgroup LDS budget in both cases.

### Dataflow

Per workgroup, per KV tile `i`:

```
# invariant at loop top: K[i%2] and V[i%2] staged and visible to all waves.
# k_next/v_next registers hold the global data for tile i+1.
load k_lo, k_hi from LDS                     (2 per (d_tile, n_tile), 128 total)
QK: 16 WMMA per wave using Q registers and K[i%2]  (transposed C: lane owns one query row)
online softmax: row max, exp2, row sum, log-domain lse, alpha/beta
P -> A: 4 packed pairs, 1 lane-group exchange, 8 selects per 16x16 fragment
PV: 16 WMMA per wave using the P fragment and V[i%2]
STS k_next -> K[(i+1)%2], STS v_next -> V[(i+1)%2]
LDG k_next, v_next = tile i+2
barrier
```

One barrier per KV tile instead of four. The global load for tile `i+2` is issued one full tile ahead of its `s_waitcnt`, so its latency is covered by a whole QK/PV block of WMMA.

### Score layout and softmax

Unchanged from the accepted row-owned idea, because it is the reason the softmax needs no cross-lane reduction along the key axis:
- Scores are computed as `K * Q^T`, so the WMMA C fragment gives each lane the 8 keys of the single query `lane%16`.
- Row max uses `v_max3_f32` plus one `v_permlanex16` to join the two lane halves, likewise the row sum.
- The softmax scale `log2(e)/sqrt(128)` is applied inside the exponentiation as `exp2(fma(score, scale, -c))`, so it costs no instruction relative to the unscaled `v_sub_f32` form and Q keeps its full bf16 precision.
- The probability tile is scaled by `beta` before the PV GEMM and the persistent fp32 accumulator by `alpha`, which is fanned out with 8 `v_readlane_b32`.

### Key permutation for the P transpose

The A fragment the PV GEMM needs is "lane `l` holds 16 contiguous keys of query `l`", while the score fragment gives a lane 8 keys of one query. One lane-group exchange is therefore unavoidable, but the cost depends on which keys a lane owns.

The kernel permutes the keys of each 16-key score tile so lane group 0 owns keys 0..7 and lane group 1 owns keys 8..15, instead of the natural even/odd split. Then each lane's own bf16 pairs are already the pairs the A fragment needs, and the transpose becomes, per 16x16 fragment: 4 packed pairs (3 ALU ops each: shift, and, or), 4 lane-group exchanges of packed pairs, and 8 selects to place the two halves. 24 instructions per fragment instead of 40. The permutation is applied while staging K and is also reflected in the tail handling.

### bf16 probability rounding

`p -> bf16` uses truncation of the fp32 mantissa (two ALU ops per pair instead of the five the compiler emits for a correct round-to-nearest-even `bf16` cast, which handles NaN and denormal edges this kernel does not have). Truncation alone biases the probability tile low by half an ulp, so `(1 + 2^-9)` is folded into `beta`, which cancels the bias to first order for a uniformly distributed mantissa. Measured cost of the alternative (round-half-up, `+0x8000` per value) is 64 extra instructions per KV tile, i.e. about 2% of wall time.

### KV tail without a per-tile mask

The existing kernel writes a `-inf` into every out-of-range score, which the compiler speculates into 32 compares plus 63 selects per KV tile, executed on every tile (measured in the loop body: the mask was 7% of all issue slots).

Instead, the out-of-range K and V rows are zeroed while staging. Their scores are then exactly `0`, so their probability is exactly `exp2(-scaled_max)`, and their PV contribution is exactly zero because V is zero. The only correction needed is to subtract `n_invalid * exp2(-scaled_max)` from the lane's row sum, where `n_invalid` is a few scalar ops per tile inside a `tail_tile` branch.

### LDS layouts

Two layouts are provided, selected per configuration:
- Padded (`kSwizzled = false`): each K row is `128 + 8` elements and each V row is `Bc + 8` elements. The pad makes the row stride coprime with the 128-byte bank cycle, so a 16-lane 128-bit read still hits its 2-cycle minimum, and it makes every LDS address affine in the tile index: one base register plus compile-time immediates, with the `ds_*` immediate field holding the whole tile stride. This removed 119 integer adds, 28 xors and the associated live ranges from the KV loop (loop size 886 -> 654 instructions including the mask removal).
- Swizzled (`kSwizzled = true`): the compact xor-swizzled layout that the previous kernel used. It is the only way to fit four 64-key buffers in the 64 KiB workgroup budget, and it costs a runtime xor, shift and add per V load.

K keeps the "chunk xor row" swizzle in the swizzled mode, which is free because it is loop-invariant, and the second d-chunk is `+/-16` bytes away depending on bit 0 of the lane row.

### Addressing

All global offsets are signed 32-bit element offsets from a per-operand base that the host validates with the existing `IsInt32Addressable` check. The query row is clamped for the tail, so the Q load and the output store need no separate specialization. The kernel has no query-tail, key-tail, or combined-tail images: tails are runtime values.

### Host policy

- `layout` selects both the grid order (head-fast for `[B,N,H,D]`, head-slow for `[B,H,N,D]`) and which of the two middle axes is the sequence axis. One layout applies to all four operands: the kernel ABI carries per-operand strides, but a mixed call (Q/K HND, V NHD, as some rows in `docs/input_shapes_attn.md` use) would need a per-operand axis hint that neither the operator schema nor the host provides, so it is not supported. Transpose the odd operand out, or extend the schema.
- `softmax_scale` is an operator argument (`1/sqrt(D)` when omitted) and travels through the ABI as `log2(e) * softmax_scale`, the same convention and default as `feather_attn_bwd`, so a forward and a backward cannot disagree about it.
- No NHD head grouping. The previous kernel grouped NHD heads into LLC-sized sublaunches, which was worth 13% there. With 256-row workgroups the K/V traffic is already halved and each workgroup streams its own tile, so the grouping is now a loss: measured +3.0%, +6.6% and +8.6% on three NHD rows when it is disabled, and it is gone.
- Config selection is `head_fast ? 2 : 1`, with the 128-row config reserved for `n_q < 256`.

## Verification

- `tests/test_attn_fwd_bf16_d128.py`: 143 cases, all passing. Every shape of `docs/input_shapes_attn.md`'s contract family is covered for both layouts against AITER at `atol=rtol=0.02`, plus an independent blocked-fp32 anchor on the small shapes. The sweep includes lengths 1 through 16,385, both tails, independent query and KV tails, odd and large head counts, `n_q > n_kv` cross-attention, batch two, and the `1023/1024/1025` boundary.
- `tests/test_attn_fwd_fp16_q8.py` (the existing fp8 kernel, 188 cases) still passes unchanged.
- `ruff check`, `ruff format`, and `ty check` are clean on the Python additions.

## Remaining optimizations on the HIP kernel

A measured hardware study of this kernel (`~/tmp/feather_attn/asm_fwd/IDEAS.md`, probes and disassembly beside it) changed the ranking of what is left. Two ceilings stack, and the shipped kernel is at 85-90% of the first one:
- LDS fragment bandwidth. Every WMMA reads one 16x16 fragment as two `ds_load_b128` whose lanes 16-31 duplicate lanes 0-15, so 1 KB is read per WMMA for 512 B of data, and the LDS delivers 234 GB/s per WGP for that access shape. At the register-only WMMA rate this caps the current dataflow at ~40 TFLOPS with the best schedule tried (8 waves/SIMD, prefetch depth 2) and at 20-30 TFLOPS with the depth-1 schedule the compiler emits. The shipped kernel at 31-35 TFLOPS is therefore near its structural ceiling, which is why removing the mask or the E5M2 Q tile bought so little.
- Non-WMMA VALU, which shares the WMMA issue resource. Per KV tile the loop has 590-740 non-WMMA instructions, and `v_exp_f32` alone is 136 cycles of them: it is quarter rate at 4.25 cycles per instruction with zero overlap with WMMA. The earlier loop accounting priced it at one cycle, so the transcendental share was undercounted fourfold.

The scheduler probes recorded in `docs/feather_attn_gfx1151_fwd_tensile_plan.md` later reframed the second ceiling: the issue cost of a softmax instruction depends on how many waves share the SIMD, 0.34 cycles per add at one wave per SIMD against 2.2 at four, so at the four waves per SIMD this kernel runs, the 590-740 non-WMMA instructions cost roughly twice their slot count. They also settled that staging is limited by a per-CU load issue port (~4-5 cycles per `b128`) rather than by bandwidth. Both point at occupancy, which is lever 19 below.

The first ceiling is a tiling change and the second is arithmetic, but only the second is reachable: the tiling change needs two query row groups per wave, and that does not fit the register file (see the log below). The schedule controls are what a generator adds, see `docs/feather_attn_gfx1151_fwd_tensile_plan.md`.

### Optimization log

Every lever was built as an extra `config` id beside the incumbent, so both variants stay in the same binary, and measured with a paired harness (`~/tmp/feather_attn/hip_opt/paired.py`): the two alternate inside every round, the statistic is the median paired log ratio with a 95% interval from a robust sigma (max of stdev, 1.4826 MAD, IQR/1.349. SE = 1.2533 sigma / sqrt(n), the estimator of `~/evotensile/docs/noisy_measurements.md`), and the promotion gate is that the pooled interval's lower bound clears a 0.5% indifference zone. Calibration is a null test, a configuration against itself over 12 rounds: effect -0.03%, interval [-0.13, +0.05], per-row sigma 0.2-0.7%. A 1% effect is resolvable, a 0.2% effect is not, which is why two of the entries below are rejections rather than wins. Raw data: `~/tmp/feather_attn/hip_opt/` and `/tmp/ab_*.log`.

Accepted, in the shipped binary:

| Id | Lever | Measured | Where it lives |
| ---: | --- | --- | --- |
| 5 | Lazy exponential base: the base stays fixed until a tile's row max moves past `base + kTheta` (8 log2 units), so the per-tile accumulator rescale, the per-tile normalisation of P and the log-sum-exp update all leave the loop. The accumulator holds the unnormalised sum of `p*v` and is divided once in the epilogue with the same per-row `readlane` fan-out. The branch is wave-uniform (`__any_sync`) because `readlane` across divergent lanes is undefined. It also lets the `(1 + 2^-9)` truncation-bias correction on beta go: the same relative error now sits in the numerator and the denominator of the final division, so it cancels to second order | +2.4% head-major (config 1 -> 5), +3.8% token-major (2 -> 6), intervals clear the gate on every row | `kLazy` in the kernel, configs 5 and 6, and `DefaultBf16D128Config` returns them |
| 6 | Four-way tree row max and row sum instead of one 16-link serial chain | Inside 5 and 6. The chain was the dependency the WMMA schedule could not hide | Same instantiation |

Rejected, with the measurement that killed it. The extra images are removed from the build:

| Id | Lever | Measured | Why |
| ---: | --- | --- | --- |
| 1 | 32 query rows per wave, so every K and V fragment read from LDS feeds two WMMAs and the LDS bytes per WMMA halve. Four variants: `Bc` 32, `Bc` 16, 64-key swizzled, and with the lazy base | Numerically correct on the first build, but 32-74% slower. At `Bc` 32 the kernel uses 256 VGPRs and still spills 404 B. At `Bc` 16 the compiler targets 192 VGPRs and spills Q plus the score tile, 524 B | A wave may use at most 256 VGPRs, and 32 rows need 128 for the fp32 accumulator plus 64 for Q before any transient. Flattening the per-group arrays into separate 1-D arrays (to keep scalar replacement) and pinning `__launch_bounds__` to one block per CU both changed nothing. Reopening it needs a different accumulator representation (fp32 in LDS, or a D-split two-pass), not a better schedule |
| 7 | `v_perm_b32` (`__builtin_amdgcn_perm`) pair pack: one instruction instead of shift/and/or | +0.21% token-major, +0.7% and -0.05% on the two head-major rows. The pooled interval straddles zero | Below the indifference zone. The loop is LDS-bound, so the two removed instructions per pair hide behind the LDS waits |
| 17 | Tail-free aligned images: no query row clamp, no tail correction, no bounds checks, selected by the host when `n_q % rows == 0` and `n_kv % Bc == 0` | -0.20% to +0.04%, intervals straddle zero | The forward's tail cost is one `cndmask` and one clamp, not a predicate on every access. The backward measured +6.6% for the same idea because its predicates sit inside the fragment traffic |
| 10 | Mask-free fast path by padding K/V to a multiple of `Bc` on the host, so one image serves every shape | Not built separately | 17 shows that removing the tail costs nothing measurable, so paying a 16 MB padding pass to reach it cannot pay either |
| 9, 18 | Affine 32-bit addressing with the tail in a cold block, and lane-base hoisting | Already the kernel's design: every LDS and global offset is a signed 32-bit affine expression of the tile index, and the row clamp is the only remaining special case. The fp8 kernel's per-tile `v_mad_u64_u32` and the backward's D128-B5 hoisting problem do not exist here | Nothing to do |

### Remaining levers

The ids are stable across this document and `docs/feather_attn_gfx1151_fwd_tensile_plan.md`. The settled ones are in the log above.

| # | Lever | Mechanism | Expected | Evidence |
| --- | --- | --- | --- | --- |
| 2 | Counted-wait prefetch depth 2 | Two LDS buffers plus a VGPR read-ahead set, one counted `s_waitcnt lgkmcnt(N)` per group instead of `lgkmcnt(0)` per fragment. `__builtin_amdgcn_s_waitcnt` is callable from HIP, so this is a source change, not asm | +35% in the LDS-fed regime. Depth 4 regresses | `1 wmma/f` probe: 19.9 -> 26.7 (4 waves) and 29.6 -> 40.3 (8) |
| 3 | Cross-boundary prefetch | The V fragments of tile i do not depend on its softmax, so their loads belong before the max/exp block. Tile i+1's first K fragments belong before the barrier that publishes its buffer | Removes the two exposed LDS round trips per tile | Disassembly: `ds_load` x2 -> `s_waitcnt lgkmcnt(0)` -> `v_wmma`, 64 times per tile, both boundaries exposed |
| 4 | fp16 P and a packed fp16 softmax tail | `v_cvt_pk_rtz_f16_f32` (one instruction per pair, reachable as `__builtin_amdgcn_cvt_pkrtz`), `v_pk_fma_f16` for the scale and base (`__hfma2`), `v_exp_f16` for the exponential (`h2exp2`), and the packed result is the A operand of an `f16` PV WMMA, so nothing is converted back. V is staged as fp16 (converted once per workgroup tile, amortised over the waves that share it) | ~60 cycles of ~1385 per tile. Deletes the bf16 pack and the `(1+2^-9)` truncation-bias correction, and raises P from 8 to 10 mantissa bits | `v_exp_f16` measures ~2 cycles per element against 4.25 for `v_exp_f32`. The conversions above were verified to come out of HIP as the intended instructions |
| 8 | Drop the exchange's `v_cndmask` | Let the two lane groups read V in different 8-key orders (group 0 reads chunks (0,1), group 1 reads (1,0) via one address bit), so both can use `[own \| partner]` directly | Removes 8 `v_cndmask` per fragment | The shipped loop carries 19-30 `v_cndmask` per tile |
| 11 | LDS pitch and access-shape audit | Confirm the shipped K/V pitches (272 B, 144 B, and the config-1 swizzle) against `probe_lds.cu` | Free: a wrong pitch class costs 5x | 136/148/276 B or >=256 B power-of-two pitches read at 46 GB/s against 234 GB/s for 144/272 B. 16 B-aligned 4-dword-multiple pitches sit in the 234 GB/s class |
| 12 | Wave priority | `s_setprio 3` inside the KV loop and `s_setprio 0` in the prologue and epilogue (`__builtin_amdgcn_s_setprio`, or LLVM's off-by-default `-mllvm -amdgpu-set-wave-priority`), so a wave that is only feeding VMEM does not win the issue slot against a wave with a ready WMMA | Unmeasured | Arbiter key is `{min(3, SysPrio + UserPrio), WaveAge}`. User priority is settable per wave at any time |
| 13 | Per-wave fragment rotation | Wave w starts at `d_tile = w mod 8` so the waves of a SIMD do not hit the same banks and the same LDS queue slot in the same cycle | Unmeasured in the real kernel. Neutral in the bandwidth-bound microbenchmark | Free in a generated schedule, one integer add in HIP |
| 14 | Two workgroups per processor | Worth +50% at depth 2 in the probe (26.7 -> 40.3). Needs <= 192 VGPR for two 4-wave workgroups, which the 32-row design only reaches with `Bc = 16` (`S` 16, `P` 8) or with Q in LDS. The backward already ran this experiment and lost: 4 workgroups fit at 176 VGPR only by reloading K/V from L2 and dropping the double-buffered Q/dO staging, which measured 0.72x-0.81x. So the constraint is not just the register count but that the VGPRs must come from the tile representation, never from the pipeline | +50% of the depth-2 gain, only if the registers come for free | `1 wmma/f` probe at 4 against 8 waves/SIMD. `docs/feather_attn_gfx1151_bwd_d128_plan.md` occupancy finding |
| 15 | Persistent grid, L2-aware tile order | One workgroup per processor looping over `(head, q_tile)` in an order that keeps a resident K/V slice in L2, instead of launching one workgroup per tile. Attacks both weak cells at once: the NHD H16/N4096 row is 512 workgroups over 20 processors (the most tail-quantised row in the matrix, 1.018x) and the long-N rows re-fetch K/V once per query tile | In-repo precedent: the backward's 64-row KV persistent owner measured 1.1016x over its control, and its 512-thread/256-row persistent Q/dO owner is retained in production | `docs/feather_attn_gfx1151_bwd_d128_plan.md`, accepted mechanisms |
| 16 | LLC-bounded head grouping, revisited | Give each workgroup a subset of heads so its K/V working set fits the LLC, the partition-aware strided mapping (`head = first + i * group_count`) that the fp8 forward already has in its ABI. The bf16 kernel removed grouping on a 256-row-workgroup measurement (3-9% slower), but the mechanism the fp8 kernel measured is working-set size, not tile height | fp8 D128: 1.135x focused geomean over ungrouped, 9/9 wins, fetch 31.4 -> 14.8 GiB at H32/N16384. fp8 D64 NHD H32/N16384: fetch 15.8 -> 13.5 GiB and L2 hits 2.08% -> 12.93%, which is the mechanism named explicitly | `docs/feather_attn_gfx1151_fwd_fp8_plan.md`, accepted optimization effects |
| 19 | Fewer waves per SIMD, so the softmax VALU is cheap | The scheduler probe measured the marginal cost of an interleaved `v_add_f32` next to WMMAs as 0.34 cycles at one wave per SIMD against 2.2 cycles at four. The shipped 256-row workgroup runs four waves per SIMD, so every non-WMMA instruction currently costs about twice its slot count. A 64- or 96-row workgroup puts one wave per SIMD and makes the VALU nearly free, at the cost of the latency hiding the four waves provide | Untested: the probe prices the VALU side at up to 2x and the LDS side negative | `~/tmp/feather_attn/hip_opt/probes/probe_sched2.hip.cpp` (`wmma` kernel), recorded in the Tensile plan |

Order of the remaining work, by expected value per unit of risk:
- Launch policy (15, 16), because it is HIP-able, independent of the schedule, and the only remaining item with a double-digit projection. Both attack the token-major long rows, which hold the two weak cells of the matrix.
- The schedule levers (2, 3, 12, 13) in a generated kernel, since HIP cannot place the LDS waits. Counted-wait depth 2 is the single largest measured headroom (the probe prices it at 40.3 TFLOPS in this dataflow against the 30-33 shipped).
- fp16 P and the packed fp16 softmax tail (4), the last arithmetic lever big enough to measure: ~2.5-3 cycles per probability against ~6 today.
- 8, 11, 14: a `v_cndmask` removal, an LDS pitch audit against `probe_lds.cu` (a wrong pitch class would show up as a 5x cliff, so it is a five-minute check rather than an optimization), and two workgroups per processor, which the backward's occupancy experiment already priced at 0.72x-0.81x unless the registers come from the tile representation.

The forward fp8 and backward plans already paid for part of this list, which is why several levers cite in-repo measurements rather than probes: the pair packing (`__builtin_amdgcn_perm`), the packed convert (`cvt_pkrtz`) and the persistent owner (1.10x) all ship in `csrc/bwd/`. Where their experience did not transfer, the log records it: their tail-free aligned images were worth +6.6% in the backward and nothing here. Three of their negative results transfer just as directly: buying occupancy by giving up the pipeline loses 20-30% (see lever 14), chasing LDS bank conflicts is wasted effort in this kernel class (the D128 forward measures 2.7% conflicts and `ALUStalledByLDS` of 0.026-0.029%, so the LDS problem is instruction bandwidth, not conflicts), and further VGPR reduction through lifetime barriers inside the current representation does not work. The fp8 plan also supplies the ranking rule this table follows: work paid once per query tile is amortized by the KV loop, work paid per KV tile is not, so only the latter is worth a lever.

## Rejected or not pursued (earlier campaign)

These were settled before the hardware study above, when the kernel was being chosen rather than tuned. The current campaign's rejections are in the optimization log.

| Candidate | Evidence | Decision |
| --- | --- | --- |
| fp8 or int8 quantization of Q | No fp8 WMMA and no fp8 conversion on gfx1151. VGPRs bind the occupancy so the LDS saving buys no wave. Q is 0.4% of traffic. 5.4% output error | Not used. bf16 Q, exact |
| 64-key tile with padded LDS and double buffering | 71,680 B > the 64 KiB per-workgroup limit | Impossible. 64 keys needs the swizzled layout |
| 64-key tile with a single K/V buffer and two barriers (config 3) | 213 VGPR, clean, but the second barrier costs 4-20% depending on the row and the occupancy gain (7 waves) does not compensate | Kept as an alternate, not the default |
| 128-row 64-key (8 waves, 240 VGPR) | 2 waves per SIMD effective, 24 TFLOPS | Dropped |
| 256-row 32-key with the swizzled layout | Compiler failure mode: 96 VGPR with 132 spills and 436 B private | Dropped |
| 128-row 32-key 64 KiB | 6 waves by VGPR, dominated by the 256-row variant | Dropped |
| Round-half-up probability rounding | 64 extra instructions per KV tile, no accuracy benefit over the unbiased truncation | Not used |
| Folding `beta` into the exponent | Requires the row sum first, so it needs a second `exp2` pass: 32 transcendental ops replace 32 multiplies | Rejected |
| Deferring the accumulator rescale to the end | The product of the per-tile `beta` underflows after ~250 tiles (`65^-256`), so the accumulator overflows | Rejected |
| NHD LLC head grouping | 3-9% slower with 256-row workgroups | Removed |
| K fragments read straight from global | Each wave would pull 1 KB per WMMA from L1, the same 330 GB/s per WGP the LDS cannot deliver. The L1 has no more | Rejected by measurement |
| Half-active LDS reads plus a cross-half copy | Duplicate lanes pay full bandwidth, but a `v_permlanex16` copy costs 8 instructions per fragment (more LDS-pipe time than the duplicate read) and `v_mov_dpp wave_shl` spends 8 VALU cycles per fragment, more than the VALU has spare at 90% density. Lifting the ceiling needs fewer fragments per WMMA, not cheaper duplicates | Rejected by measurement |
| Polynomial exp2 | Degree-2 polynomial with 8 interleaved chains costs 9.2 cycles per element against 4.25 for `v_exp_f32` | Rejected by measurement |
| Three WMMAs per LDS fragment | Feeds 49.8-51.2 TFLOPS in the probe, but 48 query rows need 192 VGPRs of output accumulator alone, on top of 96 for Q | Impossible within 256 VGPRs |
| Wave64 | A and B fragments replicate 4x instead of 2x, doubling the LDS ceiling problem | Rejected on the fragment model |
| `ds_read2_b128` to coalesce the two fragment reads | The instruction does not exist on gfx1151 (assembler probe). `ds_load_b128` pairs it is | Unavailable |
| `v_pk_exp_f16` | Does not exist. `h2exp2` lowers to two `v_exp_f16_e32`, so the win is per element (2 against 4.25 cycles), not one instruction per pair | Not available as a packed op |

Two entries above are superseded and are now levers rather than rejections:
- `v_perm_b32` does work as documented (LLVM itself emits `v_perm_b32 ... 0x5040100` in our own loop, and the backward's `PackPair` has shipped `__builtin_amdgcn_perm` for the same job), so the pair packs in one instruction instead of three. It was then measured (id 7 in the optimization log) and is a wash in this LDS-bound loop.
- The rejection of "deferring the accumulator rescale to the end" is about an unsound scheme: with no rescale at all, the product of the per-tile `beta` underflows after ~250 tiles and the accumulator overflows. The lazy-base variant that shipped (id 5 in the optimization log) is a different mechanism. It rescales whenever a tile's row max exceeds `b + theta`, which is exact to the usual FA rounding and happens less often than in FA itself, so the base is never stale by more than `theta` in log2 units.

## Open items

- Levers 2 and 3 (counted-wait prefetch depth 2, cross-boundary prefetch) are the largest measured headroom left: the probe prices the depth-2 schedule at 40.3 TFLOPS in this dataflow against the 30-33 shipped. They need the LDS waits placed by hand, so they belong to `docs/feather_attn_gfx1151_fwd_tensile_plan.md`.
- Lever 4 (fp16 P with the packed fp16 softmax tail) is the last arithmetic lever big enough to measure from HIP: ~2.5-3 cycles per probability against ~6 today.
- Levers 15 and 16 (persistent grid with an L2-aware tile order, LLC-bounded head grouping) are the only remaining double-digit projection, and they target the two weak cells of the matrix: the token-major H16 rows at 1.018x and 1.089x, which are 512 and 1024 workgroups over 20 processors. Both are launch policy that this ABI already admits (`head_start`/`head_step`).
- Levers 8, 12 and 13 (drop the exchange's `cndmask`, wave priority, per-wave fragment rotation) are small and unmeasured.
- Lever 14 (two workgroups per processor) stays blocked unless the registers come from the tile representation rather than the pipeline. The backward priced the alternative at 0.72x-0.81x.
- `n_q < 256` uses the 128-row config. Shapes below one 128-tile (the 12- and 16-token text rows) run at low efficiency because the whole grid is a handful of workgroups.
- Everything that needs the arbiter rather than the arithmetic (interleaved schedules, exact VGPR pinning, `s_sleep`/`s_wakeup` producer rings, clause policy, CU mode, `s_sendmsg` dealloc, the geometry search) is out of reach of this file and is planned in the Tensile document.

## Artifacts

- Kernel: `csrc/fwd/bf16/bf16_fwd_d128_kernel.h`, `bf16_fwd_d128_image.cu` (instantiations), `bf16_fwd_d128.cpp` (host and op). The ABI is `csrc/attention_args.h` and the host helpers are `csrc/attention_host.h`.
- Python: `feather_attn/ops_bf16.py`.
- Test: `tests/test_attn_fwd_bf16_d128.py`.
- Bench: `bench/benchmark_attn_fwd.py`.
- Measurements and the tuning campaign: `~/tmp/feather_attn/bf16_fwd/` (`bench/final_matrix.{json,log}`, `correctness_final.log`, `loop_analysis2.txt`, `probe_degenerate.py`, `tol_probe.py`).
- Optimization log campaign: `~/tmp/feather_attn/hip_opt/` (`paired.py`, the statistical harness. `final_lazy.json`, the shipped matrix. `isa_stats.py`, per-config VGPR/private/LDS inspection) and the raw paired runs `/tmp/ab_*.log`.
- Hardware study behind the remaining-optimizations table: `~/tmp/feather_attn/asm_fwd/` (`IDEAS.md`, `probe_sched.cu`, `probe_lds.cu`, raw runs, the shipped kernel's disassembly `k1_hnd_bc64_swz.s` and all eight images `bf16_image_gfx1151.s`).
- Generator plan for the schedule-only levers: `docs/feather_attn_gfx1151_fwd_tensile_plan.md`.
- Prior-art analysis for the fp8-Q kernel: `~/tmp/feather_attn/q16_analysis/REPORT.md`.
