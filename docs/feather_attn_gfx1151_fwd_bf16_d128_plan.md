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
| HND H16 N4096 | 29.98 | 32.78 | 1.093x |
| HND H16 N8192 | 28.64 | 32.11 | 1.121x |
| HND H16 N16384 | 28.49 | 31.94 | 1.121x |
| HND H32 N4096 | 27.85 | 31.38 | 1.127x |
| HND H32 N8192 | 27.67 | 31.93 | 1.154x |
| HND H32 N16384 | 27.63 | 31.31 | 1.133x |
| HND H56 N4096 | 26.87 | 31.43 | 1.170x |
| HND H56 N8192 | 27.28 | 31.30 | 1.147x |
| HND H56 N16384 | 27.10 | 30.87 | 1.139x |
| NHD H16 N4096 | 26.50 | 26.80 | 1.011x |
| NHD H16 N8192 | 26.25 | 28.40 | 1.082x |
| NHD H16 N16384 | 26.76 | 27.96 | 1.045x |
| NHD H32 N4096 | 24.31 | 28.03 | 1.153x |
| NHD H32 N8192 | 24.67 | 28.69 | 1.163x |
| NHD H32 N16384 | 21.84 | 26.84 | 1.229x |
| NHD H56 N4096 | 22.77 | 27.02 | 1.187x |
| NHD H56 N8192 | 23.27 | 27.21 | 1.169x |
| NHD H56 N16384 | 23.61 | 27.05 | 1.146x |

Measured with `python -m bench.benchmark_attn_fwd --impls aiter,feather_bf16_d128 --dtypes bf16 --head-dims 128`, i.e. both tile configurations beside the control, with the shipped per-layout choice applied: `HND -> config 1`, `NHD -> config 2`. Raw data: `~/tmp/feather_attn/bf16_fwd/bench/final2.json`:
- Geometric mean over the 18 rows: 1.132x against AITER. HND alone 1.135x, NHD alone 1.129x.
- All 18 rows beat AITER. The weakest is NHD H16 N4096 at 1.011x, which is 512 workgroups over 20 processors and therefore the most tail-quantised cell in the matrix.
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

Config 1 has the highest instruction efficiency and one barrier per KV tile but occupies only one workgroup per processor (a 16-wave workgroup needs 4 waves per SIMD and 5-wave VGPR headroom cannot hold two). Config 2 reaches two workgroups per processor at the cost of twice the per-key tile overhead. HND rows prefer 1 by 5-8%, NHD rows prefer 2 by up to 13%, so the host picks per layout.

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

- `tests/test_attn_fwd_bf16_d128.py`: 318 cases, all passing. Every shape of `docs/input_shapes_attn.md`'s contract family is covered for both layouts against AITER at `atol=rtol=0.02`, plus an independent blocked-fp32 anchor on the small shapes. The sweep includes lengths 1 through 16,385, both tails, independent query and KV tails, odd and large head counts, `n_q > n_kv` cross-attention, batch two, and the `1023/1024/1025` boundary.
- `tests/test_attn_fwd_fp16_q8.py` (the existing fp8 kernel, 188 cases) still passes unchanged.
- `ruff check`, `ruff format`, and `ty check` are clean on the Python additions.

## Rejected or not pursued

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
| `v_perm_b32` for the packed bf16 pair | Its byte-selection convention could not be established on this target (a probe of the shipped decode pattern shows the selector's nibbles do not map to bytes as documented), so the pair is packed with shift/and/or | Not used |

## Open items

- Config 1 sits at 256 VGPR and one workgroup per processor. Reaching 8 waves/SIMD needs 192 VGPR, i.e. 64 fewer than the tile registers alone (`Q 64 + output 64 + scores 32 = 160`), so it requires a structural change to the ownership scheme, not more local tuning.
- NHD is 1.108x while HND is 1.133x. The two NHD rows at H16 are tail-quantisation limited. A persistent-grid variant is the obvious follow-up.
- `n_q < 256` uses the 128-row config. Shapes below one 128-row tile (the 12- and 16-token text rows) run at low efficiency because the whole grid is a handful of workgroups.

## Artifacts

- Kernel: `csrc/fwd/bf16/bf16_fwd_d128_kernel.h`, `bf16_fwd_d128_image.cu` (instantiations), `bf16_fwd_d128.cpp` (host and op). The ABI is `csrc/attention_args.h` and the host helpers are `csrc/attention_host.h`.
- Python: `feather_attn/ops_bf16.py`.
- Test: `tests/test_attn_fwd_bf16_d128.py`.
- Bench: `bench/benchmark_attn_fwd.py`.
- Measurements and the tuning campaign: `~/tmp/feather_attn/bf16_fwd/` (`bench/final_matrix.{json,log}`, `correctness_final.log`, `loop_analysis2.txt`, `probe_degenerate.py`, `tol_probe.py`).
- Prior-art analysis for the fp8-Q kernel: `~/tmp/feather_attn/q16_analysis/REPORT.md`.
