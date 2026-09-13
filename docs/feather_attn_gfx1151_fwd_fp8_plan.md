# FeatherAttn gfx1151 Forward Kernel

## Hardware and Execution Assumptions

| Item | Value |
| --- | --- |
| GPU | Radeon 8060S, gfx1151 |
| Compute units | 40 CUs, 20 WGPs |
| Wave size | 32 |
| Main comparison | FlashAttention AITER Triton `attn_fwd` |
| AITER stack for the AITER column | Triton `3.8.0+git669b31ac.rocm10.2.0a20261003` (LLVM 23 fork), torch `2.14.0+rocm10.2.0a20261003` |
| AITER gfx1151 tiles | `~/aiter` `796961330`: D64 `128/32/w4/waves1`, D128 `128/32/w8/waves6`, D256 `128/32/w8/waves2` |
| Arithmetic | FP16 WMMA with FP32 score and output accumulation |

## Public Contract

The public operation is:

```text
out = softmax(Q @ K^T / sqrt(D)) @ V
```

| Property | Current support |
| --- | --- |
| Input/output dtype | FP16 |
| Layouts | Contiguous HND `[B,H,N,D]` and NHD `[B,N,H,D]` |
| Head dimensions | 64 and 128 |
| Head count | Any positive count within checked launch/address limits |
| Query and KV lengths | Positive, independent, and not required to align to a tile |
| Attention | Dense, non-causal, equal query/KV heads |
| Unsupported | Masks, dropout, bias, ALiBi, windows, GQA/MQA, other dtypes, non-contiguous tensors |

The host validates dimensions, contiguity, layout, dtype, device, sequence/grid bounds, and signed-int32 addressability before narrowing any device-side index. Sequence length changes only grid size and the number of KV-loop iterations. It does not change per-workgroup LDS or the compiled kernel shape.

Accuracy is checked against AITER-backed FP32 reconstruction. The public elementwise gate is `atol=rtol=0.10` for `N_KV < 1024` and `atol=rtol=0.05` for `N_KV >= 1024`. The full test also requires finite output and bounded relative-L2 and normalized error.

## Current Design

### Tile and Ownership

| Parameter | Value |
| --- | ---: |
| Query tile `Br` | 128 rows |
| KV tile `Bc` | 64 rows |
| Workgroup | 256 threads |
| Waves per workgroup | 8 wave32 waves |
| Query ownership | One wave owns 16 query rows |
| Q storage in LDS | E5M2, 1 byte per value |
| K/V storage in LDS | FP16 |
| QK and persistent output state | FP32 |

One workgroup owns one 128-row query tile for one physical head. Each wave owns 16 query rows from score generation through softmax and output accumulation. This row ownership keeps the score fragments, probability conversion, and output fragments inside the producing wave and avoids a full score/probability LDS tile.

### Dataflow

- Load one 128-row Q tile, multiply by `log2(e) / sqrt(D)`, round to E5M2 with RNE, and store the compact values in lane-major LDS.
- Stage one 64-row K tile in FP16 LDS.
- Decode Q fragments to FP16 immediately before WMMA and compute transposed score fragments as `K * Q^T`. The transposed C layout gives each lane one query row for the local softmax reduction.
- Apply the KV-tail mask, perform the online softmax update in FP32, rescale the persistent FP32 output fragments, and convert the current probabilities directly from the WMMA C layout to WMMA A registers.
- Stage V with the rotating-shared swizzle and accumulate `P @ V` into the wave-owned FP32 output fragments.
- Convert the final output fragments to FP16 and store them with query-tail guards.

The accepted recurrence keeps a log-domain `lane_lse` state. A linear `(running_max, running_sum)` recurrence was correct and removed one logarithm, but it was slower on gfx1151 and is not used.

### LDS Layout

| D | Q8 LDS | Reused K/V LDS | Total LDS |
| ---: | ---: | ---: | ---: |
| 64 | 8,192 B | 8,192 B | 16,384 B |
| 128 | 16,384 B | 16,384 B | 32,768 B |

K and V share one lifetime-reused LDS region. K uses an XOR-swizzled row/chunk layout. For D64, V uses the Triton-style rotating phase

```text
n_chunk XOR ((d % 8) XOR ((d / 8) % 8))
```

which keeps D64 conflict-free at the profiler's resolution and limits D128 to a small repeatable residual conflict.

### Specialization and Dispatch

The kernel is compile-time specialized by:
- head dimension: D64 or D128.
- physical layout: HND or NHD.
- query tail: aligned or guarded.
- KV tail: aligned or guarded.

The base set contains 16 images. Four additional D64 NHD strided images implement the accepted long-sequence partition-aware mapping.

The host selector is `SelectLauncher(int64_t head_dim, bool nhd, bool pad_q, bool pad_kv)`. It selects the D64 or D128 HND/NHD entry point from the query-tail and KV-tail flags. Aligned, query-tail, KV-tail, and combined-tail paths are distinct symbols. The separate D64 NHD strided selector is used only after the long-sequence grouping policy admits partition-aware launches. This selector topology and the raw ABI are part of the qualified contract.

HND uses flattened batch/head ownership. NHD makes head the fastest grid axis so adjacent workgroups touch adjacent head offsets. Two bounded host policies handle long NHD tensors:
- D128 may split heads into sequential LLC-sized groups when the total K/V working set is large enough to benefit.
- D64 may use the strided physical-head mapping `group_index + local_head * group_count` for any admitted positive head count. The selector avoids group counts divisible by eight because those strides recreate the gfx1151 memory-partition cliff. It does not require the physical head count to be divisible by 16.

All sublaunches run on the caller's current stream. Tail selection and grouping do not change the tensor layout or require a physical transpose.

### Q Decode Caching

D64 HND has enough register headroom to cache decoded Q fragments across KV tiles. Aligned, query-tail, and combined-tail images cache all four D16 fragments. The KV-tail image caches three. D64 NHD and all D128 images retain decode-on-use because their measured tradeoffs or resource pressure do not justify persistent decoded Q state.

### E5M2 Conversion and Packing

The numerical baseline is RNE. The current scalar encoder is equivalent to:

```text
rounded = fp16_bits + 0x007f + ((fp16_bits >> 8) & 1)
e5m2 = rounded >> 8
```

The decode path already loads packed bytes and uses `v_perm_b32`. gfx1151 has no native FP8 WMMA path, so Q quantization and decode are VALU/frontend work around FP16 WMMA. Prior tracing attributed roughly 23% of math-pipeline execution to unpacking. Explicit `uint32` packing with `0x07050301` byte-permute control is the strongest low-risk instruction-level lead found so far, but its setup cost is too small relative to the KV loop to justify a standalone production change.

The packed candidate must be treated as an integrated kernel candidate. A standalone pack microbenchmark measured `0.0563 ms` versus `0.0974 ms` for scalar packing, but the complete aligned kernel did not reproduce that gain. Future work should only revisit packing when it is fused with Q load/scale and also shortens live state or creates an occupancy transition.

## Resource Results

The table reports exact loaded gfx1151 metadata from the qualified production extension.

| Kernel group | Used VGPRs | LDS | Private/spills | Notes |
| --- | ---: | ---: | ---: | --- |
| D64 HND, all tail modes | 171-175 | 16,384 B | 0 / 0 | Expanded decoded-Q cache |
| D64 NHD, base and strided | 130-162 | 16,384 B | 0 / 0 | Includes four partition-aware images. The combined-tail image is the upper end |
| D128 HND, all tail modes | 191 | 32,768 B | 0 / 0 | Resource-pinned |
| D128 NHD, all tail modes | 191 | 32,768 B | 0 / 0 | Resource-pinned. Optional grouped launches |

The maximum SGPR count is 46. Base kernels use 56-byte arguments. Strided D64 NHD kernels use 64-byte arguments. Every image uses wave32 and has zero private segment, scratch, and register spills.

## Current Benchmarks

### Method

The authoritative matrix uses batch one, heads `{16,32,56}`, sequence lengths `{4096,8192,16384}`, D `{64,128}`, and both physical layouts. Each provider uses `triton.testing.do_bench` with `warmup=25` and `rep=100`.

Forward throughput counts the two attention GEMMs:

```text
FLOPs = 4 * B * H * NQ * NKV * D
TFLOPS = FLOPs / elapsed_seconds / 1e12
```

`Feather / AITER` is the throughput ratio. Values above `1.000x` favor FeatherAttn. AITER consumes zero-copy transposed views for the HND comparison.

Raw matrix: `~/tmp/feather_attn/fwd_autonomous_20260817/non16_selector_final_matrix/attn.csv`.

AITER matrix: `~/tmp/feather_attn/attn_triton_fwd_tune/plan_matrix_triton38.{csv,json}`, produced by `bench_plan_matrix.py` (three rounds per row, median of round medians, same `do_bench` settings, production dispatch path).

The AITER column was re-measured with the Triton build listed in the hardware table and the gfx1151 tuned tiles. Only launch knobs differ from upstream AITER. The kernel body is untouched. The same shape and tile can drift by up to about `20%` between sessions, so re-measuring Feather against this AITER column is a separate task.

A Triton or LLVM update changes the generated code for these kernels, and therefore the winning tile, so the gfx1151 tiles must be re-tuned after a Triton upgrade. The AITER column came from an LLVM-24 build. Under the LLVM-23 build used here the same `BLOCK_N=64` D128 tile grows from `233` VGPR with zero spills to `256` VGPR with `120` B of spills, and is up to `65%` slower on one row (`H24 N4096 D128 NHD`: `6.48 ms` under LLVM 24 against `10.73 ms` under LLVM 23, same launch configuration). The AITER numbers above land at `0.9713x` geometric mean against the Triton main branch.

### D64 Results

Every layout, head count, and sequence length in the authoritative matrix is listed here. `Feather / AITER` is the throughput ratio. Values above `1.000x` favor FeatherAttn.

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 33.092 | 35.164 | 1.063x |
| HND | 16 | 8192 | 31.370 | 33.716 | 1.075x |
| HND | 16 | 16384 | 30.087 | 32.957 | 1.095x |
| HND | 32 | 4096 | 31.922 | 34.279 | 1.074x |
| HND | 32 | 8192 | 28.836 | 33.300 | 1.155x |
| HND | 32 | 16384 | 29.299 | 33.076 | 1.129x |
| HND | 56 | 4096 | 29.865 | 32.508 | 1.089x |
| HND | 56 | 8192 | 29.303 | 32.990 | 1.126x |
| HND | 56 | 16384 | 29.208 | 33.278 | 1.139x |
| NHD | 16 | 4096 | 29.898 | 32.101 | 1.074x |
| NHD | 16 | 8192 | 28.695 | 30.833 | 1.074x |
| NHD | 16 | 16384 | 28.000 | 31.397 | 1.121x |
| NHD | 32 | 4096 | 28.936 | 30.543 | 1.056x |
| NHD | 32 | 8192 | 25.691 | 29.233 | 1.138x |
| NHD | 32 | 16384 | 23.817 | 27.303 | 1.146x |
| NHD | 56 | 4096 | 27.439 | 30.388 | 1.107x |
| NHD | 56 | 8192 | 27.058 | 30.803 | 1.138x |
| NHD | 56 | 16384 | 27.648 | 30.947 | 1.119x |

### D128 Results

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 30.767 | 33.885 | 1.101x |
| HND | 16 | 8192 | 30.459 | 32.848 | 1.078x |
| HND | 16 | 16384 | 30.883 | 33.334 | 1.079x |
| HND | 32 | 4096 | 29.630 | 34.161 | 1.153x |
| HND | 32 | 8192 | 28.775 | 33.196 | 1.154x |
| HND | 32 | 16384 | 29.985 | 33.940 | 1.132x |
| HND | 56 | 4096 | 28.582 | 33.373 | 1.168x |
| HND | 56 | 8192 | 29.208 | 32.915 | 1.127x |
| HND | 56 | 16384 | 29.375 | 34.064 | 1.160x |
| NHD | 16 | 4096 | 27.424 | 30.817 | 1.124x |
| NHD | 16 | 8192 | 26.960 | 30.655 | 1.137x |
| NHD | 16 | 16384 | 28.226 | 30.425 | 1.078x |
| NHD | 32 | 4096 | 25.025 | 30.121 | 1.204x |
| NHD | 32 | 8192 | 26.262 | 30.361 | 1.156x |
| NHD | 32 | 16384 | 21.514 | 26.542 | 1.234x |
| NHD | 56 | 4096 | 25.377 | 30.479 | 1.201x |
| NHD | 56 | 8192 | 26.350 | 31.168 | 1.183x |
| NHD | 56 | 16384 | 26.521 | 31.180 | 1.176x |

Short, independent-tail, arbitrary-head, and batch-two cases are correctness requirements rather than part of this throughput matrix. The `188/188` public test covers lengths from 1 through 16,385, tails on both sides of every primary sequence length, the `1023/1024/1025` tolerance boundary, independent query/KV tails, odd head counts, and batch two.

## Current Profile Results

### Accepted Optimization Effects

| Change | Measured effect | Current use |
| --- | --- | --- |
| Expanded D64 HND decoded-Q cache | `+0.789%` geomean, 9/9 wins, dynamic VALU `-3.39%`, LDS instructions `-3.83%` | Enabled for D64 HND |
| Partition-aware D64 NHD strided groups | `+21.280%` selected-domain geomean, 12/12 wins | Guarded long-NHD selector |
| Arbitrary-head D64 NHD strided eligibility | Non-16 paired screen `+2.406%`, 7/8 wins. Two complete nine-row brackets pooled to `+1.274%`, 15/18 wins. Newly selected H56/N8192 and H56/N16384 won every bracket by `3.99-5.60%`. | Removed the obsolete head-count divisibility guard. All LLC and partition gates remain |
| D64 NHD H32/N16384 traffic | Fetch `15.830 -> 13.554 GiB`. GCEA reads `15.858 -> 13.460 GiB`. L2 hit `2.08% -> 12.93%` | Explains the `1.146x` AITER ratio at this row |
| Bounded D128 NHD LLC grouping | `1.135x` focused geomean over ungrouped, 9/9 wins. H32/N16384 fetch approximately `31.435 -> 14.822 GiB` | Enabled when the K/V working set crosses the LLC gate |

### LDS Bank Conflicts

| Layout/domain | AITER | Feather | Interpretation |
| --- | ---: | ---: | --- |
| D64 HND/NHD | 0.000% | 0.000% | Conflict-free at counter resolution |
| D128 H16/N4096 | 2.849% | 2.702% | Small repeatable residual |
| D128 NHD H32/N16384 | 2.855% | 2.702% | Grouped path does not add conflict |

The D128 residual is not the primary bottleneck: prior pressure profiles measured Feather `ALUStalledByLDS` at only `0.026-0.029%`. D128 is instead constrained by its 191-VGPR, 32-KiB image and by instruction/dependency work. Long NHD cases are primarily controller-traffic and cross-workgroup reuse problems, which is why launch grouping produced larger gains than local arithmetic edits.

### D128 Compiler Pressure

Optimized gfx1151 MIR at `-stop-after=phi-node-elimination` reports a maximum virtual VGPR pressure of 216 at the KV-loop backedge, with 225 virtual registers live. That point is dominated by overlapping old and new phi-copy versions of the 64 scalar FP32 output components. Linked allocation coalesces those versions and remains authoritative at 191 VGPRs with no private memory or spills. The MIR peak is not evidence of a linked spill.

The actionable WMMA boundaries range from 176 to 209 virtual VGPRs as score and PV fragments become live alongside the eight FP32 output fragments. Direct source attempts did not produce a usable reduction: splitting D128 V staging preserved 191 VGPRs, replacing the tied WMMA assembly worsened at least one layout, and constraining QK address lifetimes introduced spills. Further D128 work therefore needs to eliminate persistent fragment state or change its representation without extra passes or LDS traffic. More local lifetime barriers are not an open direction.

## Investigation Closeout

The final exploratory question was whether the Q FP16-to-E5M2 path could remove RNE work or use explicit packed operations without changing the production contract.

| Candidate | Numerical result | Linked resource result | Timing result | Decision |
| --- | --- | --- | --- | --- |
| Packed RNE encoding | `168/168` public cases. Bitwise identical to the scalar-RNE aligned image in the focused comparison | Same 20-image resource envelope. D64/D128 stayed at the existing VGPR and LDS values | 12 aligned paired rows: `0.9969x` geometric mean, 4/12 wins, range `0.9828x-1.0063x` | Promising ISA experiment, no promotion |
| Scalar and packed E5M2 truncation | Exhaustive 65,536-pattern conversion probe had zero GPU-reference mismatches, but the production-tree scalar candidate passed only `52/168` public cases. Scalar and packed attention runs reached `Rel-L2` about `0.12-0.125` and failed elementwise cases in HND/NHD and D64/D128. | No resource issue in the disposable aligned image | Not relevant after numerical failure | Rejected. RNE restored |
| Half-up alternative | Finite and contract-qualified in the prior campaign, but not output-exact and had no repeatable timing gain | No useful allocation transition | Prior robust result was `-0.176%` overall | Closed. Retain RNE |

The packed-RNE candidate was compiled with the recovered production-equivalent command, linked with all aligned, query-tail, KV-tail, combined-tail, and D64 NHD strided translation units, and exercised through the host selector. Its ISA reduction is real: in representative HND symbols, D64 changed from 8,224 to 7,500 code bytes and from 1,250 to 1,158 encoded instructions. D128 changed from 13,104 to 11,620 bytes and from 1,949 to 1,739 instructions. Scalar `v_lshrrev_b16` and `v_add_nc_u16` packing operations were replaced by packed 16-bit operations and byte permutes. The work is paid once per query tile and is therefore mostly amortized by the KV loop, which explains the neutral complete-kernel timing.

The truncation probe is useful for understanding conversion semantics, not for promotion. It preserved signs and did not increase finite magnitudes, but 510 of 2,046 subnormal inputs became zero and 510 of 2,046 NaN inputs became infinity. The later scalar production-tree experiment confirmed the attention-level cost: only `52/168` public cases passed, despite exercising the same ABI, dispatch, and tail paths as the qualified kernel. Removing RNE is consequently a rejected numerical-policy change, not an instruction substitution.

The screening timing used the frozen phase-11c aligned baseline and 20 alternating samples per row. It did not replace the authoritative 36-row matrix or exercise a production selector change. Production arithmetic, ABI, and registration remain unchanged. The later selector-only grouping refinement is documented separately below.

## Accepted Work

| Work | Result |
| --- | --- |
| Row-owned 128x64x8 kernel | Accepted as the common D64/D128 topology |
| E5M2 Q in LDS with FP16 decode before WMMA | Accepted. Reduces LDS without requiring unavailable FP8 WMMA |
| RNE Q encoding and log2 pre-scaling | Accepted numerical policy |
| Transposed `K * Q^T` score ownership | Accepted. Removed the expensive full C-fragment transpose |
| Register online softmax and direct C-to-A probability conversion | Accepted |
| Rotating V LDS swizzle | Accepted. Removes the severe store-conflict pattern |
| Compile-time query/KV tails | Accepted. Aligned path remains unchanged |
| D64 specialization | Accepted at 16 KiB LDS |
| Native HND and NHD layouts | Accepted. No physical transpose |
| Head-fast NHD grid order | Accepted. Removes the original partition aliasing |
| Bounded D128 NHD head grouping | Accepted for large K/V working sets |
| Expanded D64 HND decoded-Q cache | Accepted |
| Partition-aware D64 NHD strided grouping | Accepted |
| Arbitrary-head D64 NHD strided grouping | Accepted. `168/168` public cases and repeatable complete-path gains with unchanged kernel images and ABI |

## Rejected Work

| Work | Evidence | Decision |
| --- | --- | --- |
| Full score/probability LDS and scalar-row softmax | Poor ownership and occupancy. Incompatible with the accepted row-owned design | Replaced structurally |
| In-thread V transpose with the original XOR layout | LDS conflict `29.4028`. `5.965 ms` versus AITER `4.467 ms` at H16/N4096/D128 | Rejected |
| Persistent linear `(m,l)` softmax state | Correct and spill-free, but `-0.938%` paired geomean | Rejected |
| Four-pair progressive QK LDS issue | HND `-2.263%`, NHD `-1.149%` | Rejected |
| Two-pair progressive QK LDS issue | HND `+0.312%`, NHD `+0.239%`, below the `0.5%` local gate | Rejected |
| Progressive PV LDS issue | HND `-3.165%` | Rejected |
| Hoist D64 V global loads before the barrier | `-0.349%` paired geomean | Rejected |
| DPP row-share alpha fan-out | HND `-0.743%`, NHD `-0.602%` despite cleaner ISA | Rejected |
| Temporary FP16 score exponentials | LLVM retained FP32 exponentials and added 64 conversions | Rejected before timing |
| Half-up E5M2 Q encoding | Overall `-0.176%`. Work removed only once per query tile | Rejected. Retain RNE |
| D64 NHD `Br=256,Bc=64` | Two-shape geomean `-1.889%` | Rejected |
| D64 HND `Br=128,Bc=128` | 242 VGPRs, 50 above the hard gate | Rejected before timing |
| Tile-local FP16 PV WMMA buffer | Correct at existing tolerances, but HND geomean `-8.739%` and near-192-VGPR pressure | Rejected |
| D128 decoded-Q cache or double buffering | D128 already uses 191 VGPRs and the full 32 KiB LDS budget | Not admitted without prior resource reduction |
| D128 two-by-two V row staging | Halving the live V-load tuple did not change either aligned D128 image from 191 VGPRs and replaced wide LDS stores with additional narrow stores | Rejected before timing |
| CK WMMA builtin in place of tied inline assembly | Aligned VGPRs changed from `175/130/191/191` to `184/132/192/191` for HND-D64, NHD-D64, HND-D128, and NHD-D128. D128 instruction counts fell slightly, but no layout gained an allocation transition. | Rejected before correctness or timing. Retain the tied accumulator assembly |
| D128 per-tile opaque QK address dependency | Both aligned D128 images rose to 192 VGPRs and introduced private storage and VGPR spills: 20 bytes/four spills for HND and 28 bytes/six spills for NHD. | Rejected before correctness or timing. Retain compiler-scheduled LDS addressing |
| D128 packed persistent FP16 output state | Aligned HND-D128 fell from 191 to 182 VGPRs with 32 KiB LDS, zero private memory, and zero spills. `168/168` public cases passed. Baseline-relative `Rel-L2` stayed at or below `0.00237` and cosine stayed above `0.999997`. Complete paired HND timing regressed all 9 rows for `-2.624%` geometric mean. | Rejected. Retain the FP32 cross-key-tile output state |
| Packed RTZ probability conversion | `168/168` public cases passed, but the complete 36-row screen reached only `0.9806x` geometric mean against the frozen RNE matrix and won `3/36` rows. Linked VGPRs were `171/131/191/191`. The numerical policy also changes FP32-to-FP16 probabilities from RNE to RTZ. | Rejected. Restore scalar RNE probability conversion |
| Packed RNE Q encoding | Representative HND code size fell 8.8% at D64 and 11.3% at D128, but the 12-row aligned screening geomean was `0.9969x` with no resource transition | Keep as a future fused-frontend lead, not a standalone change |
| Direct packed-RNE D64 HND Q cache | Exact packed RNE plus direct register decode reduced aligned HND-D64 from 175 to 174 VGPRs, 16 KiB to 8 KiB LDS, 8,224 to 8,012 symbol bytes, and 1,250 to 1,199 instructions. The public fixture passed `168/168` and paired outputs were bitwise exact, but all 9 HND rows regressed for a `0.99277x` geometric mean. | Rejected. The duplicated scale/decode frontend costs more than the removed Q LDS traffic |
| Scalar and packed E5M2 truncation | The scalar production-tree candidate passed only `52/168`. Packed focused rows also failed. The policy changes subnormal and NaN behavior despite a clean exhaustive bit-conversion probe. | Rejected. RNE restored |

## Historical Forward FP8 Campaign (Superseded)

The notes below came from the pre-row-owned FP8 forward design that materialized the `Si` score/probability tile in LDS and ran scalar row softmax. They are retained as historical evidence only. The current accepted row-owned 128x64 kernel and its results are the sections above. Do not treat this as the current implementation plan.

### Structural Comparisons

- AITER: the old HIP image exposed the `Si` LDS round trip, three barriers per KV iteration, scalar row softmax, and repeated Q loads. AITER kept score/softmax/P in registers and only loaded V after P was ready. The old root-cause analysis correctly identified `Si` materialization as the main structural deficit. The present row-owned design removes that whole class of work.
- CK-Tile: useful transferable ideas were K double-buffering in the D-loop, post-softmax V load on gfx11, K-column inter-chunk padding, K/V LDS lifetime sharing, and `sched_group_barrier` hints. On gfx1151, K/V sharing and strict DS:MFMA scheduling were measured and rejected: sharing regressed 1-4% and scheduler hints regressed 13-19%. K double-buffering is not viable without asynchronous copy. gfx1151 lacks the needed global-to-LDS path. Do not copy CDNA MFMA-specific implementation details. Only the transferable dataflow/layout ideas apply.
- SageAttention gfx12: int8 QK is not useful on gfx1151 because int8 WMMA has the same throughput as FP16 WMMA. The benefit would be memory/register pressure only. `global_load_tr_b128` is gfx12xx-only and is unavailable here. Two-Q-group and streaming-column ideas were blocked by the old register-P/PV state. They would require a new row-owned layout design, not a local port.

### Historical Rejected and Blocked Work

| Candidate | Historical result | Disposition |
| --- | --- | --- |
| Standalone V transpose/prepack | Slower forward and higher prepack cost | Rejected. Only revisit inside a new dataflow |
| Local softmax split/cache patches | Did not remove the `Si`/softmax bottleneck | Rejected |
| Row-owned skeleton keeping full `Si` | Higher VGPR and worse V scalar-load footprint | Rejected |
| First register-P/PV wave-shuffle sentinel | Correctness failure | Rejected. Require a minimal fragment fixture |
| Q LDS load-once on the row-owned skeleton | LDS 44 KiB, VGPR 252, private segment | Rejected |
| Raw K-only LDS staging | Added LDS/barrier/wait pressure. V scalar loads remained | Rejected |
| K/V LDS stride `D + 8` | Forced profile regressed to `12443.569 us` | Rejected. Keep `D + 16` |
| `Bc=64` raw staging | Slower forced profile and nonzero bank conflicts | Rejected |
| Register-P/PV `(64,32,4)` retry | VGPR 256, private segment `0x38`, `16274.958 us` at S=4096 | Rejected |
| QK-produced partial row maxima | S=4096 fell to `18.416490` TFLOPS. Added hot-path `ds_bpermute` | Rejected |
| Parallel per-wave register softmax | Regressed 34-39%. Only 4/8 waves active during softmax | Rejected |
| Q-in-registers additive plan | Blocked by register-P/PV rejection. Adds about 64 VGPR/thread before any benefit | Not pursued |
| Integrated register-P/PV benchmark/tuning | Blocked by the register-P/PV failure | Not pursued |
| Warp-parallel softmax with `Si` in LDS | Per-wave softmax either halves active waves or requires cross-wave LDS merging | Blocked |
| JIT V load | Added a V-load barrier. Same mechanism regressed 1-4% in K/V sharing | Blocked |
| K double-buffering in the D-loop | No asynchronous global-to-LDS path on gfx1151. Global loads in the hot loop compete with Q | Blocked |
| K/V LDS sharing + JIT V load | Regressed 1-4%. LDS savings did not change residency | Rejected |
| `sched_group_barrier` DS:MFMA hints | Regressed 13-19% and shifted autotune | Rejected |
| Streaming col-tile softmax | `Bc=32` made it a no-op. `Bc=64` rejected | Blocked |
| Lane-major K zero-decode layout | NaN from the byte-layout conversion. Theoretical gain under 0.2% | Rejected |
| 2Q architecture | Doubles per-wave state. Blocked by register-P/PV failure | Not pursued |

### Durable Historical Conclusions

- The old `Si` LDS materialization forced three barriers, doubled score-element LDS traffic, serialized row softmax, and drove the Q tile down to 64 rows. This was the dominant structural bottleneck.
- The QK/output tile mismatch was fundamental: for a 64-row Q tile and 32-column KV tile, each wave produced one QK tile but needed four output tiles. Removing `Si` required a row-owned redesign, not a local patch.
- `Bc=32` was the safest KV width in the old design. Larger tiles increased conflicts and did not beat the staged baseline.
- HND was the optimized layout. NHD remained primarily a layout-conversion/memory-partition problem.
- The 128-bit load/store audit found Q/global and LDS tile movement already wide. V global scalar loads were the key memory-width problem.
- Truncation and int8 Q were not viable numerical policies on this architecture. Keep RNE and FP16 WMMA.
- These conclusions are historical evidence for why the current row-owned kernel exists. They do not reopen the old `Si` design.

## Ranked Forward Plan

The accepted selector refinement above is the only new production change in this campaign. No further forward source change is currently admitted: packed-RNE frontend integration and D128 live-state reduction both failed their complete-path gates. Any reopening requires new symbol-matched evidence and the promotion gates below.

### Promotion Gates

Every candidate must satisfy all gates before any production source or selector change:
- The repository public fixture remains `188/188`, including HND/NHD, D64/D128, independent tails, odd heads, and batch two. Outputs must be finite. Directional screening requires cosine at least `0.997`, with `Rel-L2 <= 0.10` preferred and `< 0.20` exploratory, in addition to the public elementwise envelope.
- Every affected linked gfx1151 image stays within 192 campaign-rounded VGPRs, 32,768 bytes LDS, zero private/scratch memory, and zero spills. Inspect the exact runtime symbol, metadata, ISA, and code-object hash.
- The raw ABI, operator IDs, selector behavior, current-device validation, and current-stream execution remain unchanged unless the candidate is explicitly a host-policy experiment.
- Timing uses complete execution, including all helper launches, conversions, synchronization, and selector sublaunches. A local change needs a repeatable at least `0.5%` target-domain geomean gain, majority wins, and no regression above `1%`. The full 36-row matrix remains the final regression gate.

### Closed Directions

RNE removal by truncation, half-up conversion, forced register allocation, generic bandwidth/cache claims, and direct FP8 compression of persistent attention state are closed for the current design. Causal attention, masks, dropout, bias, GQA/MQA, additional dtypes, and wide-index kernels remain separate feature campaigns.

Hard resource gates remain 192 allocated VGPRs, 32,768 bytes LDS, zero private/scratch memory, zero spills, and no material LDS-bank-conflict regression.

## Artifacts

- Final matrix: `~/tmp/feather_attn/fwd_autonomous_20260817/non16_selector_final_matrix/attn.csv`.
- Frozen pre-refinement matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Final loaded metadata: `~/tmp/feather_attn/phase11_final/metadata.txt`.
- Final LDS conflicts: `~/tmp/feather_attn/phase11_final/lds_bank_conflicts/summary.json`.
- D64 HND cache timing/profile: `~/tmp/feather_attn/phase11b_qcache_full_paired_robust.json` and `phase11b_qcache_profiles/`.
- D64 NHD strided timing/profile: `~/tmp/feather_attn/phase11c_selected_domain_paired.json` and `phase11c_strided_profiles/`.
- Broader AITER/Feather counter review: `~/tmp/feather_attn/review/counter_matrix_summary.txt`.

## References

- `~/amd-llvm-project/llvm/docs/AMDGPUUsage.rst` and `~/rdna35-isa-markdown/`: ISA and code-object reference.
- `~/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`, `utils.py`, and `common.py`: AITER forward kernel, architecture/config selection, and shared helpers.
- `~/triton/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp` and `LoadStoreOpToLLVM.cpp`: AMD WMMA/LDS lowering and layout references.
- `~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`, `block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`, and `~/rocm-libraries/shared/rocroller/docs/src/LDSSwizzling.md`: CK dataflow, K/V policy, and LDS-swizzle references.
- `~/SageAttention/sageattention/triton/quant_per_block.py`, `quant_per_thread.py`, and `attn_qk_int8_per_block.py`: SageAttention int8 quantization and QK references.
