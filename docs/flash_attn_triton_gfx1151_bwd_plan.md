# FlashAttention gfx1151 Triton Backward Kernel Optimization Plan

## Target and rules

- AITER AMD Triton FlashAttention backward, `bwd_kernel_fused_noncausal`.
- GPU: `gfx1151` / RDNA3.5. Primary shape: `B=1, S=9216, H=16, D=128, bf16, non-causal`.
- Secondary: `S=4096`, `S=16384`. Use case: Anima model training, dense BSHD, no dropout/ALiBi/varlen.
- AITER source changes allowed. Triton source changes only after concrete bounded hypothesis.
- Do not patch LLVM without asking. Experiments <= 10 min (`BENCH_WARMUP=1 BENCH_STEPS=3`).
- Temp scripts/outputs under `~/sd-scripts/tmp_attn_bwd/`.

## Current baselines

| S | median ms | output |
|---:|---:|---|
| 4096 | 26.1 | `tmp_attn_bwd/outputs/dense_v2_s4096_steps3.log` |
| 9216 | 127.1 | `tmp_attn_bwd/outputs/dense_v2_outer_masks_s9216_steps3.log` |
| 16384 | 396.3 | `tmp_attn_bwd/outputs/dense_v2_s16384_steps3.log` |

## Codegen snapshot (dense v2, S=9216)

```text
VGPR: 256, SGPR: 83, scratch: 536 bytes, LDS: 16384
scratch ops: 172, ds_*: 1012, s_waitcnt: 456
v_cmp: 269, v_cndmask: 338, v_add: 632, v_dual: 416, s_delay: 143
WMMA: 112, buffer_load: 82, buffer_store: 192
TTGIR: local_alloc 8, local_load 18, local_store 8, tt.dot 14
```

Hotspot mapping (from source-loc analysis):
- `ds_*`: dpT dot 256, dQ dot 256, dK dot 128 - rest from global-load-to-LDS staging.
- `s_waitcnt`: dQ dot 132, dpT dot 124, dK dot 101 - mostly `lgkmcnt` waits.
- `scratch_load`/`scratch_store`: concentrated near dpT/qkT/dk dots (dK/dV section).
- Store-side `v_cmp_o_f32`/`v_cndmask` (64 per store) = fp32->bf16 conversion+NaN canonicalization, not residual source masks.

Diagnostic dK/dV-only vs dQ-only split:

```text
full:      112 WMMA, 172 scratch ops, 1012 ds_*, 456 s_waitcnt -> 127.1 ms
dK/dV-only: 64 WMMA, 158 scratch ops,  636 ds_*, 286 s_waitcnt -> 90.5 ms
dQ-only:    48 WMMA,  18 scratch ops,  376 ds_*, 169 s_waitcnt -> 44.0 ms
```

dK/dV is the dominant pressure source and carries most spills. Naive split is slower than fused.

G0 layout/waitcnt inspection: `tmp_attn_bwd/outputs/pathG0_layout_waitcnt_inspection_20260616.md`.

## Hard decisions and avoid-list

Keep `bwd_kernel_fused_noncausal` as target:
- `mode=split`, `fused_atomic`, one-kernel variants all much slower (324-850+ ms).
- Do not reopen broad config sweeps: `(16,64,64,16)` remains best/near-best for S ∈ {4096,9216,16384}.

Compiler/environment decisions:
- Keep buffer ops. `AMDGCN_USE_BUFFER_OPS=0` -> 750 ms.
- Keep `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` enabled. Disabling -> 148 ms.
- `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` -> 750 ms. Forced async-copy -> pass manager failure.
- pingpong/scalarize-packed-fops: no meaningful win.
- `maxnreg` unsupported on installed Triton 3.7.
- `schedule_hint="attention"` / `"memory-bound-attention"`: rejected (G1/G2, increased spills/runtime).
- `TRITON_HIP_USE_BLOCK_PINGPONG=1`: no effect on this kernel shape (G3).

## Completed experiment ledger

### Committed wins

#### Dense v1: inner dense mask removal

```text
S=9216: 131.647 -> 128.363 ms.  S=4096: 27.176 -> 26.834 ms.  S=16384: 411.561 -> 397.067 ms.
correctness: max_abs=0, rel_norm=0.
```

#### Dense v2: outer dense load/store mask removal

```text
S=4096: 26.086 ms, S=9216: 127.091 ms, S=16384: 396.267 ms.
correctness: max_abs=0, rel_norm=0.
```

### Rejected AITER source experiments (reason: did not improve runtime or traded off artifact)

| Path | What | Median | Scratch | Key tradeoff |
|------|------|--------|---------|---------------|
| C4 | dQ Delta unmasked load | ~126.7 ms | same | no confirmed win |
| v3 | remove `m==-inf` guard + exact-grid split | 125.2 ms | 536->736 | S=16384 regression |
| A1 | `GROUP_SIZE==1` source split | 130.5 ms | 536->804 | spill explosion |
| A2 | separate dense BSHD kernel | 128.2 ms | 536->692 | scheduling/liveness worsened |
| B1 | register-source-style dK rewrite | 176.4 ms | 536->888 | LDS bars/spills massive |
| B2 | `evict_last` on inner loads | ~126.5 ms | same | codegen unchanged |
| C1 | inline `ds/dsT` into dot | ~127.1 ms | same | canonicalized away |
| C2 | compute dK before dV | ~129.1 ms | 536->468 | better spills, worse runtime |
| C3 | late `Delta` load in dK/dV | ~126.5 ms | same | canonicalized away |
| D1 | explicit bf16 store casts | ~126.9 ms | same | canonicalized away |
| D2 | manual finite RTNE bf16-bit stores | ~125.2 ms | 536->588 | worse int/global-store lowering |
| D3 | `fp_downcast_rounding="rtz"` stores | 133.1 ms | 536->572 | clear regression, output changed |
| E1 | dense preprocess mask removal | ~126.4 ms | same | target hot kernel unchanged |
| F1 | N2=32 tile | 154.1 ms | 800 | scratch/WMMA/ds explosion |
| F2 | M1=32 tile | 199.3 ms | 988 | scratch/WMMA/ds explosion |

### Rejected backend/compiler knobs

| Path | Knob | Median | Artifact | Verdict |
|------|------|--------|----------|---------|
| G1 | `schedule_hint="attention"` | 130.2 ms | scratch 536->564, v_add↑, v_dual↓, s_delay↓ | Regression, more spills |
| G2 | `schedule_hint="memory-bound-attention"` | 351.3 ms | scratch 536->1064, s_waitcnt↓ but spills exploded | Catastrophic |
| G3 | `TRITON_HIP_USE_BLOCK_PINGPONG=1` | 126.8 ms | Identical to v2 | No transformation applied |

## FlashAttention/RDNA3.5 research takeaways

- FA2-style shared score/dS tile could avoid recomputing qk/dP/dS between dK/dV and dQ.
- Hard part on gfx1151: partial dQ accumulation. Existing `fused_atomic` is not competitive.
- RDNA3.5 LDS capacity (64 KiB workgroup) is not the limit (current uses 16 KiB). Traffic/layout/waiting likely is.
- RDNA3.5 lacks NVIDIA TMA/WGMMA/TMEM/2-CTA-MMA. Port principles only: reduce LDS intermediates, improve overlap, choose WMMA-friendly layouts.

## Current interpretation

Mixed bottleneck:
- LDS/layout: `ds_* ~1012`, many `s_waitcnt` tied to dot operand staging (G0).
- Register/spills: 256 VGPR, ~536-byte scratch, concentrated in dK/dV.
- Scalar/conversion: remaining `v_cmp`/`v_cndmask` is store-conversion, not source masks.
- Algorithm: fused avoids slow split/atomic paths but recomputes score/dS.

Key lessons from rejected work:
- Reducing scalar ops is not enough if live ranges/spills worsen (v3, A1, A2).
- Reducing scratch alone is not sufficient if scheduling worsens (C2).
- Reducing LDS ops alone is not sufficient if Triton adds barriers/spills (B1).
- Source reorderings that Triton canonicalizes are noise (C1, C3, B2, D1).
- Instruction scheduling hints do not help this kernel on gfx1151 (G1, G2).
- Block-pingpong pass does not transform this kernel shape (G3).

## Next-step plan

Root cause: dK/dV has two large accumulators (dk + dv, 64x128 f32 each, ~64KB total) live simultaneously in the inner loop, vs dQ's single accumulator (dq, 32KB). This is why dK/dV has 92% of all scratch ops. The high-impact path is reducing simultaneously live large tensors in dK/dV.

### Tier 1 - directly attacks root cause - EXHAUSTED

All Tier 1 approaches tested and rejected:

| Path | Change | Median | Scratch | Verdict |
|------|--------|--------|---------|--------|
| H1 | `num_stages=1` | 359.8 ms | 536->1112 | Pipeline overlap essential |
| H2 | Separate dV/dK passes | 187.4 ms | 536->980 | Recomputation + duplicate loads worse |
| H3 | `BLOCK_M1=8` | 1971 ms | - | WMMA unsupported below M=16 |
| H4 | `BLOCK_N1=32` | 338.1 ms | 536->916 | Grid overhead dominates |

Root cause confirmed: two large accumulators (dk+dv) cause register pressure, but directly separating them or reducing tile sizes either breaks WMMA compatibility or adds too much overhead. Tile sizes are already well-tuned.

### Tier 2 - incremental (likely 0-3%)

#### 2a. Skip `add_move_up_prologue_loads` - REJECTED

Median 128.7 ms (baseline 127.1), artifact nearly identical (VGPR/scratch unchanged). The pass has negligible effect on this kernel. Reverted env-guard from installed `compiler.py`.

Outputs: `tmp_attn_bwd/outputs/pathI1_no_prologue_move_s9216_steps6.log`, `pathI1_no_prologue_move_artifact_analysis.json`.

#### 2b. In-thread transpose `dsBitWidth` 64->128

Requires Triton C++ rebuild. Only affects ds_write width for local stores (small fraction of total ds_* ops). Low priority unless clean evidence of a ds_write bottleneck.

### Tier 3 - step-change potential (20%+) but architecturally hard - ANALYZED

#### 3. Algorithmic shared-score/dS

Inspected FA2 backward kernel (`~/flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h`) and RDNA3.5 ISA (`~/rdna35-isa-markdown/`).

FA2 approach: Each CTA processes one K-block, iterates Q-blocks. For each (K-block, Q-block) pair, computes S=Q@K^T and dS=P*(dP-dP_sum) ONCE, then derives dV, dK, AND dQ_partial from the same dS. dQ_partial is written to a global dQ_accum buffer and reduced across K-block CTAs (via atomics or split buffer + merge kernel).

Why this doesn't help gfx1151: FA2 holds THREE large accumulators (dk + dv + dQ_partial, each 64*128 f32 = 32KB) live simultaneously in the inner loop. On RDNA3.5 with 256 VGPR/thread, we already spill 536 bytes with TWO accumulators. Adding a third would explode scratches (cf. H2 where separate dV/dK passes alone worsened scratch 536->980).

Score recomputation overhead in AITER fused: Only the single (K-block=k, Q-block=k) pair is processed by both sections of the same CTA - 1 out of 288 total pairs per CTA, or 0.35%. Negligible.

dQ accumulation alternatives: Global float atomics -> 772 ms (fused_atomic). Separate dQ_accum buffer + reduction kernel -> adds buffer allocation (75 MB), write traffic, and kernel launch overhead. Unlikely to beat current fused approach.

Conclusion: The AITER fused design (separate dK/dV and dQ sections, no cross-CTA accumulation) is the correct architecture for RDNA3.5's register-constrained VGPR budget. FA2's approach is suited for NVIDIA's register-rich SM80+ (255 registers/thread, larger register file). The current 127 ms baseline may be near-optimal for this architecture with current Triton codegen quality. Do not pursue shared-score/dS without a concrete register-pressure solution first.

### Avoid repeating

- Store-conversion / preprocess-mask experiments.
- Tile sweeps, GROUP_SIZE==1 splits, dK-before-dV reordering, explicit bf16 casts, eviction_policy tweaks.
- `schedule_hint`, block-pingpong env knobs, `num_stages` variants.
- Softmax/exp approximation without user approval.
- DPP/VOPD tuning without concrete evidence.
- FA2-style shared-score/dS: unsuited to RDNA3.5 register constraints.

## Benchmark and correctness policy

- Run S=9216 first: `BENCH_S=9216 BENCH_WARMUP=1 BENCH_STEPS=3`.
- If >=1-2% improvement and artifact not worse, confirm S=4096 and S=16384.
- Check correctness against generic or saved gradients before commit.
- `rocprofv3` only after a confirmed win. Commit only confirmed speed improvements.

Useful scripts: `tmp_attn_bwd/scripts/bench_fa_bwd.py`, `analyze_triton_artifact.py`, `loc_prefix_analysis.py`, `save_fa_bwd_direct_grads.py`.
Saved generic gradients: `tmp_attn_bwd/outputs/generic_grads_s9216.pt`.
