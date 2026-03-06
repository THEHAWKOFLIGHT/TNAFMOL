# TNAFMOL — Results
**Last updated:** 2026-03-06 after hyp_009

## Status
Incremental patching of model.py to match tarflow_apple.py is EXHAUSTED after 4 experiments (hyp_006–hyp_009). hyp_009 (pre-norm + layers_per_block=2) gave 14% VF — WORSE than baseline. Pivoting to use tarflow_apple.py + TarFlow1DMol directly for multi-molecule training. Best multi-molecule result remains hyp_007: 55.8% ethanol, 34.7% mean VF, 2/8 above 50%.

## Experiments

| ID | Method | Valid % (best) | Key Finding | Status |
|----|--------|---------------|-------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all) | Log-det exploitation (3 collapse modes) | FAILURE |
| hyp_003 | TarFlow stabilization (clamp + reg) | 18.3% mean | Alpha_pos saturation equilibrium | FAILURE |
| hyp_004 | TarFlow architectural ablation (pos_enc + SBG) | 29.5% mean (sweep) | Improvements within buggy equilibrium | PARTIAL |
| und_001 | TarFlow Diagnostic Ladder | 98.2% mean (no pad) | Padding is sole failure; bugs found + fixed | DONE |
| hyp_005 | Padding-Aware TarFlow (PAD token + query zeroing) | 4.7% (ethanol) | Padding fixes have zero effect; log-det exploitation persists | FAILURE |
| hyp_006 | Output-Shift TarFlow (Apple architecture) | 24.8% (ethanol) | Hypothesis CONFIRMED: log-det exploitation eliminated. VF plateau at 13-25%. | FAILURE |
| hyp_007 | Output-Shift + ldr=5.0 + 20k steps | 55.8% (ethanol), 34.7% mean | Padding neutral (4pp drop). ldr=5.0 critical. 2/8 molecules above 50%. | PARTIAL |
| hyp_008 | Per-dim scale (3 log_scales/atom) | 39.2% (ethanol, T=9) | <1pp effect vs shared scale (und_001 Phase 4). True gap: pre-norm + layers_per_block. | FAILURE |
| **hyp_009** | **Pre-norm + layers_per_block=2** | **14% (ethanol, T=9)** | **WORSE than baseline (39%). Incremental patching exhausted.** | **FAILURE** |

## Best Result
**und_001:** Architecture ceiling = 98.2% mean VF across all 8 molecules with T=n_real (no padding). Range: 94.3% (aspirin) to 100% (naphthalene, benzene).

**hyp_007 (multi-molecule with output-shift + ldr=5.0):** Best VF=55.8% on ethanol, 53.2% on malonaldehyde, 42.8% benzene, 39.4% uracil, 29.8% toluene, 24.6% salicylic acid, 22.4% naphthalene, 9.2% aspirin. Mean VF=34.7%. Config: ldr=5.0, lr=3e-4, cosine, 20k steps, d_model=128, n_blocks=8. Best checkpoint at step 12000.

## What's Next
hyp_010: Use tarflow_apple.py + TarFlow1DMol (the und_001 architecture that achieves 96-98% VF per-molecule) directly for multi-molecule joint training on all 8 MD17 molecules. This bypasses model.py entirely. Goal: VF >= 50% on >= 4/8 molecules.
