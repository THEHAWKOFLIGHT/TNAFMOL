# TNAFMOL — Results
**Last updated:** 2026-03-07 after hyp_010

## Status
TarFlow Apple architecture (tarflow_apple.py + TarFlow1DMol) achieves **71.6% mean VF** across all 8 MD17 molecules in joint multi-molecule training. **All 8 molecules exceed 50% VF** — far surpassing the 4/8 target. Two critical padding bugs fixed in train_phase3.py (sampling noise at padding positions + attention key masking in PermutationFlip). No log-det regularization needed. Aspirin recovered from 9.2% (hyp_007) to 67.4%.

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
| hyp_009 | Pre-norm + layers_per_block=2 | 14% (ethanol, T=9) | WORSE than baseline (39%). Incremental patching exhausted. | FAILURE |
| **hyp_010** | **TarFlow Apple (TarFlow1DMol) multi-mol** | **71.6% mean, all 8 >50%** | **Apple arch generalizes to multi-mol. No ldr needed. 2 padding bugs fixed.** | **DONE** |

## Best Result
**und_001:** Architecture ceiling = 98.2% mean VF across all 8 molecules with T=n_real (no padding). Range: 94.3% (aspirin) to 100% (naphthalene, benzene).

**hyp_010 (multi-molecule with TarFlow Apple architecture):** All 8 molecules, T=21, 20k steps. Per-molecule VF: malonaldehyde 82.6%, naphthalene 81.0%, benzene 79.4%, aspirin 67.4%, salicylic acid 67.4%, toluene 67.4%, ethanol 64.0%, uracil 63.6%. Mean VF=71.6%. Config: d_model=256, n_blocks=4, layers_per_block=2, lr=1e-3, cosine, 20k steps, ldr=0. W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/tw349mhw

## What's Next
hyp_011: DDPM diffusion baseline — implement transformer-based denoiser with comparable parameter count. Train on same multi-molecule MD17 data. Head-to-head comparison with TarFlow on all evaluation metrics.
