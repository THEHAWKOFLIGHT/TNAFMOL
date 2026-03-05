# TNAFMOL — Results
**Last updated:** 2026-03-06 after hyp_007

## Status
Multi-molecule output-shift TarFlow with log-det regularization (ldr=5.0) achieves 55.8% ethanol VF and 34.7% mean VF — first multi-molecule result with 2 molecules above 50%. Padding confirmed neutral with output-shift (4pp max variation across 5 padding sizes). Key bottleneck: larger molecules (aspirin 9.2%) need more capacity or training.

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
| **hyp_007** | **Output-Shift + ldr=5.0 + 20k steps** | **55.8% (ethanol), 34.7% mean** | **Padding neutral (4pp drop). ldr=5.0 critical. 2/8 molecules above 50%.** | **PARTIAL** |

## Best Result
**und_001:** Architecture ceiling = 98.2% mean VF across all 8 molecules with T=n_real (no padding). Range: 94.3% (aspirin) to 100% (naphthalene, benzene).

**hyp_007 (multi-molecule with output-shift + ldr=5.0):** Best VF=55.8% on ethanol, 53.2% on malonaldehyde, 42.8% benzene, 39.4% uracil, 29.8% toluene, 24.6% salicylic acid, 22.4% naphthalene, 9.2% aspirin. Mean VF=34.7%. Config: ldr=5.0, lr=3e-4, cosine, 20k steps, d_model=128, n_blocks=8. Best checkpoint at step 12000.

## What's Next
2/8 molecules above 50% (target: 4/8). Remaining gap is molecule-size-dependent: small molecules (9 atoms) achieve 50%+, large molecules (18-21 atoms) are below 25%. Next steps: (A) SCALE — d_model=256, n_blocks=12 to increase per-molecule capacity, (B) per-molecule normalization instead of global std, (C) longer training with plateau-aware LR (model peaks at step 12000/20000 — cosine may over-decay), (D) DDPM baseline for head-to-head comparison.
