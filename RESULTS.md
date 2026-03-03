# TNAFMOL — Results
**Last updated:** 2026-03-03 after und_001

## Status
TarFlow achieves 98.2% mean valid fraction across all 8 MD17 molecules when trained per-molecule (no padding). The architecture is sound — the multi-molecule failure (hyp_002/003/004) was caused by two implementation bugs (logdet normalization + causal mask) and padding-induced latent space corruption. Per-molecule TarFlow (T=n_real) is the viable path. Next: hyp_005 (per-molecule TarFlow full training for DDPM comparison).

## Experiments

| ID | Method | Valid % (best) | Key Finding | Status |
|----|--------|---------------|-------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all) | Log-det exploitation (3 collapse modes) | FAILURE |
| hyp_003 | TarFlow stabilization (clamp + reg) | 18.3% mean | Alpha_pos saturation equilibrium | FAILURE |
| hyp_004 | TarFlow architectural ablation (pos_enc + SBG) | 29.5% mean (sweep) | Improvements within buggy equilibrium | PARTIAL |
| **und_001** | **TarFlow Diagnostic Ladder** | **98.2% mean (no pad)** | **Padding is sole failure; bugs found + fixed** | **DONE** |

## Best Result
**und_001:** TarFlow Diagnostic Ladder. Architecture ceiling = 98.2% mean VF across all 8 molecules with T=n_real (no padding). Range: 94.3% (aspirin, 21 atoms) to 100% (naphthalene, benzene). Multi-molecule padded (T=21): 20.8% mean VF. The gap is entirely explained by padding tokens corrupting the flow's latent space. Two implementation bugs discovered: (1) logdet normalization T*D vs n_real*D, (2) self-inclusive causal mask creating non-triangular Jacobian. Both fixed.

Prior experiments (hyp_002/003/004) operated within these bugs — their "alpha_pos saturation equilibrium" was an artifact of the normalization bug, not a fundamental architectural limitation.

## What's Next
hyp_005: Per-molecule TarFlow. Train one TarFlow model per molecule at T=n_real (no padding). Expected 94-100% VF based on und_001 Config A. This provides the TarFlow baseline for head-to-head comparison with DDPM.
