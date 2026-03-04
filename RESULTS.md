# TNAFMOL — Results
**Last updated:** 2026-03-03 after hyp_005

## Status
TarFlow achieves 98.2% mean VF per-molecule (no padding) but only 4.7% VF in multi-molecule (padded) mode even with full padding mitigation. The padding fixes (PAD token, query zeroing) are correct but have zero measurable effect — the bottleneck is log-det exploitation in the SOS+causal architecture, which persists regardless of padding treatment. The 10x degradation from single-molecule to multi-molecule is unexplained. Next direction requires user decision: test hyp_003-best config in multi-molecule, or switch to Apple's output-shift architecture.

## Experiments

| ID | Method | Valid % (best) | Key Finding | Status |
|----|--------|---------------|-------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all) | Log-det exploitation (3 collapse modes) | FAILURE |
| hyp_003 | TarFlow stabilization (clamp + reg) | 18.3% mean | Alpha_pos saturation equilibrium | FAILURE |
| hyp_004 | TarFlow architectural ablation (pos_enc + SBG) | 29.5% mean (sweep) | Improvements within buggy equilibrium | PARTIAL |
| und_001 | TarFlow Diagnostic Ladder | 98.2% mean (no pad) | Padding is sole failure; bugs found + fixed | DONE |
| **hyp_005** | **Padding-Aware TarFlow (PAD token + query zeroing)** | **4.7% (ethanol)** | **Padding fixes have zero effect; log-det exploitation persists** | **FAILURE** |

## Best Result
**und_001:** TarFlow Diagnostic Ladder. Architecture ceiling = 98.2% mean VF across all 8 molecules with T=n_real (no padding). Range: 94.3% (aspirin, 21 atoms) to 100% (naphthalene, benzene). Multi-molecule padded (T=21): 20.8% mean VF.

**hyp_005 (multi-molecule with padding fixes):** Best VF=4.7% on ethanol with Config D (PAD token + query zeroing) + reg_weight=2.0. Far below 50% target. SANITY 2x2 factorial showed all 4 padding configs produce identical VF=0% and log_det/dof=7.3 — padding fixes have zero effect on the training dynamics.

## What's Next
User decision needed. Options: (1) Test alpha_pos=0.02 + reg_weight=5 + Config D in multi-molecule (hyp_003's best recipe was never tested with padding fixes), (2) Switch to Apple's output-shift architecture for multi-molecule training (different Jacobian structure may avoid the log-det issue), (3) Proceed with per-molecule TarFlow (T=n_real, no padding) for DDPM comparison — bypasses the padding problem entirely.
