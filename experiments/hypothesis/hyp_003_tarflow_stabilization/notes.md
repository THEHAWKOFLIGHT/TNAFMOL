## [hyp_003] — TarFlow Stabilization via Soft Clamping + Soft Equivariance
**Date:** 2026-03-01 | **Type:** Hypothesis | **Tag:** `hyp_003`

### Motivation
hyp_002 showed TarFlow collapses completely (valid_fraction=0 on all molecules) due to log_det exploitation: any unconstrained scale DOF is exploited by the MLE objective to maximize log_det without learning the data distribution. Three collapse modes: affine scale, shift, ActNorm scale.

hyp_003 addresses this with three targeted interventions:
1. Asymmetric soft scale clamping (Andrade et al. 2024): bounds expansion to exp(alpha_pos)~1.105x per layer while allowing contraction, preventing runaway log_det.
2. Log-det regularization penalty: explicit L2 penalty on log_det_per_dof discourages exploitation at the loss level.
3. Soft equivariance via SO(3) rotation + CoM noise augmentation + unit-variance normalization (SBG, Tan et al. 2025): makes the model robust to rotational orientation and puts input in ~N(0,1) scale.

Connection to research story: TarFlow is the primary candidate for exact-likelihood molecular conformation generation. If hyp_003 fails, the DDPM baseline becomes hyp_004.

### Method
OPTIMIZE: SANITY (all three fixes combined) → HEURISTICS (SBG training recipe) → SCALE (larger model)

See reports/diagnostic_report.md, reports/plan_report.md for detailed angle specifications.

### Results
*(populated after experiment completes)*

### Interpretation
*(populated after experiment completes)*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
