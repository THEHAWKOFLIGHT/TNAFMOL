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
**FAILURE** — Primary criterion (valid_fraction ≥ 0.5 on ≥ 4/8 molecules) not met.

Best result: HEURISTICS sweep at 3000 steps → mean valid fraction 18.3%, 0/8 molecules ≥ 50%.

SANITY full (10k steps): mean 13.1% — ethanol 33.0%, malonaldehyde 32.6%, others < 20%
HEURISTICS full (20k steps): mean 14.3% — malonaldehyde 38.0%, ethanol 33.4%, others < 20%

Root cause: alpha_pos saturation equilibrium. The NLL gradient (pushing log_det up) and the regularization gradient (pushing toward 0) reach a stable fixed point at exactly log_det/dof = alpha_pos (= 0.02). This produces a ~38% compression of generated samples relative to reference, causing close-collision failures. More training steps, higher LR, EMA — none break the equilibrium. SCALE skipped (not capacity-limited — model saturates at step 150).

See results/: valid_fraction_comparison.png, best_results_summary.png, sweep_comparison.png, angle_summary.png

See reports/final_report.md for full analysis.

W&B runs: SANITY sweep rccehd8m | SANITY full o5naez7a | HEURISTICS val o6pnle0k | HEURISTICS sweep cmgrp6jo | HEURISTICS full 4079op64

![Valid Fraction Comparison](../results/valid_fraction_comparison.png)
**Valid fraction by molecule — SANITY vs HEURISTICS full runs** — Neither angle reaches the 50% threshold. Small molecules (9 atoms) perform best (~33-38%). Large molecules (18-21 atoms) are near 0%.

![Best Results Summary](../results/best_results_summary.png)
**Best results summary** — Left: per-molecule valid fraction showing size-dependent scaling failure. Right: clear inverse correlation between n_atoms and valid fraction.

### Interpretation
The interventions partially worked: log_det is now bounded (hyp_002 had log_det/dof → 50+, hyp_003 has it at 0.02). However, partial suppression is insufficient — the regularization creates a new stable equilibrium at log_det/dof = alpha_pos rather than at 0. The equilibrium is a mathematical fixed point between the NLL gradient (wants expansion) and the regularization gradient (penalizes deviation from 0). No reasonable hyperparameter tuning escapes it.

This result falsifies the hypothesis. TarFlow with asymmetric clamping + log-det regularization cannot generate valid molecular conformations under standard MLE training. The fundamental issue is the interaction between the affine scale DOFs and the MLE objective in high-dimensional molecular systems.

**Research implication:** TarFlow is likely not viable for molecular conformation generation. The next experiment should either abandon TarFlow entirely or test a fundamentally different loss formulation (e.g., asymmetric regularization targeting only positive log_det).

**Status:** [x] Conflict — escalate to Postdoc | Explanation: Two consecutive TarFlow failures. Research story expects further investigation or pivot to DDPM.
