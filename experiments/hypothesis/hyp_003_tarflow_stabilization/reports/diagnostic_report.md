# Diagnostic Report — hyp_003 TarFlow Stabilization

**Date:** 2026-03-01
**W&B run:** https://wandb.ai/kaityrusnelson1/tnafmol/runs/hna3dmkz (hyp_003_sanity_diag)

---

## Baseline Failure Analysis

**Diagnostic run config:** 500 steps, all three fixes enabled (alpha_pos=0.1, alpha_neg=2.0, log_det_reg_weight=0.1, augment_train=True, normalize_to_unit_var=True), lr=3e-4, batch_size=128.

**What fraction of samples fail and why?**
- valid_fraction = 0.000 on ALL 8 molecules (200 samples per molecule).
- min_dist_mean = 0.2–0.35 Å across all molecules (bond distances should be 1.0–1.5 Å, minimum valid distance is 0.8 Å).
- Pairwise distance divergence = 0.26–0.45 (high, indicating poor distribution match).

**Loss behavior (good):**
- Loss decreased from 1.39 → 0.14 over 500 steps. Real learning is happening.
- log_det/dof stabilized at 0.78 — within the [-2, 2] target range. ✓
- The asymmetric clamping successfully prevents the catastrophic log_det explosion seen in hyp_002 (where log_det/dof reached 50+).

## Root Cause

**New collapse mode: alpha_pos saturation collapse.**

Per-block log_det analysis shows each of the 8 blocks contributes exactly 0.0977 log_det/dof. This is the saturation value of `(2/π) * alpha_pos * arctan(∞)` = `alpha_pos = 0.1`. The model has learned to set log_scale ≈ +0.1 (its maximum) uniformly for all atoms in all blocks.

**Mechanism:**
1. In the FORWARD direction (data → latent): positions are expanded by `exp(0.1)^8 ≈ 2.23×` per atom.
2. Latent distribution is approximately N(0, 2.23²) rather than N(0, 1).
3. In the INVERSE direction (latent → data): noise z ~ N(0, 1) is CONTRACTED by the same factor.
4. Generated samples have std ≈ 0.54 normalized = 0.70 Å (vs reference 0.92 Å).
5. Atoms are compressed together → min pairwise distances are too small → valid_fraction = 0.

**Why log_det_reg_weight=0.1 was insufficient:**
The NLL gradient pushes log_scale toward its maximum (to maximize log_det and minimize NLL). At log_det_per_dof = 0.78, the regularization penalty = 0.1 × 0.78² = 0.061. This is smaller than the NLL benefit from the expansion, so the optimizer still finds it profitable to saturate alpha_pos.

**Comparison to hyp_002:**
Same root class (log_det exploitation), different mechanism. hyp_002 used tanh*3 (log_scale_max=3.0), giving log_det/dof = 50+. hyp_003 with alpha_pos=0.1 gives log_det/dof = 0.78. Progress: we reduced exploitation by 64×, but exploitation is not yet zero.

## Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY | Yes | The current configuration (alpha_pos=0.1, reg_weight=0.1) is a miscalibration — insufficient regularization for the regularization weight. This is a parameter tuning issue, not a fundamental limitation. Sweeping reg_weight and alpha_pos is a SANITY fix. |
| KNOWN HEURISTICS | Yes | If stronger regularization doesn't solve it, the SBG training recipe (AdamW betas=(0.9, 0.95), OneCycleLR, EMA, batch_size=512) from Tan et al. 2025 may help by providing better optimization dynamics. |
| SCALE | Likely No | Collapse is still architectural/loss-landscape, not capacity-limited. Larger model will exploit the same log_det bound more efficiently. Only relevant if loss looks capacity-limited after SANITY/HEURISTICS succeed at preventing collapse. |

## Proposed Angles (preliminary)

**Primary insight:** The SANITY angle needs to find a combination of (alpha_pos, log_det_reg_weight) that drives log_det_per_dof → 0. The current alpha_pos=0.1 is still being fully exploited. Options:
1. Dramatically increase log_det_reg_weight (to 1.0, 5.0, or 10.0)
2. Reduce alpha_pos further (to 0.02–0.05) to limit the maximum exploitation
3. Combined approach: alpha_pos=0.05 + reg_weight=1.0

The SANITY sweep will explore these combinations via W&B Bayesian optimization.

Full angle specification in plan_report.md.
