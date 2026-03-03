# Adaptation Ladder Report — und_001 Phase 3

**Date:** 2026-03-02
**Branch:** `exp/und_001`
**Status:** COMPLETE

---

## Summary

Phase 3 runs 6 incremental steps, each adding ONE architectural change to Apple TarFlow, training
for 5000 steps on ethanol (MD17, 9 atoms, 444k train). The goal is to identify which change
causes performance degradation. The full report is in `phase3_report.md`. This document provides
the ladder summary and primary diagnostic conclusions.

---

## Adaptation Ladder Results

| Step | Change Added | Valid % | Final Loss | logdet/dof | Delta VF |
|------|-------------|---------|-----------|------------|---------|
| A | Baseline: Apple TarFlow1D, 9 atoms, raw coords | **89.1%** | -2.802 | 0.122 | — |
| B | + Atom type embedding (nn.Embedding 4→16, concat) | **92.9%** | -2.772 | 0.121 | +3.8% |
| C | + Padding to T=21 (9 real + 12 pad), causal+pad mask | **2.7%** | -2.801 | 0.122 | **-90.2%** |
| D | + Gaussian noise augmentation (sigma=0.05) | **14.3%** | -1.867 | 0.088 | +11.6% |
| E | Shared scale: 1 scalar/atom × 3 coords (KEY TEST) | **40.2%** | -1.864 | 0.088 | +25.9% |
| F | + Asymmetric clamp (alpha_pos=0.1) + log-det reg (0.01) | **10.4%** | -1.857 | 0.087 | -29.8% |

**Training:** 5000 steps, batch_size=256, lr=5e-4, cosine schedule, gradient clip 1.0, seed=42
**Model:** channels=256, num_blocks=4, layers_per_block=2, head_dim=64

---

## Primary Finding: Padding is the Failure Point

The expected failure at Step E (shared scale) did NOT occur. The actual primary failure is at
Step C (padding): valid fraction collapses from 89.1% → 2.7% when 12 padding atoms are added.

**Crucially**: the Step C model achieves similar NLL as Step A (-2.801 vs -2.802). The model
learns the training distribution correctly — but the learned latent space under T=21 does not
produce valid molecular geometries when sampled.

This decoupling of NLL and valid fraction is the central diagnostic finding of Phase 3.

---

## Finding on Shared Scale (Step E vs Hypothesis)

The original hypothesis was: shared scale → log-det exploitation → VF collapse.

**Result: Shared scale IMPROVES valid fraction** when padding and noise are both present
(14.3% → 40.2%). No saturation was observed. The hyp_002/hyp_003 failures were due to
two implementation bugs, NOT shared scale per se:

1. **Causal mask bug**: self-inclusive attention mask (non-triangular Jacobian)
2. **Logdet normalization bug**: normalizing by T*D instead of n_real*D shifted the NLL
   equilibrium from z≈1 Å to z≈0.655 Å, making log-det exploitation gradient-favorable

Both bugs are fixed in `src/train_phase3.py` (commit `901d6c5`).

---

## Finding on Stabilization (Step F)

The asymmetric clamp (alpha_pos=0.1) limits scale to exp(-0.1) ≈ 0.905 max contraction
per block. This HURTS performance (40.2% → 10.4%) — the clamp is too tight. The model
needs larger scale freedom to represent valid molecular geometries.

The log-det regularizer (weight=0.01) at this NLL level is mild and doesn't drive the
degradation — the clamp is the primary constraint.

---

## Open Questions for Next Experiments

1. **Why does padding cause VF collapse when NLL is correct?**
   - Is it the 12 degenerate zero-padding tokens disrupting the latent manifold?
   - Is it the 21-atom autoregressive ordering creating too many unnecessary causal dependencies?
   - Does removing padding and keeping T=9 with Step E architecture achieve >40.2% VF?

2. **Can Step E (T=21, shared scale) be improved further?**
   - More training steps (10k, 20k)?
   - Better data ordering (random vs sorted by distance from CoM)?
   - Different padding strategy (no padding, using only T=9 with masking in the metric)?

3. **Is the Step C VF collapse recoverable?**
   - Step D (noise) recovers to 14.3%, Step E to 40.2% — suggesting partial recovery is possible
   - A systematic ablation of padding fraction (T=10, 12, 15, 21) could isolate the mechanism

---

## W&B Runs

| Step | Run ID | URL |
|------|--------|-----|
| A | `und_001_phase3_step_a` | https://wandb.ai/kaityrusnelson1/tnafmol/runs/ |
| B | `und_001_phase3_step_b` | https://wandb.ai/kaityrusnelson1/tnafmol/runs/ |
| C | `und_001_phase3_step_c` | https://wandb.ai/kaityrusnelson1/tnafmol/runs/j0mjo68n |
| D | `und_001_phase3_step_d` | https://wandb.ai/kaityrusnelson1/tnafmol/runs/w4ux853y |
| E | `und_001_phase3_step_e` | https://wandb.ai/kaityrusnelson1/tnafmol/runs/2rm7zvm3 |
| F | `und_001_phase3_step_f` | https://wandb.ai/kaityrusnelson1/tnafmol/runs/0d4b0nne |

---

## Key Files

| File | Description |
|------|-------------|
| `src/train_phase3.py` | All 6 step implementations (commit `901d6c5`) |
| `reports/phase3_report.md` | Full step-by-step analysis with embedded plots |
| `results/phase3/step_{a-f}_{name}/results.json` | Per-step metrics |
| `results/phase3/step_{a-f}_{name}/loss_curve.png` | Training loss + logdet/dof |
| `results/phase3/step_{a-f}_{name}/pairwise_dist.png` | Generated vs reference distances |
| `results/phase3/step_{a-f}_{name}/best.pt` | Best checkpoint per step |
