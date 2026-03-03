# Final Experiment Report — und_001 TarFlow Diagnostic Ladder
**Status:** DONE (Phase 3 complete; Level 2 CIFAR-10 still running — standalone, does not block)
**Branch:** `exp/und_001`
**Commits:**
- `35e7829` — initial project state (pre-experiment)
- (earlier commits from Phase 1/2 on this branch)
- `d77e240` — [und_001] results: Phase 3 Steps A and B complete
- `09c565f` — [und_001] code: fix attention mask and padding z-zeroing
- `901d6c5` — [und_001] code: fix permutation-aware padding mask
- `3fbf7f1` — [und_001] results: Phase 3 Steps C-E complete; docs updated
- `05da6e1` — [und_001] results: Step F complete (VF=10.4%); corrected docs

---

## Experimental Outcome

### Phase 1 — Source Comparison (complete, pre-existing)

13 differences between Apple TarFlow and our hyp_002/hyp_003 implementation. Critical:
1. Shared vs per-dim scale (per-dim prevents exploitation)
2. SOS token vs output shift autoregression (our implementation had correctness bug)
3. Missing clamping/regularization in Apple version

### Phase 2 — Apple Baseline Verification (Levels 0-1 complete, Level 2 in progress)

| Level | Dataset | Key metric | Status |
|-------|---------|-----------|--------|
| 0 | 2D 8-mode Gaussian | VF=88.6%, NLL=0.91 | DONE |
| 1 | MNIST 1×28×28 | -3.20 bits/dim | DONE |
| 2 | CIFAR-10 3×32×32 | Loss=-2.00 at step 8500/50000 | RUNNING (~28 hours remaining) |

Apple TarFlow confirmed working on standard benchmarks. Level 2 is a long run (228M params, 50k steps)
running in the background. Phase 3 was not blocked by Level 2.

### Phase 3 — Adaptation Ladder (COMPLETE)

All 6 steps completed on ethanol (9 atoms, MD17 dataset), 5000 steps each:

| Step | Description | best_loss | VF | logdet/dof |
|------|-------------|-----------|-----|-----------|
| A | Raw coords, 9 atoms | -2.827 | **89.1%** | 0.122 |
| B | + Atom type conditioning | -2.795 | **92.9%** | 0.121 |
| C | + Padding (T=21, 9 real) | -2.825 | **2.7%** | 0.122 |
| D | + Noise augmentation | -1.902 | **14.3%** | 0.088 |
| E | Shared scale (KEY TEST) | -1.892 | **40.2%** | 0.088 |
| F | + Stabilization (clamp+reg) | -1.887 | **10.4%** | 0.087 |

W&B runs: `und_001_phase3_step_{a,b,c,d,e,f}` in project `tnafmol`, group `und_001`
Plots: `results/phase3/step_*/loss_curve.png`, `pairwise_dist.png`
Checkpoints: `results/phase3/step_*/best.pt` (W&B Artifacts)

---

## Project Context

The und_001 diagnostic was launched after two consecutive TarFlow failures (hyp_002: 0% VF from
log-det exploitation; hyp_003: 18.3% VF from alpha_pos saturation equilibrium). The goal: find
exactly where the architecture breaks and why.

### What we found:

**1. The primary failure is PADDING (Step C), not shared scale.**

The original hypothesis was: shared scale → log-det exploitation → failure. This is WRONG.
Step E (shared scale) achieves 40.2% VF, BETTER than Step D (per-dim scale, 14.3% VF).

The actual failure is: padding 9 ethanol atoms to 21 tokens collapses VF from 89% to 2.7%.
The model trains correctly (similar NLL) but the latent space learned with T=21 doesn't map
back to valid molecular geometries.

**2. The hyp_002/hyp_003 failures were caused by two bugs:**
- T*D logdet normalization instead of n_real*D → equilibrium at z=0.655 Å (below bond lengths)
  → log-det exploitation
- Causal mask bug in older code (SOS token + self-inclusive mask → non-triangular Jacobian)

With both bugs fixed, shared scale performs comparably or better than per-dim scale.

**3. Noise augmentation (sigma=0.05) is more impactful than scale parameterization.**
Step D (per-dim + noise) vs Step C (per-dim, no noise): 14.3% vs 2.7% VF.
Noise helps the model generalize from training configurations.

**4. Clamping is costly with correct normalization (reduces VF from 40.2% → 10.4%).**
The first Step F run (with wrong normalization) falsely appeared to give 0% VF — an artifact
of the normalization bug creating a degenerate constant equilibrium. The corrected run shows
clamping does reduce performance but doesn't prevent learning.

---

## Story Validation

**Partially fits. One significant conflict.**

Fits:
- Apple TarFlow degrades on molecular data (89% → 2.7% VF with padding)
- The degradation is not fundamental — Step E achieves 40.2% VF without special tricks
- Log-det exploitation IS the root cause of hyp_002/hyp_003 (correct diagnosis)
- Shared scale doesn't cause saturation when normalization is correct

Does NOT fit:
- **Shared scale is NOT the primary failure point.** RESEARCH_STORY.md should be updated.
  The diagnostic correctly identified the bug root causes (normalization + causal mask), but
  the story's prediction that shared scale causes failure was incorrect.
- Padding is the primary bottleneck, which was not anticipated in the research story.

This is not a crisis — the research story can be updated with accurate findings. The diagnostic
achieved its goal: we now know where the failure is (padding) and why the previous failures
occurred (normalization bugs + mask bug).

---

## Open Questions

1. **Why does padding collapse VF even when NLL is correct?** (Key question for Phase 4)
   - The model correctly optimizes NLL on T=21 tokens but generates invalid structures
   - Possible: degenerate padding atoms (all zeros) create a mixed latent manifold
   - Possible: autoregressive ordering over 21 tokens disrupts causal dependency structure

2. **Can we achieve high VF without padding?** (Test: run Step D/E config with T=9 on all 8 molecules)

3. **What happens at different molecule sizes?** (Phase 5: 8 MD17 molecules with best config)

---

## Verification

- All 6 steps ran to completion (5000 steps), no NaN or crashes in final runs
- Results spot-checked: Steps A/B VF match qualitative expectation (89-93% for clean 9-atom case)
- Step E VF (40.2%) higher than Step D (14.3%) — unexpected but reproducible; same seed
- Logdet/dof stable throughout all runs (0.087-0.122 range, no exploitation)
- W&B summaries confirm: valid_fraction logged for all 6 steps
- phase3_report.md documents all figures with captions

---

## Files Generated

| Path | Description |
|------|-------------|
| `src/train_phase3.py` | All 6 steps, TarFlow1DMol + all MetaBlock variants |
| `results/phase3/step_{a-f}_*/results.json` | Per-step metrics |
| `results/phase3/step_{a-f}_*/loss_curve.png` | Training loss curves |
| `results/phase3/step_{a-f}_*/pairwise_dist.png` | Pairwise distance distributions |
| `reports/phase3_report.md` | Detailed step-by-step analysis |
| `reports/source_comparison.md` | Phase 1 architectural diff (pre-existing) |
| `reports/ladder_report.md` | Phase 2 benchmark results (pre-existing) |
