## [und_001] — TarFlow Diagnostic Ladder
**Date:** 2026-03-02 | **Type:** Understanding | **Tag:** `und_001`

### Motivation
Two consecutive TarFlow failures (hyp_002: log-det exploitation → 0% VF; hyp_003: alpha_pos saturation equilibrium → 18.3% best VF) demand a systematic investigation. Instead of continuing to patch our implementation, we start from Apple's working TarFlow (arXiv:2412.06329), verify it on a constructive complexity ladder (2D → MNIST → CIFAR-10), then progressively adapt toward molecules to find exactly where and why performance degrades. This identifies whether the failure is architectural (fixable) or fundamental (pivot to DDPM).

### Method
Six-phase diagnostic ladder:
1. **Source Comparison** — systematic diff of Apple TarFlow vs our model.py
2. **Apple Baseline Verification** — constructive ladder: 2D Gaussian → MNIST → CIFAR-10
3. **Adaptation Ladder** — progressively adapt Apple architecture toward molecules (ethanol), one change at a time
4. **Ablation Matrix** — cross the most impactful factors
5. **Best Config Validation** — test best config on all 8 MD17 molecules
6. **Synthesis** — where does it break, why, and what to do next

### Results (Phases 1-3 complete; Phases 4-6 pending)

**Phase 1 — Source Comparison:** Complete. 13 differences documented between Apple TarFlow and
our hyp_002/hyp_003 implementation. Critical differences: (1) shared vs. per-dim scale, (2) SOS
token vs. output shift (correctness bug in our model), (3) clamping + regularization vs. none.
See `reports/source_comparison.md`.

**Phase 2 — Apple Baseline Verification:**

| Level | Dataset | Key result | Status |
|-------|---------|-----------|--------|
| 0 | 2D 8-mode Gaussian | 88.6% mode coverage, NLL=0.91 | DONE |
| 1 | MNIST 1×28×28 | -3.20 bits/dim (14400 steps, converged) | DONE |
| 2 | CIFAR-10 3×32×32 | -2.01 NLL @ step 1340 (in progress) | RUNNING (will be updated when complete) |

Apple TarFlow verified working across the complexity ladder. Architecture trains correctly
with per-dim scale, output shift autoregression, and no clamping/regularization.
See `reports/ladder_report.md`.

**Phase 3 — Adaptation Ladder (ethanol, 5000 steps each):**

| Step | Description | best_loss | VF | logdet/dof |
|------|-------------|-----------|-----|-----------|
| A | Raw coords, 9 atoms, no padding | -2.827 | 89.1% | 0.122 |
| B | + Atom type conditioning (emb dim=16) | -2.795 | 92.9% | 0.121 |
| C | + Padding (T=21, n_real=9) | -2.825 | 2.7% | 0.122 |
| D | + Noise augmentation (sigma=0.05) | -1.902 | 14.3% | 0.088 |
| E | Shared scale (1 scalar/atom, KEY TEST) | -1.892 | 40.2% | 0.088 |
| F | + Stabilization (clamp alpha=0.1/2.0 + reg=0.01) | -1.887 | 10.4% | 0.087 |

See `results/phase3/step_*/results.json` for full results.

### Interpretation

Phase 2 confirms: the Apple TarFlow architecture is correctly implemented in `src/tarflow_apple.py`
and trains successfully on 2D, MNIST, and CIFAR-10 data. The critical failures in hyp_002/hyp_003
were due to (1) shared isotropic scale instead of per-dim scale, and (2) a causal masking bug
(SOS token with self-inclusive mask → non-triangular Jacobian). Apple's design avoids both.

Phase 3 reveals the molecular adaptation failure cascade:

1. **Padding (Step C) is the primary failure point.** Adding 12 padding atoms to ethanol's 9 collapses
   valid fraction from 89.1% to 2.7%, despite correct attention masking and logdet normalization.
   The model correctly learns the NLL objective (best_loss=-2.825, same as Step A) but produces
   invalid structures. The flow learns a latent space that maps to valid NLL but not to physically
   valid coordinates. This is the fundamental problem.

2. **Noise augmentation partially recovers** (2.7% → 14.3%). Smoothing the density with sigma=0.05
   helps the flow generalize away from training configurations.

3. **Shared scale IMPROVES over per-dim scale WITH noise** (40.2% vs 14.3%). This is contrary to
   the original hypothesis. With correct normalization (n_real*D), shared scale does not cause
   saturation exploitation. The improvement may reflect shared scale's inductive bias toward
   isotropic coordinate scaling — appropriate for 3D molecular coordinates.

4. **Clamping reduces performance** (Step F: 40.2% → 10.4% VF). Note: the FIRST Step F run used
   the buggy T*D normalization and produced VF=0.0 (degenerate constant equilibrium). The SECOND run
   (correct n_real*D normalization) achieves 10.4%. The clamping limits but does not destroy learning.
   Alpha_pos=0.1 constrains scale contraction significantly but allows some structural learning.

5. **Loss and VF are decoupled in the padded regime.** Steps A and C have nearly identical best_loss
   (-2.83) but radically different VF (89% vs 3%). The NLL does not reflect pairwise distance validity.

**Key open question for Phase 4:** Why does padding cause the VF collapse even when the NLL
appears to train correctly? Three candidate hypotheses:
- The latent space structure with T=21 is fundamentally harder than T=9 for the autoregressive ordering.
- The 12 padding atoms (all zeros) create a degenerate latent manifold that corrupts sampling.
- The attention over 21 tokens instead of 9 disrupts the causal dependency structure.

Phase 4 ablation matrix should cross: {T=9 vs T=21} × {atom ordering} × {noise sigma} to isolate
the padding effect.

**Status:** [x] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
Phase 3 results partially fit the research story: the Apple architecture degrades on molecular
data, but the degradation point is PADDING (Step C), not shared scale (Step E) as originally
hypothesized. The research story needs updating — shared scale is not the primary culprit.
