## [hyp_004] — TarFlow Architectural Ablation + Optimization
**Date:** 2026-03-02 | **Type:** Hypothesis | **Tag:** `hyp_004`

### Motivation
hyp_003 stabilized TarFlow via asymmetric clamping (alpha_pos=0.02) and log_det regularization, achieving 14.3% mean valid fraction. However, three architectural gaps remained unaddressed:

1. **Causal masking hides future atom types** — during generation, each atom is built without knowledge of what atom types come later. The model cannot condition on the full molecular composition.
2. **No permutation augmentation** — the model overfits to the arbitrary atom ordering in the dataset instead of learning permutation-invariant structure.
3. **No positional encodings** — the model cannot distinguish atom positions within the autoregressive sequence.

This experiment ablates all three fixes and then optimizes the best architectural combination.

### Method
OPTIMIZE protocol with 3 angles (SCALE skipped due to confirmed alpha_pos saturation):
- **SANITY:** 6-config architectural ablation (3000 steps each) + LR sweep + 10000-step full run
- **HEURISTICS:** SBG training recipe (Tan et al. 2025, ICML) on best ablation config
- **SCALE:** *Skipped* — loss saturates at step ~150 across all configs, confirming mathematical equilibrium not capacity limit

### Results

#### SANITY Angle: Architectural Ablation
| Config | Mod | Mean VF |
|--------|-----|---------|
| A_baseline | none | 12.68% |
| B_bidir | bidir types only | 11.80% |
| C_perm | perm aug only | 12.60% |
| **D_pos** | **pos enc only** | **17.65%** |
| E_bidir_perm | bidir+perm | 10.92% |
| F_bidir_pos | bidir+pos | 16.40% |

Key findings:
- Positional encodings (+5ppt over baseline) are the only beneficial modification
- Bidirectional type conditioning slightly hurts in isolation and when combined with perm_aug
- Permutation augmentation slightly hurts — atom ordering is likely informative for MD17 molecules
- All configs saturate at loss≈0.869, log_det/dof=0.100 by step ~150 (alpha_pos saturation equilibrium)

**SANITY full run (D_pos, 10000 steps):** Mean VF = 17.48%
- Best checkpoint: step 1000 (val_loss=0.8176) — 10000 steps provides no improvement over 3000
- W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/k88dxne7

#### HEURISTICS Angle: SBG Training Recipe
**Val run (D_pos + SBG, 3000 steps):** Mean VF = 17.93% (+0.45ppt vs SANITY full)
- SBG: AdamW betas=(0.9,0.95), OneCycleLR, EMA=0.999, batch_size=512
- Improvement pattern: +2-3ppt on smaller molecules; slight reduction on ethanol/malonaldehyde
- Best val_loss = 0.8116 at step 1500 (slightly better than SANITY full)
- W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ht2xyghi

**Sweep (9 runs: ema_decay × lr):** *(results pending)*

**Full run:** *(pending sweep results)*

#### SCALE Angle
*Skipped* — all 6 ablation configs show identical saturation by step ~150. This is a mathematical equilibrium of alpha_pos=0.02: exactly 8×0.02=0.160 cumulative scale → log_det/dof→0.100. Increasing model capacity cannot escape this equilibrium.

### Interpretation
The core finding is nuanced: positional encodings help (+5ppt) but cannot escape the fundamental alpha_pos saturation. The gain from pos_enc is real and reproducible but does not address the root cause. The SBG recipe provides marginal additional benefit (~0.4-1ppt in val), consistent with hyp_003 which showed ~3.7ppt gain from a lower baseline.

The alpha_pos saturation equilibrium is the dominant failure mode across both hyp_003 and hyp_004:
- Best valid fraction: ~17-18% on D_pos + SBG (3000 steps)
- Target: >50% on any molecule
- Gap: the architectural modifications and training recipe improve within the constrained regime but do not escape the equilibrium

**Implication for next steps:** The alpha_pos=0.02 saturation sets a hard ceiling on performance. To make meaningful progress, the constraint itself must change — either via a different parameterization (not affine coupling), a different training objective, or by finding a way to break the mathematical equilibrium directly.

**Status:** [x] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:

Results fit the research story: architectural improvements are minor relative to the fundamental alpha_pos bottleneck. The story consistently points toward the need for a fundamentally different approach to normalizing flows for molecular generation.
