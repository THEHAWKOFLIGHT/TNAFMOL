## [hyp_004] — TarFlow Architectural Ablation + Optimization
**Date:** 2026-03-02 | **Type:** Hypothesis | **Tag:** `hyp_004`

### Motivation
hyp_002 and hyp_003 demonstrated that TarFlow's autoregressive affine flow fails on molecular conformations due to log_det exploitation and alpha_pos saturation. However, three architectural gaps were identified that may contribute to the poor valid fraction:

1. **Causal masking hides future atom types** — during generation, each atom is built without knowledge of what atom types come later. The model cannot condition on the full molecular composition.
2. **No permutation augmentation** — the model overfits to the arbitrary atom ordering in the dataset instead of learning permutation-invariant structure.
3. **No positional encodings** — the model cannot distinguish atom positions within the autoregressive sequence.

This experiment ablates all three fixes (bidirectional type conditioning, permutation augmentation, positional encodings) on top of the hyp_003 stabilization baseline, then optimizes the best combination with SBG training recipe and hyperparameter tuning.

### Method
OPTIMIZE protocol with 3 angles:
- **SANITY:** 6-config architectural ablation (3000 steps each) — baseline, bidir_types, perm_aug, pos_enc, bidir+perm, bidir+pos
- **HEURISTICS:** SBG training recipe (Tan et al. 2025) + hyperparameter sweep on best ablation config
- **SCALE:** Capacity increase (d_model=256, n_blocks=12, 50k steps) — conditional on loss curve analysis

### Results
*(To be filled after experiment completes)*

### Interpretation
*(To be filled after experiment completes)*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
