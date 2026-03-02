## Diagnostic Report — hyp_004 TarFlow Architectural Ablation

### Baseline Failure Analysis

**Diagnostic run:** 500 steps, hyp_003 best config + use_bidir_types=True.
**W&B run:** https://wandb.ai/kaityrusnelson1/tnafmol/runs/8s3kfzri

Per-molecule valid fractions (500 samples each, from best checkpoint at step 100):

| Molecule | Atoms | Valid Fraction | Min Dist Mean | PW Divergence |
|----------|-------|---------------|---------------|---------------|
| aspirin | 21 | 0.2% | 0.380 | 0.150 |
| benzene | 12 | 11.4% | 0.551 | 0.256 |
| ethanol | 9 | 33.4% | 0.689 | 0.080 |
| malonaldehyde | 9 | 30.6% | 0.679 | 0.077 |
| naphthalene | 18 | 1.4% | 0.420 | 0.189 |
| salicylic_acid | 16 | 2.6% | 0.454 | 0.112 |
| toluene | 15 | 6.4% | 0.486 | 0.114 |
| uracil | 12 | 17.4% | 0.574 | 0.102 |
| **Mean** | — | **12.9%** | **0.530** | **0.135** |

**Comparison to hyp_003 baselines:**
- hyp_003 SANITY full (10k steps): 13.1% mean VF
- hyp_003 HEURISTICS sweep best: 18.3% mean VF
- hyp_004 diagnostic (500 steps, +bidir_types): 12.9% mean VF

The diagnostic with bidirectional type conditioning produces comparable results to hyp_003 at much shorter training (500 vs 10k steps), which is expected since the model saturates early. No new collapse modes observed.

### Root Cause

The root cause is unchanged from hyp_003: **alpha_pos saturation equilibrium**. Training metrics confirm:

- loss plateaus at 0.8692 from step ~150 onward (identical to hyp_003's 0.8689-0.8690)
- log_det/dof locks at exactly 0.100 from step 100 onward (= alpha_pos cumulative across blocks)
- Val log_det/dof = 0.104 (consistent with train)
- Gradient norm drops to 0.006-0.007 — the optimizer has converged to the equilibrium

The bidirectional type encoder does not break the equilibrium. It adds global molecular composition information to the type embeddings, but the fundamental limitation is in the affine coupling layers — each layer's scale saturates at alpha_pos regardless of what information the transformer has access to.

**Molecule-size scaling confirmed:** Clear inverse correlation between atom count and valid fraction:
- 9-atom molecules (ethanol, malonaldehyde): ~30-33% valid
- 12-atom molecules (benzene, uracil): ~11-17% valid
- 15-18 atom molecules (toluene, salicylic_acid, naphthalene): ~1-6% valid
- 21-atom aspirin: 0.2% valid

### Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY | Yes | The 6-config ablation is the core of this experiment — it tests whether architectural gaps (no bidirectional types, no permutation augmentation, no positional encodings) contribute to the poor valid fraction. The diagnostic confirms the baseline behavior is stable with bidir_types enabled. |
| KNOWN HEURISTICS | Yes | SBG recipe (Tan et al. 2025) with best ablation config. This was partially tested in hyp_003 (improved from 13.1% to 14.3%) but not combined with architectural fixes. |
| SCALE | Conditional | Only if SANITY or HEURISTICS show loss curve is still dropping at training end, suggesting underfitting. Current evidence (saturation at step 150) suggests SCALE will not help. |

### Proposed Angles (preliminary)

1. **SANITY — 6-config architectural ablation** (3000 steps each): Test all combinations of bidir_types, perm_aug, pos_enc. Identify which modifications improve valid fraction beyond baseline.

2. **HEURISTICS — SBG recipe on best ablation config**: Apply SBG training recipe (Tan et al. 2025) — OneCycleLR, AdamW betas=(0.9, 0.95), EMA, batch_size=512 — to the winning ablation config.

3. **SCALE — capacity increase** (conditional): d_model=256, n_blocks=12, 50k steps. Only if loss curves from SANITY/HEURISTICS show continued descent.
