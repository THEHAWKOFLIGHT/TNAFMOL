# OPTIMIZE Plan Sub-report — hyp_011: Crack MD17 Multi-Molecule TarFlow

## Diagnostic Summary

The Phase 1 SANITY runs revealed that model capacity is the primary driver of multi-molecule VF improvement. Run B (channels=384, blocks=6, 20k steps) achieved mean VF = 83.9% (+12.3pp), while Run A (same model as hyp_010, 50k steps) achieved only 73.3% (+1.7pp) and collapsed on toluene (3.2%). The HEURISTICS sweep will use Run B architecture as the base to push past 85%.

## Angles

**Angle 1 — SANITY: Longer training + bigger model (COMPLETE)**

Phase 1 SANITY tested two hypotheses in parallel:
- Run A: Same architecture as hyp_010 (256ch, 4blk), 50k steps, lr=5e-4 → Mean VF = 73.3% (toluene collapse: 3.2%)
- Run B: Bigger model (384ch, 6blk), 20k steps, lr=5e-4 → Mean VF = 83.9% (all 8 molecules > 67%)

Result: **Promising criterion met.** Run B exceeds 78% threshold by +5.9pp. Run B is the winning architecture for HEURISTICS.

Key insight: Capacity > training budget for multi-molecule generalization. The 384ch/6blk model is significantly better at 20k steps than 256ch/4blk at 50k steps.

**Angle 2 — KNOWN HEURISTICS: Sweep over learning rate, log-det regularization, noise sigma**
- **What changes**: Systematic hyperparameter sweep using Run B architecture (channels=384, blocks=6, lpb=2) as base. Three axes swept:
  - `lr`: {1e-4, 3e-4, 5e-4} — Run B used 5e-4 which may not be optimal; lower LR with more steps could be better
  - `ldr` (log_det_reg_weight): {0.0, 2.0, 5.0} — hyp_007 showed ldr=5.0 was critical for training stability with model.py. Apple architecture omitted it. Worth testing if it helps at higher VF.
  - `noise_sigma`: {0.03, 0.05, 0.1} — current 0.05 is inherited from und_001; more/less noise augmentation may change convergence
  - `n_steps` fixed at 20k (same as Run B — avoid long-training collapse risk)
- **Literature citation**:
  - LR sweep: standard practice for normalizing flows (Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and Inference", JMLR 2021, Section 4.4).
  - Log-det regularization: encourages stability by penalizing extreme determinants. Standard technique in flow training (Grathwohl et al., "FFJORD", ICLR 2019).
  - Data augmentation noise: Gaussian perturbation of training targets reduces overfitting, standard in structure generation (Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D", ICML 2022).
- **Why it applies here**: Run B shows capacity helps but aspirin/ethanol/salicylic_acid remain below 75%. The lower-VF molecules likely need different training dynamics. LR affects convergence speed vs final quality tradeoff. ldr=5.0 was shown in hyp_007 to prevent training instability in multi-molecule settings.
- **Validation run**: Run one config (lr=3e-4, ldr=2.0, noise_sigma=0.05) for 5k steps on cuda:8. Promising if: loss decreases smoothly (no spikes) and better trajectory than Run B at same step.
- **Sweep**: W&B sweep, 27 total configs (3×3×3), run on escher/germain via Slurm array (27 jobs × ~45 min = ~45 min wall with full parallelism).
- **Promising criterion after sweep**: Best config achieves mean VF > 86%.

**Angle 3 — SCALE: Maximum capacity**
- **What changes**: channels=512, num_blocks=8, layers_per_block=2 (~25M params vs Run B's ~14M). Best lr and ldr from Phase 2 sweep. n_steps=100k with warmup=2000. batch_size=256.
- **Validation run**: 5k steps on cuda:8 with the big model. Promising if: loss at step 5k is better than Run B's loss at step 5k. If training is unstable (loss spikes or diverges), lower lr further.
- **Sweep**: Temperature sweep at eval time only (no training) — sweep temp ∈ {0.7, 0.8, 0.9, 0.95, 1.0} on the best checkpoint. This is free (eval only) and can boost VF without additional training.
- **Promising criterion**: Mean VF > 90%.

## Validation Run for HEURISTICS (before sweep)

Before launching the full 27-config sweep, validate the setup with one representative config:
- Config: channels=384, blocks=6, lpb=2, lr=3e-4, ldr=2.0, noise_sigma=0.05, n_steps=5000, batch=128
- GPU: cuda:8 (test GPU — <5 min)
- Promising if: loss at step 5k <= -1.0 (training is proceeding normally)

## Questions / Concerns

1. **Toluene collapse in Run A**: At 50k steps, toluene collapsed to 3.2% VF while other molecules improved. This suggests molecule-specific late-training instability. The HEURISTICS sweep should monitor per-molecule VF at intermediate checkpoints, not just final. If ldr helps stabilize this, it would be visible.

2. **val_loss divergence**: Both runs show high final val_loss (Run A: 5.79, Run B: 2.58) vs best val_loss (Run A: 1.35 at step 1000, Run B: 0.47 at step 500). This indicates overfitting to training distribution. The final checkpoint is used (per und_001 convention) because it gives better sample quality despite higher val_loss. This convention should be validated for the sweep: check if VF improves by using best-val-loss checkpoint vs final.

3. **aspirin/salicylic_acid gap**: These molecules remain at ~68-71% VF even with Run B. They are structurally complex (more heavy atoms, benzene ring + functional groups). The SCALE angle targeting 90%+ will need to particularly help these molecules.
