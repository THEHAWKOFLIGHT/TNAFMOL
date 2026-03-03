## OPTIMIZE Plan Sub-report — hyp_004 TarFlow Architectural Ablation

### Diagnostic Summary
The alpha_pos saturation equilibrium (root cause from hyp_003) persists with bidirectional type conditioning enabled. Loss plateaus at 0.8692 by step 150, log_det/dof locks at 0.100. Mean valid fraction is 12.9% — comparable to hyp_003 baselines. The proposed ablation tests whether architectural modifications (type conditioning, permutation symmetry, positional information) can improve sample quality within this equilibrium.

### Angles

**Angle 1 — SANITY: 6-config architectural ablation**
- What changes: Ablate all combinations of three boolean flags (use_bidir_types, use_perm_aug, use_pos_enc) on the hyp_003 best config.
- Configurations:
  - A: baseline (all False) — establishes hyp_004 baseline on hyp_003 config
  - B: use_bidir_types=True — bidirectional molecular composition context
  - C: use_perm_aug=True — permutation augmentation for ordering invariance
  - D: use_pos_enc=True — positional encodings for sequence position
  - E: use_bidir_types=True + use_perm_aug=True — composition + invariance
  - F: use_bidir_types=True + use_pos_enc=True — composition + position
  - Note: perm_aug + pos_enc is intentionally excluded (antagonistic — positional encodings on randomized ordering are meaningless)
- Validation run: 3000 steps each, promising if: any config achieves mean VF > 0.20 (above hyp_003 best of 18.3%)
- If promising — Sweep: lr in [5e-5, 1e-4, 2e-4], batch_size in [128, 256] on the best config
- Justification: The diagnostic confirms these modifications do not introduce new collapse modes. The ablation isolates which architectural gap (if any) is responsible for low valid fraction.

**Base config for all ablation runs (hyp_003 best):**
- d_model=128, n_blocks=8, n_heads=4, d_ff=512
- alpha_pos=0.02, alpha_neg=2.0, log_det_reg_weight=5.0
- lr=1e-4, batch_size=128, n_steps=3000
- augment_train=True, normalize_to_unit_var=True
- eval_every=500, n_eval_samples=500, eval_at_end=True

**Angle 2 — KNOWN HEURISTICS: SBG training recipe (Tan et al. 2025)**
- What changes: Apply Stochastic Boltzmann Generator training recipe to the best config from Angle 1.
- Literature citation: Tan et al., "Scalable Training of Normalizing Flows for Molecular Systems" (2025). SBG recipe: AdamW with betas=(0.9, 0.95), OneCycleLR schedule, EMA with decay=0.999, batch_size=512.
- Why it applies here: SBG recipe was designed for normalizing flows on molecular systems and showed improvement in hyp_003 (13.1% -> 14.3% mean VF). Combined with architectural improvements from Angle 1, it may push valid fraction higher.
- Validation run: 3000 steps with SBG recipe on best Angle 1 config, promising if: mean VF > best Angle 1 result
- If promising — Sweep: ema_decay in [0.99, 0.999, 0.9999], lr in [5e-4, 1e-3, 2e-3], batch_size in [256, 512]

**Angle 3 — SCALE: capacity increase (conditional)**
- What changes: d_model=256, n_blocks=12, n_steps=50000
- Validation run: 5000 steps, promising if: loss is still decreasing at step 5000 (underfitting evidence)
- If promising — Sweep: d_model in [192, 256], n_blocks in [10, 12]
- Skip justification (if applicable): Will skip if SANITY and HEURISTICS both show saturation by step 150 (same as hyp_003), indicating the bottleneck is architectural equilibrium, not capacity.

### Execution Plan

1. Launch all 6 SANITY configs in parallel as a W&B sweep on escher (GPUs 3-7 via Slurm)
2. Wait for all 6 runs to complete
3. Analyze results: identify best config, check if promising criterion met
4. If promising: run hyperparameter sweep on best config
5. Launch HEURISTICS with best config from SANITY
6. Evaluate SCALE necessity based on loss curves

### Questions / Concerns
None. The diagnostic confirms the modifications are safe and the baseline behavior is preserved. Proceeding with the 6-config ablation.
