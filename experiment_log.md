# TNAFMOL — Experiment Log

PhD student-maintained. Append-only record of all experiments.

---

### hyp_001 — MD17 Data Pipeline
**Date:** 2026-02-28
**Branch:** `exp/hyp_001`
**Command:** EXECUTE
**Status:** DONE

Downloaded and preprocessed all 8 MD17 molecules (aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil) into canonical frame representation. Total ~3.6M conformations across all molecules.

**Preprocessing steps:**
1. Download from quantum-machine.org (md17_*.npz format)
2. CoM subtraction (mass-weighted)
3. Kabsch alignment to mean structure per molecule
4. Zero-padding to 21 atoms with attention mask
5. Atom type encoding: H=0, C=1, N=2, O=3
6. Deterministic 80/10/10 split (seed=42)

**Output:** 8 dataset directories in `data/md17_{mol}_v1/`, each with dataset.npz, metadata.json, ref_stats.pt, README.md.

**Verification:** All datasets verified for correct shapes, padding, CoM centering, atom type encoding, and split consistency. Energy and pairwise distance distributions are physically plausible.

---

### hyp_002 — TarFlow OPTIMIZE
**Date:** 2026-03-01
**Branch:** `exp/hyp_002`
**Command:** OPTIMIZE
**Status:** FAILURE (angle budget exhausted)
**Success criterion:** valid_fraction > 0.5 on 5+/8 molecules at T=1.0

**Architecture:** TarFlow (Transformer Autoregressive Normalizing Flow) — 8 alternating forward/reverse autoregressive blocks, d_model=128, n_heads=4, atom type embedding dim=16. SOS token prepended to atom sequence to guarantee valid context for atom 0. Base distribution: N(0,I) over real atom coordinates.

**Angles attempted:**

**Angle 1 — SANITY: shift_only=True (volume-preserving flow)**
- Rationale: diagnostics showed affine blocks chain log_scale to max → z≈0 collapse
- shift_only prevents log_det exploitation; optimal solution = conditional mean predictor
- Results: loss plateau at 0.919 (Gaussian entropy floor), z.std=0.0007 (shift collapse)
- Model learned shift≈x → z≈0 for all inputs. At T=2σ, raw N(0,T²) gives >70% valid on all molecules — model adds nothing above Gaussian baseline
- FAILED: shift collapse is equivalent global minimum of NLL for shift-only

**Angle 2 — HEURISTICS: ActNorm (Kingma & Dhariwal 2018, GLOW)**
- Rationale: ActNorm normalizes per-atom output to N(0,1) with data-dependent init; prevents cumulative scale drift; adds log_det contribution; cites established normalizing flow component
- Re-enabled full affine flow (shift_only=False) + use_actnorm=True
- 5k validation run (lr=3e-4): loss converged to -9.67 (excellent in forward direction)
- Forward pass verified: z ~ N(0,1) properly, total log_det = 428, NLL per dof = -15.38
- Samples: valid_fraction = 0.000 on all 8 molecules. min_pairwise_dist ≈ 0.16 Å (atoms clustered)
- Root cause: model learned negative ActNorm log_scale (≈-0.81 per layer), giving large log_det forward contribution (408 total). In sampling inverse, cumulative contraction ≈ 0.45^8 = 0.0013. Temperature has zero effect on sample diversity (verified T=0.5 to T=50.0)
- FAILED: ActNorm created new collapse mode — same class as affine collapse, different mechanism

**Angle 3 — SCALE: skipped**
- Justification: collapse is architectural, not capacity-limited. Larger model with same unconstrained affine+ActNorm will produce same collapse faster. Skipped as non-applicable.

**Best result:** valid_fraction = 0 on all molecules across all angles. Loss curves show excellent NLL improvement but all due to log_det exploitation rather than data distribution learning.

**Root cause diagnosis:** TarFlow's autoregressive affine coupling objective (maximize log_det) always finds degenerate solutions when any scale DOF is unconstrained. Shift-only eliminates the scale DOF but creates shift collapse. ActNorm adds a new scale DOF that gets exploited. This is a fundamental tension in the architecture design.

**W&B runs:**
- Diagnostic: `hyp_002_diag`
- SANITY val: `hyp_002_sanity_val`
- SANITY val_shift: `hyp_002_sanity_val_val_shift`
- SANITY sweep: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/...
- HEURISTICS val: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ras6geue

---

### hyp_003 — TarFlow Stabilization via Soft Clamping + Soft Equivariance
**Date:** 2026-03-01
**Branch:** `exp/hyp_003`
**Command:** OPTIMIZE
**Status:** FAILURE (angle budget exhausted, primary criterion not met)
**Success criterion:** valid_fraction ≥ 0.5 on ≥ 4/8 molecules

**Architecture:** TarFlow (same as hyp_002) + three stabilization changes:
1. Asymmetric soft scale clamping: _asymmetric_clamp(s, alpha_pos=0.02, alpha_neg=2.0) — replaces tanh, bounds expansion to ≈0.02 per layer
2. Log-det regularization: loss += reg_weight × (log_det_per_dof)² — penalizes deviation from log_det=0
3. Soft equivariance: random SO(3) rotation + CoM noise augmentation during training, unit-variance normalization (global_std=1.2905 Å)

**Diagnostic (500 steps):**
- log_det/dof = 0.78 (stable, not exploding like hyp_002)
- valid_fraction = 0 on all molecules (new collapse mode: alpha_pos saturation)
- Root cause: model sets log_scale ≈ +alpha_pos uniformly across all 8 blocks → log_det/dof = 8 × alpha_pos/π ≈ 0.78 → samples compressed by exp(-0.78 × dof) per atom

**Angles attempted:**

**Angle 1 — SANITY: All three fixes + regularization tuning**
- SANITY val (2000 steps, α=0.05, reg=1.0, lr=3e-4): malonaldehyde 11.6%, ethanol 7.8% — PROMISING
- SANITY sweep (24/30 runs, 3000 steps, Bayesian): best = α=0.02, reg=5, lr=1e-4 → mean VF 17.5%
  - Top results: ethanol 39.3%, malonaldehyde 38.0%, uracil 26.3%, benzene 22.0%
  - W&B sweep: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/rccehd8m
- SANITY full (10000 steps, best sweep config): mean VF 13.1%
  - ethanol 33.0%, malonaldehyde 32.6%, benzene 16.2%, uracil 13.8%
  - Loss flat at 0.8689 from step 300 onward; log_det/dof locked at 0.100; best ckpt at step 500
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/o5naez7a
- FAILED: 0/8 molecules ≥ 50%

**Angle 2 — HEURISTICS: SBG training recipe (Tan et al. 2025)**
- Changes: AdamW betas=(0.9,0.95), OneCycleLR pct_start=0.05, EMA decay=0.999, batch_size=512
- Best SANITY hyperparams (α=0.02, reg=5) retained
- HEURISTICS val (2000 steps, lr=3e-4): mean VF 16.8%, ethanol 41.3%, malonaldehyde 40.7% — PROMISING
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/o6pnle0k
- HEURISTICS sweep (12 runs, 3000 steps): best = bs=512, ema=0.999, lr=1e-3 → mean VF 18.3% (BEST OVERALL)
  - W&B sweep: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/cmgrp6jo
- HEURISTICS full (20000 steps, best sweep config): mean VF 14.3%
  - malonaldehyde 38.0%, ethanol 33.4%, uracil 13.6%, benzene 15.2%
  - Same saturation pattern: loss flat at 0.8689, log_det locked at 0.100, best ckpt at step 2000
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/4079op64
- FAILED: 0/8 molecules ≥ 50%

**Angle 3 — SCALE: SKIPPED**
- Justification: model saturates at step 150 — not capacity-limited. Loss flat from step 150 onward.
  Alpha_pos saturation is a mathematical equilibrium, not a capacity limitation. Larger model would hit the same equilibrium at the same step count.

**Best result:** mean valid fraction 18.3% (HEURISTICS sweep best config at 3000 steps). 0/8 molecules ≥ 50%.

**Root cause diagnosis (deeper):** The NLL gradient pushes log_scale to the upper bound (+alpha_pos) while the log-det regularization gradient pulls toward 0. The two gradients reach a stable fixed point at log_det/dof = alpha_pos regardless of regularization weight, LR, or batch size. The alpha_pos saturation equilibrium is not a local minimum — it is an attractor. No gradient-based optimization can escape it without fundamentally changing the loss formulation.

**Per-molecule pattern:** Clear inverse correlation between molecule size (n_atoms) and valid fraction. 9-atom molecules (ethanol, malonaldehyde): 33-38%. 21-atom molecules (aspirin): <1%. Each additional atom pair is an independent opportunity for close-collision failure due to residual compression.

**Story fit:** CONFLICT — TarFlow is not viable for molecular conformations under standard MLE + regularization. Two consecutive failures confirm the architectural issue. Research story needs to pivot.

---

### hyp_004 — TarFlow Architectural Ablation + Optimization
**Date:** 2026-03-02
**Branch:** `exp/hyp_004`
**Command:** OPTIMIZE
**Hypothesis:** Three architectural modifications (bidirectional type conditioning, permutation augmentation, positional encodings) will improve over hyp_003 baseline (14.3% mean VF).

**Diagnostic:**
- Bidirectional type encoder (BidirectionalTypeEncoder) implemented in src/model.py
- All 3 modifications are independent flags (use_bidir_types, use_perm_aug, use_pos_enc)
- W&B diagnostic run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/8s3kfzri

**Angle 1 — SANITY: 6-config architectural ablation**
- 6 configs tested (3000 steps each): A_baseline, B_bidir, C_perm, D_pos, E_bidir_perm, F_bidir_pos
- Results (mean VF): D_pos 17.65% > F_bidir_pos 16.40% > A_baseline 12.68% > C_perm 12.60% > B_bidir 11.80% > E_bidir_perm 10.92%
- Best: **D_pos** (use_pos_enc=True only) — positional encodings add +5ppt
- Permutation augmentation hurts: atom ordering is informative for MD17. Bidir types slightly hurts.
- ALL configs: loss saturates at 0.869, log_det/dof=0.100 by step ~150 (alpha_pos equilibrium)
- LR sweep (D_pos only): lr=5e-5 marginally best at 17.73% (spread <0.5ppt). Best config: D_pos, lr=5e-5, bs=128.
- SANITY full run (D_pos, 10000 steps, lr=5e-5): mean VF = 17.48% (no improvement over 3000 steps)
  - Best checkpoint: step 1000 (val_loss=0.8176). W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/k88dxne7
  - ethanol 44.2%, malonaldehyde 39.8%, benzene 22.2%, uracil 18.4%, toluene 7.4%, salicylic_acid 5.6%, naphthalene 1.8%, aspirin 0.4%
- FAILED primary criterion (no molecule ≥ 50%)

**Angle 3 — SCALE: SKIPPED**
- Justification: loss saturates at step ~150 across ALL 6 ablation configs — confirmed alpha_pos equilibrium.
  Not capacity-limited. Increasing model size would reach same equilibrium faster.

**Angle 2 — HEURISTICS: SBG training recipe (Tan et al. 2025, ICML)**
- SBG recipe: AdamW betas=(0.9,0.95), OneCycleLR, EMA decay=0.999, batch_size=512
- Applied to D_pos config (best architectural variant)
- HEURISTICS val (3000 steps, lr=3e-4): mean VF 17.93% (+0.45ppt vs SANITY full) — PROMISING
  - Better val_loss: 0.8116 at step 1500 (vs 0.8176 for SANITY full)
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ht2xyghi
- HEURISTICS sweep (9 runs: ema_decay=[0.99,0.999,0.9999] x lr=[1e-4,3e-4,1e-3], bs=512 fixed):
  *(results pending — sweep running)*
- HEURISTICS full run: *(pending sweep completion)*

**Best result so far:** *(pending HEURISTICS sweep + full)*

**Pattern consistent with hyp_003:** alpha_pos saturation is the dominant failure mode. Architectural improvements (pos_enc +5ppt) and training recipe (SBG +0.4ppt) improve within the constrained regime but cannot escape the equilibrium. 20 architectural combinations and training recipe variants all converge to the same loss plateau by step 150.
