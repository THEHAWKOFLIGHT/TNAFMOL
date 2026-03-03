# TNAFMOL — Process Log

PhD student-maintained. Append-only decisions, reasoning, file manifest.

---

## 2026-02-28 — hyp_001: MD17 Data Pipeline
**Branch:** `exp/hyp_001`

### Decisions & Reasoning
- Used MD17 (not revised rMD17) dataset — more samples per molecule (~130k-993k vs 100k subsampled)
- Download URL: `http://quantum-machine.org/gdml/data/npz/md17_{mol}.npz` (newer format with z/R/E/F keys)
- Canonical frame uses mass-weighted CoM subtraction + Kabsch (SVD) alignment to mean structure
- Two-pass alignment: first align to initial mean, then recompute mean and re-align for stability
- Padding to 21 atoms (aspirin is the largest). Mask: 1.0 for real atoms, 0.0 for padding
- Atom type indices (not one-hot) as primary encoding. One-hot also stored for convenience
- Used deterministic split with seed=42 for reproducibility
- Fixed PyTorch DLL load failure by enabling Windows long paths and reinstalling torch 2.10.0+cu126

### New Files Created
- `src/data.py` — data pipeline: download, preprocessing, padding, encoding, splitting, reference stats, PyTorch datasets
- `src/preprocess.py` — main preprocessing script (runs full pipeline)
- `src/metrics.py` — evaluation metrics stubs (valid_fraction, pairwise distance divergence, energy Wasserstein, RMSD coverage, bond length MAE)
- `src/visualize.py` — verification visualization script
- `data/md17_{mol}_v1/` (8 dirs) — processed datasets with metadata.json, ref_stats.pt, README.md
- `experiments/hypothesis/hyp_001_data_pipeline/results/*.png` — verification plots
- `documentation/environment_setup.md` — environment documentation

### Commits
- `2d25fd6` — [hyp_001] docs: add environment setup documentation
- `31f4199` — [hyp_001] code: add data pipeline, metrics, and visualization modules
- `8345dd2` — [hyp_001] data: add processed MD17 metadata, ref_stats, and READMEs for all 8 molecules
- `26cd199` — [hyp_001] results: add verification plots for all 8 MD17 molecules

### Notes
- Environment: PyTorch 2.10.0+cu126, Python 3.11.9, CUDA 13.0, RTX 2000 Ada 8GB
- Benzene has the most conformations (627,983) and narrowest energy distribution (std=2.3 kcal/mol) — rigid molecule
- Uracil is the only molecule with all 4 atom types (H, C, N, O)
- Aspirin is the largest molecule (21 atoms) — defines the padding size
- Total dataset size across all molecules: ~3.6M conformations

---

## 2026-03-01 — hyp_002: TarFlow OPTIMIZE
**Branch:** `exp/hyp_002`

### Decisions & Reasoning
- TarFlow: transformer autoregressive normalizing flow. Autoregressive ordering over ATOMS (not coordinates).
- Each atom gets affine params (shift + log_scale) from causal attention over previous atoms.
- Within atom, all 3 coords transformed simultaneously with same affine params.
- Alternating direction: even layers forward (0→N), odd layers reverse (N→0).
- Atom type conditioning via learned embedding concatenated to position features at every layer.
- Base distribution: isotropic Gaussian N(0,I) over all real atom coordinates.
- Training loss: NLL = -sum(log_s) - log p_z(z) where z = forward(x).
- Padding handled by zeroing out attention to/from padding positions AND zeroing log-det contributions from padded atoms.
- INTENTION (write-before-execute): Implement TarFlow in src/model.py, then training loop in src/train.py.
  Run diagnostic on GPU 8, then OPTIMIZE angles SANITY → HEURISTICS → SCALE.
- Architecture choices: L=8 blocks, d_model=128, n_heads=4, FFN ratio 4x.
  Start with per-atom feature dim = 3 (positions) + atom_type_emb_dim (16).
- Input representation: input to each layer is (batch, max_atoms, 3) continuous positions
  plus (batch, max_atoms, atom_type_emb_dim) atom type embeddings.
  Transformer reads (3 + atom_type_emb_dim) features per atom, outputs (6) per atom: [shift(3), log_scale(3)].
  Actually simpler: predict scalar log_scale + 3D shift from context — but spec says per-atom affine with same params for all 3 coords. Using: predict shift(3) + log_scale(1) per atom from context.

### New Files Created
- `src/model.py` — TarFlow model (TarFlowBlock, TarFlow)
- `src/train.py` — training loop, evaluation, W&B logging
- `experiments/hypothesis/hyp_002_tarflow/` — experiment directory (already exists)

### Decisions & Reasoning (continued)
- **Diagnostic (1000 steps, no shift_only):** Model learned to chain large log_scale across all 8 blocks, mapping all samples to z≈0 (log_det ≈ 568, max with tanh*3 clamp). Root cause: affine flow can always exploit log_det by expanding forward direction. Valid fraction = 0 on all molecules.
- **SANITY Fix 1 (log_scale_max=0.5):** Reduced tanh bound from 3.0 to 0.5. Insufficient — model finds chain-shift alternative: expand by e^0.5 per block, then shift to cancel, log_det still hits max (108). Valid fraction still 0.
- **SANITY Fix 2 (shift_only=True):** Volume-preserving flow (log_det=0). Prevents exploit entirely. Optimal solution becomes conditional mean predictor z_i = x_i - E[x_i|x_{<i}]. 1000-step validation: loss converges to 0.919 = -log N(0,1)/dof, ethanol/malonaldehyde valid_fraction ~22% at T=1.0. PROMISING.
- SOS (start-of-sequence) token prepended to atom sequence: prevents NaN in attention when atom 0 has no causal context.
- Combined float additive causal+padding mask: eliminates PyTorch deprecation warning for mixed mask types.
- **SANITY Sweep (INTENTION):** Grid sweep over n_steps=(5k,10k,20k) × lr=(1e-4,3e-4) with warmup_steps=500, batch_size=256, shift_only=True. Using W&B sweep agent via /tmp/sanity_sweep.py. Run on cuda:0.

### New Files Created
- `src/model.py` — TarFlow model (TarFlowBlock with SOS token, combined causal+padding mask; TarFlow with alternating direction blocks, shift_only mode)
- `src/train.py` — training loop, per-molecule evaluation, W&B logging with artifacts
- `experiments/hypothesis/hyp_002_tarflow/angles/diagnostic/` — diagnostic run outputs
- `experiments/hypothesis/hyp_002_tarflow/angles/sanity/val/` — SANITY val run (affine, collapsed)
- `experiments/hypothesis/hyp_002_tarflow/angles/sanity/val_shift/` — SANITY shift_only val run (promising)

### Commits
- `949135f` — [hyp_002] code: implement TarFlow model and training loop
- `0388079` — [hyp_002] code: add log_scale_max param to prevent coordinate collapse
- `ba4af6b` — [hyp_002] code: add shift_only mode to prevent log-det exploitation

- **SANITY Sweep results:** Grid sweep (n_steps=5k/10k, lr=1e-4) — ALL runs plateau at loss=0.919 with grad_norm~0.001. No improvement with more steps or different LR. DIAGNOSIS: shift-only collapse — model learns shift≈x_i, mapping all data to z≈0. Samples from N(0,T²) with model≈raw Gaussian. Raw N(0, (2σ)²) gives >70% valid_fraction on all 8 molecules, but model provides no improvement over baseline.
- **SANITY Angle EXHAUSTED:** shift_only flow converges to degenerate solution (shift≈x, z≈0). This is equivalent to the affine collapse but via shift alone. Root cause: the global minimum of NLL with shift_only is shift=x → z=0 → ||z||²→0.
- **HEURISTICS ANGLE (INTENTION):** Re-enable affine flow but add per-atom ActNorm (Activation Normalization, Kingma & Dhariwal 2018, GLOW) after each coupling block to prevent cumulative scale drift. ActNorm: after each block, normalize per-atom-position output to zero mean and unit variance with learned scale/bias. This is a standard normalizing flow component missing from the current implementation. Combined with proper data normalization (per-molecule global std scaling), this should prevent both affine and shift collapses.

### New Files Created (continued)
- `experiments/hypothesis/hyp_002_tarflow/angles/sanity/sweep/runs/` — sweep run outputs (2 completed: 5k/lr=1e-4, 10k/lr=1e-4)

### Commits
- `a4d27f7` — [hyp_002] code: fix W&B sweep integration and unique per-run output dirs

### Notes
- GPU 8 (~17GB free) for test/validation runs; GPU 0 for full training
- Target: valid_fraction > 0.5 on 5+/8 molecules
- Shift-only flow is volume-preserving: log_det = 0 always. NLL = -log p_z(z) / n_dof = 0.919 at optimum (standard Gaussian entropy).
- Val_shift results at T=1.0: ethanol=22.8%, malonaldehyde=22.4%, uracil=7.2%, benzene=6.6%. More steps needed.
- Shift collapse: model learns shift≈x, z≈0. At T=2σ (T≈2 for most molecules), raw N(0,T²) gives >70% valid on all 8 molecules.
- Model NOT helping — samples equivalent to raw Gaussian at T=2σ_data.

---

## 2026-03-01 — hyp_002 HEURISTICS: ActNorm validation run
**Branch:** `exp/hyp_002`

### Decisions & Reasoning (INTENTION)
- SANITY angle (shift_only) exhausted: loss plateau at 0.919 (Gaussian entropy floor), shift collapse confirmed (z.std=0.0007), model equivalent to raw Gaussian sampler.
- HEURISTICS angle: ActNorm (Kingma & Dhariwal 2018, GLOW paper) after each affine coupling block. Re-enable full affine flow (shift_only=False). ActNorm normalizes per-atom-position output to N(0,1) via data-dependent initialization, adds log_det contribution, breaks both shift and scale collapse mechanisms.
- ActNorm implemented in src/model.py as dedicated nn.Module with learned shift+log_scale per atom per coord. TarFlow supports use_actnorm=True parameter.
- Validation run: 5000 steps, lr=3e-4, batch_size=256, cuda:0. Promising if valid_fraction > 0.1 on any molecule at T=1.0.
- Quick test (200 steps) showed: loss=-1.47 (vs 0.919 floor), latent z.std=1.0 (properly normalized), grad_norm~20 (model is learning). PROCEEDING to full 5k val run.

### HEURISTICS Val Run Results — FAILURE (ActNorm collapse)
- Loss converged to -9.67 at step 5000 (vs 0.919 for shift-only — clear improvement in forward direction)
- Forward pass verified: z ~ N(0,1) correctly, total log_det = 428, NLL per dof = -15.38 (excellent)
- Samples: valid_fraction = 0.000 on all 8 molecules
- Root cause: **ActNorm log_scale exploitation** — model learned negative log_scale (≈-0.81) in ActNorm layers
  - Forward: large log_det from ActNorm contraction (contribution ≈ 51 per layer × 8 = 408)
  - Inverse (sampling): cumulative contraction = exp(-0.81)^8 ≈ 0.0013 on the noise
  - Temperature has ZERO effect on sample diversity (verified T=0.5 to T=50.0 all give std=0.2675)
  - Sample spread (std=0.27 Å) is 3.3× smaller than real data (std=0.91 Å)
  - All samples cluster together with min pairwise dist ≈ 0.16 Å (atoms on top of each other)
- Same root class as shift and affine collapse: model exploits any unconstrained scale DOF to maximize log_det
- HEURISTICS ANGLE FAILED: ActNorm did not prevent collapse, it created a new collapse mode

### SCALE Angle Assessment — SKIPPED (not applicable)
- SCALE (bigger model, more steps) will NOT fix collapse: this is NOT a capacity problem
- The collapse mechanism is architectural: unconstrained affine + ActNorm always finds degenerate log_det solutions
- Increasing d_model or n_steps will produce the same collapse faster, not fix it
- SCALE skipped with justification: degenerate solution is independent of capacity

### Optimize Failure Report
- All 3 angles exhausted (SANITY: shift collapse; HEURISTICS: ActNorm scale collapse; SCALE: skipped as non-applicable)
- Root cause: autoregressive affine flow objective (maximize log_det) always finds degenerate solutions with unconstrained scale parameters
- Recommended next: either (a) use continuous normalizing flows (CNF/FFJORD) which don't have closed-form log_det and thus can't exploit it, or (b) implement diffusion model instead (no log_det at all), or (c) use fixed-scale coupling blocks (only shifts, but per-molecule normalized) with much more training

### Commits
- `caa321f` — [hyp_002] results: HEURISTICS val run complete — ActNorm collapse diagnosis and final experiment results

---

## 2026-03-01 — hyp_003: TarFlow Stabilization
**Branch:** `exp/hyp_003`

### Decisions & Reasoning (INTENTION — write before execute)
- Implementing three targeted interventions from literature to fix log_det collapse:
  1. Asymmetric soft scale clamping (Andrade et al. 2024): _asymmetric_clamp with alpha_pos=0.1, alpha_neg=2.0
  2. Log-det regularization: log_det_reg_weight * (log_det_per_dof)^2 penalty term
  3. Soft equivariance: random SO(3) rotation + CoM noise augmentation + global std normalization
  Source: SBG training recipe (Tan et al. 2025 ICML)
- Environment note: escher.lbl.gov, Slurm unavailable, using direct CUDA. Physical GPU 1 = CUDA_VISIBLE_DEVICES=1 -> logical cuda:0.
- Global std computed from all 8 molecules train split = 1.2905 Angstroms. Plausible (expected 1.3-1.4 Å).

### New Files Created
- `src/model.py` — updated: _asymmetric_clamp function, TarFlowBlock updated (alpha_pos/alpha_neg replacing log_scale_max), TarFlow.nll_loss updated (log_det_reg_weight), EMAModel class added
- `src/data.py` — updated: augment_positions(), compute_global_std(), MD17Dataset/MultiMoleculeDataset with augment+global_std
- `src/train.py` — updated: DEFAULT_CONFIG for hyp_003, global_std computation, augmentation passthrough, log_det_reg_weight in training loop, OneCycleLR option, EMA support, evaluate_molecule denormalization
- `experiments/hypothesis/hyp_003_tarflow_stabilization/notes.md` — experiment notes
- `experiments/hypothesis/hyp_003_tarflow_stabilization/config/diag_config.json` — diagnostic config

### DIAGNOSTIC RUN RESULTS (500 steps)
- log_det/dof stable at 0.78 — within [-2, 2] range. Asymmetric clamp IS working.
- Loss decreases: 1.39 -> 0.14. Real learning is happening.
- BUT valid_fraction = 0.000 on all 8 molecules. min_dist_mean = 0.2-0.35 Å (should be >0.8 Å)
- ROOT CAUSE (new collapse mode): model exploits alpha_pos=0.1 by setting log_scale ≈ +0.1 uniformly across all 8 blocks. This gives log_det/dof = 8 * 0.0977 = 0.78 (the saturation value). In INVERSE (sampling) direction, noise is contracted by exp(-0.78 * per_dof * 3) per atom. Sample std = 0.54 Å normalized = 0.70 Å real (vs 0.92 Å reference). Atoms are too close.
- The model IS learning structure (NLL improves significantly) but samples are compressed by forward expansion.
- CONCLUSION: log_det_reg_weight=0.1 is insufficient. Need stronger regularization OR reduce alpha_pos further.

### Decisions for SANITY angle
- INTENTION: Increase log_det_reg_weight to force model toward log_det_per_dof ≈ 0
  Strong regularization candidates: reg_weight = 1.0, 5.0, 10.0
- Alternative: reduce alpha_pos to 0.02-0.05 (tighter expansion bound)
- Combined approach: alpha_pos=0.05 + reg_weight=1.0
- Key insight: log_det_per_dof needs to stay NEAR 0, not just bounded. The clamp alone is not enough.
  The regularization penalty (log_det_per_dof)^2 needs to be strong enough to overcome the NLL gradient pushing log_det up.

### SANITY Val Run Results
- alpha_pos=0.05, log_det_reg_weight=1.0, lr=3e-4, 2000 steps
- W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/l9r3k0sf
- malonaldehyde: 11.6% valid, ethanol: 7.8% valid, others low
- log_det/dof = 0.396 (down from 0.78 with reg_weight=0.1)
- PROMISING (malonaldehyde > 0.1). Proceeding to sweep.

### SANITY Sweep
- Sweep URL: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/rccehd8m
- Sweep ID: rccehd8m
- Parameters: alpha_pos [0.02, 0.05, 0.1], log_det_reg_weight [0.5, 1.0, 2.0, 5.0, 10.0], lr [1e-4, 3e-4, 1e-3], run_cap=30
- Agents running on GPUs 1-4 in parallel

### Commits
- `8faf809` — [hyp_003] code: implement asymmetric clamping, log-det regularization, soft equivariance
- `d42ffb2` — [hyp_003] results: diagnostic run — new collapse mode identified, plan written

---

## 2026-03-01 — Re-spawn: SANITY full run + HEURISTICS (if needed) + final report
**Branch:** `exp/hyp_003`

### Context at Re-spawn
Previous PhD agent context exhausted after completing:
1. Implementation (8faf809), diagnostic (d42ffb2), plan (d42ffb2)
2. SANITY val run (2000 steps): malonaldehyde 11.6%, ethanol 7.8% — PROMISING
3. SANITY sweep (24/30 runs, W&B rccehd8m): best config = alpha_pos=0.02, log_det_reg_weight=5, lr=1e-4 → 17.5% mean valid fraction at 3000 steps
.state.json was never updated — fixed at re-spawn start to mark steps 1-5 as completed.

### INTENTION: SANITY Full Run (before executing)
- Best sweep config: alpha_pos=0.02, log_det_reg_weight=5, lr=1e-4, batch_size=128
- n_steps=10000 (3x more than sweep), eval_n_samples=500
- d_model=128, n_blocks=8, n_heads=4 (unchanged)
- augment_train=True, normalize_to_unit_var=True (global_std=1.290462613105774)
- lr_schedule=cosine, warmup_steps=300, grad_clip_norm=1.0
- GPU: CUDA_VISIBLE_DEVICES=1 (physical GPU 1 → logical cuda:0)
- Save to: experiments/hypothesis/hyp_003_tarflow_stabilization/angles/sanity/full/
- W&B run name: hyp_003_sanity_full_ap002_rw5_lr1e-4
- Expecting: best sweep run was 17.5% at 3000 steps. 10k steps may improve but the alpha_pos=0.02 saturation (log_det/dof≈0.1) persists. Success criterion (4/8 molecules ≥ 50%) is unlikely to be met.
- Plausibility checks: log_det_per_dof near 0.1 (alpha_pos saturation), small molecules > large molecules in valid fraction.

### INTENTION: HEURISTICS angle (if SANITY fails)
- SBG recipe from Tan et al. 2025 (ICML): AdamW betas=(0.9, 0.95), OneCycleLR pct_start=0.05, EMA decay=0.999, batch_size=512
- Keep best alpha_pos=0.02, log_det_reg_weight=5 from SANITY
- Val run: 2000 steps, promising if valid_fraction > 0.1 on any molecule
- Full run: 20000+ steps if promising

### Decisions & Reasoning
- Starting from freshly initialized model (no checkpoint reuse between angles — spec compliance)
- Will skip SCALE if the collapse mechanism is still alpha_pos saturation (SCALE doesn't fix saturation)

### New Files to be Created
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/sanity/full/config.json` — full run config
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/sanity/full/best.pt` — best checkpoint
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/sanity/full/raw/` — raw arrays
- `experiments/hypothesis/hyp_003_tarflow_stabilization/results/` — canonical plots after winning angle

### SANITY Full Run Results (10000 steps)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/o5naez7a
- Config: alpha_pos=0.02, log_det_reg_weight=5, lr=1e-4, batch_size=128, cosine LR
- CONFIRMED: Loss plateaued at 0.8689 from step 300 onward. log_det/dof locked at 0.100 (alpha_pos saturation).
- Best checkpoint: step 500 (val_loss=0.8145) — very early
- Evaluation results:
  - ethanol: 33.0%, malonaldehyde: 32.6% (best)
  - benzene: 16.2%, uracil: 13.8%
  - toluene: 4.4%, salicylic_acid: 3.0%, naphthalene: 1.8%, aspirin: 0.2%
  - Mean: 13.1%, 0/8 molecules ≥ 50%
- SANITY FAILS primary criterion. Proceeding to HEURISTICS.
- Plausibility check: ✓ smaller molecules > larger ones. ✓ log_det=0.100 = exact alpha_pos saturation. ✓ best checkpoint early = loss plateaus immediately.

### HEURISTICS Val Run Results (2000 steps)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/o6pnle0k
- Config: AdamW betas=(0.9,0.95), OneCycleLR pct_start=0.05, EMA decay=0.999, batch_size=512, lr=3e-4
- alpha_pos=0.02, log_det_reg_weight=5 (same as SANITY best)
- Results:
  - ethanol: 41.3%, malonaldehyde: 40.7%
  - uracil: 20.7%, benzene: 17.7%
  - toluene: 8.0%, salicylic_acid: 4.0%, naphthalene: 1.3%, aspirin: 0.3%
  - Mean: 16.8%, 0/8 molecules ≥ 50%
- PROMISING (>0.1 on multiple molecules). Best checkpoint at step 2000 (end of run).
- log_det/dof still at 0.100 — saturation persists. SBG recipe improves results but doesn't fix root cause.
- Proceeding to HEURISTICS sweep (ema_decay × lr × batch_size, 12 runs, W&B sweep cmgrp6jo)

### HEURISTICS Sweep
- Sweep ID: cmgrp6jo
- URL: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/cmgrp6jo
- Parameters: ema_decay [0.995, 0.999], lr [1e-4, 3e-4, 1e-3], batch_size [256, 512], run_cap=12
- Agents running on GPUs 1-4 in parallel (4 agents × 3 runs each)
- Best config: batch_size=512, ema_decay=0.999, lr=1e-3 → mean VF 18.3% (3 independent runs confirm)

### HEURISTICS Full Run Results (20000 steps)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/4079op64
- Config: batch_size=512, ema_decay=0.999, lr=1e-3, OneCycleLR, alpha_pos=0.02, reg=5
- Results: mean VF 14.3%, 0/8 molecules ≥ 50%
  - malonaldehyde 38.0%, ethanol 33.4%, uracil 13.6%, benzene 15.2%
  - toluene 7.4%, salicylic_acid 3.8%, naphthalene 2.0%, aspirin 0.8%
- Best checkpoint at step 2000 — same saturation pattern
- HEURISTICS FAILS primary criterion

### SCALE Decision
- SKIPPED with justification: model saturates at step 150 (loss flat, log_det locked at 0.100)
- Not capacity-limited — the alpha_pos saturation is a mathematical equilibrium
- Increasing model size would hit same equilibrium faster, not escape it
- Per plan spec: "If the collapse mechanism is still alpha_pos saturation, SCALE will NOT help"

### Visualizations Generated
- valid_fraction_comparison.png: SANITY vs HEURISTICS full runs per molecule
- min_pairwise_distance.png: reference distribution vs generated stats for top molecules
- sweep_comparison.png: HEURISTICS sweep 12 runs, LR × batch_size analysis
- angle_summary.png: all angles table with status
- best_results_summary.png: HEURISTICS full per-molecule breakdown + size correlation scatter
- All saved to experiments/hypothesis/hyp_003_tarflow_stabilization/results/
- Source: src/visualize_hyp003.py (new file)

### New Files Created
- `src/visualize_hyp003.py` — hyp_003-specific visualization script
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/sanity/full/config.json` — SANITY full config
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/heuristics/val/config.json` — HEURISTICS val config
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/heuristics/sweep/sweep_config.json` — sweep config
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/heuristics/sweep/run_sweep.py` — sweep runner
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/heuristics/sweep/summary.json` — sweep results
- `experiments/hypothesis/hyp_003_tarflow_stabilization/angles/heuristics/full/config.json` — HEURISTICS full config
- `experiments/hypothesis/hyp_003_tarflow_stabilization/reports/final_report.md` — final experiment report
- `experiments/hypothesis/hyp_003_tarflow_stabilization/results/*.png` — canonical plots

### Commits (to be created)
- `[hyp_003] results: SANITY full run — mean 13.1%, saturation confirmed`
- `[hyp_003] results: HEURISTICS val + sweep + full — mean 18.3% best, FAIL`
- `[hyp_003] results: canonical plots and notes.md update`
- `[hyp_003] docs: final report, experiment_log, process_log`
- `[hyp_003] integrate: clean experiment directory`

---

## 2026-03-02 — und_001 Phase 2: Apple TarFlow Baseline Verification
**Branch:** `exp/und_001`

### Decisions & Reasoning (INTENTION — written before execution)
- Implementing Apple TarFlow clean re-implementation in `src/tarflow_apple.py`
- Must match Apple reference exactly: per-dim scale, output shift autoregression, no clamping
- Training unified script `src/train_ladder.py` for levels 0-3
- Level 0 (2D Gaussian): seq_length=2, in_channels=1 (NOT seq_length=1 which is degenerate)
  - seq_length=1 → output shift always returns zeros → permanent identity → no learning
  - seq_length=2 splits (x,y) into two scalar tokens with autoregressive conditioning
- Level 1 (MNIST): channels=256, 4 blocks × 4 layers, patch_size=4 → 49 tokens × 16 dims
- Level 2 (CIFAR-10): channels=768, 8 blocks × 4 layers, 228M params
  - LR reduced to 1e-4 (3e-4 caused NaN at step 8 with large model)
  - num_workers=0 to avoid DataLoader deadlock in background process
- All three levels run on GPUs 5, 6, 7

### Results (Phase 2)
- Level 0: DONE. Best NLL=0.91, mode coverage=88.6%, 5000 steps, 2.9 min
- Level 1: DONE. Best NLL=-2.245, bits/dim=-3.20, converged at 14400/20000 steps
- Level 2: RUNNING. Loss 0.13→-2.01 in 1340 steps (in progress, GPU 7 at 100%)

### New Files Created
- `src/tarflow_apple.py` — Apple TarFlow clean re-implementation (TarFlowApple + TarFlow1D)
- `src/train_ladder.py` — unified training script for levels 0-3
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase2/level0_2d_gaussian/` — Level 0 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase2/level1_mnist/` — Level 1 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase2/level2_cifar10/` — Level 2 outputs (in progress)
- `experiments/understanding/und_001_tarflow_diagnostic/reports/ladder_report.md` — Phase 2 report

### Commits
- `94d7677` — [und_001] code: implement Apple TarFlow clean re-implementation (tarflow_apple.py)
- `7522813` — [und_001] code: implement unified train_ladder.py for benchmark levels 0-2
- `85687e2` — [und_001] code: fix Level 0 to use seq_length=2 in_channels=1
- `aaf9d19` — [und_001] code: lower CIFAR-10 lr to 1e-4 to prevent early training instability
- `4360f13` — [und_001] code: set cifar10 num_workers=0 to prevent dataloader deadlock
- `15c29f0` — [und_001] results: Level 0 (88.6% mode coverage) and Level 1 MNIST (-3.20 bpd) complete

### Notes
- GPU 5: Level 0 (done), GPU 6: Level 1 (done), GPU 7: Level 2 (in progress)
- W&B runs: Level 0 = nkuogsf4, Level 1 = n8fokhe6, Level 2 = rlvxam2e
- Level 2 at 50k steps estimated ~8.3 hours; run in background
- Plausibility checks PASS: Level 0 loss ~6.38 at init (correct for ring data with radius=5),
  Level 1 loss negative (expected with [-1,1] normalization + noise augmentation),
  Level 2 rapid improvement in first 1000 steps confirms no collapse or instability

---

## 2026-03-02 — und_001 Phase 3: Adaptation Ladder (Steps A-F)
**Branch:** `exp/und_001`

### INTENTION (write before execute)
- Implementing the adaptation ladder: 6 incremental steps from Apple TarFlow (raw coords) to our full molecular model
- PURPOSE: isolate which architectural change causes performance degradation (log-det exploitation)
- HYPOTHESIS: Step E (shared scale) is the break point — ONE scalar log_scale per atom × 3 coords gives 3× leverage on log_det exploitation vs per-dimension scale
- Each step trains from SCRATCH on ethanol (9 real atoms, 444k train samples) for 5000 steps
- After all steps: write adaptation_report.md identifying the degradation point

**Step plan:**
- Step A: TarFlow1D on raw 9 ethanol atoms — pure Apple architecture, no modifications
- Step B: Add atom type conditioning (nn.Embedding 4→16, concat to input)
- Step C: Add padding (21 atoms) + causal+padding attention mask
- Step D: Add Gaussian noise augmentation (sigma=0.05 to real atoms only)
- Step E: Switch to shared scale — ONE scalar per atom applied to all 3 coords (KEY TEST)
- Step F: Add stabilization (asymmetric clamp alpha_pos=0.1 + log_det_reg_weight=0.01)

**Architecture for Step A:**
- TarFlow1D: seq_length=9, in_channels=3 (raw xyz per atom)
- channels=256, num_blocks=4, layers_per_block=2, head_dim=64
- Apple loss: 0.5 * z.pow(2).mean() - logdets.mean()
- NO clamping, NO log-det reg, NO noise, NO padding

**Key insight being tested:**
- Apple uses per-DIMENSION scale: xa shape (B, T, 3), log_det = -xa.mean([1,2])
  → 3 independent scale params per atom, each contributing 1/3 to log_det per dof
- Our model uses shared scale: 1 scalar per atom applied to all 3 coords, log_det = -3*s.mean([1,2])
  → Same scalar × 3 means the optimizer gets 3× more benefit per unit of log_scale exploitation
  → This concentration of influence per parameter is what we hypothesize causes the saturation equilibrium

**MetaBlock modification for shared scale (Step E):**
- MetaBlock proj_out produces in_channels*2 = 6 dims for NVP (xa=3 dims, xb=3 dims)
- For shared scale: proj_out produces in_channels+1 = 4 dims (xa=1 shared scalar, xb=3 shift)
- Forward: z = (x - xb) * exp(-xa.expand_as(x)); log_det = -xa.mean([1,2]) * 3
- This requires modifying MetaBlock or creating a variant

**Implementation plan:**
- New file: src/train_phase3.py — unified runner for all 6 steps
- New classes: MetaBlockSharedScale, TarFlow1DSharedScale for Step E
- TarFlow1DMol for Steps B-F: extends TarFlow1D with atom type conditioning
- Each step's model defined in train_phase3.py (no separate files per step)

### New Files to be Created
- `src/train_phase3.py` — Phase 3 adaptation ladder training script
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase3/step_a_raw_coords/`
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase3/step_b_atom_type/`
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase3/step_c_padding_mask/`
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase3/step_d_noise_aug/`
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase3/step_e_shared_scale/`
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase3/step_f_stabilization/`
- `experiments/understanding/und_001_tarflow_diagnostic/reports/adaptation_report.md`

### Commits (planned)
- `[und_001] code: implement Phase 3 adaptation ladder (train_phase3.py)`
- `[und_001] results: Phase 3 step_a complete`
- `[und_001] results: Phase 3 step_b complete`
- `[und_001] results: Phase 3 step_c complete`
- `[und_001] results: Phase 3 step_d complete`
- `[und_001] results: Phase 3 step_e complete (KEY TEST)`
- `[und_001] results: Phase 3 step_f complete`
- `[und_001] docs: adaptation_report.md`

---

## 2026-03-02 (session continuation) — Phase 3 pre-flight + CIFAR-10 monitoring
**Branch:** `exp/und_001`

### INTENTION (written before action)
Context was restored after previous session ran out. Level 2 CIFAR-10 is still running (PID 1282962, GPU 7, step 3000+). In this session: verify Phase 3 script (train_phase3.py) is bug-free before CIFAR-10 finishes, so Phase 3 can begin immediately after Phase 2.

### Decisions & Reasoning
- Verified CIFAR-10 run healthy: step 3000 at 21:41, loss=-2.02, GPU at 87% utilization
- Committed Level 2 config.json and interim samples.png (step 1340 → already committed earlier; same file)
- Updated .state.json with step 2000 progress marker
- Ran end-to-end sanity tests for all 6 Phase 3 steps (A-F) on CPU with 5 training steps each
- **Found and fixed bug in train_phase3.py**: attention mask shape `(B, T, T)` must be `(B, 1, T, T)` for `F.scaled_dot_product_attention` to broadcast over num_heads. Bug affected Steps C, D, E, F (all that use padding mask). Fixed by adding `.unsqueeze(1)` in both `MetaBlockWithCond.forward` and `MetaBlockSharedScale.forward`.
- All 6 steps pass CPU sanity tests after fix.

### Verification
- Step A: best_loss=0.043 (5 steps, CPU) ✓
- Step B: best_loss=0.118 ✓
- Step C: best_loss=0.254 ✓  (previously failed with RuntimeError: size mismatch dim 1)
- Step D: best_loss=0.222 ✓
- Step E: best_loss=0.281 ✓
- Step F: best_loss=0.292 ✓

### New Files Created
- `finalize_level2.sh` — shell script to commit Level 2 final results after training completes

### Commits
- `00d27f5` — [und_001] config: Level 2 CIFAR-10 config + .state.json progress update (step 2000, loss=-2.00)
- `34fd7dd` — [und_001] code: fix attention mask broadcasting bug in train_phase3.py (B,T,T) → (B,1,T,T)

### Notes
- Level 2 CIFAR-10 expected to complete in ~7 more hours from 21:40 PST (March 2)
- Phase 3 steps A-F can begin immediately after Level 2 completes
- Phase 3 will run steps in parallel: each step 5000 steps on a single GPU (cuda:5 or :6)
- The KEY TEST is Step E (shared scale) — expect significant degradation vs. Step A (per-dim scale)

## 2026-03-02 (context restored) — Phase 3 Step C NaN fix + GPU runs
**Branch:** `exp/und_001`

### INTENTION (written before action)
Previous context ran out mid-debugging. Step C was producing NaN from step 9 onwards. Steps A and B completed successfully. This session: diagnose root cause of Step C NaN, fix it, run all remaining steps (C-F).

### Decisions & Reasoning
- **Root cause of Step C NaN**: Found via diagnostic: at step 9, z_max = 2.6e23 (overflow). The model weights were NOT yet NaN when loss was computed — z itself was overflowing.
  - First fix attempt (zero xa/xb for padding): partially correct but missed a critical detail.
  - **Actual root cause**: PermutationFlip (used in even-numbered blocks) reverses token order. The padding mask was applied in ORIGINAL token order (atoms 0-8 real, 9-20 padding), but after PermutationFlip the order is reversed (atoms 20-12 real in new positions 0-8, atoms 11-0 padding in positions 9-20). Zeroing xa/xb using the original mask was zeroing REAL atoms in PermutationFlip blocks and leaving PADDING positions unconstrained — exactly backwards!
  - **Fix**: Permute padding_mask the same way as x before using it to zero xa/xb and build the attention mask. `mask_perm = permutation(padding_mask.unsqueeze(-1)).squeeze(-1)`.
  - Verified: 20-step diagnostic shows stable loss descent, z_max < 5.2 throughout.
- Also changed logdet normalization in padded blocks from T*D to n_real*D to match get_loss z² normalization (per real dof). This ensures NLL is balanced and not artificially deflated by the padding tokens.
- Also changed the assert to a soft warning: skip NaN steps rather than crash, so skipping batch anomalies doesn't kill the run.

### New Files Created
- None (only modified `src/train_phase3.py`)

### Commits
- `901d6c5` — [und_001] code: fix permutation-aware padding mask in MetaBlockWithCond and MetaBlockSharedScale

---

## 2026-03-02 (session 3) — Phase 3 Steps C-F completion
**Branch:** `exp/und_001`

### INTENTION (written before action)
Context restored from summary. Continuing from mid-debug of NaN in Steps C-F. n_real normalization fix was in place (commit 901d6c5). Previous session analyzed equilibrium math but didn't confirm if the fix worked. This session: verify fix works, complete all Phase 3 steps, collect results.

### Decisions & Reasoning
- **Discovered Step F already completed** (from a prior background run at 21:49 PST): loss=-3.047, VF=0.0. Step F uses clamping+regularization which prevents NaN but results in saturation equilibrium. The asymmetric clamp (alpha_neg=2.0) limits scale to exp(2.0)≈7.4× max, which prevents log-det explosion but the model doesn't learn the distribution — stuck in near-identity transform.
- **Step C was running successfully** (PID 1304900) at step 1700, loss=-2.17 when session began. The 901d6c5 fix (permutation-aware padding mask + n_real normalization) worked.
- **Accident**: killed PID 1306820 thinking it was a duplicate Step C, but it was actually a DataLoader worker for the ORIGINAL Step C run. This caused Step C to crash at step 4200 (checkpoint saved at step 4182, best_loss=-2.747). Cleared the directory and relaunched Step C fresh.
- **Step D and E ran successfully**: both completed without NaN using 901d6c5 code.
- **Decided to relaunch Step C from scratch** (not from checkpoint) since the evaluation and plots were not generated in the crashed run.
- **Did not change lr for Steps C-F**: the n_real normalization fix alone was sufficient to prevent NaN at lr=5e-4. Earlier Adam overshoot hypothesis was incorrect — the permutation-aware mask was the real fix.

### Key Results (Phase 3 Complete)

| Step | Description | best_loss | VF | logdet/dof |
|------|-------------|-----------|-----|-----------|
| A | Raw coords, no padding | -2.827 | 89.1% | 0.122 |
| B | + Atom type conditioning | -2.795 | 92.9% | 0.121 |
| C | + Padding (T=21, n_real=9) | -2.825 | 2.7% | 0.122 |
| D | + Noise augmentation | -1.902 | 14.3% | 0.088 |
| E | Shared scale (KEY TEST) | -1.892 | 40.2% | 0.088 |
| F | + Stabilization (clamp+reg) | -3.047 | 0.0% | 0.113 |

**Interpretation**:
1. **Padding is the primary failure point** (Step C: 89% → 2.7% VF). Adding 12 padding atoms to 9 real atoms collapses valid fraction dramatically, even though the architecture handles padding correctly via masking.
2. **Noise augmentation partially recovers** (Step D: 2.7% → 14.3%). The noise sigma=0.05 smooths the distribution enough to help with pairwise distance validity.
3. **Shared scale IMPROVES over per-dim with noise** (Step E vs D: 14.3% → 40.2%). This is the opposite of the expected result. The shared scale does NOT cause saturation in our diagnostic setting — the normalization fix resolved that issue. The improvement suggests shared scale may have better inductive bias for molecular coords (same scaling for all 3 spatial dims, which respects isotropy).
4. **Clamping destroys performance** (Step F: 40.2% → 0.0%). The asymmetric clamping prevents the model from learning the necessary scale transformations. The clamping is too tight (alpha_pos=0.1 → max scale contraction = exp(-0.1) ≈ 0.905).
5. **Loss magnitude vs VF are decoupled**: Steps C and A have similar best_loss (-2.83 vs -2.83) but radically different VF (2.7% vs 89.1%). The NLL doesn't capture pairwise distance validity. Loss alone is insufficient to diagnose molecular structure quality.

### New Files Created
- `reports/phase3_report.md` — detailed Phase 3 step-by-step analysis with figures and interpretation
- `reports/final_report.md` — final experiment report for Postdoc synthesis

### Commits
- `3fbf7f1` — [und_001] results: Phase 3 Steps C-E complete; docs updated
- `05da6e1` — [und_001] results: Step F complete (VF=10.4%); corrected docs
