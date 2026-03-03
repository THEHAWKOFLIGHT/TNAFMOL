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

## 2026-03-02 — hyp_004: TarFlow Architectural Ablation + Optimization
**Branch:** `exp/hyp_004`

### Context at Resume
Previous PhD agent context exhausted after:
1. Implementation: BidirectionalTypeEncoder, permutation augmentation, positional encodings (commit 44d4ec9)
2. Reports: diagnostic_report.md and plan_report.md (commit 8177c42)
3. Configs: all ablation and sweep configs (commit 9288f75)
4. Executed: all 6 ablation configs (3000 steps each) + 3 sweep runs (bs=128) on escher
5. NOT done: .state.json updates, analysis, remaining sweep runs, full run, HEURISTICS, visualizations, reports

### Ablation Analysis Results (write-before-execute)
- Best config: **D_pos** (use_pos_enc=True only): mean VF = 17.65%
- Rankings: D_pos 17.65% > F_bidir_pos 16.40% > A_baseline 12.68% > C_perm 12.60% > B_bidir 11.80% > E_bidir_perm 10.92%
- Loss curve: ALL configs saturate at loss ~0.869 by step ~150 (same alpha_pos equilibrium as hyp_003)
- Best checkpoints all at step 500-1000 (early training) — val loss increases after that (overfitting)
- pos_enc adds ~+5ppt; bidir_types slightly hurts when combined with pos_enc; perm_aug slightly hurts overall
- Sweep (bs=128): lr=5e-5 marginally best (17.73%). LR spread <0.5ppt. Remaining bs=256 runs unnecessary.
- Promising criterion NOT met (0.20 mean VF). But D_pos is clearly the best config.

### SCALE Angle Assessment (write-before-execute)
- SKIPPED with justification: training saturates at step ~150 across ALL 6 ablation configs
- Same alpha_pos=0.02 saturation equilibrium as hyp_003 (loss=0.869, log_det/dof=0.100 locked)
- Not capacity-limited — this is a mathematical equilibrium, not underfitting
- Plan condition: "Skip SCALE if both SANITY and HEURISTICS show saturation by step 150" — confirmed

### INTENTION: SANITY Full Run (10000 steps)
- Config: D_pos with best sweep LR
  - use_pos_enc=True, use_bidir_types=False, use_perm_aug=False
  - lr=5e-5 (best from sweep), batch_size=128
  - alpha_pos=0.02, alpha_neg=2.0, log_det_reg_weight=5.0
  - cosine LR, warmup_steps=500, n_steps=10000, eval_n_samples=500
  - augment_train=True, normalize_to_unit_var=True
- Output: experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/
- W&B run name: hyp_004_sanity_full_pos_lr5e-5
- GPU: cuda:0 (production)
- Expecting: ~17-18% mean VF (sweep best was 17.73% at 3000 steps; 10k steps may be slightly better but saturation is early)
- Plausibility check: log_det/dof near 0.100, smaller molecules > larger, best checkpoint early

### INTENTION: HEURISTICS Val Run (3000 steps)
- SBG recipe (Tan et al. 2025): AdamW betas=(0.9, 0.95), OneCycleLR, EMA decay=0.999, batch_size=512
- Apply to D_pos config (best from ablation)
- Promising if: mean VF > SANITY full result
- In hyp_003: SBG on plain config gave 16.8% vs 13.1% baseline (+3.7ppt). Similar gain expected.
- Fresh initialization — no checkpoint reuse

### Decisions & Reasoning
- D_pos is the clear winner despite not meeting promising criterion — it improves over baseline by 5ppt
- perm_aug hurts: likely because positional ordering is informative for atom types in MD17 molecules
- bidir_types hurts when combined with pos_enc: may be redundant information or conflicting inductive biases
- SANITY full run is necessary to establish canonical best result for this angle
- HEURISTICS is mandatory next step per plan (SBG recipe citation: Tan et al. 2025)
- SCALE skip is clean: saturation confirmed by step 150 in all 6 independent runs

### New Files to be Created
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/config.json` — full run config
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/best.pt` — best checkpoint
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/raw/mol_results.pt`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/val/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/val/best.pt`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/val/raw/mol_results.pt`
- (sweep and full dirs to follow)
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/results/` — canonical plots
- `reports/final_report.md`

### Commits (planned)
- `[hyp_004] docs: process_log and state.json update at resume`
- `[hyp_004] results: SANITY full run — pos_enc config 10000 steps`
- `[hyp_004] results: HEURISTICS val run`
- `[hyp_004] results: HEURISTICS sweep`
- `[hyp_004] results: HEURISTICS full run`
- `[hyp_004] results: canonical plots and notes.md`
- `[hyp_004] docs: final report, experiment_log, process_log`

---

## 2026-03-02 — hyp_004 continued: SANITY full + HEURISTICS val
**Branch:** `exp/hyp_004`

### SANITY Full Run Results (10000 steps, D_pos, lr=5e-5)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/k88dxne7
- Config: use_pos_enc=True, lr=5e-5, batch_size=128, cosine LR, no EMA
- Results per molecule:
  - ethanol: 44.2%, malonaldehyde: 39.8% (best)
  - benzene: 22.2%, uracil: 18.4%
  - salicylic_acid: 5.6%, toluene: 7.4%, naphthalene: 1.8%, aspirin: 0.4%
  - Mean: 17.48%, 0/8 molecules ≥ 50%
- Best checkpoint: step 1000 (val_loss=0.8176) — saturation same as ablation
- Confirms: 10000 steps provides NO improvement over 3000 steps (17.65%)
  → alpha_pos saturation is not a training-budget issue. Best checkpoint always early.
- Plausibility check: ✓ smaller molecules > larger ✓ log_det/dof=0.100 locked ✓ early checkpoint
- SANITY FAILS primary criterion (no molecule ≥ 50%). Proceeding to HEURISTICS.

### INTENTION: HEURISTICS Val Run (currently running)
- Config: D_pos + SBG recipe (betas=(0.9,0.95), OneCycleLR, EMA=0.999, bs=512, lr=3e-4)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ht2xyghi
- Promising if: mean VF > 17.48% (SANITY full baseline)
- In hyp_003: SBG improved from 13.1% → 16.8% (+3.7ppt). Expecting similar gain here.
- PID: 1222332, cuda:0, 3000 steps, ~659s total (at step 1500/3000 at t=175s)

### New Files Created (SANITY full)
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/best.pt` (gitignored)
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/full/raw/mol_results.pt` (gitignored)
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/diag/ablation_comparison.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/diag/per_mol_heatmap.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/sweep/sweep_summary.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/val/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/summary.json`
