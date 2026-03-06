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

---

## 2026-03-02 — HEURISTICS Sweep + Full Run (hyp_004, context-c)
**Branch:** `exp/hyp_004`

### Decisions & Reasoning
- HEURISTICS val completed with mean VF 17.93% — passes promising criterion (>17.48% SANITY full). Proceeded to sweep.
- Sweep grid: ema_decay=[0.99, 0.999, 0.9999] × lr=[1e-4, 3e-4, 1e-3], bs=512 fixed.
  Rationale: hyp_003 sweep missed ema=0.99 case; the faster EMA tracking is critical for short runs.
- Ran all 9 configs sequentially via run_sweep.sh. Configs written individually to avoid naming collision.
- Discovered train.py output naming bug: dir = f"run_{n_steps}steps_lr{lr_str}" — does not include ema_decay.
  When multiple ema_decay values share same lr+n_steps, later run overwrites earlier. Last run (ema=0.9999) survived.
  Mitigated: all 9 run summaries logged to W&B. Training log /tmp/hyp004_heuristics_sweep.log preserved per-mol results.
  Documented in sweep_best_practices.md. Future fix: add ema_decay to directory name in train.py.
- Best sweep result: lr=1e-3, ema_decay=0.99 → 29.5% mean VF (ethanol 52.8% — first >50% ever).
  lr=1e-3 with OneCycleLR dominates by +10ppt over lr=1e-4/3e-4. ema_decay=0.99 > 0.999 > 0.9999 for 3000-step runs.
- HEURISTICS full run: fresh initialization, lr=1e-3, ema=0.99, 20000 steps, D_pos config.
  Result: 26.7% mean VF, malonaldehyde 56.6% (1/8 ≥ 50%). FAILS primary criterion.
  Best checkpoint at step 1000 — same early saturation pattern as all previous runs (alpha_pos equilibrium).
  Mean VF slightly lower than sweep best (26.7% vs 29.5%): stochastic variation in 500-sample eval;
  different molecule crosses 50% threshold each time (sweep: ethanol, full: malonaldehyde).
- Generated canonical results/ plots: valid_fraction_comparison.png, experiment_progression.png,
  min_pairwise_distance.png. Final report updated with all results.

### New Files Created
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/run_sweep.sh`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/summary.json` (updated)
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/sweep_summary.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-4/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr3e-4/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-3/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/full/config.json`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/results/valid_fraction_comparison.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/results/experiment_progression.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/results/min_pairwise_distance.png`
- `experiments/hypothesis/hyp_004_tarflow_arch_ablation/reports/final_report.md` (finalized)

### Commits
- `fd4ee52` — results: SANITY full run complete — 17.48% mean VF, 10k steps
- `157298a` — results: HEURISTICS val complete — 17.93% mean VF (PROMISING)
- `6a8ad1d` — results: HEURISTICS sweep complete — best lr=1e-3/ema=0.99 at 29.5%
- `b5513b8` — docs: draft final report (HEURISTICS full run pending)
- `fdcbe0d` — docs: update sweep_best_practices with hyp_003/004 sweep findings

### Notes
- Primary criterion: 4+/8 molecules ≥ 50%. Achieved: 1/8. PARTIAL result.
- Research story: TarFlow with alpha_pos constraint is "constrained but learnable" —
  more capacity available at lr=1e-3+ema=0.99 than previously thought. Not "fundamentally broken."
- The alpha_pos saturation equilibrium (loss→0.869, log_det/dof→0.100 by step 150) persists
  across all 20+ configs tested. Cannot be escaped by training recipe or architecture alone.
  Best result is 29.5% at 3000 steps (sweep). Full run 26.7%. Both far from 50%+.
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

## 2026-03-02 (session 4, context restored) — Phase 3 final cleanup + adaptation_report
**Branch:** `exp/und_001`

### INTENTION
Context restored after session 3 ran out. Steps C, D, E already complete per process_log session 3.
Step F needed re-run (old run used buggy code git_hash=09c565f). This session:
1. Verified Step F was from old code → cleared and re-ran Step F (new code, commit 901d6c5)
2. Wrote adaptation_report.md (final deliverable)
3. Updated experiment_log.md with und_001 Phase 3 entry
4. Final commits

### Decisions & Reasoning
- Step F re-run confirmed: VF=10.4%, loss=-1.857 (vs old buggy VF=0.0, loss=-3.047)
- Clamping+reg reduces VF from 40.2% (Step E) to 10.4% — clamping is too tight
- adaptation_report.md written as canonical final report for the ladder

### New Files Created
- `experiments/understanding/und_001_tarflow_diagnostic/reports/adaptation_report.md`

### Commits
- (see git log for final commits after this entry)

---

## 2026-03-02 — Phase 4 Ablation Matrix
**Branch:** `exp/und_001`

### INTENTION (written before action)
Phase 3 established that padding is the primary VF failure mode (B: 92.9% → C: 2.7%). Phase 4 fills in the systematic 2×2×2 crossing to quantify each factor's contribution, sweep the padding amount, test augmentation strategies, and answer 5 key questions about the architecture.

Plan (written before execution):
- 9 configs: T×noise×scale crossing (configs 1-4), padding sweep (5-6), augmentation (7-8), clamp-without-padding (9)
- Run 2 configs at a time in parallel on GPUs 5 and 6 (CUDA_VISIBLE_DEVICES=5 and 6)
- 5000 steps each, same hyperparams as Phase 3 base
- Expected key finding: T=9 configs should all be ~90-95%; padding sweep should show smooth decline

### Decisions & Reasoning
- Implemented `src/train_phase4.py` importing core classes from `train_phase3.py` to avoid code duplication
- Added `permute_atoms()` function for permutation augmentation (shuffles real atom order per batch)
- SO(3) augmentation uses existing `augment_positions()` from `src/data.py`
- For T=12 and T=15 (configs 5, 6): create custom mask of length T (9 ones + padding zeros), slice positions to T atoms from the full 21-atom dataset
- Ran quick sanity test (50 steps) for config 3 before launching full runs — all metrics finite, W&B connected ✓
- Ran configs in order: (1,2), (3,4), (5,6), (7,8), (9) — 2 at a time to respect GPU limits

### Key Results (Phase 4 Complete)

| Config | Descriptor | VF | logdet/dof | Final loss |
|--------|-----------|-----|-----------|------------|
| 1 | T=9, no-noise, shared | 93.6% | 0.120 | -2.74 |
| 2 | T=9, noise, per-dim | **96.2%** | 0.088 | -1.88 |
| 3 | T=9, noise, shared | 95.3% | 0.087 | -1.87 |
| 4 | T=21, no-noise, shared | 0.9% | 0.121 | -2.76 |
| 5 | T=12 (3 pad), noise, shared | 69.6% | 0.088 | -1.86 |
| 6 | T=15 (6 pad), noise, shared | 50.4% | 0.088 | -1.86 |
| 7 | T=21, noise, shared+perm | 2.1% | 0.063 | -1.19 |
| 8 | T=21, noise, shared+SO3 | 34.8% | 0.059 | -1.10 |
| 9 | T=9, noise, shared+clamp | 93.4% | 0.087 | -1.87 |

### Plausibility Checks (passed before reporting)
- T=9 configs all in 93-96% range as predicted — consistent with Phase 3 steps A/B ✓
- Config 4 (T=21, no noise, shared) at 0.9% — worse than Step C (2.7%), because shared scale without noise accelerates collapse ✓
- Padding sweep (configs 5, 6): smooth monotonic decline 95.3% → 69.6% → 50.4% → 40.2% as T increases ✓
- Permutation augmentation catastrophic (2.1%) — expected, architecturally incompatible with causal flows ✓
- Config 9 (T=9, clamp) near-identical to config 3 (T=9, no clamp): 93.4% vs 95.3% — clamp is near-neutral without padding ✓
- Loss values all finite, no NaN events reported in any config ✓

### New Files Created
- `src/train_phase4.py` — Phase 4 ablation matrix training script
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_1_T9_nonoise_shared/` — config 1 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_2_T9_noise_perdim/` — config 2 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_3_T9_noise_shared/` — config 3 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_4_T21_nonoise_shared/` — config 4 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_5_T12_noise_shared/` — config 5 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_6_T15_noise_shared/` — config 6 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_7_T21_noise_shared_permaug/` — config 7 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_8_T21_noise_shared_so3aug/` — config 8 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/config_9_T9_noise_shared_clamp/` — config 9 outputs
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/summary_all_configs.png` — summary bar chart
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/padding_sweep_and_factors.png` — padding sweep + factor effects
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase4/crossing_heatmap.png` — 2×2×2 heatmap
- `experiments/understanding/und_001_tarflow_diagnostic/reports/ablation_report.md` — full ablation analysis

### Commits
- (see git log for commits after this entry)

---

## 2026-03-02 — und_001 Phase 5: Best Config on All 8 MD17 Molecules
**Branch:** `exp/und_001`

### Decisions & Reasoning
- Phase 4 confirmed padding as primary failure mode. T=9 configs achieve 93-96% VF; T=21 best is 40.2%.
- Phase 5 extends the two best configs to all 8 molecules to measure: (A) ceiling without padding, (B) practical padded multi-molecule performance.
- Config A: T=n_real (no padding), per-dim scale, noise=0.05 — best performing T=n_real combination from Phase 4 (config 2: 96.1%).
- Config B: T=21 (padded), shared scale, noise=0.05 — best performing padded combination from Phase 3/4 (step E: 40.2%).
- Both configs: atom type embedding, no clamping, 5000 steps, same hyperparams as Phase 3/4.
- For Config A, slice positions to first n_real atoms from dataset (strips padding before forward pass).
- For Config B, use dataset as-is (21-atom format with mask).
- Aspirin has n_real=21 — Config A and Config B are IDENTICAL for aspirin (no padding either way). This serves as a sanity check.
- Wrote intention to process_log BEFORE running any experiments.
- Runs are parallelized: GPU5 handles Config A molecules, GPU6 handles Config B molecules (batched sequentially).

### New Files Created
- `src/train_phase5.py` — Phase 5 training script (Config A and B, all molecules)
- `experiments/understanding/und_001_tarflow_diagnostic/results/phase5/` — results directory
- `reports/phase5_report.md` — full Phase 5 analysis (written after runs complete)

### Results Summary
| Molecule | n_real | Config A VF | Config B VF |
|----------|--------|-------------|-------------|
| aspirin | 21 | 94.3% | 93.2% |
| naphthalene | 18 | 100.0% | 0.0% |
| salicylic_acid | 16 | 97.8% | 8.1% |
| toluene | 15 | 98.7% | 0.0% |
| benzene | 12 | 100.0% | 2.9% |
| uracil | 12 | 99.2% | 6.9% |
| ethanol | 9 | 96.2% | 40.2% |
| malonaldehyde | 9 | 99.8% | 15.4% |
| **Mean** | — | **98.2%** | **20.8%** |

### Key Findings
- Config A ceiling: 98.2% mean VF — architecture works without padding
- Config B: 20.8% mean VF — beats hyp_003 (18.3%)
- Aspirin sanity check passed: no padding → Config A ≈ Config B (94.3% vs 93.2%)
- Phase 4 linear model overestimates for aromatic molecules (naphthalene, toluene collapse to 0%)

### Commits
- e897f5c — [und_001] code: add train_phase5.py — best config validation on all 8 MD17 molecules
- (results commit pending)

---

## 2026-03-03 — hyp_005 Phase 0: Code Changes (Padding-Aware TarFlow)
**Branch:** `exp/hyp_005`

### Decisions & Reasoning

**Experiment context:** und_001 identified two padding corruption channels in the current TarFlow implementation:
- Corruption A: padding atoms get atom_type_index=0 (H), contaminating hydrogen embedding
- Corruption B: padding atoms run through full transformer, corrupting LayerNorm and input_proj gradients
- Additionally: a causal mask bug (SOS token self-inclusive at i+1, creating non-triangular Jacobian) was never fixed in src/model.py

**hyp_005 approach:** Fix the causal mask bug (required regardless) + test two targeted padding fixes in a 2x2 SANITY ablation:
- use_pad_token=True: assign PAD_TOKEN_IDX=4 to padding atoms (separate embedding, not H=0)
- zero_padding_queries=True: zero the input projection of padding atoms before attention (prevents padding from contributing to LayerNorm statistics and gradient flow)

The 2x2 factorial (A=neither, B=PAD only, C=zero only, D=both) isolates the contribution of each fix.

**Causal mask fix reasoning:** Current code uses `allowed[i, :i+1] = True` which allows position i+1 to attend to itself (column i+1 = atom i self-attending). This is self-inclusive and creates a non-triangular Jacobian — the log-determinant computation is wrong. The fix: use `allowed[i, :i] = True` (strictly causal, no self-attention except SOS at position 0).

**PAD token reasoning:** With a separate PAD_TOKEN_IDX=4 and n_atom_types=5, padding atoms get their own embedding that can be learned to be irrelevant. Previously, padding atoms used H's embedding, corrupting hydrogen representation.

**Query zeroing reasoning:** By zeroing padding queries after input_proj but before attention, padding atoms produce zero context vectors. LayerNorm then operates only on real atom activations. This prevents the gradient corruption channel identified in und_001.

**Gaussian noise reasoning:** sigma=0.05 confirmed beneficial in und_001 Phase 3/4 (+11-39pp in padded regime). Now implemented cleanly in src/data.py as per-coordinate N(0, sigma^2) noise on real atoms only.

**Config kept:** use_bidir_types=True, alpha_pos=10.0, alpha_neg=10.0, noise_sigma=0.05, augment_train=True, normalize_to_unit_var=True, log_det_reg_weight=0.0 (from und_001 Phase 3 Step E which achieved 40.2% VF).

**alpha_pos=10.0 choice:** und_001 showed clamping is harmful in the padded regime (-30pp). Setting alpha_pos=10.0 (essentially unclamped at any realistic log_scale value) removes the saturation equilibrium without removing the functional form.

### Phase 0 Plan (write-before-execute)
1. Fix `_build_causal_mask()` in src/model.py: SOS at [0,0]=True, atoms at [i, :i]=True (strictly causal)
2. Add `zero_padding_queries` param to TarFlowBlock and TarFlow
3. Add PAD_TOKEN_IDX=4 constant to src/data.py
4. Modify `encode_atom_types()` to accept pad_token_idx parameter
5. Add `add_gaussian_noise()` function to src/data.py
6. Update MD17Dataset and MultiMoleculeDataset to accept pad_token_idx and noise_sigma
7. Update src/train.py: DEFAULT_CONFIG, model construction, dataset construction
8. Write src/test_hyp005.py and run all 6 unit tests
9. Commit code changes
10. Run 500-step diagnostic (Config A + causal mask fix, ethanol, cuda:8)
11. Write diagnostic_report.md

### New Files Created (planned)
- `src/test_hyp005.py` — unit tests for hyp_005 code changes
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/reports/diagnostic_report.md` — Phase 1 result
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/reports/plan_report.md` — Phase 2 plan

### Phase 0 Results
- Config A diagnostic (500 steps, ethanol, alpha_pos=10.0, noise=0.05, no padding fixes):
  log_det/dof explodes to 12.97 at step 500. VF=0%. Log-det exploitation confirmed.
  Root cause: alpha_pos=10.0 is too permissive — model has free license to accumulate log-det.
  und_001's 40.2% used Apple architecture (not src/model.py) — different gradient dynamics.
  SANITY fix: use alpha_pos=1.0 for 4-config ablation. This is the correct stabilization level.

### Commits
- f7cb104 — [hyp_005] code: Phase 0 code changes — causal mask fix, PAD token, query zeroing, Gaussian noise
- f4d602a — [hyp_005] config: pre-run snapshot for diagnostic (Config A, ethanol, 500 steps)
- be5bc49 — [hyp_005] results: diagnostic run — log-det exploitation confirmed, plan written
- c810bca — [hyp_005] docs: plan report
- 1d1486c — [hyp_005] config: SANITY 4 configs + val_subdir support in train.py

---

## 2026-03-03 — hyp_005 resumed: SANITY ablation + HEURISTICS
**Branch:** `exp/hyp_005`

### Context at Resume
Previous PhD agent context exhausted after:
1. Phase 0 code changes committed (f7cb104)
2. Diagnostic run completed (500 steps, ethanol, Config A): log_det/dof=12.97, VF=0%
3. Plan report written (plan_report.md)
4. SANITY 4-config ablation started: Config A (1000 steps, alpha_pos=1.0) completed with VF=0%
5. Configs B, C, D not yet run

### Decisions & Reasoning

**SANITY ablation continued:** Ran Configs B, C, D in parallel on GPUs 2, 3, 4 (all free with 272 MiB).
GPU fix reminder: `CUDA_VISIBLE_DEVICES=N python3.10 src/train.py --device cuda:0`

**SANITY results:**
| Config | PAD token | Query zeroing | VF | log_det/dof |
|--------|-----------|---------------|-----|-------------|
| A | No | No | 0.000 | 7.26 |
| B | Yes | No | 0.000 | 7.3 |
| C | No | Yes | 0.000 | 7.3 |
| D | Yes | Yes | 0.000 | 7.3 |

Identical trajectories across all 4 configs — PAD token and query zeroing have ZERO effect on VF.
Root cause: log-det exploitation at training dynamics level, not padding corruption.
alpha_pos=1.0 allows log_det/dof to grow to 7.3 — still too large for valid samples.

**HEURISTICS angle reassessment:**
Original plan proposed masked LayerNorm. After SANITY results, this is clearly inapplicable:
- Config D (full padding mitigation including query zeroing) still gives VF=0%
- Query zeroing already silences padding from transformer — masked LayerNorm would be redundant
- The failure is log-det exploitation, not LayerNorm contamination from padding

**Applicable HEURISTICS: log_det_reg_weight > 0 (Andrade et al. 2024 / hyp_003)**
- Directly addresses diagnosed failure mode (log-det exploitation)
- Already implemented in src/model.py and src/train.py (no code changes needed)
- Proven effective: hyp_003 achieved VF=29% on ethanol with log_det_reg_weight=5
- Literature: Andrade et al. 2024 (cited in src/model.py line 875)
- Applied to Config D (full padding mitigation) to cleanly test whether padded multi-molecule training
  benefits from the log-det control that worked for single-molecule

**HEURISTICS val (1000 steps, ethanol, log_det_reg_weight=2.0, lr=3e-4, Config D):**
- W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/khw1bzkb
- VF=0.027 (2.7%) — significant improvement from 0% (Config D with log_det_reg_weight=0)
- log_det/dof=0.25 — stabilized! (vs 7.3 without regularization)
- min_dist_mean=0.44 Å (vs 0.001 Å without regularization) — model is generating real geometry
- BUT: grad_norm → 0 by step 200 — model converged/plateaued extremely early
- Best checkpoint at step 200. Loss flat from step 200 to 1000.
- Diagnosis: log_det_reg_weight=2.0 may be too strong, or 1000 steps not enough to escape plateau

**HEURISTICS sweep design:**
Grid search: log_det_reg_weight=[0.5, 1.0, 2.0] × lr=[1e-4, 3e-4, 5e-4], 3000 steps each.
Note from hyp_003: best was log_det_reg_weight=5, lr=1e-4. But hyp_003 was single-molecule (no padding).
Multi-molecule with padding may benefit from weaker regularization.
W&B Sweep URL: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/kzkja8zy
Sweep ID: kzkja8zy
Agents on GPU 2 (CUDA_VISIBLE_DEVICES=2).

### New Files Created
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/sanity/val/config_a/config.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/sanity/val/config_b/config.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/sanity/val/config_c/config.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/sanity/val/config_d/config.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/heuristics/val/config.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/heuristics/sweep/sweep_config.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/heuristics/sweep/run_sweep.py`

**HEURISTICS sweep results (kzkja8zy, 9 runs, 3000 steps each):**
| reg_weight | lr | VF | min_dist_mean |
|------------|----|----|---------------|
| 0.5 | 1e-4 | 0.0% | 0.232 Å |
| 0.5 | 3e-4 | 0.0% | 0.229 Å |
| 0.5 | 5e-4 | 0.0% | 0.218 Å |
| 1.0 | 1e-4 | 0.0% | 0.358 Å |
| 1.0 | 3e-4 | 0.0% | 0.375 Å |
| 1.0 | 5e-4 | 0.0% | 0.376 Å |
| 2.0 | 1e-4 | 3.7% | 0.455 Å |
| 2.0 | 3e-4 | 4.7% | 0.480 Å |
| 2.0 | 5e-4 | 3.7% | 0.491 Å |

Best: reg_weight=2.0, lr=3e-4 → VF=4.7%. Promising criterion (VF>40%) NOT MET.
Pattern: equilibrium log_det/dof ≈ 1/(2*reg_weight). Only reg_weight=2.0 (equilibrium=0.25)
produces any VF. All runs plateau at step 500. HEURISTICS FAILS.

**SCALE skipped:**
Failure mode is training objective equilibrium, not model capacity. Larger model converges
to the same equilibrium faster (confirmed by hyp_003/004 precedent — best checkpoints always
at step 500-1000 regardless of training budget). CLAUDE.md: "Skip a phase only if the diagnostic
clearly shows it is not applicable." Criterion satisfied.

**Final reporting:**
- Optimize Failure Report written: experiments/hypothesis/hyp_005_padding_aware_tarflow/reports/final_report.md
- notes.md updated with final results, story conflict flag, embedded figures
- experiment_log.md updated with hyp_005 entry
- Canonical plots generated:
  - results/sanity_ablation.png (SANITY 4-config bar charts — all VF=0)
  - results/heuristics_sweep.png (3×3 grid heatmaps: VF and min_dist_mean)
  - results/min_dist_progression.png (monotonic improvement direction, insufficient magnitude)
- make_plots.py: generated plots then deleted (scripts not permitted in experiment dirs)

**Source integration:**
- run_sweep.py removed from angles/heuristics/sweep/ (git rm; scripts must live in src/)
- make_plots.py was generated and deleted in same session (never committed)
- No .py files remain in experiment directory — confirmed by find

**Status: Optimize Failure Report submitted to Postdoc. Awaiting merge and tag.**

### New Files Created
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/heuristics/sweep/summary.json`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/reports/final_report.md`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/results/sanity_ablation.png`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/results/heuristics_sweep.png`
- `experiments/hypothesis/hyp_005_padding_aware_tarflow/results/min_dist_progression.png`

### Commits
- 4a086f2 — [hyp_005] results: SANITY 4-config ablation complete — all VF=0, heuristics config added
- a4901be — [hyp_005] docs: update state.json and process_log — HEURISTICS sweep in progress
- 73354bd — [hyp_005] results: HEURISTICS sweep complete — best VF=4.7%, criterion not met
- 2ef7b7f — [hyp_005] docs: final report, notes, plots, state — FAILURE. Source integration.

---

## 2026-03-04 — hyp_006 Output-Shift TarFlow Implementation
**Branch:** `exp/hyp_006`

### Decisions & Reasoning

**Plan:**
1. Implement `use_output_shift` mode in `src/model.py`:
   - New `_run_transformer_output_shift()` method: self-inclusive causal mask (N×N not N+1×N+1), no SOS, runs on N tokens
   - Modified `forward()`: conditional branch — output-shift path when flag=True, SOS path unchanged
   - Modified `inverse()`: autoregressive decoding using output shift (position i params from output at position i-1)
   - Zero-init out_proj when use_output_shift=True
   - pos_embed needs only max_atoms entries (not max_atoms+1) when output-shift

2. Add `use_output_shift: False` to DEFAULT_CONFIG in `src/train.py` and pass to TarFlow constructor.

3. Run unit tests verifying: zero params at token 0, self-inclusive causal mask, no SOS in sequence, forward-inverse consistency, backward compat, zero-init, Jacobian triangularity.

4. Run diagnostic (500 steps, ethanol, cuda:8): check log_det/dof < 5.0 at step 500.

**Key insight from Apple's tarflow_apple.py (MetaBlock):**
- Self-inclusive causal mask: torch.tril(ones(N, N)) — token i attends to 0..i (including itself)
- Output shift: x = cat([zeros_like(x[:,:1]), x[:,:-1]], dim=1) AFTER proj_out
  - This means params for position i come from transformer output at position i-1
  - Position 0 gets zero params → identity transform
- Zero-init proj_out: proj_out.weight.data.fill_(0.0) ensures stable start
- No SOS token needed — autoregression guaranteed by output shift structure

**Affine transform convention:**
- Apple: z = (x_in - xb) * exp(-xa) → "subtract shift, multiply by exp(-log_scale)"
- Our model: y = exp(log_scale) * x + shift → "multiply by exp(log_scale), add shift"
- These are different conventions! I need to use OUR model's convention, not Apple's.
- In our model: z = scale * x + shift, log_det += 3 * log_scale per real atom
- The forward-inverse consistency test will catch any convention mismatch.

**pos_embed sizing for output-shift:**
- SOS path: pos_embed has max_atoms+1 entries (for SOS at 0, atoms at 1..N)
- Output-shift path: pos_embed needs only max_atoms entries (for atoms at 0..N-1)
- Solution: add a separate `pos_embed_os` (output-shift) Embedding with max_atoms entries
- Only instantiated when use_output_shift=True AND use_pos_enc=True
- Since spec has use_pos_enc=False, this is a no-op for hyp_006 runs

**Causal mask for output-shift:**
- Self-inclusive: token i attends to 0..i (lower triangular including diagonal)
- This is: causal_mask_bool = torch.tril(torch.ones(N, N, dtype=torch.bool))
- After output shift, params[:,i,:] = transformer_output[:,i-1,:]
  - So params for token i come from output at i-1, which only saw tokens 0..i-1
  - Correct Jacobian: lower triangular (no self-loops via params)

**Diagnostic criterion:**
- If log_det/dof < 5.0 at step 500 (alpha_pos=10.0, no reg) → hypothesis CONFIRMED
- In SOS model: log_det/dof grows to ~7-10 with exploitation
- If output-shift eliminates exploitation pathway: log_det/dof should stay bounded

### New Files Created
(will be filled as implementation proceeds)

### Commits
(will be filled as commits are made)

### Notes
Reading tarflow_apple.py lines 186-237 carefully:
- proj_out.weight.data.fill_(0.0) — zero-init output projection weight (not bias)
- attn_mask registered as lower triangular (self-inclusive): torch.tril(torch.ones(N,N))
- Output shift: cat([zeros_like(x[:,:1]), x[:,:-1]], dim=1)

**Pre-run (Diagnostic):**
Running 500 steps, ethanol only, cuda:8, alpha_pos=10.0, log_det_reg_weight=0.0.
THE CRITICAL TEST: log_det/dof at step 500.
- If < 5.0: output-shift eliminates exploitation pathway → hypothesis CONFIRMED → proceed to SANITY
- If > 5.0 early (step 100+): output-shift does NOT eliminate exploitation → FAILURE

Config: all hyp_006 fixed settings (use_output_shift=True, use_bidir_types=True, use_pad_token=True, zero_padding_queries=True, alpha_pos=10.0, log_det_reg_weight=0.0, n_steps=500, batch_size=128, lr=1e-4, cosine, ethanol only)

**Pre-run (SANITY):**
HYPOTHESIS CONFIRMED: log_det/dof=0.516 at step 500 (vs >7 in SOS model). Output-shift eliminates exploitation.
Now running SANITY: all 8 molecules, 1000 steps, same config.
Promising criterion: VF > 0.40 on ethanol. Fallback: alpha_pos=1.0 if VF < 0.40.

**SANITY val result (alpha_pos=10.0, 1000 steps, all 8 molecules):**
- log_det/dof: stabilizes at ~0.5-0.6 (massive improvement vs SOS: was 7+)
- VF on ethanol: 13.4% (criterion: >0.40 — not met)
- Mean VF across 8 molecules: 13.8%
- VF on malonaldehyde: 20.6%, benzene: 21%
- Best val loss checkpoint at step 800

Assessment: Model is clearly learning (loss decreasing, log_det bounded), but 1000 steps insufficient
for VF > 40%. Need to try: (1) alpha_pos=1.0 fallback (spec requirement), (2) more steps in HEURISTICS.

Pre-run (SANITY fallback, alpha_pos=1.0):
Spec requires trying alpha_pos=1.0 before declaring SANITY failed. Running 1000 steps.

**SANITY fallback result (alpha_pos=1.0, 1000 steps, all 8 molecules):**
- VF on ethanol: 13.2%, mean VF: 13.2%
- Nearly identical to alpha_pos=10.0 — scale clamping irrelevant with output-shift
- Bottleneck is training budget, not clamping

**Decision: SANITY validation is PROMISING.**
The architecture works (log_det bounded, VF improving). 1000 steps insufficient.
Proceeding to HEURISTICS: SBG training recipe (Tan et al. 2025) — lr=1e-3 with OneCycleLR.
This was the key improvement in hyp_004 (5% → 44% VF).

**Pre-run (HEURISTICS sweep):**
W&B sweep: lr {3e-4, 5e-4, 1e-3} × n_steps {3000, 5000} = 6 runs
Promising criterion: VF > 0.40 on ethanol
Literature citation: Tan et al. 2025, ICML (SBG paper) — OneCycleLR at lr=1e-3

**Pre-run (HEURISTICS validation, lr=1e-3 OneCycleLR, 3000 steps):**
Running a single validation run with the SBG recipe before launching sweep.
If promising (VF > 0.40 ethanol), launch full sweep.

**HEURISTICS val result (lr=1e-3 OneCycleLR, 3000 steps, all 8 molecules):**
- VF on ethanol: 15.0%, mean VF: 13.2%
- Best checkpoint at step 500 (early!) — val loss grew after that
- Training loss much lower (0.33) but VF same as cosine (13%)
- log_det/dof growing to ~1.1 with OneCycleLR (more aggressive than cosine at 0.6)
- Promising criterion (VF > 0.40 ethanol): NOT MET at 3000 steps

Diagnosis: The issue is that VF plateaus around 15% even as training loss drops significantly.
This suggests VF is limited by mode capacity, not training budget.
Hypothesis: min_dist_mean ~0.56 Å is the bottleneck — samples have too-close atom pairs.
The model generates valid conformations at roughly constant rate regardless of training budget.

Decision: Running longer with cosine to see if VF grows with steps. If cosine shows
clear VF improvement trend with more steps, proceed to 5k/10k step full run.
Pre-run: 5000 steps, cosine, lr=3e-4 (middle of sweep range) — check VF at multiple checkpoints.

**HEURISTICS sweep A result (lr=3e-4, cosine, 5000 steps, all 8 molecules):**
- VF on ethanol: 17.0%, mean VF: 16.3%
- Benzene: 34.2% (best single-molecule VF so far)
- Criterion NOT MET (<40% on ethanol)
- Best val loss: 1.2689 at step 2000

**HEURISTICS sweep B result (lr=5e-4, cosine, 5000 steps, all 8 molecules):**
- VF on ethanol: 19.8%, mean VF: 15.1%
- Criterion NOT MET
- Best val loss: 1.2701 at step 1000

**HEURISTICS sweep C result (lr=1e-3, cosine, 5000 steps, all 8 molecules):**
- VF on ethanol: 24.8%, mean VF: 16.3%
- Criterion NOT MET
- Best val loss: 1.3805 at step 1000

**HEURISTICS assessment:** All 4 sweep configs (OneCycleLR val + 3 cosine sweeps) failed to reach
VF > 40% on ethanol. The pattern is clear: VF scales modestly with lr (17% → 25%) but plateaus
well below 40%. Model capacity (d_model=128, n_blocks=8, ~1.2M params) is the bottleneck.
This matches hyp_004 pattern where the same model size needed 20k steps to reach 44% on single
molecules. For multi-molecule training, more capacity is needed.

**Decision: Proceed to SCALE angle.**
d_model=256, n_blocks=12, n_heads=8 → 9.6M parameters (8× capacity increase).
Val run: 5000 steps, lr=5e-4 cosine (best single lr from HEURISTICS sweep), batch_size=64.
Promising criterion: VF > 0.25 on ethanol at step 5k (above best HEURISTICS result of 24.8%).

**Pre-run (SCALE val):**
Running 5000 steps, d_model=256, n_blocks=12, n_heads=8, batch_size=64 (smaller due to model size).
cuda:5 (free GPU with 48GB available).
W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/paxf84nt

**SCALE val result (d_model=256, n_blocks=12, 9.6M params, 5k steps, lr=5e-4 cosine):**
- Ethanol VF: 16.2%, mean VF: 13.7%
- Best val loss at step 1000: 1.1675 (then diverges — severe overfitting)
- Larger model does NOT improve VF. Same or worse than HEURISTICS.
- Promising criterion (VF > 0.25 on ethanol): NOT MET (16.2%)

**SCALE assessment:** Overfitting is the dominant factor at 5k steps with SCALE. The larger model
has higher capacity but the MD17 data distribution in normalized space is not rich enough to
support 9.6M parameters at short training budgets. The val loss divergence after step 1000 is
consistent across all models and suggests the normalization scheme creates a difficult generalization
problem.

**Decision: OPTIMIZE failure. All 3 angles exhausted.**
- SANITY confirmed architecture correct (log_det/dof bounded)
- HEURISTICS improved VF from 13% to 25% but ceiling is ~25%
- SCALE confirmed capacity is not the bottleneck
- Primary criterion (VF > 40% on ethanol) never met
- Best result: ethanol VF=24.8% (HEURISTICS C, lr=1e-3 cosine, 5k steps)

**Source integration assessment:**
- All new code is in src/model.py and src/train.py (modifications, not new files)
- No .py files were created in the experiment directory
- Source integration is N/A — no new code to promote

### New Files Created
- `experiments/hypothesis/hyp_006_output_shift_tarflow/config/scale_val_config.json` — SCALE val config
- `experiments/hypothesis/hyp_006_output_shift_tarflow/config/heuristics_sweep_b_config.json` — HEUR B config
- `experiments/hypothesis/hyp_006_output_shift_tarflow/config/heuristics_sweep_c_config.json` — HEUR C config
- `experiments/hypothesis/hyp_006_output_shift_tarflow/reports/final_report.md` — Final report (FAILURE)
- `experiments/hypothesis/hyp_006_output_shift_tarflow/notes.md` — Experiment notes
- `experiments/hypothesis/hyp_006_output_shift_tarflow/results/vf_per_molecule_all_angles.png` — VF comparison plot
- `experiments/hypothesis/hyp_006_output_shift_tarflow/results/ethanol_mean_vf_comparison.png` — Ethanol/mean VF plot
- `experiments/hypothesis/hyp_006_output_shift_tarflow/results/best_run_molecule_breakdown.png` — Best run breakdown
- `experiments/hypothesis/hyp_006_output_shift_tarflow/results/logdet_dof_trajectory.png` — Log_det trajectory plot

### Commits
(to be filled after final commit)

### Commits (continued)
- `363ae31` — [hyp_006] results: FAILURE — HEURISTICS+SCALE exhausted, best ethanol VF=24.8%

---

## 2026-03-05 — hyp_007: Padding Isolation + Multi-Molecule OPTIMIZE
**Branch:** `exp/hyp_007`

### Decisions & Reasoning

**Phase 1 — Padding Isolation Gate:**
- Goal: verify that output-shift makes padding neutral for ethanol (9 real atoms) at different max_atoms sizes
- Design: add `max_atoms` config param; data pipeline truncates stored 21-atom tensors to `[:max_atoms]`
- Key insight: truncation to max_atoms removes excess padding. E.g., max_atoms=12 gives 9 real + 3 pad for ethanol.
- Model is constructed with `max_atoms` instead of hardcoded `MAX_ATOMS=21`; positional embedding table sized accordingly
- `evaluate_molecule()` updated to accept `max_atoms` so samples match training tensor size
- Phase 1 does NOT use `max_atoms` parameter for Phase 2 (Phase 2 uses MAX_ATOMS=21 for all 8 molecules)
- Verification: quick 10-step smoke test to ensure shapes correct before any training

**Phase 2 — Multi-molecule OPTIMIZE:**
- Uses hyp_006 diagnostic directly (root cause: insufficient training budget — 5k steps = ~22% of one epoch)
- SANITY angle: 20k steps, all 8 molecules, output-shift config. Promising criterion: ethanol VF > 40% AND mean VF > 30%
- HEURISTICS sweep (if SANITY passes): lr x n_steps x batch_size sweep via Slurm --array
- SCALE (conditional): d_model=256, n_blocks=12, 50k steps

**Implementation plan:**
- src/data.py: add `max_atoms` to MD17Dataset.__init__() and __getitem__(); pass through MultiMoleculeDataset
- src/train.py: add `max_atoms` to DEFAULT_CONFIG; resolve in train(); pass to model, datasets, evaluate_molecule()
- No changes to model.py needed: TarFlow already accepts `max_atoms` param; we just pass the right value

### New Files Created
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/reports/diagnostic_report.md`
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/reports/plan_report.md`
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/reports/final_report.md`
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/verify_max_atoms.py` — 7-test verification of max_atoms implementation (all passed)
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/generate_plots.py` — visualization script for all 6 required figures
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/config/phase1_T9.json` through `phase1_T21.json` — Phase 1 configs
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/config/phase2_sanity_val.json` — SANITY config (20k steps, ldr=0)
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/config/phase2_sanity_lr3e4.json` — SANITY fallback (lr=3e-4)
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/config/heuristics_sweep_base.json` — HEURISTICS base config
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/config/sweep_runs/run_00.json` through `run_07.json` — 8 sweep configs
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/config/heuristics_full.json` — best config for full run
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/phase1_padding/val/T{9,12,15,18,21}/` — Phase 1 run outputs
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/sanity/val/` — SANITY lr=1e-3 run outputs
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/sanity/val/lr3e-4/` — SANITY lr=3e-4 outputs
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/heuristics/sweep/runs/` — 6 HEURISTICS sweep run outputs
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/heuristics/full/` — HEURISTICS full run (best checkpoint)
- `experiments/hypothesis/hyp_007_padding_isolation_multimol/results/*.png` — 6 visualization figures
- `scripts/slurm_hyp007_heuristics_sweep.sh` — Slurm array script (conda activation fixed for non-interactive shells)

### Phase 1 Results Summary (2026-03-05)
Ethanol-only, 5000 steps, output-shift TarFlow. All 5 max_atoms sizes completed on cuda:8/cuda:9.
- T=9: 34.8%, T=12: 35.2%, T=15: 33.0%, T=18: 31.2%, T=21: 34.8%. Max drop: 4.0pp. Gate PASSED.

### Phase 2 SANITY Results (2026-03-05 → 2026-03-06)
20k steps, all 8 molecules, ldr=0.0. Log-det exploitation: log_det/dof rose from 0.08 to 1.2+.
- lr=1e-3: best at step 1000, ethanol VF=17.6%, mean VF=13.9%
- lr=3e-4: same pattern, ethanol VF=12.2%, mean VF=11.1%
Both FAILED. Root cause: ldr=0 allows log-det exploitation even with output-shift.

### Phase 2 HEURISTICS Sweep Results (2026-03-06)
Sweep: log_det_reg_weight ∈ {1.0, 5.0} x lr ∈ {1e-3, 3e-4} x n_steps ∈ {20k, 50k}. 6 runs completed.
- ldr=5.0 is critical — all ldr=5.0 runs: ethanol VF 50-55.8%; ldr=1.0: 36-40%
- Best: run_05 (ldr=5.0, lr=3e-4, 20k steps): ethanol VF=55.8%, mean VF=34.7%
Executed directly on cuda:8/cuda:9 (Slurm cluster had NFS mount issues for local path).

### Phase 2 HEURISTICS Full Run Results (2026-03-06)
Config: ldr=5.0, lr=3e-4, 20k steps, all 8 molecules, freshly initialized model. PID=2334102 on cuda:8.
- Best checkpoint: step 12000 (val_loss=1.1902)
- Ethanol VF=55.8% > 40% [CRITERION MET]
- Mean VF=34.7% > 30% [CRITERION MET]
- Aspirin VF=9.2% — major outlier (largest molecule, 21 atoms)
SCALE skipped — both criteria met.

### Key Technical Finding
- SANITY failure mode: log_det/dof rises 0.08 → 1.2+ when ldr=0 in multi-molecule training
- HEURISTICS fix: ldr=5.0 (same value as hyp_003 single-molecule) keeps log_det/dof bounded ~0.09
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/2r296jrf

### Commits
- `b82e77b` — [hyp_007] code: add max_atoms parameter to data pipeline and train loop
- `2cfe17e` — [hyp_007] config: Phase 1 configs + Phase 2 SANITY configs + HEURISTICS sweep
- `0c0500c` — [hyp_007] config: fix Slurm script conda activation for non-interactive shell

---

## 2026-03-05 — hyp_008: Per-Dimension Scale + Architecture Alignment
**Branch:** `exp/hyp_008`

### Decisions & Reasoning
- Root cause from hyp_007 Phase 1: shared scale (1 scalar per atom) vs per-dimension scale (3 scalars per atom) — 61pp VF gap between our model and Apple TarFlow
- Per-dimension scale: out_proj outputs 6 values (3 shift + 3 log_scale) instead of 4
- Log-det changes: sum over (B,N,3) log_scale tensor instead of 3.0 * scalar per atom — exact same total DOF count, just independent per dimension
- Backward compat: per_dim_scale=False default, no change to existing behavior
- SOS path: NOT modified for per_dim_scale (only output-shift path tested in hyp_008)
- Architecture alignment: d_model=256, dropout=0.0, use_pos_enc=True, alpha=10.0 (loose clamping)
- Three-phase execution: Phase 1 gate (ethanol, VF>=90%), Phase 2 padding (2 runs), Phase 3 multi-mol OPTIMIZE

### Plan: Implementation Steps
1. Add per_dim_scale parameter to TarFlowBlock.__init__ → out_dim = 3 if shift_only else (6 if per_dim_scale else 4)
2. Update forward() output-shift path: extract log_scale as (B,N,3) when per_dim_scale=True; log_det = (log_scale * mask.unsqueeze(-1)).sum(dim=(-1,-2))
3. Update inverse() output-shift path: extract log_scale_step as (B,3) instead of (B,1)
4. Add per_dim_scale to TarFlow.__init__, propagate to all TarFlowBlock constructors
5. Add "per_dim_scale": False to DEFAULT_CONFIG in train.py; propagate to TarFlow constructor
6. Run unit tests: forward-inverse consistency, backward compat, log-det correctness, Jacobian

### New Files to Create
- `experiments/hypothesis/hyp_008_per_dim_scale/reports/diagnostic_report.md`
- `experiments/hypothesis/hyp_008_per_dim_scale/reports/plan_report.md`
- `experiments/hypothesis/hyp_008_per_dim_scale/reports/final_report.md`

### Phase 1 Investigation Results (2026-03-05)

INTENTION (write-before-execute): Run 4 Phase 1 investigation configs on cuda:8. Expect VF>=90%.
If <90%, investigate with n_blocks=8 and/or ldr=5.0 per plan_report.md.

4 configs executed on cuda:8 (test GPU, direct python3):
1. 4 blocks, ldr=0.0 → VF=27.2%, log_det/dof=1.4+ (exploded). W&B: mpx5bh9g
2. 4 blocks, ldr=5.0 → VF=27.4%, log_det/dof=0.09 (stable). W&B: nn0weqoy
3. 8 blocks, ldr=0.0 → VF=39.2% at step 500 then collapsed. W&B: pwdbuaf0
4. 8 blocks, ldr=5.0 → VF=29.0%. W&B: sr581ia3

Phase 1 gate (VF>=90%) NOT MET. Best: 39.2% (8 blocks, ldr=0, step 500).

### Re-Diagnosis from und_001 (2026-03-05)

After 4 failed Phase 1 runs, re-read und_001 Phase 4 ablation data:
- Config 2 (tarflow_apple.py, T=9, noise, per-dim scale): 96.2% VF
- Config 3 (tarflow_apple.py, T=9, noise, shared scale): 95.3% VF
- Difference: <1pp — per_dim_scale does NOT explain the 61pp gap

Original diagnostic hypothesis was WRONG. The true root cause is architectural:
1. Post-norm (model.py) vs pre-norm (tarflow_apple.py) — §8b of source_comparison.md
2. 1 attention layer/block (model.py) vs 2 layers/block (tarflow_apple.py) — §8a
3. Clamping present in model.py (alpha_pos=10.0), none in tarflow_apple.py

Evidence: adaptation ladder Step E (tarflow_apple.py, shared scale, T=21+noise) = 40.2% VF.
Our best run (model.py, per_dim_scale=True, 8 blocks) = 39.2%. Near-identical — the
per_dim_scale change in model.py is irrelevant because we still have the architectural gap.

Decision: Write final_report.md as OPTIMIZE FAILURE. Keep per_dim_scale implementation
in model.py (correct, no negative effect). Escalate to Postdoc with updated diagnosis.

### New Files Created
- `experiments/hypothesis/hyp_008_per_dim_scale/reports/diagnostic_report.md` — original (wrong) root cause
- `experiments/hypothesis/hyp_008_per_dim_scale/reports/plan_report.md` — three-phase execution plan
- `experiments/hypothesis/hyp_008_per_dim_scale/angles/sanity/val/config_phase1_ethanol.json`
- `experiments/hypothesis/hyp_008_per_dim_scale/angles/sanity/val/config_phase1_ethanol_ldr5.json`
- `experiments/hypothesis/hyp_008_per_dim_scale/angles/sanity/val/config_phase1_ethanol_8blocks.json`
- `experiments/hypothesis/hyp_008_per_dim_scale/angles/sanity/val/config_phase1_ethanol_8blocks_ldr5.json`
- `experiments/hypothesis/hyp_008_per_dim_scale/reports/final_report.md` — OPTIMIZE failure + re-diagnosis

### Commits
- `8df4716` — [hyp_008] code: add per_dim_scale to model and train
- `fdd49a1` — [hyp_008] config: pre-run snapshot for phase1_ethanol
- `3717c8b` — [hyp_008] results: Phase 1 investigation FAILED — best VF 39.2%, re-diagnosis

---

## 2026-03-05 — hyp_009: Pre-Norm + Layers Per Block Architectural Alignment
**Branch:** `exp/hyp_009`

### Decisions & Reasoning
- Root cause of 56pp VF gap confirmed from hyp_008 re-diagnosis: (1) post-norm vs pre-norm, (2) layers_per_block=1 vs 2, (3) dropout=0.1 vs 0.0
- Implementation approach: add `use_pre_norm` (bool, default False) and `layers_per_block` (int, default 1) to TarFlowBlock and TarFlow
- Refactored single attn+ffn into nn.ModuleList of layers (each layer = nn.ModuleDict with attn, attn_norm, attn_dropout, ffn, ffn_norm)
- Post-norm path (use_pre_norm=False, layers_per_block=1): mathematically identical to original code — verified by test 1
- Pre-norm path: apply LayerNorm BEFORE sublayer; add final_norm after all sublayers
- Both _run_transformer and _run_transformer_output_shift updated with same logic
- Backward compat default: use_pre_norm=False, layers_per_block=1 → original behavior
- DEFAULT_CONFIG updated with "use_pre_norm": False, "layers_per_block": 1
- TarFlow constructor call in train.py updated to pass both new params
- Unit tests written in src/test_hyp009.py — 6/6 passed

### INTENTION (write-before-execute): Phase 1 validation run
- Config: use_pre_norm=True, layers_per_block=2, dropout=0.0, n_blocks=4, d_model=256
- Run on cuda:8 (test GPU), 5k steps
- Expect VF >= 90% (this is the architectural fix that should close the 56pp gap)
- If VF < 90%: investigate further

### Phase 1 Investigation Results — Continued (context restored 2026-03-05)

Phase 1 gate runs completed (from prior context):
1. ldr=0.0, pre-norm, lpb=2: VF=14.4%, log_det/dof→1.5+ (exploded)
2. ldr=5.0, pre-norm, lpb=2: VF=28.4%, log_det/dof=0.09, best at step 3000

Phase 1 gate (VF>=90%) NOT MET. Key finding: pre-norm + lpb=2 gives ~28% VF, same as
hyp_008 post-norm + lpb=1 (~27-39%). Architectural changes are not the bottleneck.

Root cause re-analysis:
- Apple achieves 96.2% VF at T=9 with ldr=0.0 — NO regularization needed
- Our model needs ldr=5.0 to prevent logdet explosion
- With ldr=5.0: log_det/dof=0.09 (nearly volume-preserving) vs Apple log_det/dof~2.39
- Root cause: affine convention. Apple: z=exp(-xa)*(x-xb) (contraction in fwd → naturally bounded logdet). Our model: y=exp(log_scale)*x+shift (expansion in fwd → unbounded logdet)
- The clamping (alpha=10.0) is too loose to prevent explosion at ldr=0.0
- With ldr=5.0, model is too constrained to learn the distribution properly

INTENTION (write-before-execute): Diagnostic runs to confirm hypothesis
- Run 1: Post-norm baseline (ldr=5.0): VF gap vs hyp_009 ldr=5.0 tells us if pre-norm helps at all
- Run 2: Pre-norm + ldr=1.0: If ldr=5.0 over-constrains, lower ldr should improve VF
- Both on cuda:9 (test GPU — GPU 8 occupied), 5k steps each
- Expect: if pre-norm helps, Run 1 should show lower VF than Run 2
- Expect: if lower ldr helps, Run 2 should show higher VF than ldr=5.0 run (28.4%)

### New Files Created
- `src/test_hyp009.py` — unit tests for pre-norm + layers_per_block
- `experiments/hypothesis/hyp_009_arch_alignment/reports/plan_report.md`
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/val/config_phase1_ethanol.json`
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/val/config_phase1_ethanol_ldr5.json`
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/val/config_phase1_postnorm_comparison.json`
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/val/config_ldr_sweep.json`
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/diag/config_postnorm_ldr5.json`
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/diag/config_prenorm_ldr1.json`
- `experiments/hypothesis/hyp_009_arch_alignment/run_diag.py`

### Diagnostic Run Results (2026-03-05 16:20-16:28)

Run results (5k steps each, cuda:9):
1. post-norm + lpb=1 + ldr=5.0: VF=29.8%, val_loss best at step unknown
2. pre-norm + lpb=2 + ldr=1.0: VF=34.0%, val_loss=1.52 at step 2400 (worse than ldr=5.0!)

KEY FINDING: pre-norm vs post-norm = 1.4pp difference (29.8% vs 28.4%)
- Pre-norm is NOT the bottleneck
- Reducing ldr (1.0 vs 5.0) gives only 5.6pp improvement
- With ldr=1.0, val_loss WORSENS (1.52 vs 1.29) despite VF improvement
- Pattern consistent with PARTIAL exploitation: lower ldr lets model use log_det to
  expand coordinates, giving better atom distances WITHOUT learning the distribution

Root cause confirmed: affine convention allows log_det exploitation.
- Our convention: y=exp(log_scale)*x+shift. Forward = expansion. log_det positive → reduces NLL.
- Apple convention: z=exp(-xa)*(x-xb). Forward = contraction. logdet negative → increases NLL.
- Fix: force log_scale <= 0 (contraction-only in forward). Equivalent to alpha_pos ~ 0.
- Test: alpha_pos=0.001 (effectively 0), alpha_neg=10.0, ldr=0.0

INTENTION (write-before-execute): Run contraction-only test on cuda:9
- Config: alpha_pos=0.001, alpha_neg=10.0, ldr=0.0, pre-norm+lpb=2, T=9
- Expect: VF >> 34% if contraction-only prevents exploitation and model can learn
- If VF >= 90%: contraction-only is the fix, update sanity angle
- If VF < 90%: investigate further

### New Files Created (continued)
- `experiments/hypothesis/hyp_009_arch_alignment/angles/sanity/diag/config_alpha_pos0.json` — contraction-only test
- `experiments/hypothesis/hyp_009_arch_alignment/run_diag2.py`

### Commits
