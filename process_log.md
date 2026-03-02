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

### Commits
- `8faf809` — [hyp_003] code: implement asymmetric clamping, log-det regularization, soft equivariance
