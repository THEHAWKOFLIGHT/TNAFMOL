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

### Notes
- GPU 8 (~17GB free) for test/validation runs; GPU 0 for full training
- Target: valid_fraction > 0.5 on 5+/8 molecules
- Shift-only flow is volume-preserving: log_det = 0 always. NLL = -log p_z(z) / n_dof = 0.919 at optimum (standard Gaussian entropy).
- Val_shift results at T=1.0: ethanol=22.8%, malonaldehyde=22.4%, uracil=7.2%, benzene=6.6%. More steps needed.
- W&B sweep URL will be logged here once initialized.
