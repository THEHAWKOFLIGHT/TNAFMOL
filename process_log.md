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
