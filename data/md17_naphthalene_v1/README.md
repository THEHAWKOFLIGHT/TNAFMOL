# MD17 Naphthalene — Preprocessed Dataset

## Overview
- **Molecule:** naphthalene
- **Atoms:** 18 (C, C, C, C, C, C, C, C, C, C, H, H, H, H, H, H, H, H)
- **Conformations:** 326250
- **Energy range:** [-241923.48, -241868.56] kcal/mol
- **Split:** train=261000, val=32625, test=32625

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (326250, 21, 3), energies (326250,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (326250, 21, 3) float32
energies = data["energies"]        # (326250,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
