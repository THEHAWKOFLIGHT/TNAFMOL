# MD17 Toluene — Preprocessed Dataset

## Overview
- **Molecule:** toluene
- **Atoms:** 15 (C, C, C, C, C, C, C, H, H, H, H, H, H, H, H)
- **Conformations:** 442790
- **Energy range:** [-170244.28, -170192.20] kcal/mol
- **Split:** train=354232, val=44279, test=44279

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (442790, 21, 3), energies (442790,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (442790, 21, 3) float32
energies = data["energies"]        # (442790,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
