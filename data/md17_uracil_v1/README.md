# MD17 Uracil — Preprocessed Dataset

## Overview
- **Molecule:** uracil
- **Atoms:** 12 (C, C, N, C, N, C, O, O, H, H, H, H)
- **Conformations:** 133770
- **Energy range:** [-260120.68, -260080.75] kcal/mol
- **Split:** train=107016, val=13377, test=13377

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (133770, 21, 3), energies (133770,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (133770, 21, 3) float32
energies = data["energies"]        # (133770,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
