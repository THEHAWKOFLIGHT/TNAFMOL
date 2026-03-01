# MD17 Ethanol — Preprocessed Dataset

## Overview
- **Molecule:** ethanol
- **Atoms:** 9 (C, C, O, H, H, H, H, H, H)
- **Conformations:** 555092
- **Energy range:** [-97208.41, -97171.79] kcal/mol
- **Split:** train=444073, val=55509, test=55510

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (555092, 21, 3), energies (555092,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (555092, 21, 3) float32
energies = data["energies"]        # (555092,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
