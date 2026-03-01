# MD17 Malonaldehyde — Preprocessed Dataset

## Overview
- **Molecule:** malonaldehyde
- **Atoms:** 9 (C, C, C, O, O, H, H, H, H)
- **Conformations:** 993237
- **Energy range:** [-167514.21, -167470.39] kcal/mol
- **Split:** train=794589, val=99323, test=99325

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (993237, 21, 3), energies (993237,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (993237, 21, 3) float32
energies = data["energies"]        # (993237,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
