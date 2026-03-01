# MD17 Salicylic Acid — Preprocessed Dataset

## Overview
- **Molecule:** salicylic_acid
- **Atoms:** 16 (C, C, C, O, C, C, C, C, O, O, H, H, H, H, H, H)
- **Conformations:** 320231
- **Energy range:** [-311050.43, -311002.96] kcal/mol
- **Split:** train=256184, val=32023, test=32024

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (320231, 21, 3), energies (320231,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (320231, 21, 3) float32
energies = data["energies"]        # (320231,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
