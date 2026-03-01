# MD17 Benzene — Preprocessed Dataset

## Overview
- **Molecule:** benzene
- **Atoms:** 12 (C, C, C, C, C, C, H, H, H, H, H, H)
- **Conformations:** 627983
- **Energy range:** [-146536.12, -146513.64] kcal/mol
- **Split:** train=502386, val=62798, test=62799

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (627983, 21, 3), energies (627983,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (627983, 21, 3) float32
energies = data["energies"]        # (627983,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
