# MD17 Aspirin — Preprocessed Dataset

## Overview
- **Molecule:** aspirin
- **Atoms:** 21 (C, C, C, C, C, C, C, O, O, O, C, C, O, H, H, H, H, H, H, H, H)
- **Conformations:** 211762
- **Energy range:** [-406757.59, -406702.30] kcal/mol
- **Split:** train=169409, val=21176, test=21177

## Preprocessing
1. Center of mass subtraction
2. Kabsch alignment to mean structure
3. Zero-padded to 21 atoms with attention mask

## Files
- `dataset.npz`: positions (211762, 21, 3), energies (211762,), atom_types (21,), mask (21,), split indices
- `metadata.json`: generation parameters, statistics, conventions
- `ref_stats.pt`: reference statistics for evaluation

## Loading
```python
data = np.load("dataset.npz")
positions = data["positions"]      # (211762, 21, 3) float32
energies = data["energies"]        # (211762,) float32
atom_types = data["atom_types"]    # (21,) int64
mask = data["mask"]                # (21,) float32
train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]
```
