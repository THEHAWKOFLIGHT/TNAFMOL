# TNAFMOL — Results
**Last updated:** 2026-02-28 after hyp_001

## Status
Data pipeline complete. All 8 MD17 molecules preprocessed into canonical frame representation with reference statistics. Ready for model training.

## Experiments

| ID | Method | Status |
|----|--------|--------|
| hyp_001 | MD17 data pipeline (download, canonical frame, ref stats) | DONE |

## Best Result
**hyp_001:** 8 MD17 molecules (aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil) downloaded and preprocessed. ~3.6M total conformations. Canonical frame: CoM-centered, Kabsch-aligned, padded to 21 atoms. Verified energy distributions, pairwise distances, and atom type encoding.

## What's Next
hyp_002: TarFlow (transformer autoregressive normalizing flow) implementation and OPTIMIZE.
