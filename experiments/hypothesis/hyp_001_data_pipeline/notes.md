## [hyp_001] -- MD17 Data Pipeline
**Date:** 2026-02-28 | **Type:** Hypothesis | **Tag:** `hyp_001`

### Motivation
This is the data foundation for the entire TNAFMOL project. All downstream experiments (TarFlow, DDPM, head-to-head comparison) depend on having correctly preprocessed molecular conformations in canonical frame representation. Getting this right is critical -- a preprocessing error here silently corrupts every downstream result.

### Method
1. Download MD17 for all 8 molecules (aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic acid, toluene, uracil)
2. Canonical frame preprocessing:
   - Subtract center of mass per conformation
   - Compute mean structure per molecule
   - Kabsch alignment to mean structure (principal axis alignment)
   - Pad to max atom count (21) with attention mask
3. One-hot atom type encoding (H, C, N, O)
4. 80/10/10 train/val/test split per molecule
5. Compute reference statistics: energy distributions, pairwise distance distributions, bond lengths
6. Save in CLAUDE.md-compliant data directory structure with metadata.json and ref_stats.pt

### Results
*(To be filled after experiment completes.)*

### Interpretation
*(To be filled after experiment completes.)*

**Status:** [ ] Fits | [ ] Conflict | [ ] Inconclusive
