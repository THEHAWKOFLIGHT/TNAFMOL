## [hyp_005] — Padding-Aware Multi-Molecule TarFlow
**Date:** 2026-03-03 | **Type:** Hypothesis | **Tag:** `hyp_005`

### Motivation
und_001 showed TarFlow achieves 94-100% VF per molecule (no padding) but collapses to 0-40% when padded to T=21 for multi-molecule training. Two concrete padding corruption channels were identified:

**Corruption A — Padding = Hydrogen embedding:** Padding positions get atom_type index 0 (H). Gradient from padding contaminates hydrogen's learned representation.

**Corruption B — Padding as active queries:** Padding positions run through the full transformer. Their outputs are zeroed at the end, but they contaminate LayerNorm statistics, send gradients through input_proj, and waste compute.

Additionally, a causal mask bug (self-inclusive, non-triangular Jacobian) discovered by und_001 must be fixed before any experiments.

### Method
OPTIMIZE with 3 angles:
1. **SANITY**: Fix causal mask bug + 2x2 factorial ablation (PAD token × query zeroing)
2. **HEURISTICS**: Masked LayerNorm, zero PAD embedding
3. **SCALE**: d_model=256, n_blocks=12, 50k steps

### Results
*(To be filled after experiment)*

### Interpretation
*(To be filled after experiment)*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
