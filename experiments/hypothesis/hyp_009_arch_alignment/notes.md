## [hyp_009] — Architecture Alignment (Pre-Norm + Layers Per Block)
**Date:** 2026-03-06 | **Type:** Hypothesis | **Tag:** `hyp_009`

### Motivation
hyp_008 identified a 61pp VF gap between model.py (~39%) and tarflow_apple.py (~96%) on single-molecule ethanol T=9. The two remaining architectural differences were: (1) post-norm vs pre-norm LayerNorm, and (2) layers_per_block=1 vs 2 (Apple uses 2 attention+FFN sublayers per flow block).

### Method
Added `use_pre_norm` and `layers_per_block` parameters to model.py's TarFlowBlock. Phase 1 gate: ethanol T=9, use_pre_norm=True, layers_per_block=2, dropout=0.0, d_model=256, n_blocks=4, 5k steps. Additional diagnostic runs explored ldr, contraction-only, and postnorm+ldr5.

### Results
Phase 1 gate FAILED: best VF = 14% on ethanol T=9 (target: 90%). Diagnostic investigation runs:
- Pre-norm + ldr=0: 14% VF (WORSE than post-norm baseline ~39%)
- Pre-norm + ldr=1: investigated
- Post-norm + ldr=5: investigated
- Contraction-only (alpha_pos=0.001): investigated

The incremental patching strategy of adding Apple features to model.py has failed. After 4 experiments (hyp_006 through hyp_009), each fixing a "root cause," model.py still cannot match tarflow_apple.py. The two architectures diverge in 13+ ways, making isolation impossible.

### Interpretation
**Status:** [x] Conflict — The incremental patching approach to closing the gap between model.py and tarflow_apple.py is exhausted. The correct approach is to use the proven architecture (tarflow_apple.py + TarFlow1DMol) directly for multi-molecule training (hyp_010).
