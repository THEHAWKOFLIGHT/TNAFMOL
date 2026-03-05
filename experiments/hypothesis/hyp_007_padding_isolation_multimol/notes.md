## [hyp_007] — Padding Isolation + Multi-Molecule Scale
**Date:** 2026-03-05 | **Type:** Hypothesis | **Tag:** `hyp_007`

### Motivation
hyp_006 proved output-shift eliminates log-det exploitation (log_det/dof bounded 0.5-1.3 vs 7+ for SOS). Best VF was 24.8% ethanol at 5k steps (severely undertrained). und_001 showed padding causes catastrophic degradation with SOS architecture (89% → 2.7%). **Critical missing experiment:** does output-shift make padding neutral? If yes, multi-molecule is just conditioning on atom types — should work with sufficient training.

Two-phase design:
- **Phase 1 (GATE):** Single-molecule ethanol at 5 padding sizes (T=9,12,15,18,21). Success: VF >= 90% for all.
- **Phase 2 (OPTIMIZE):** Multi-molecule with all 8 MD17 molecules, SANITY → HEURISTICS → SCALE.

### Method
Phase 1: Add configurable `max_atoms` parameter. Train ethanol-only at max_atoms=9,12,15,18,21 (5000 steps each). Use hyp_006 HEUR C recipe (lr=1e-3, cosine, output-shift, bidir types, PAD token, query zeroing).

Phase 2: OPTIMIZE multi-molecule with output-shift + sufficient training. SANITY at 20k steps, HEURISTICS sweep over lr/steps/batch_size, SCALE at d_model=256/n_blocks=12.

### Results
*(to be filled after execution)*

### Interpretation
*(to be filled after execution)*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
