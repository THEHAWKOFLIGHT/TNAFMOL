# Diagnostic Report — hyp_007: Padding Isolation + Multi-Molecule

## Baseline Failure Analysis

This experiment builds directly on hyp_006 (output-shift TarFlow). The hyp_006 postdoc synthesis identified the failure mode explicitly — see the hyp_006 final report for full detail.

**Key hyp_006 findings (from synthesis):**
- Output-shift eliminates log-det explosion: training stable, loss converges
- Single-molecule ethanol: VF = 24.8% (best, HEURISTICS+SCALE exhausted)
- Multi-molecule (all 8): ethanol VF < 15% in all configurations tried
- Root cause confirmed by SCALE failure: 5k steps ≈ 22% of one epoch for the full 8-molecule dataset (~240k train samples / batch=128 = ~1875 steps/epoch). Model sees each sample less than once before evaluation. VF failure is a training budget failure, NOT a modeling failure.
- Secondary hypothesis: padding atoms from shorter molecules may still corrupt learning even with output-shift. Needs explicit isolation test.

**Two failure modes to address in hyp_007:**
1. **Training budget** (primary): too few steps — need 20k+ steps for meaningful learning on 8 molecules
2. **Padding neutrality** (gate): unknown whether output-shift sufficiently neutralizes padding for multi-molecule training. Ethanol (9 atoms) padded to 21 means 12/21 = 57% of atom slots are padding. Need to verify this is neutral before attributing all failure to budget.

## Root Cause

Primary: training budget — insufficient steps for any meaningful convergence on the 8-molecule dataset.
Secondary (to verify): padding atoms may not be fully neutral, potentially inflating the loss contribution from padding positions and preventing the model from focusing on real atom geometry.

## Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY (Phase 1) | Yes | Padding neutrality isolation test — verify output-shift makes padding neutral across different max_atoms values. Gate before Phase 2. |
| SANITY (Phase 2) | Yes | Training budget fix — 20k steps, all 8 molecules. This is the primary likely fix. |
| KNOWN HEURISTICS | Yes | If SANITY passes but VF still low — sweep lr/n_steps/batch_size to find better training dynamics |
| SCALE | Yes | If HEURISTICS insufficient — increase model capacity (d_model=256, n_blocks=12) |

## Proposed Angles (preliminary)

**Phase 1 (padding isolation gate):** Train ethanol only at max_atoms ∈ {9, 12, 15, 18, 21}. Expect VF ≥ 90% for all sizes if output-shift is truly padding-neutral. Any >20pp drop signals padding contamination.

**Phase 2 SANITY:** 20k steps, all 8 molecules, output-shift config unchanged. Promising criterion: ethanol VF > 40% AND mean VF > 30%.

**Phase 2 HEURISTICS:** If SANITY promising — sweep lr ∈ {3e-4, 5e-4, 1e-3}, n_steps ∈ {20k, 50k}, batch_size ∈ {128, 256}.

**Phase 2 SCALE:** d_model=256, n_blocks=12, ~9.6M params, lr=3e-4, 50k steps.
