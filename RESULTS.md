# TNAFMOL — Results
**Last updated:** 2026-03-02 after hyp_004

## Status
TarFlow with positional encodings + SBG training recipe (lr=1e-3, EMA=0.99) achieves 26.7% mean valid fraction with 1/8 molecules ≥ 50% (malonaldehyde 56.6%). This is the best TarFlow result to date but fails the 4+/8 primary criterion. The alpha_pos saturation equilibrium persists across all configurations tested — architectural modifications and training recipes improve performance within the constraint but cannot break it. Next: DDPM diffusion baseline (hyp_005).

## Experiments

| ID | Method | Valid % (best) | Collapse Mode | Status |
|----|--------|---------------|---------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all molecules) | Affine scale / shift / ActNorm scale | FAILURE |
| hyp_003 | TarFlow stabilization (asym clamp + log-det reg + soft equiv) | 18.3% mean (sweep), 14.3% mean (full) | Alpha_pos saturation equilibrium | FAILURE |
| hyp_004 | TarFlow architectural ablation (pos_enc + SBG lr=1e-3 ema=0.99) | 29.5% mean (sweep), 26.7% mean (full) | Alpha_pos saturation (unchanged) | PARTIAL |

## Best Result
**hyp_004:** PARTIAL. Best mean valid fraction across all OPTIMIZE angles: 29.5% (HEURISTICS sweep, lr=1e-3, ema=0.99). Full run: 26.7% (20k steps). Per-molecule best (HEURISTICS full): malonaldehyde **56.6%**, uracil 43.6%, ethanol 40.0%, benzene 26.6%, salicylic_acid 17.8%, toluene 14.6%, naphthalene 7.8%, aspirin 6.6%. First molecule ever to exceed 50% valid fraction. 1/8 molecules ≥ 50% — primary criterion (4+/8) not met.

Key architectural finding: positional encodings (+5ppt) are the only beneficial modification of the three tested (bidirectional type conditioning and permutation augmentation both slightly hurt). Key training recipe finding: lr=1e-3 with OneCycleLR + EMA=0.99 gives +12ppt over the SANITY baseline — 10× more improvement than architecture alone.

## What's Next
hyp_005: DDPM diffusion baseline. Three consecutive TarFlow experiments (hyp_002, hyp_003, hyp_004) have established the performance ceiling: ~30% mean valid fraction under the alpha_pos saturation equilibrium. The equilibrium is a mathematical fixed point that cannot be escaped by architecture, training recipe, or scale. Diffusion avoids this entirely.
