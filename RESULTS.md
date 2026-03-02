# TNAFMOL — Results
**Last updated:** 2026-03-01 after hyp_003

## Status
TarFlow normalizing flow has failed twice on molecular conformation generation: hyp_002 collapsed via unbounded log_det exploitation (valid fraction 0%), hyp_003 stabilized with asymmetric clamping + log-det regularization but saturated at a mathematical equilibrium (best 18.3% mean valid fraction, 0/8 molecules ≥ 50%). The failure is architectural. Next: DDPM diffusion baseline (hyp_004).

## Experiments

| ID | Method | Valid % (best) | Collapse Mode | Status |
|----|--------|---------------|---------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all molecules) | Affine scale / shift / ActNorm scale | FAILURE |
| hyp_003 | TarFlow stabilization (asym clamp + log-det reg + soft equiv) | 18.3% mean (sweep), 14.3% mean (full) | Alpha_pos saturation equilibrium | FAILURE |

## Best Result
**hyp_003:** FAILURE. Best mean valid fraction across all OPTIMIZE angles: 18.3% (HEURISTICS sweep, bs=512, ema=0.999, lr=1e-3). Per-molecule best (HEURISTICS full, 20k steps): malonaldehyde 38.0%, ethanol 33.4%, benzene 15.2%, uracil 13.6%, toluene 7.4%, salicylic_acid 3.8%, naphthalene 2.0%, aspirin 0.8%. Strong inverse correlation with molecule size -- 9-atom molecules reach ~35% but 21-atom aspirin is <1%. No molecule reaches the 50% primary criterion.

## What's Next
hyp_004: DDPM diffusion baseline. Two consecutive TarFlow failures confirm the architecture is not viable for molecular conformations under MLE training. Diffusion avoids log_det exploitation entirely (no explicit density in the training objective).
