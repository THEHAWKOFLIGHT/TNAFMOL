# TNAFMOL — Results
**Last updated:** 2026-03-01 after hyp_002

## Status
TarFlow (transformer autoregressive flow) failed to generate valid molecular conformations — valid fraction = 0 on all 8 MD17 molecules. The failure is architectural: autoregressive affine flow MLE training exploits any unconstrained scale DOF via log_det maximization. Next: DDPM diffusion baseline (hyp_003).

## Experiments

| ID | Method | Valid % (best) | Collapse Mode | Status |
|----|--------|---------------|---------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all molecules) | Affine scale / shift / ActNorm scale | FAILURE |

## Best Result
**hyp_002:** FAILURE. Best valid fraction across all OPTIMIZE angles: 22.8% (ethanol, SANITY shift_only at T=1.0) — but this is equivalent to raw Gaussian sampling, not meaningful model output. The HEURISTICS angle (ActNorm from GLOW, Kingma & Dhariwal 2018) achieved valid_fraction = 0% on all molecules due to ActNorm scale collapse (cumulative sampling contraction of 0.0013 across 8 layers).

## What's Next
hyp_003: DDPM diffusion baseline. The diffusion approach avoids the log_det exploitation problem entirely (no explicit density computation). This makes it the stronger candidate for molecular conformation generation in the head-to-head comparison.
