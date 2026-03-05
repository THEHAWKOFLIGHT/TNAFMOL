# TNAFMOL — Results
**Last updated:** 2026-03-04 after hyp_006

## Status
Output-shift architecture eliminates log-det exploitation (log_det/dof bounded 0.5-1.3 vs 7+ for SOS). Best multi-molecule VF: 24.8% ethanol (hyp_006 HEURISTICS C, lr=1e-3, cosine, 5k steps). Architecture ceiling per-molecule (no padding): 98.2% mean VF (und_001). The remaining VF gap (25% vs 98%) is due to atom overlaps in generated samples, likely caused by normalization mismatch — not log-det exploitation or padding corruption.

## Experiments

| ID | Method | Valid % (best) | Key Finding | Status |
|----|--------|---------------|-------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all) | Log-det exploitation (3 collapse modes) | FAILURE |
| hyp_003 | TarFlow stabilization (clamp + reg) | 18.3% mean | Alpha_pos saturation equilibrium | FAILURE |
| hyp_004 | TarFlow architectural ablation (pos_enc + SBG) | 29.5% mean (sweep) | Improvements within buggy equilibrium | PARTIAL |
| und_001 | TarFlow Diagnostic Ladder | 98.2% mean (no pad) | Padding is sole failure; bugs found + fixed | DONE |
| hyp_005 | Padding-Aware TarFlow (PAD token + query zeroing) | 4.7% (ethanol) | Padding fixes have zero effect; log-det exploitation persists | FAILURE |
| **hyp_006** | **Output-Shift TarFlow (Apple architecture)** | **24.8% (ethanol)** | **Hypothesis CONFIRMED: log-det exploitation eliminated. VF plateau at 13-25%.** | **FAILURE** |

## Best Result
**und_001:** Architecture ceiling = 98.2% mean VF across all 8 molecules with T=n_real (no padding). Range: 94.3% (aspirin) to 100% (naphthalene, benzene).

**hyp_006 (multi-molecule with output-shift):** Best VF=24.8% on ethanol with HEURISTICS C (lr=1e-3, cosine, 5k steps). log_det/dof bounded at 0.5-1.3 throughout training — exploitation pathway eliminated. Benzene 27.6%, malonaldehyde 23.8%, toluene 19.6%. SCALE (9.6M params) overfits and performs worse (16.2%).

## What's Next
The output-shift architecture is now the correct platform for multi-molecule TarFlow. The remaining VF gap (25% vs 98%) is a normalization/training dynamics issue, not architectural. Next steps: (A) per-atom-type normalization instead of global std, (B) extend HEUR C to 20k-50k steps with careful lr scheduling, (C) internal coordinates (bonds, angles, dihedrals) instead of Cartesian, (D) per-molecule training on output-shift platform for DDPM comparison.
