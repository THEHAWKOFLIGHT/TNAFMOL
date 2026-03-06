# TNAFMOL — Results
**Last updated:** 2026-03-06 after hyp_011

## Status
TarFlow Apple architecture (TarFlow1DMol, 512ch/8blk, 50.6M params) achieves **98.9% mean VF** across all 8 MD17 molecules at sampling temperature T=0.7 — essentially matching the per-molecule ceiling (98.2%). All 8 molecules above 95.6%. The multi-molecule gap is closed: a single shared model matches dedicated per-molecule models.

## Experiments

| ID | Method | Valid % (best) | Key Finding | Status |
|----|--------|---------------|-------------|--------|
| hyp_001 | MD17 data pipeline | N/A | N/A | DONE |
| hyp_002 | TarFlow (autoregressive affine flow) | 0% (all) | Log-det exploitation (3 collapse modes) | FAILURE |
| hyp_003 | TarFlow stabilization (clamp + reg) | 18.3% mean | Alpha_pos saturation equilibrium | FAILURE |
| hyp_004 | TarFlow architectural ablation (pos_enc + SBG) | 29.5% mean (sweep) | Improvements within buggy equilibrium | PARTIAL |
| und_001 | TarFlow Diagnostic Ladder | 98.2% mean (no pad) | Padding is sole failure; bugs found + fixed | DONE |
| hyp_005 | Padding-Aware TarFlow (PAD token + query zeroing) | 4.7% (ethanol) | Padding fixes have zero effect; log-det exploitation persists | FAILURE |
| hyp_006 | Output-Shift TarFlow (Apple architecture) | 24.8% (ethanol) | Hypothesis CONFIRMED: log-det exploitation eliminated. VF plateau at 13-25%. | FAILURE |
| hyp_007 | Output-Shift + ldr=5.0 + 20k steps | 55.8% (ethanol), 34.7% mean | Padding neutral (4pp drop). ldr=5.0 critical. 2/8 molecules above 50%. | PARTIAL |
| hyp_008 | Per-dim scale (3 log_scales/atom) | 39.2% (ethanol, T=9) | <1pp effect vs shared scale (und_001 Phase 4). True gap: pre-norm + layers_per_block. | FAILURE |
| hyp_009 | Pre-norm + layers_per_block=2 | 14% (ethanol, T=9) | WORSE than baseline (39%). Incremental patching exhausted. | FAILURE |
| hyp_010 | TarFlow Apple (TarFlow1DMol) multi-mol | 71.6% mean, all 8 >50% | Apple arch generalizes to multi-mol. No ldr needed. 2 padding bugs fixed. | DONE |
| **hyp_011** | **TarFlow scaling + HP tuning** | **98.9% mean (T=0.7), 97.4% (T=1.0)** | **Capacity is primary driver. noise_sigma=0.03 key. Multi-mol gap closed.** | **DONE** |

## Best Result
**hyp_011 SCALE (512ch, 8blk, 50.6M params, T=0.7):** All 8 molecules, T=21, 50k steps. Per-molecule VF: benzene 100%, naphthalene 100%, toluene 100%, malonaldehyde 99.8%, salicylic_acid 99.8%, uracil 99.6%, ethanol 96.2%, aspirin 95.6%. **Mean VF=98.9%.** Config: channels=512, num_blocks=8, layers_per_block=2, lr=5e-4, ldr=2.0, noise_sigma=0.03, cosine+1000 warmup, 50k steps, sampling temp=0.7. W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/z7dwsfdj

**und_001:** Architecture ceiling = 98.2% mean VF (per-molecule, no padding). hyp_011 SCALE at T=0.7 exceeds this ceiling (98.9% > 98.2%), demonstrating that the multi-molecule model has effectively matched or exceeded per-molecule performance.

## What's Next
DDPM diffusion baseline — implement transformer-based denoiser with comparable parameter count. Train on same multi-molecule MD17 data. Head-to-head comparison with TarFlow on all evaluation metrics.
