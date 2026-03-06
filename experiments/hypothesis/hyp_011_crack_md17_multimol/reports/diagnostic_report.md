# Diagnostic Report — hyp_011: Crack MD17 Multi-Molecule TarFlow

## Baseline Failure Analysis

hyp_010 established the multi-molecule TarFlow baseline at **mean VF = 71.6%** (all 8 molecules > 50%).

Per-molecule VF comparison (hyp_010 baseline):
| Molecule | hyp_010 VF | und_001 ceiling | Gap |
|----------|-----------|-----------------|-----|
| aspirin | 67.4% | 94.3% | -26.9pp |
| benzene | 79.4% | 100.0% | -20.6pp |
| ethanol | 64.0% | 96.2% | -32.2pp |
| malonaldehyde | 82.6% | 98.6% | -16.0pp |
| naphthalene | 81.0% | 100.0% | -19.0pp |
| salicylic_acid | 67.4% | 95.0% | -27.6pp |
| toluene | 67.4% | 99.8% | -32.4pp |
| uracil | 63.6% | 100.0% | -36.4pp |
| **Mean** | **71.6%** | **98.2%** | **-26.6pp** |

hyp_010 config: channels=256, num_blocks=4, layers_per_block=2, lr=1e-3, cosine, 20k steps, batch_size=128, ~6.3M params.

## Root Cause Assessment

The 26.6pp gap has multiple potential causes:
1. **Training budget**: 20k steps with lr=1e-3 — model likely undertrained or LR too high for convergence
2. **Model capacity**: 6.3M params shared across 8 diverse molecules — may be insufficient
3. **Learning rate**: lr=1e-3 may cause loss instability at long training horizons

Phase 1 tested both levers in parallel to determine which matters more.

## Phase 1 Results — Parallel Validation Runs

### Run A — Long training, same model (256ch, 4blk, 50k steps, lr=5e-4, batch=256)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ls03bb0g
- **Mean VF: 73.3%** (+1.7pp over hyp_010)

| Molecule | Run A VF | hyp_010 VF | Delta |
|----------|---------|-----------|-------|
| aspirin | 83.2% | 67.4% | +15.8pp |
| benzene | 94.6% | 79.4% | +15.2pp |
| ethanol | 86.8% | 64.0% | +22.8pp |
| malonaldehyde | 94.6% | 82.6% | +12.0pp |
| naphthalene | 68.4% | 81.0% | -12.6pp |
| salicylic_acid | 70.2% | 67.4% | +2.8pp |
| **toluene** | **3.2%** | 67.4% | **-64.2pp COLLAPSE** |
| uracil | 85.6% | 63.6% | +22.0pp |

**Critical finding:** Toluene collapsed to 3.2% VF at 50k steps. This is a per-molecule overfitting/collapse phenomenon. The model achieved strong VF on most molecules (6/7 improved) but catastrophically failed on toluene at long training. The final checkpoint convention (which we use per und_001 precedent) captured the collapsed state. The mean VF of 73.3% is misleadingly low — without toluene the 7-molecule mean would be ~83.4%.

### Run B — Bigger model (384ch, 6blk, 20k steps, lr=5e-4, batch=128)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/820m2ely
- **Mean VF: 83.9%** (+12.3pp over hyp_010)

| Molecule | Run B VF | hyp_010 VF | Delta |
|----------|---------|-----------|-------|
| aspirin | 70.6% | 67.4% | +3.2pp |
| benzene | 97.2% | 79.4% | +17.8pp |
| ethanol | 72.4% | 64.0% | +8.4pp |
| malonaldehyde | 97.0% | 82.6% | +14.4pp |
| naphthalene | 91.0% | 81.0% | +10.0pp |
| salicylic_acid | 67.6% | 67.4% | +0.2pp |
| toluene | 91.0% | 67.4% | +23.6pp |
| uracil | 84.2% | 63.6% | +20.6pp |

**All 8 molecules improved. No collapse.** Run B exceeds the promising criterion (>78%) with a margin of +5.9pp.

## Root Cause — Updated Assessment

The comparison reveals:
1. **Model capacity is the primary driver.** Run B (bigger model, same steps as hyp_010) achieves +12.3pp while Run A (same model, 2.5x more steps) achieves only +1.7pp. The capacity increase is far more effective than extended training with the current architecture size.
2. **Long training with small model causes per-molecule collapse.** Toluene's 3.2% VF at 50k steps vs 91.0% at 20k steps with bigger model shows that overtraining a small model on a diverse multi-molecule task leads to catastrophic forgetting/collapse on specific molecules.
3. **Lower LR (5e-4 vs 1e-3) helps.** Run A's improved per-molecule scores (6/8 molecules improved, some by 15-22pp) suggest lower LR is beneficial. The issue is the training budget + model size interaction, not LR alone.
4. **The gap is capacity-dominated.** aspirin (70.6%), ethanol (72.4%), and salicylic_acid (67.6%) remain below 75% even with Run B. These harder molecules need more capacity or more careful training.

## Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY | Yes — DONE | Tested longer training and bigger model. Clear winner: Run B (384ch, 6blk). |
| KNOWN HEURISTICS | Yes | Sweep key hyperparameters: lr, ldr, noise_sigma. Use Run B architecture (384ch, 6blk). |
| SCALE | Yes | If HEURISTICS doesn't reach 85%+, channels=512/blocks=8 is the next lever. |

## Proposed Angles (preliminary — full spec in Plan Sub-report)

1. **HEURISTICS — Hyperparameter sweep**: Use Run B config (384ch, 6blk) as base. Sweep lr ∈ {1e-4, 3e-4, 5e-4}, ldr ∈ {0.0, 2.0, 5.0}, noise_sigma ∈ {0.03, 0.05, 0.1}, n_steps ∈ {20k, 50k}. W&B sweep on escher/germain via Slurm array.
2. **SCALE — Maximum capacity**: channels=512, blocks=8, lpb=2 (~25M params). Best heuristics config from Phase 2. 100k steps on Slurm.
