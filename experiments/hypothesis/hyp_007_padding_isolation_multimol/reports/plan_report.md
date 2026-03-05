# OPTIMIZE Plan Sub-report — hyp_007: Padding Isolation + Multi-Molecule

**Status:** READY_TO_START

## Understanding

hyp_006 established that output-shift TarFlow trains stably on MD17 molecules. The primary failure was training budget: 5k steps is only ~22% of one epoch on all 8 molecules. This experiment has two phases:

1. **Phase 1 (gate):** Verify padding neutrality — train ethanol at different max_atoms sizes and confirm VF ≥ 90% across all padding amounts. This validates that output-shift genuinely neutralizes padding.
2. **Phase 2:** Multi-molecule OPTIMIZE starting with SANITY (20k steps). If that passes, sweep hyperparameters (HEURISTICS) and/or scale model (SCALE).

## Execution Plan

**Phase 1 — Padding Isolation Test:**
- Add `max_atoms` parameter to src/data.py and src/train.py
- Run ethanol training at max_atoms ∈ {9, 12, 15, 18, 21}, 5k steps each on test GPUs
- Run extremes (T=9, T=21) in parallel first
- Gate criterion: ALL runs achieve VF ≥ 90% (or ≥ 70% minimum with < 20pp drop)
- If any run below 70%: stop Phase 2, report padding contamination failure

**Phase 2 — Multi-Molecule OPTIMIZE (conditional on Phase 1 pass):**

**SANITY (Angle 1):** 20k steps, all 8 molecules, cuda:8 (test GPU exception: urgency + ≤20k steps validation run)
- Promising criterion: ethanol VF > 40% AND mean VF > 30%
- Fallback if not met: try lr=3e-4 with 20k steps before declaring SANITY failed

**HEURISTICS (Angle 2):** W&B sweep via Slurm --array
- Variables: lr ∈ {3e-4, 5e-4, 1e-3}, n_steps ∈ {20k, 50k}, batch_size ∈ {128, 256}
- Max 8 runs
- Citation: Cosine LR schedule with extended budget is standard practice in flow matching literature (Lipman et al. 2022 Flow Matching, Albergo & Vanden-Eijnden 2023 Building Normalizing Flows); batch size and learning rate scaling follow standard Adam tuning guidelines (Kingma & Ba 2015).
- Select best config by ethanol VF + mean VF after full sweep completion
- Full training run with best config

**SCALE (Angle 3, if needed):**
- d_model=256, n_blocks=12, ~9.6M params (vs 3.4M base)
- lr=3e-4, 50k steps
- Must use Slurm for production run

## Proposed Milestones

- Milestone 1: Phase 1 complete — VF for all 5 max_atoms values reported
- Milestone 2: Phase 2 SANITY complete — VF per molecule reported
- Milestone 3: Phase 2 HEURISTICS sweep complete (if applicable)
- Milestone 4: Final training run + evaluation + all plots

## Questions / Concerns

None. Implementation is clear from codebase reading.
