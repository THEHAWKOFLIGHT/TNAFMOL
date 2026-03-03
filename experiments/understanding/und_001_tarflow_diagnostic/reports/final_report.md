# Final Experiment Report — und_001 TarFlow Diagnostic Ladder

**Status:** DONE
**Branch:** `exp/und_001`
**Type:** Understanding — DIAGNOSE
**Date:** 2026-03-03

**Commits:**
- `cff2c17` — [und_001] meta: initialize experiment
- `34fd7dd` — [und_001] code: fix attention mask broadcasting bug
- `b07b9b5` — [und_001] docs: process_log session continuation
- `d77e240` — [und_001] results: Phase 3 Steps A and B complete
- `09c565f` — [und_001] code: fix attention mask and padding z-zeroing
- `901d6c5` — [und_001] code: fix permutation-aware padding mask
- `3fbf7f1` — [und_001] results: Phase 3 Steps C-E complete
- `05da6e1` — [und_001] results: Step F complete (VF=10.4%)
- `da9ac3a` — [und_001] docs: final report and process log for Phase 3
- `8e03ca9` — [und_001] results: Phase 3 adaptation_report.md
- `7a53b43` — [und_001] docs: update final_report with commit list
- `e308108` — [und_001] code: implement Phase 4 ablation matrix (train_phase4.py)
- `5495e93` — [und_001] results: Phase 4 ablation matrix — all 9 configs complete
- `8b211a5` — [und_001] docs: Phase 4 ablation report
- `e897f5c` — [und_001] code: add train_phase5.py
- `10ca9e8` — [und_001] results: Phase 5 — best config validation on all 8 MD17 molecules

---

## Executive Summary

The und_001 diagnostic ladder systematically traced TarFlow's failure on molecular conformations through 5 experimental phases covering 31 training runs across all 8 MD17 molecules. The central finding:

**TarFlow achieves 98.2% mean valid fraction across all 8 MD17 molecules when each molecule is trained at its natural size (T = n_real atoms, no padding).** The architecture is fundamentally sound for molecular conformation generation.

**The multi-molecule failure (hyp_003: 18.3% mean VF) is caused entirely by padding.** When molecules are padded to a common sequence length (T=21) for multi-molecule training, valid fraction collapses to 20.8% on average, with aromatic molecules (naphthalene, toluene) collapsing to 0%.

The prior hypothesis — that shared scale parameterization causes log-det exploitation — was wrong. The hyp_002/hyp_003 failures were caused by two implementation bugs (logdet normalization by T*D instead of n_real*D, and a self-inclusive causal mask). With both bugs fixed, shared scale performs comparably to or better than per-dim scale.

---

## Phase-by-Phase Results

### Phase 1: Source Comparison

13 architectural differences documented between Apple TarFlow and our hyp_002/hyp_003 implementation.

Three CRITICAL differences:
1. **Per-dim vs. shared scale** — Apple uses D separate scales per token; ours uses 1 shared scalar. Phase 3 proved this is NOT the primary failure.
2. **Causal mechanism** — Apple uses output shift (correct triangular Jacobian); ours used SOS + self-inclusive mask (incorrect Jacobian). Bug fix: commit `901d6c5`.
3. **Scale clamping** — Apple uses none; ours applied asymmetric arctan clamping that created the alpha_pos saturation equilibrium in hyp_003.

Two HIGH-severity differences:
4. **Log-det regularization** — Apple none; ours penalized log-det deviation from zero.
5. **Noise augmentation** — Apple sigma=0.05; ours none. Phase 3/4 confirmed this is important.

Full report: `reports/source_comparison.md`

### Phase 2: Apple Baseline Verification

| Level | Dataset | Key Result | Status |
|-------|---------|-----------|--------|
| 0 | 2D 8-mode Gaussian | 88.6% mode coverage, NLL=0.91 | DONE |
| 1 | MNIST 1x28x28 | -3.20 bits/dim, recognizable digits | DONE |
| 2 | CIFAR-10 3x32x32 | Loss=-2.06 at step 23500/50000 | IN PROGRESS |

Apple TarFlow confirmed working on standard benchmarks. CIFAR-10 training continues on GPU 0 (non-blocking; does not affect molecular diagnostic conclusions).

Full report: `reports/ladder_report.md`

### Phase 3: Adaptation Ladder (ethanol, 5000 steps each)

| Step | Change Added | VF | Delta |
|------|-------------|-----|-------|
| A | Apple TarFlow1D, raw coords, 9 atoms | **89.1%** | baseline |
| B | + Atom type conditioning | **92.9%** | +3.8 pp |
| C | + Padding to T=21 | **2.7%** | **-90.2 pp** |
| D | + Noise augmentation (sigma=0.05) | **14.3%** | +11.6 pp |
| E | Shared scale (1 per atom) | **40.2%** | +25.9 pp |
| F | + Clamping + log-det reg | **10.4%** | -29.8 pp |

**Primary finding:** Padding (Step C) is the catastrophic failure point, not shared scale (Step E). NLL is decoupled from VF in the padded regime — the model learns the distribution correctly by NLL but generates invalid structures.

Two bugs discovered and fixed:
1. `PermutationFlip` did not permute the padding mask (commit `901d6c5`)
2. Logdet normalization by T*D instead of n_real*D (commit `901d6c5`)

Full reports: `reports/phase3_report.md`, `reports/adaptation_report.md`

### Phase 4: Ablation Matrix (15 configs total: 6 Phase 3 + 9 Phase 4)

Crossed T (9 vs 21) x Noise (yes/no) x Scale (shared/per-dim) + padding sweep (T=12, 15) + augmentation tests (perm, SO3) + clamp without padding.

| Finding | Effect Size | Key Numbers |
|---------|-------------|-------------|
| **Padding** | -90 pp (dominant) | T=9: 93-96% VF; T=21: 0.9-40% VF |
| Permutation augmentation | -38 pp (catastrophic) | 2.1% VF — architecturally incompatible |
| Clamping with padding | -30 pp (harmful) | 40.2% → 10.4% |
| Noise augmentation | +11-39 pp (beneficial) | Essential in padded regime |
| Shared vs per-dim scale | <1 pp (irrelevant at T=9) | 93.6% vs 92.9% without padding |
| SO(3) augmentation | -5 pp (mild harm) | 34.8% vs 40.2% |

**Padding scaling:** VF declines linearly with padding fraction on ethanol: VF ~ 95.3% - 96.4% * pad_fraction (R^2 = 0.98 across T=9, 12, 15, 21).

**Interaction effect:** Noise x shared scale has a large positive interaction ONLY in the padded regime (+39 pp jointly vs +12 pp for noise alone with per-dim scale). Without padding, noise and scale type are independently small effects.

Full report: `reports/ablation_report.md`

### Phase 5: Best Config Validation on All 8 MD17 Molecules (16 runs)

Two configs tested on all 8 molecules:
- **Config A (ceiling):** T = n_real (no padding), per-dim scale, noise=0.05
- **Config B (practical):** T = 21 (padded), shared scale, noise=0.05

| Molecule | n_real | Config A VF | Config B VF | pad_frac |
|----------|--------|-------------|-------------|----------|
| aspirin | 21 | 94.3% | 93.2% | 0.000 |
| naphthalene | 18 | 100.0% | 0.0% | 0.143 |
| salicylic_acid | 16 | 97.8% | 8.1% | 0.238 |
| toluene | 15 | 98.7% | 0.0% | 0.286 |
| benzene | 12 | 100.0% | 2.9% | 0.429 |
| uracil | 12 | 99.2% | 6.9% | 0.429 |
| ethanol | 9 | 96.2% | 40.2% | 0.571 |
| malonaldehyde | 9 | 99.8% | 15.4% | 0.571 |
| **Mean** | — | **98.2%** | **20.8%** | — |

**Reference:** hyp_003 best mean VF = **18.3%**

**Key findings:**
- Config A achieves 94-100% VF on every molecule — the architecture is not the bottleneck.
- Config B (20.8% mean) modestly beats hyp_003 (18.3%) via noise + shared scale improvements.
- Aspirin (no padding) achieves 93.2% in Config B, confirming padding fraction is the sole driver of the A-B gap.
- Phase 4's ethanol-derived linear model overestimates VF for non-ethanol molecules. Aromatic molecules (naphthalene 0%, toluene 0%) collapse at much lower padding fractions than predicted.

Full report: `reports/phase5_report.md`

---

## Answers to the Four Diagnostic Questions

### 1. Where on the ladder does performance degrade?

**Step C: adding padding.** This is the single point of catastrophic failure. The transition from T=n_real (89-100% VF) to T=21 (0-40% VF) accounts for the entire performance gap. Every other adaptation (atom types, noise, scale type, clamping) causes changes of 1-30 pp — none is catastrophic in isolation.

The degradation is smooth, not binary: VF declines approximately linearly with padding fraction on ethanol (96% at 0%, 70% at 25%, 50% at 40%, 40% at 57%). However, the decline is molecule-specific: aromatic molecules collapse to 0% at much lower padding fractions than the ethanol-based fit predicts.

### 2. Which adaptation factor matters most?

**Padding fraction**, by a factor of 3x over the next largest effect.

Per-factor effect sizes (from Phase 4, ethanol):

| Factor | Effect (pp) | Direction |
|--------|-------------|-----------|
| Padding (T=9 → T=21) | 90 | Harmful |
| Permutation augmentation | 38 | Harmful |
| Clamping (with padding) | 30 | Harmful |
| Noise augmentation | 11-39 | Beneficial |
| Shared scale (with noise+padding) | 26 | Beneficial |
| SO(3) augmentation | 5 | Harmful |
| Scale type (without padding) | <1 | Neutral |

Note: permutation augmentation and clamping-with-padding are both harmful, but they are NOT adaptations we would use — they are mitigations that don't work. The only genuinely harmful adaptation is padding itself.

### 3. Is the failure architectural or fundamental?

**Neither — the failure is in the multi-molecule interface (padding), not in the flow architecture itself.**

Config A demonstrates that TarFlow achieves 94-100% VF on all 8 MD17 molecules at their natural sizes. The architecture — autoregressive affine flow with transformer blocks — correctly learns molecular conformation distributions when the sequence length matches the actual molecule size.

The failure occurs specifically when molecules are padded to a common length for multi-molecule training. This is an interface problem: the padding tokens corrupt the flow's latent space and log-determinant computation. The root mechanism is that padding tokens participate as key/value context in causal attention, diluting the mutual information budget for real atoms and creating gradient imbalances in the log-det objective.

This means:
- **TarFlow IS viable for single-molecule models** (per-molecule T = n_real). This is immediately actionable.
- **TarFlow is NOT viable as a multi-molecule model with naive padding.** Solving this requires a padding-free variable-length architecture.

### 4. Recommended next experiment

**Two paths forward, in priority order:**

**Path A — Per-molecule TarFlow (immediate, low risk):**
Train one TarFlow model per molecule with T = n_real. Expected 94-100% VF based on Config A results. This bypasses the padding problem entirely and provides a strong flow baseline for head-to-head comparison with DDPM (the original project goal).

- Pro: Phase 5 Config A already demonstrates this works (98.2% mean VF).
- Con: 8 separate models, no parameter sharing across molecules.
- Effort: Minimal — the code exists (train_phase5.py Config A), just needs full training runs (longer than 5000 steps) with proper evaluation.

**Path B — Padding-free multi-molecule architecture (research, higher risk):**
Design a variable-length architecture that does not require padding. Options:
1. **Molecule-specific positional encodings** that absorb molecule identity, so the model knows where real atoms end without padding tokens.
2. **Graph neural flow** where the autoregressive structure follows molecular bonds rather than a linear sequence.
3. **Set-based flow** (non-autoregressive) that is inherently permutation-invariant and handles variable sizes via masking rather than padding.
4. **Separate log-det normalization** per sample by actual sequence content, with padding tokens excluded from attention entirely (not just masked in queries but removed from key/value context).

- Pro: Would enable a single multi-molecule model.
- Con: Significant architectural change; may introduce new failure modes.

**Recommendation:** Start with Path A. It answers the original research question (TarFlow vs DDPM) immediately. Path B is a research direction worth exploring but should not block the comparison.

---

## Story Fit

**FITS with correction.** The diagnostic achieved its goal: we now know exactly where TarFlow fails and why.

**What the story got right:**
- TarFlow degrades on molecular data (confirmed)
- Log-det exploitation was the mechanism in hyp_002/hyp_003 (confirmed — but caused by bugs, not architecture)
- The degradation is not fundamental to the flow architecture (confirmed — 98.2% ceiling)

**What the story got wrong:**
- "Shared scale causes log-det exploitation" — WRONG. With correct normalization, shared scale performs comparably to per-dim scale. The exploitation in hyp_002/hyp_003 was caused by (1) T*D normalization bug and (2) causal mask bug.
- "TarFlow is not viable for molecular conformations" — TOO STRONG. TarFlow achieves 94-100% VF per molecule. The statement should be: "TarFlow with naive padding is not viable for multi-molecule models."

**RESEARCH_STORY.md updates needed:**
- Correct the shared scale hypothesis
- Add padding as the identified failure mechanism
- Update the TarFlow viability assessment: viable per-molecule, problematic multi-molecule
- Note the two bugs discovered (normalization, causal mask) and their role in prior failures

---

## Verification

### Phase 3 (6 runs)
- All 6 steps completed to 5000 steps, no NaN events
- Steps A/B VF (89-93%) plausible for clean 9-atom ethanol
- Step C NLL matches Step A despite 90 pp VF drop — confirms NLL-VF decoupling
- Logdet/dof stable (0.087-0.122) across all steps — no exploitation

### Phase 4 (9 runs)
- All 9 configs completed, no NaN events
- T=9 configs (1-3, 9): all 93-96% VF — consistent with Phase 3 Steps A/B
- T=21 no-noise (config 4): 0.9% VF — consistent with Phase 3 Step C
- Ethanol Step E (40.2%) exactly reproduced across sessions
- Padding sweep (T=12, 15): intermediate VF values fall on the linear fit (R^2=0.98)
- Clamping without padding (config 9): 93.4% VF confirms clamping is not intrinsically harmful
- 3 Phase 4 plots reviewed: summary, padding sweep, crossing heatmap

### Phase 5 (16 runs)
- All 16 runs completed, no NaN events
- Aspirin Config A ~ Config B (94.3% vs 93.2%) — expected since aspirin has no padding
- Ethanol Config B (40.2%) exactly matches Phase 3 Step E and Phase 4 config E
- Config A range (94.3-100%) is physically plausible across all molecule sizes
- 2 Phase 5 plots reviewed: summary bars, padding scaling

### Cross-phase consistency
- Phase 3 Step E ethanol (40.2%) = Phase 4 config E (40.2%) = Phase 5 Config B ethanol (40.2%) — three independent reproductions
- Phase 3 Step A ethanol (89.1%) ~ Phase 4 config 1 (93.6%) — close, small difference from shared vs per-dim scale
- Phase 4 linear fit (VF = 95.3% - 96.4% * pad_frac) accurate for ethanol, overestimates for other molecules — molecule-specific effects documented

---

## Bugs Discovered

| Bug | Impact | Fix | Commit |
|-----|--------|-----|--------|
| Causal mask broadcasting (B,T,T) → should be (B,1,T,T) | Incorrect attention pattern | Fixed in train_phase3.py | `34fd7dd` |
| Attention mask + padding z-zeroing | Padding tokens not properly handled | Fixed | `09c565f` |
| PermutationFlip not permuting padding mask | Mask corrupted after permutation flip | Fixed in MetaBlockWithCond + MetaBlockSharedScale | `901d6c5` |
| Logdet normalization T*D vs n_real*D | Shifted NLL equilibrium, enabled exploitation | Fixed (all train scripts use n_real) | `901d6c5` |

**Note on hyp_002/hyp_003:** The causal mask bug (#2 in source comparison — SOS + self-inclusive mask) was present in the original `src/model.py` used by hyp_002 and hyp_003. The Phase 3 code (`src/train_phase3.py`) uses Apple's correct output-shift approach, bypassing this bug entirely. The normalization bug was also present in the original code (normalizing by T*D rather than n_real*D). Both bugs contributed to the exploitation behavior in hyp_002 and the saturation equilibrium in hyp_003.

---

## Files Generated

| Path | Description |
|------|-------------|
| **Phase 1** | |
| `reports/source_comparison.md` | 13 architectural differences, severity assessment |
| **Phase 2** | |
| `src/tarflow_apple.py` | Apple TarFlow implementation |
| `src/train_ladder.py` | Training script for 2D/MNIST/CIFAR-10 |
| `reports/ladder_report.md` | Baseline verification results |
| `results/phase2/level0_2d_gaussian/` | 2D Gaussian results |
| `results/phase2/level1_mnist/` | MNIST results |
| `results/phase2/level2_cifar10/` | CIFAR-10 results (in progress) |
| **Phase 3** | |
| `src/train_phase3.py` | 6-step adaptation ladder (TarFlow1DMol + all MetaBlock variants) |
| `reports/phase3_report.md` | Detailed step-by-step analysis |
| `reports/adaptation_report.md` | Phase 3 summary and diagnostic conclusions |
| `results/phase3/step_{a-f}_*/` | Per-step results (results.json, loss_curve.png, pairwise_dist.png, best.pt) |
| **Phase 4** | |
| `src/train_phase4.py` | 9-config ablation matrix |
| `reports/ablation_report.md` | Full results table, per-factor effects, interaction analysis |
| `results/phase4/config_{1-9}_*/` | Per-config results |
| `results/phase4/summary_all_configs.png` | All 15 configs bar chart |
| `results/phase4/padding_sweep_and_factors.png` | VF vs T + factor effect sizes |
| `results/phase4/crossing_heatmap.png` | T x Scale x Noise interaction |
| **Phase 5** | |
| `src/train_phase5.py` | Best config validation on all 8 molecules |
| `reports/phase5_report.md` | Config A vs Config B across 8 molecules |
| `results/phase5/config_{a,b}_{molecule}/` | Per-molecule results |
| `results/phase5/phase5_summary.png` | Config A vs B bars + VF vs n_real scatter |
| `results/phase5/padding_scaling_all_molecules.png` | Config B VF vs padding fraction |

---

## Summary of Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Architecture ceiling (no padding) | **98.2%** mean VF | Phase 5 Config A |
| Best single-molecule VF | **100%** (naphthalene, benzene) | Phase 5 Config A |
| Worst single-molecule VF | **94.3%** (aspirin) | Phase 5 Config A |
| Multi-molecule padded (T=21) | **20.8%** mean VF | Phase 5 Config B |
| hyp_003 reference | **18.3%** mean VF | Prior experiment |
| Padding effect size | **90 pp** degradation (T=9 → T=21) | Phase 3 Step B→C |
| NLL-VF decoupling | NLL identical, VF drops 90 pp | Phase 3 Steps A/C |
| Noise benefit (padded) | **+39 pp** (with shared scale) | Phase 4 interaction |
| Clamping cost (padded) | **-30 pp** | Phase 3 Step E→F |
| Scale type difference (no padding) | **<1 pp** | Phase 4 T=9 configs |

---

## Conclusion

TarFlow works for molecules. The multi-molecule padding problem is solvable — either by training per-molecule models (immediate, 98% VF) or by designing a padding-free variable-length architecture (future work). The diagnostic ladder has fully resolved the ambiguity that remained after hyp_002 and hyp_003: the failures were implementation bugs and a padding interface problem, not a fundamental architectural limitation.
