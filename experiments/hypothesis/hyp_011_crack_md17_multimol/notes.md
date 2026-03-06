## [hyp_011] — Crack MD17 Multi-Molecule TarFlow
**Date:** 2026-03-06 | **Type:** Hypothesis | **Tag:** `hyp_011`

### Motivation
hyp_010 proved the Apple TarFlow architecture (TarFlow1DMol) works for multi-molecule MD17, achieving 71.6% mean VF with all 8 molecules above 50%. The per-molecule ceiling (und_001) is 98.2% mean VF. This leaves a 26.6pp gap. hyp_011 pushes VF as high as possible by scaling model capacity, training budget, and tuning hyperparameters.

### Method
OPTIMIZE with 3 angles:
- **SANITY**: Longer training (50k steps) and larger model (channels=384, blocks=6) — two parallel validation runs
- **HEURISTICS**: W&B sweep over lr, ldr, noise_sigma using best architecture from Phase 1
- **SCALE**: Maximum capacity (channels=512, blocks=8, 100k steps) if needed

Training script: src/train_apple.py (config changes only, no code modifications).

### Results
*(to be filled after experiment)*

### Interpretation
*(to be filled after experiment)*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
