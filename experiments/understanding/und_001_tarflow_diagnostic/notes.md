## [und_001] — TarFlow Diagnostic Ladder
**Date:** 2026-03-02 | **Type:** Understanding | **Tag:** `und_001`

### Motivation
Two consecutive TarFlow failures (hyp_002: log-det exploitation → 0% VF; hyp_003: alpha_pos saturation equilibrium → 18.3% best VF) demand a systematic investigation. Instead of continuing to patch our implementation, we start from Apple's working TarFlow (arXiv:2412.06329), verify it on a constructive complexity ladder (2D → MNIST → CIFAR-10), then progressively adapt toward molecules to find exactly where and why performance degrades. This identifies whether the failure is architectural (fixable) or fundamental (pivot to DDPM).

### Method
Six-phase diagnostic ladder:
1. **Source Comparison** — systematic diff of Apple TarFlow vs our model.py
2. **Apple Baseline Verification** — constructive ladder: 2D Gaussian → MNIST → CIFAR-10
3. **Adaptation Ladder** — progressively adapt Apple architecture toward molecules (ethanol), one change at a time
4. **Ablation Matrix** — cross the most impactful factors
5. **Best Config Validation** — test best config on all 8 MD17 molecules
6. **Synthesis** — where does it break, why, and what to do next

### Results
*(To be filled after experiment completes)*

### Interpretation
*(To be filled after experiment completes)*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
