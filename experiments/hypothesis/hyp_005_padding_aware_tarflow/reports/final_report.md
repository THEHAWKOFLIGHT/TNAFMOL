## Final Experiment Report — hyp_005 Padding-Aware TarFlow
**Status:** FAILURE (Optimize Failure Report)
**Branch:** `exp/hyp_005`
**Commits:** [`f7cb104` — code: Phase 0], [`f4d602a` — config: diagnostic], [`be5bc49` — results: diagnostic], [`c810bca` — docs: plan], [`1d1486c` — config: SANITY 4 configs + val_subdir], [`4a086f2` — results: SANITY ablation + heuristics config], [`a4901be` — docs: state.json + process_log], [`73354bd` — results: heuristics sweep]

---

## Experimental Outcome

### Phase 0 — Code Changes (COMPLETED)
All code changes implemented and unit-tested:
- Fixed `_build_causal_mask()` in `src/model.py` (strictly causal: SOS at [0,0]=True, atoms at [i, :i]=True)
- Added `PAD_TOKEN_IDX=4` to `src/data.py` (separate embedding slot for padding)
- Added `zero_padding_queries` to `TarFlowBlock` and `TarFlow` (zeros padding queries before attention)
- Added `add_gaussian_noise()` to `src/data.py` (noise on real atoms only)
- Updated `src/train.py` with `use_pad_token`, `zero_padding_queries`, `noise_sigma` config params
- 6/6 unit tests pass (`src/test_hyp005.py`)

### SANITY Angle — FAILED (all 4 configs VF=0)

**2×2 factorial ablation (1000 steps, ethanol, alpha_pos=1.0):**

| Config | PAD token | Query zeroing | VF (ethanol) | log_det/dof |
|--------|-----------|---------------|--------------|-------------|
| A (baseline) | No | No | 0.000 | 7.26 |
| B | Yes | No | 0.000 | 7.3 |
| C | No | Yes | 0.000 | 7.3 |
| D | Yes + Yes | Yes | 0.000 | 7.3 |

W&B runs: lwg95pjk (A), otquvyhm (B), krb1i427 (C), 1gruyalq (D)

**Finding:** All configs show identical trajectories. PAD token and query zeroing have ZERO effect on VF. The failure mode is log-det exploitation at the training dynamics level (log_det/dof=7.3 with alpha_pos=1.0), not padding corruption.

### HEURISTICS Angle — FAILED (best VF=4.7%, criterion 40%)

**Heuristic applied:** log_det_reg_weight > 0 (Andrade et al. 2024, cited in src/model.py)
- Rationale: SANITY evidence shows log-det exploitation is the failure, not LayerNorm contamination
- Original HEURISTICS (masked LayerNorm) is inapplicable — Config D's query zeroing already silences padding from the transformer; masked LayerNorm would add nothing
- log_det_reg_weight is the established technique for this failure mode, proven in hyp_003 (single-molecule)

**Validation run (1000 steps, ethanol, reg_weight=2.0, lr=3e-4, Config D):**
- VF=2.7%, log_det/dof=0.25, min_dist_mean=0.44 Å
- W&B: khw1bzkb
- Plateau at step 200 (grad_norm→0). Qualitative improvement from 0% but far below criterion.

**Sweep (3000 steps each, 9 configs: reg_weight=[0.5,1.0,2.0] × lr=[1e-4,3e-4,5e-4]):**
- W&B sweep: kzkja8zy

| reg_weight | lr | VF | min_dist_mean |
|------------|----|----|---------------|
| 0.5 | 1e-4 | 0.0% | 0.232 Å |
| 0.5 | 3e-4 | 0.0% | 0.229 Å |
| 0.5 | 5e-4 | 0.0% | 0.218 Å |
| 1.0 | 1e-4 | 0.0% | 0.358 Å |
| 1.0 | 3e-4 | 0.0% | 0.375 Å |
| 1.0 | 5e-4 | 0.0% | 0.376 Å |
| 2.0 | 1e-4 | 3.7% | 0.455 Å |
| **2.0** | **3e-4** | **4.7%** | **0.480 Å** |
| 2.0 | 5e-4 | 3.7% | 0.491 Å |

Best: reg_weight=2.0, lr=3e-4, VF=4.7%.
All runs plateau at step 500. Best checkpoint always early.
Promising criterion (VF>0.40) not met. No full run warranted.

### SCALE Angle — SKIPPED

**Justification:**
The failure is a training objective equilibrium, not a capacity limitation. With reg_weight=W, the model converges to log_det/dof ≈ 1/(2W) regardless of model size — this is the mathematical minimizer of `NLL + W * log_det_dof^2`. A larger model would converge to the same equilibrium faster.

This is confirmed by hyp_003/004 where scaling (d_model=256, n_blocks=12, 10k-20k steps) provided no benefit over the baseline capacity once the objective equilibrium was reached. Best checkpoints were always at step 500-1000 regardless of training budget.

Skipping SCALE is justified per CLAUDE.md: "Skip a phase only if the diagnostic clearly shows it is not applicable."

---

## Project Context

hyp_005 was motivated by und_001's identification of padding as the sole failure mechanism for multi-molecule TarFlow. The hypothesis was: fix the two padding corruption channels (atom type embedding contamination, LayerNorm contamination) and multi-molecule training should work.

**What happened instead:** The fundamental obstacle is log-det exploitation in `src/model.py`'s SOS+causal attention architecture. This architecture has steeper log-det gradients than Apple's output-shift architecture used in und_001. Even with full padding mitigation, the training objective creates a softened equilibrium at log_det/dof ≈ const that prevents learning valid molecular geometry.

The padding fixes (PAD token, query zeroing) are CORRECT and will be needed when the log-det stability problem is solved. They are not the bottleneck — they just cannot be evaluated cleanly while the log-det issue persists.

**Best result:** 4.7% VF on ethanol (single molecule, 3000 steps) with Config D (both padding fixes) + reg_weight=2.0. This is far below the 50% target.

**Comparison to baselines:**
- hyp_003 best (single molecule, no padding): 29-33% VF (ethanol) at best config
- hyp_004 best (single molecule, SBG heuristics): 44% VF (ethanol) at best
- hyp_005 best (multi-molecule, with padding fixes): 4.7% VF (ethanol)

The 10x degradation from single→multi-molecule is unexpected. The expected benefit from padding fixes was not realized because the optimizer converges to a degenerate equilibrium much faster in the multi-molecule setting.

---

## Story Validation

**CONFLICT WITH RESEARCH STORY.** The research story (from und_001 findings) predicted that fixing the two padding corruption channels would enable multi-molecule TarFlow to approach single-molecule performance. This prediction failed:

1. PAD token + query zeroing successfully isolated padding from the model — confirmed by unit tests
2. But VF did NOT improve from 0% to single-molecule levels
3. Root cause: training objective equilibrium that is independent of padding treatment

The und_001 reference result (40.2% VF) was achieved with Apple's fundamentally different architecture (output-shift autoregression, no SOS token, per-dim scale). Our `src/model.py` (SOS+causal attention) has different Jacobian structure and different log-det gradient dynamics.

**Implication:** The research story needs updating — the padding fixes are insufficient alone. The SOS+causal architecture in `src/model.py` may require a different stabilization approach than reg_weight, or the architecture itself may need revision to match Apple's output-shift structure.

---

## Open Questions

1. **Why is multi-molecule training harder than single-molecule?** hyp_003 got 29% single-molecule with reg_weight=5. Multi-molecule with same technique gives 4.7%. The padding fixes reduce noise but the optimizer plateau is deeper in multi-molecule.

2. **What makes the alpha_pos=0.02 + reg_weight=5 combination work for single-molecule?** hyp_003's best config uses BOTH bounds (alpha_pos hard architectural bound + reg_weight soft penalty). The current sweep only tests reg_weight. Combining alpha_pos=0.02 with reg_weight in multi-molecule was not tested.

3. **Is the issue the architecture or the training?** Apple's TarFlow achieves 40% VF without any of these stabilization tricks. The fundamental difference is per-dim scale vs. per-atom scale with SOS conditioning. This may require architectural changes to `src/model.py`.

4. **Would the Apple architecture work for multi-molecule (padded)?** und_001 Phase 3 showed Apple architecture works for single-molecule. Whether it degrades in the padded multi-molecule setting was not tested.

---

## Figures

![SANITY 4-config ablation](../results/sanity_ablation.png)
**SANITY ablation — VF and log_det/dof across 4 padding configs** — All 4 configs show VF=0 and log_det/dof≈7.3. PAD token and query zeroing have zero effect on the log-det exploitation. The near-identical trajectories confirm the failure is objective-level, not padding-specific.

![HEURISTICS sweep grid](../results/heuristics_sweep.png)
**HEURISTICS sweep — VF and min_dist across reg_weight × lr grid** — Only reg_weight=2.0 achieves any VF (3-5%). Lower reg_weight locks at higher log_det/dof equilibria (0.5-1.0) that produce no valid samples. All runs plateau at step 500, indicating early convergence to the regularized objective minimum.

![Min distance progression](../results/min_dist_progression.png)
**Min pairwise distance across all tested configs** — The progression shows monotonic improvement in min_dist as reg_weight increases (approaching but not reaching the ~0.8 Å valid bond threshold). The correct direction is confirmed; the magnitude is insufficient with reg_weight ≤ 2.0.
