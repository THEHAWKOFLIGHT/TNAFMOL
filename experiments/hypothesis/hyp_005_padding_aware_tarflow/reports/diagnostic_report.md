## Diagnostic Report — hyp_005 Padding-Aware TarFlow

**Date:** 2026-03-03
**Branch:** `exp/hyp_005`
**W&B Run:** https://wandb.ai/kaityrusnelson1/tnafmol/runs/3ti6nuqw

---

### Baseline Failure Analysis

**Run:** Config A (causal mask fix applied, no PAD token, no query zeroing, alpha_pos=10.0, noise_sigma=0.05, use_bidir_types=True, log_det_reg_weight=0.0, 500 steps, ethanol)

| Metric | Step 100 | Step 500 |
|--------|----------|----------|
| log_det/dof | 3.44 | 12.97 |
| train loss | -2.42 | -11.84 |
| val loss | -2.86 | -13.25 |
| valid_fraction | — | **0.000** |

**Failure mode: log-det exploitation runaway.** The model achieves valid_fraction=0% because it has learned to maximize log_det/dof (→ 13) rather than fit the data distribution. With alpha_pos=10.0 (effectively unclamped), there is NO gradient opposing the log-det accumulation. 

This is the same failure mode as hyp_002/hyp_003, but now confirmed post-causal-mask-fix. The strictly causal mask fix does NOT prevent log-det exploitation — it was necessary for NLL correctness but not sufficient for stable training.

**Critical architectural difference from und_001 Phase 3:**
und_001's 40.2% VF result used Apple's `MetaBlockSharedScale` architecture (`train_phase3.py`), which:
1. Does NOT use an SOS token — output shift from causal attention of preceding positions
2. Had much smaller channel size (256 hidden), fewer blocks (4)
3. Achieved log_det/dof ≈ 0.088 — no exploitation

Our `src/model.py` uses SOS+causal attention, which creates different gradient dynamics. Even with the strict causal mask, our architecture appears to have steeper log-det gradients.

### Root Cause

**The model is exploiting log-det freedom.** With alpha_pos=10.0 (soft bound only active above s=10), the log_scale output can grow continuously, and the NLL objective's log-det term provides positive gradient for unlimited scale growth. The causal mask fix is necessary but not sufficient — we also need to limit the log-det magnitude.

The underlying mechanism: with padding atoms producing zero positions, the affine transforms for real atoms can achieve low reconstruction error at low NLL by accumulating large log_det contributions. With no regularization, this is the gradient-descent minimum of the NLL objective.

**The und_001 40.2% result is NOT directly applicable** to our architecture because:
- und_001 used Apple's output-shift architecture (different Jacobian structure)
- No SOS token means different attention patterns and log-det gradient dynamics
- Our architecture is the one in `src/model.py` which must remain the reference for hyp_005

### Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY | **Yes** | Log-det stability is a fundamental misconfiguration. alpha_pos=10.0 is wrong — need alpha_pos~1.0 or mild log_det_reg_weight to stabilize. PAD token and query zeroing are the primary research questions but cannot be evaluated while log-det explodes. |
| KNOWN HEURISTICS | Possibly | After log-det is stable, masked LayerNorm or zero PAD embedding may help. |
| SCALE | If needed | Only if SANITY + HEURISTICS fail. |

### Proposed Angles (preliminary)

**SANITY: Stabilize log_det + test padding fixes (2x2 factorial)**

The 4-config ablation from the task spec remains valid, but must be run WITH log-det stability. The SANITY fix is: use alpha_pos=1.0 (bounded expansion per layer) to prevent runaway. This restores the stabilization from hyp_003/hyp_004 which prevented the saturation equilibrium by limiting per-layer expansion.

Key difference from hyp_003: alpha_pos=1.0 (not 0.1). hyp_003 used alpha_pos=0.02, which was too tight. alpha_pos=1.0 limits expansion to exp(1.0)=2.7x per layer — strong enough to prevent runaway while not creating the saturation equilibrium at alpha_pos saturation level.

All 4 SANITY configs will use alpha_pos=1.0. The 2x2 ablation (PAD token × query zeroing) can then measure the padding-specific effects cleanly.

