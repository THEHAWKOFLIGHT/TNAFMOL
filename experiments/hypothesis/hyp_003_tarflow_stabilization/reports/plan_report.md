# OPTIMIZE Plan Sub-report — hyp_003 TarFlow Stabilization

**Date:** 2026-03-01
**Status:** READY_TO_START

---

## Diagnostic Summary

Root cause: **alpha_pos saturation collapse**. Model learns to set log_scale ≈ +alpha_pos uniformly across all 8 blocks, giving log_det/dof = 8 × alpha_pos = 0.78. In the inverse (sampling) direction, this contracts generated samples by exp(-0.78) ≈ 0.46× per dof, compressing atoms together. The log_det_reg_weight=0.1 is insufficient to overcome the NLL gradient that pushes log_scale toward its maximum.

**Proposed fix:** The SANITY angle sweeps over (alpha_pos, log_det_reg_weight, lr) to find a combination where the regularization penalty is strong enough to force log_det_per_dof ≈ 0 during training. Key target: log_det/dof < 0.2 AND valid_fraction > 0 on at least one molecule.

---

## Angles

### Angle 1 — SANITY: Sweep alpha_pos and log_det_reg_weight

**What changes:** Tune (alpha_pos, log_det_reg_weight, lr) together via W&B sweep. The diagnostic showed that alpha_pos=0.1 and reg_weight=0.1 are insufficient. We need stronger regularization or tighter clamping.

**Exact modifications:**
- alpha_pos in [0.02, 0.05, 0.1, 0.2] — lower values enforce tighter per-block expansion
- log_det_reg_weight in [0.1, 0.5, 1.0, 5.0, 10.0] — stronger values penalize log_det exploitation
- lr in [1e-4, 3e-4, 1e-3] — learning rate interaction with regularization
- shift_only=False, use_actnorm=False (same as diagnostic)
- augment_train=True, normalize_to_unit_var=True (same as diagnostic)

**Validation run (2000 steps):**
- Config: alpha_pos=0.05, log_det_reg_weight=1.0, lr=3e-4, batch_size=128
- Promising if: valid_fraction > 0.1 on at least 1 molecule OR log_det_per_dof < 0.2

**If promising — Sweep:**
- Method: W&B Bayesian optimization
- Parameters: alpha_pos [0.02, 0.05, 0.1], log_det_reg_weight [0.5, 1.0, 2.0, 5.0, 10.0], lr [1e-4, 3e-4]
- N runs: ~30, run_cap=30
- Success metric: valid_fraction (maximize)
- Run on GPUs 1-4 in parallel (4 sweep agents)

**Full run:**
- Best sweep config, fresh init, 10000+ steps
- Success: valid_fraction >= 0.5 on at least 4/8 molecules

**Justification:** The asymmetric clamp + log-det regularization is the theoretically correct fix for log_det exploitation (Andrade et al. 2024). The diagnostic confirmed the clamp is working (no infinite log_det). The issue is parameter calibration. The SANITY sweep is a direct calibration search.

---

### Angle 2 — KNOWN HEURISTICS: SBG Training Recipe

**Citation:** Tan, H., Tong, A., et al. "Scalable Equilibrium Sampling with Sequential Boltzmann Generators," ICML 2025.

**Why it applies:** SBG trains a normalizing flow for molecular systems and faces the same optimization challenges. Their specific training recipe (AdamW betas=(0.9, 0.95), OneCycleLR with 5% warmup, EMA, larger batch_size=512) provides better gradient dynamics and more stable convergence.

**What changes (one targeted change: optimizer + schedule):**
- AdamW betas=(0.9, 0.95) instead of (0.9, 0.999) — lower beta_2 adapts faster
- OneCycleLR with pct_start=0.05 — better LR schedule for flows
- EMA decay=0.999 for evaluation
- batch_size=512 — larger batches provide more stable gradient estimates
- Best SANITY config for alpha_pos/log_det_reg_weight

**Validation run:** 2000 steps, promising if valid_fraction > 0.1 on any molecule.

**If promising — Sweep:** EMA decay [0.995, 0.999], lr [1e-4, 3e-4, 1e-3], batch_size [256, 512]

**Full run:** 20000+ steps with best HEURISTICS config.

---

### Angle 3 — SCALE (conditional)

**Condition:** Only run if diagnostic shows loss is still decreasing at end of full HEURISTICS run (capacity-limited behavior).

**What changes:** d_model=256, n_blocks=12, 50000 steps.

**Validation run:** Promising if valid_fraction > prior full run at 2000 steps.

**Note:** If collapse is not resolved by SANITY+HEURISTICS, SCALE will not help — the collapse mechanism is loss-landscape, not capacity-limited.

---

## Questions / Concerns

None. Diagnostic clearly identifies the issue as parameter miscalibration (alpha_pos saturation). SANITY sweep directly addresses this.
