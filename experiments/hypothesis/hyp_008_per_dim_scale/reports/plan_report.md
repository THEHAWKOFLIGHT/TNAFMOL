# OPTIMIZE Plan Sub-report — hyp_008: Per-Dimension Scale

## Diagnostic Summary

Root cause: shared log_scale (1 scalar per atom) cannot represent anisotropic molecular geometry. Fix: per_dim_scale=True gives 3 independent log_scales per atom (one per coordinate), matching Apple's architecture. Secondary: d_model=128→256, dropout=0.1→0.0, pos_enc disabled→enabled.

## Angles

**Angle 1 — SANITY: per_dim_scale + architecture alignment**

Three-phase execution:

### Phase 1: Per-Molecule Gate (run ASAP)
- What changes: per_dim_scale=True, d_model=256, dropout=0.0, use_pos_enc=True, alpha=10.0, use_bidir_types=False
- Config: molecules=["ethanol"], max_atoms=9, n_steps=5000, lr=5e-4, cosine, batch_size=256
- Device: cuda:8 (test GPU — 5000 steps, ~15 min)
- Validation run (IS Phase 1): 5000 steps
- Promising criterion: VF >= 90%
- If 70-90%: investigate remaining gaps (try n_blocks=8, alpha_pos=100.0, or remove clamping)
- If < 70%: stop, report, deeper investigation needed

### Phase 2: Padding Re-Validation (only if Phase 1 passes)
- Two parallel runs: ethanol T=9 (cuda:8) and T=21 (cuda:9)
- Same config as Phase 1 except max_atoms changes
- Promising criterion: both >= 85%, |VF_diff| < 10pp
- NOTE: runs are short (~15 min each), run on test GPUs, NOT Slurm

### Phase 3: Multi-Molecule OPTIMIZE (only if Phase 2 passes)
- **SANITY angle (20k steps):**
  - Config: molecules=None (all 8), max_atoms=21, per_dim_scale=True, use_bidir_types=True, use_pad_token=True, zero_padding_queries=True
  - d_model=256, n_blocks=8, dropout=0.0, log_det_reg_weight=5.0 (proven in hyp_007)
  - lr=3e-4, cosine, batch_size=128, n_steps=20000
  - Device: cuda:8 (20k steps = ~2 hours, no Slurm needed as this is a test GPU)
  - Promising criterion: VF > 50% on ethanol AND mean VF > 40%

- **HEURISTICS angle (if SANITY promising but <4/8 molecules >= 50%):**
  - Sweep: lr in {1e-4, 3e-4, 5e-4}, log_det_reg_weight in {2.0, 5.0, 10.0}
  - Literature: hyp_007 showed ldr=5.0 critical; sweeping around proven values
  - Sweep via Slurm --array (production run)

- **SCALE angle (if HEURISTICS fails):**
  - d_model=384, n_blocks=12, lr=1e-4, 50k steps
  - Slurm (production run: >500 steps on non-test GPU needed for this scale)

## Questions / Concerns

None — implementation complete, all unit tests pass (forward-inverse consistency, Jacobian log-det, backward compat, output dimensions). Phase 1 is the make-or-break gate.
