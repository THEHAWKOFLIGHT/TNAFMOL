## OPTIMIZE Plan Sub-report — hyp_010: TarFlow Apple Multi-Molecule

### Diagnostic Summary

T=21 VF failure (33%) caused by two bugs in `src/train_phase3.py`:
1. Sampling bug: Gaussian noise at padding positions corrupted PermutationFlip autoregressive chain.
2. Attention key masking bug: padding keys masked in permuted space starved first real atom of context.

Both bugs fixed. T=21 VF = 93.6%, T=9 VF = 95%, gap = 1.4pp. Phase 2 PASSED.

Phase 3 proceeds with the corrected implementation on all 8 molecules at T=21.

### Angles

**Phase 3 SANITY: Multi-molecule, T=21, 20k steps**
- What changes: extend from single-molecule (ethanol) to all 8 MD17 molecules in multi-task training. Same architecture and hyperparameters as successful T=21 ethanol run (channels=256, num_blocks=4, layers_per_block=2, lr=5e-4 cosine, batch_size=256).
- This is categorized as SANITY because the code changes are already validated — it's scaling the data scope, not changing the model.
- Validation run: N/A — Phase 3 IS the validation run. 20k steps is the full run per task spec.
- Promising criterion at step 5k: mean VF > 20% across 8 molecules (training signal detectable).
- Success criterion (full 20k): VF > 50% ethanol AND mean VF > 40%.
- Justification: T=21 ethanol already achieves 93.6% with the fixed code. Multi-molecule should be achievable at the target given hyp_007 showed 55.8% ethanol VF with the old (buggy) code + ldr=5.0 regularization. With the Apple architecture (no logdet exploitation), multi-molecule at 20k should exceed the old results.

**Angle 2 — KNOWN HEURISTICS: lr and ldr sweep (conditional)**
- What changes: sweep lr ∈ {3e-4, 5e-4, 1e-3} × log_det_reg_weight ∈ {0.0, 1.0} — 6 configs.
- Literature citation: Tan et al. 2025 (SBG) — lr=1e-3 with schedule. This was the key finding from hyp_007 HEURISTICS sweep (ldr=5.0 + lr=3e-4 → 55.8% ethanol VF vs ldr=0 → 17% VF). However, with the Apple architecture (contraction-in-forward convention), ldr=0 should be stable — no logdet explosion. So the sweep here tests whether any ldr penalty is needed.
- Why it applies: the hyp_007 result showed ldr matters for logdet-exploitation prevention. The Apple convention may eliminate this need, but the sweep confirms it.
- Validation run: 5k steps each config (parallel via Slurm --array).
- If promising: select best config, run 20k full.

**Angle 3 — SCALE: Larger model (conditional)**
- What changes: channels=384, num_blocks=6 (from 256/4) → ~3× parameter count.
- Training: 50k steps via Slurm.
- Only if SANITY and HEURISTICS both fail.
- Promising criterion: mean VF > 30% at step 10k.

### Questions / Concerns

None. Both code fixes are validated. Phase 3 config uses the same hyperparameters as the successful T=21 ethanol run. Multi-molecule task has precedent from hyp_007 (55.8% ethanol VF with older, less-capable architecture).
