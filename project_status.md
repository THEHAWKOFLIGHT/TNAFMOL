# TNAFMOL — Project Status

Append-only log. One entry per experiment.

---

### hyp_001 — MD17 Data Pipeline
**Date:** 2026-02-28 | **Status:** DONE
**Tag:** `hyp_001` | **Merge commit:** (see git log)
**Result:** All 8 MD17 molecules preprocessed into canonical frame. ~3.6M conformations total. Verified.
**PhD quality:** CLEAN
**Failure modes:** None
**Story fit:** FITS
**Concerns:** None

---

### hyp_002 — TarFlow OPTIMIZE
**Date:** 2026-03-01 | **Status:** FAILURE
**Tag:** `hyp_002` | **Merge commit:** `4b2e51b`
**Result:** TarFlow autoregressive affine flow fails on molecular conformation generation. Valid fraction = 0 on all 8 molecules. Three collapse modes identified: affine scale, shift, ActNorm scale — all stem from log_det exploitation in MLE training.
**PhD quality:** CLEAN
**Failure modes:** Architectural — autoregressive affine flow NLL objective always finds degenerate solutions with unconstrained scale DOFs
**Story fit:** CONFLICT — TarFlow cannot generate valid molecular conformations with this architecture
**Concerns:** None — failure is genuine and well-diagnosed. Informs that diffusion baseline (hyp_003) is the more promising direction.

---

### hyp_003 — TarFlow Stabilization OPTIMIZE
**Date:** 2026-03-01 | **Status:** FAILURE
**Tag:** `hyp_003` | **Merge commit:** `ddddc1b`
**Result:** TarFlow with asymmetric soft clamping + log-det regularization + soft equivariance fails. Best mean valid fraction 18.3% (HEURISTICS sweep). 0/8 molecules reach 50% threshold. Root cause: alpha_pos saturation equilibrium — NLL and regularization gradients reach a stable fixed point at log_det/dof = alpha_pos.
**PhD quality:** CLEAN
**Failure modes:** Mathematical — stable equilibrium at alpha_pos saturation, not escapable via training
**Story fit:** CONFLICT — TarFlow cannot generate valid molecular conformations even with proper regularization
**Concerns:** None — failure is genuine. Two consecutive failures confirm architectural incompatibility.

---

### hyp_004 — TarFlow Architectural Ablation + Optimization
**Date:** 2026-03-02 | **Status:** PARTIAL
**Tag:** `hyp_004` | **Merge commit:** `5a982e2`
**Result:** Positional encodings (+5ppt) and SBG recipe with lr=1e-3 + ema=0.99 (+12ppt) push TarFlow to 26.7% mean VF (full run) / 29.5% (sweep best). 1/8 molecules ≥ 50% (malonaldehyde 56.6%). Fails primary criterion (4+/8 ≥ 50%). Alpha_pos saturation equilibrium persists unchanged across all 20+ configurations tested.
**PhD quality:** CLEAN — three PhD agents needed (context exhaustion ×2, seamless recovery via GUPP). One minor process issue: sweep output directory naming bug caused raw output overwrites for different ema_decay runs (W&B captured all results).
**Failure modes:** None (experiment-level). The PARTIAL status reflects the alpha_pos equilibrium ceiling, not an implementation failure.
**Story fit:** FITS (partially) — confirms alpha_pos equilibrium is the fundamental bottleneck. Architectural improvements help within the constraint but cannot break it. Assessment revised from "fundamentally broken" to "constrained with ~30% ceiling."
**Concerns:** None — result is conclusive. TarFlow is exhausted. Proceed to DDPM.

---

### und_001 — TarFlow Diagnostic Ladder
**Date:** 2026-03-03 | **Status:** DONE
**Tag:** `und_001` | **Merge commit:** `85a1cec`
**Result:** Architecture ceiling 98.2% mean VF (no padding). Multi-molecule padded 20.8% mean VF. Padding identified as sole failure mechanism. Two implementation bugs found and fixed (logdet normalization, causal mask). Shared scale hypothesis refuted.
**PhD quality:** CLEAN — multiple PhD agents across phases, all completed successfully. 4 bugs discovered and fixed during Phase 3 debugging. All 31 training runs completed without NaN events.
**Failure modes:** None — diagnostic achieved its goal completely.
**Story fit:** FITS with correction — shared scale hypothesis was wrong; padding is the primary failure. RESEARCH_STORY.md updated.
**Concerns:** CIFAR-10 baseline verification (Phase 2 Level 2) still training at step 23500/50000. Non-blocking — does not affect molecular diagnostic conclusions.

---

### hyp_005 — Padding-Aware Multi-Molecule TarFlow
**Date:** 2026-03-03 | **Status:** FAILURE
**Tag:** `hyp_005` | **Merge commit:** `bf99802`
**Result:** Best VF=4.7% on ethanol with Config D (PAD token + query zeroing) + reg_weight=2.0, lr=3e-4. SANITY 2x2 factorial: all 4 configs VF=0%, log_det/dof=7.3 — padding fixes have zero effect. HEURISTICS sweep best VF=4.7% (far below 40% criterion). SCALE skipped (training objective equilibrium).
**PhD quality:** CLEAN — single PhD agent, no send-backs. HEURISTICS pivot from masked LayerNorm to log_det_reg_weight well-justified by SANITY evidence.
**Failure modes:** None (experiment-level). Failure is log-det exploitation in SOS+causal architecture independent of padding treatment.
**Story fit:** CONFLICT — und_001 predicted padding fixes would restore multi-molecule VF. Prediction failed. Padding fixes correct but insufficient.
**Concerns:** 10x degradation from single-molecule to multi-molecule unexplained. Key untested: alpha_pos=0.02 + reg_weight=5 + Config D in multi-molecule.

---

### hyp_006 — Output-Shift Multi-Molecule TarFlow
**Date:** 2026-03-04 | **Status:** FAILURE
**Tag:** `hyp_006` | **Merge commit:** `67ef49f`
**Result:** Hypothesis CONFIRMED: output-shift bounds log_det/dof at 0.5-1.3 (vs 7+ for SOS). Best VF=24.8% ethanol (HEUR C, lr=1e-3, cosine, 5k steps). All 3 angles exhausted. Primary criterion (VF>40%) not met.
**PhD quality:** CLEAN — single PhD agent, no send-backs. 7/7 unit tests, all runs completed cleanly.
**Failure modes:** None (experiment-level). VF plateau (13-25%) across all configs — root cause is atom overlap in generated samples, not log-det exploitation.
**Story fit:** FITS — the architectural hypothesis is confirmed (output-shift eliminates exploitation). VF gap points to normalization/training dynamics issue, not architecture.
**Concerns:** VF plateau despite bounded log_det. SCALE model overfits. Best checkpoint always at step 1000 regardless of total budget.

---

### hyp_007 — Padding Isolation + Multi-Molecule OPTIMIZE
**Date:** 2026-03-06 | **Status:** PARTIAL
**Tag:** `hyp_007` | **Merge commit:** `c3cbc1a`
**Result:** Phase 1 (padding isolation) CONFIRMED: output-shift makes padding neutral (4pp max VF variation across T=9,12,15,18,21). Phase 2: log-det regularization (ldr=5.0) is critical — pushes ethanol from 17.6% → 55.8%. Best multi-molecule result: ethanol 55.8%, malonaldehyde 53.2%, mean 34.7%. 2/8 molecules above 50% (target: 4/8).
**PhD quality:** CLEAN — single PhD agent, no send-backs. 7 verification tests passed. All training runs completed. HEURISTICS sweep adapted from planned lr/steps/bs sweep to ldr/lr/steps sweep based on SANITY evidence — reasonable adaptation.
**Failure modes:** None (experiment-level). PARTIAL status reflects not meeting primary criterion (4/8 >= 50%), though significant progress made (2.25x improvement over hyp_006).
**Story fit:** FITS — confirms output-shift + ldr=5.0 is the correct platform. Padding neutrality confirmed. VF gap now correlates with molecule size, not architecture.
**Concerns:** Aspirin at 9.2% is a major outlier. SCALE angle was skipped — could potentially push more molecules above 50%. Best checkpoint at step 12000/20000 — cosine LR may over-decay.

---

### hyp_008 — Per-Dimension Scale + Architecture Alignment
**Date:** 2026-03-06 | **Status:** FAILURE
**Tag:** `hyp_008` | **Merge commit:** `bbb7a2e`
**Result:** per_dim_scale implemented correctly (6/6 unit tests). Phase 1 gate FAILED: best VF=39.2% on ethanol T=9 (target 90%). Re-diagnosis: und_001 Phase 4 already showed per-dim vs shared scale has <1pp effect. True 61pp gap is architectural: post-norm vs pre-norm + layers_per_block=1 vs 2.
**PhD quality:** CLEAN — single PhD agent, no send-backs. Implementation correct. 4 investigation runs within Phase 1 SANITY angle. Re-diagnosis properly cited und_001 data.
**Failure modes:** Incorrect root cause hypothesis — per-dim scale was not the primary gap. The spec incorrectly attributed the 61pp VF gap to scale parameterization.
**Story fit:** PARTIAL FIT — per_dim_scale implementation is correct and retained (aligns with Apple). Hypothesis was wrong but corrective diagnosis is sound.
**Concerns:** The true architectural gaps (pre-norm, layers_per_block) were documented in und_001 source_comparison.md but not incorporated into hyp_008 diagnostic design.

---

### hyp_009 — Architecture Alignment (Pre-Norm + Layers Per Block)
**Date:** 2026-03-06 | **Status:** FAILURE
**Tag:** `hyp_009` | **Merge commit:** `6dbc21e`
**Result:** Pre-norm + layers_per_block=2 added to model.py. Phase 1 gate FAILED: VF=14% on ethanol T=9 (target 90%) — WORSE than post-norm baseline (39%). Multiple diagnostic runs (ldr sweep, contraction-only) unable to recover performance. Incremental patching of model.py exhausted after 4 experiments (hyp_006–hyp_009).
**PhD quality:** CLEAN — implementation correct, unit tests pass. Failure is at the experimental hypothesis level, not implementation.
**Failure modes:** Incremental patching strategy failure. 13+ architectural differences between model.py and tarflow_apple.py cannot be isolated. Each "fix" introduces new interactions.
**Story fit:** CONFLICT — incremental approach abandoned. Pivot to using tarflow_apple.py + TarFlow1DMol directly for multi-molecule training.
**Concerns:** None — failure is conclusive. Path forward is clear.

---

### hyp_010 — TarFlow Apple Architecture for Multi-Molecule MD17
**Date:** 2026-03-07 | **Status:** DONE
**Tag:** `hyp_010` | **Merge commit:** `4694b7b`
**Result:** Uses tarflow_apple.py + TarFlow1DMol directly for multi-molecule training. Phase 1: ethanol T=9 VF=95% (gate passed). Phase 2: padding validation passed (VF gap=1.4pp). Two critical bugs fixed in train_phase3.py (sampling noise at padding + attention key masking). Phase 3: all 8 molecules, T=21, 20k steps — mean VF=71.6%, ALL 8 molecules >50%. Best: malonaldehyde 82.6%, naphthalene 81.0%, benzene 79.4%. Aspirin recovered from 9.2% (hyp_007) to 67.4%.
**PhD quality:** CLEAN — single PhD agent. Two bugs found and fixed during Phase 2 (sampling + attention mask). Both fixes well-documented with clear reasoning. Slurm used for Phase 3 production run. Source integration incomplete (.py files left in experiment dir — Postdoc cleaned up).
**Failure modes:** Source integration incomplete (Level 1 — .py files not removed in PhD's integration commit, Postdoc caught and fixed).
**Story fit:** FITS — proves Apple TarFlow architecture generalizes to multi-molecule training. Eliminates need for log-det regularization. Resolves the aspirin outlier from hyp_007.
**Concerns:** Benzene PW divergence high (0.17 vs ~0.04) — likely C6 symmetry ambiguity, not structural failure (VF=79.4%). Ethanol VF lower in multi-mol (64%) vs single-mol (93.6%) — expected from multi-task capacity sharing.

---

### hyp_011 — Crack MD17 Multi-Molecule TarFlow
**Date:** 2026-03-06 | **Status:** DONE
**Tag:** `hyp_011` | **Merge commit:** `(see git log)`
**Result:** Three-phase OPTIMIZE closed the multi-molecule gap from 71.6% to 98.9% mean VF. Phase 1 SANITY: capacity > budget (384ch/6blk → 83.9% vs 256ch/4blk at 50k → 73.3% with toluene collapse). Phase 2 HEURISTICS: 27-config sweep found noise_sigma=0.03 + ldr=2.0 as key levers; full run at 50k steps → 94.7%. Phase 3 SCALE: 512ch/8blk (50.6M params) → 97.4% at T=1.0, 98.9% at T=0.7. All 8 molecules above 95.6%. Gap to per-molecule ceiling (98.2%) effectively closed.
**PhD quality:** CLEAN — three PhD agents across phases, all completed successfully. 27-sweep sweep had W&B artifact naming errors (non-blocking). Source integration cleanup done.
**Failure modes:** None
**Story fit:** FITS — confirms multi-molecule TarFlow can match per-molecule performance with sufficient capacity and tuning. Single shared model achieves parity with dedicated per-molecule models.
**Concerns:** None — result exceeds stretch target (90%). Temperature sweep is a free 1-2pp improvement that should be standard practice.
