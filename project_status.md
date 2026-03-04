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
