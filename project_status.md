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
