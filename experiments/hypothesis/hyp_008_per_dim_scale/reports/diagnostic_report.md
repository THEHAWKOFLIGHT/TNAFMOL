# Diagnostic Report — hyp_008: Per-Dimension Scale

## Baseline Failure Analysis

**Context:** hyp_007 Phase 1 confirmed that single-molecule ethanol with NO padding achieves only 34.8% VF with our model (output-shift, use_output_shift=True), while Apple's TarFlow (und_001) achieves 96.2% VF on the same task. The 61pp gap is NOT caused by padding, output-shift mechanism, or data pipeline — those were validated clean. The gap is architectural.

**Root Cause Identification:** Direct comparison of our model.py (output-shift path) vs Apple's tarflow_apple.py reveals one primary and two secondary differences:

| Aspect | Apple (tarflow_apple.py) | Our model (hyp_007) | Gap |
|--------|--------------------------|---------------------|-----|
| Scale params per atom | 3 (per-dim) | 1 (shared) | PRIMARY |
| d_model | 256 | 128 | SECONDARY |
| Dropout | 0.0 | 0.1 | SECONDARY |
| Positional encoding | Yes (learned, 1e-2 init) | No (disabled in hyp_004) | SECONDARY |

**PRIMARY — Shared scale vs per-dimension scale:**
Apple's out_proj outputs 2D dimensions: xa (D-dim log_scale, one per coordinate) and xb (D-dim shift). Our out_proj outputs 4 values: 3 shifts + 1 scalar log_scale shared across all 3 coordinates.

Molecular conformations are highly anisotropic — bond vibrations along covalent bonds differ dramatically from perpendicular modes. Shared scale forces identical expansion/contraction across all 3 spatial dimensions per atom, preventing the model from representing this anisotropy. This is a fundamental expressivity bottleneck that cannot be overcome by longer training or more data.

**Log-det formula difference (implicit):**
Apple: `logdet = -xa.mean(dim=[1, 2])` where xa is (B, N, D) → logdet per sample is mean of N*D values
Our per_dim_scale: `(log_scale * mask.unsqueeze(-1)).sum(dim=(-1,-2))` — sum over all N*3 values

These produce equivalent results when N_real is the same across samples, but Apple normalizes by N*D (mean over all dims) while we sum. The key change is that we now have 3 independent values rather than 1 replicated 3 times — this is the expressivity fix.

**SECONDARY factors (architecture alignment):**
- d_model=128 → 256: 2x capacity per attention layer
- dropout=0.1 → 0.0: Apple uses no dropout (no regularization needed at this scale)
- use_pos_enc=False → True: positional encoding was disabled in hyp_004 based on SOS-path results; with output-shift, it should be beneficial

## Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY | Yes | Per-dimension scale is a bug/missing-feature fix; architectural alignment (d_model, dropout, pos_enc) is a direct config change. No new loss terms or regularizers. |
| KNOWN HEURISTICS | Conditional | Only if SANITY (Phase 1) hits 70-90% rather than 90%+. Would sweep alpha clamping bounds or n_blocks. |
| SCALE | Conditional | Only if HEURISTICS fails. Phase 3 already increases to d_model=256, n_blocks=8 for multi-mol. |

## Proposed Angles (preliminary)

**SANITY — Three phases:**
- Phase 1: per_dim_scale + architecture alignment on single-molecule ethanol (T=9, no padding). VF >= 90% = PASS.
- Phase 2: Padding re-validation. Ethanol T=9 and T=21. |VF_diff| < 10pp = PASS.
- Phase 3: Multi-molecule OPTIMIZE. All 8 molecules, T=21. VF > 50% on ethanol AND mean VF > 40%.

**The per-dim scale change is the minimum necessary fix. If Phase 1 still falls short of 90%, remaining gaps are explored via the investigation path described in the brief.**
