# TNAFMOL — Synthesis Log

Append-only. All reviews, corrections, project state, and escalations.

---

### 2026-02-28 — Project initialization
Project initialized from approved spec. RESEARCH_STORY.md authored. Git repo created. First experiment: hyp_001 (data pipeline).

### 2026-02-28 — hyp_001 synthesis
**Status:** DONE | **Failure level:** None
**Branch:** `exp/hyp_001` | **Merge commit:** see `git log --oneline` | **Tag:** `hyp_001`

Data pipeline executed cleanly. All 8 MD17 molecules downloaded, preprocessed into canonical frame, reference statistics computed, verification plots generated. No issues encountered.

Source integration: code already in src/ (data.py, preprocess.py, metrics.py, visualize.py). No .py files in experiment directory. Pre-merge checks all passed.

Environment note: PyTorch was broken on this machine (Windows DLL load failure). Fixed by enabling Windows long paths and reinstalling torch 2.10.0+cu126. Required packages installed: h5py, ase, wandb.

---

### 2026-03-01 — hyp_002 synthesis
**Status:** FAILURE | **Failure level:** None
**Branch:** `exp/hyp_002` | **Tag:** `hyp_002`

**Experiment:** TarFlow (Transformer Autoregressive Normalizing Flow) OPTIMIZE on MD17.

**PhD execution review:**
- TarFlow model implemented correctly in src/model.py: causal masked self-attention, SOS token, alternating direction blocks, ActNorm support
- Training loop in src/train.py with proper W&B logging, cosine schedule, per-molecule evaluation
- OPTIMIZE protocol followed correctly: diagnostic → SANITY → HEURISTICS → SCALE (skip)
- Three distinct collapse modes identified and mechanistically explained
- All visualizations generated with proper captions and annotations
- Logs maintained (process_log.md, experiment_log.md)
- No hardcoded reference values; all ref stats loaded from files
- Source code written directly in src/ — no promotion needed

**OPTIMIZE angle review:**
1. **SANITY (shift_only=True):** Correct identification that affine scale parameters cause log_det exploitation. Shift-only eliminates scale DOF. However, the model collapses to shift≈x → z≈0 (identity mapping). Loss plateaus at 0.919 (Gaussian entropy floor). Best valid_fraction = 22.8% (ethanol) — no improvement over raw Gaussian at T=2σ. Sweep partially completed (2/6 runs, background task failed exit 144), but finding is conclusive — shift collapse is the global optimum for shift-only flows. Valid conclusion.
2. **HEURISTICS (ActNorm, Kingma & Dhariwal 2018):** Citation verified — ActNorm is from GLOW paper, well-established normalizing flow component. The technique addresses the diagnosed failure mode (cumulative scale drift). However, ActNorm introduces a new unconstrained scale DOF that gets exploited: negative log_scale (≈-0.81/layer) gives large forward log_det but cumulative sampling contraction of 0.0013. Valid_fraction = 0 on all molecules. No sweep run — val run was not promising (correct per OPTIMIZE protocol). Valid conclusion.
3. **SCALE (skipped):** Justified — collapse is architectural (log_det exploitation via unconstrained scale DOF), not capacity-limited. Larger model would find the same degenerate solution faster. Valid skip.

**Postdoc verification:**
- Reviewed src/model.py: NLL formula correct, log_det computation correct, inverse sampling correct
- Reviewed all 3 plots: loss curves clearly show the collapse mechanisms, min pairwise distance histograms confirm catastrophic failure (all generated samples at ~0.16 Å), valid fraction comparison shows no angle meets criterion
- Verified .state.json reflects failure status with all steps completed

**Source integration:** N/A — code already in src/. No .py files in experiment directories.

**Pre-merge checks:**
- [x] No .py files in experiments/
- [x] No __pycache__/
- [x] No TASK_BRIEF.md
- [x] data/ exists with 8 versioned datasets
- [x] results/ folder contains plots and best.pt
- [x] experiment_log.md has hyp_002 entry
- [x] process_log.md has hyp_002 entries with commits section
- [x] synthesis_log.md is append-only
- [x] All commits follow convention
- [x] Branch name follows convention (exp/hyp_002)
- [x] Working tree clean

**PhD execution quality:** CLEAN — no send-backs needed. The failure is an experimental outcome, not an implementation error.

**W&B runs:**
- HEURISTICS val: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ras6geue

**Root cause of failure:** The TarFlow architecture with MLE training creates a fundamental tension: any unconstrained affine scale parameter becomes a "log_det pump" that the optimizer exploits to minimize NLL without learning the data distribution. Three collapse modes observed (affine scale, shift, ActNorm scale) all stem from the same root cause. This is a genuine negative result about autoregressive affine flows on molecular coordinate data.
