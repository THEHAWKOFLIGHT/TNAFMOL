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

---

### 2026-03-01 — hyp_003 synthesis
**Status:** FAILURE | **Failure level:** None
**Branch:** `exp/hyp_003` | **Merge commit:** `ddddc1b` | **Tag:** `hyp_003`

**Experiment:** TarFlow Stabilization OPTIMIZE on MD17. Attempted to fix hyp_002's log_det exploitation with three interventions: asymmetric soft scale clamping (Andrade et al. 2024), log-det regularization lambda*(log_det_per_dof)^2, and soft equivariance via SO(3) augmentation + unit-variance normalization (SBG, Tan et al. 2025).

**PhD execution review:**
- Two PhD agents spawned during execution (PhD #1 context exhausted during HEURISTICS sweep monitoring; PhD #2 completed remaining work cleanly)
- OPTIMIZE protocol followed correctly: diagnostic → SANITY (val→sweep→full) → HEURISTICS (val→sweep→full) → SCALE (skipped with justification)
- Root cause correctly diagnosed: alpha_pos saturation equilibrium (mathematical fixed point)
- All visualizations generated with proper captions (5 canonical plots)
- Logs maintained (process_log.md, experiment_log.md)
- .state.json NOT updated during execution (all steps remained "pending" despite completion) — process discipline violation, noted but not blocking
- Source integration completed (src/visualize_hyp003.py added)
- No hardcoded reference values; global_std loaded from data

**OPTIMIZE angle review:**
1. **SANITY (alpha_pos + reg_weight sweep):** Correct identification that alpha_pos and reg_weight need calibration. Val run (alpha_pos=0.05, reg=1.0): malonaldehyde 11.6% valid — promising criterion met. Sweep (24 runs): best alpha_pos=0.02, reg=5, lr=1e-4 → 17.5% mean VF. Full run (10k steps): 13.1% mean VF — degraded from sweep due to longer training hitting saturation harder. Root cause: log_det/dof locks at exactly alpha_pos from step 100-300 onward, loss plateaus at 0.8689-0.8690. Valid conclusion.
2. **HEURISTICS (SBG recipe, Tan et al. 2025):** Citation verified — SBG is a published normalizing flow training recipe for molecular systems. The technique addresses gradient dynamics (faster adaptation, better convergence). Val run: 16.8% mean VF — promising. Sweep (12 runs): best bs=512, ema=0.999, lr=1e-3 → 18.3% mean VF. Full run (20k steps): 14.3% mean VF — same saturation plateau. SBG recipe improves results slightly but does not break the alpha_pos equilibrium (designed for a different failure mode). Valid conclusion.
3. **SCALE (skipped):** Justified — model saturates by step 150, no underfitting evidence. The collapse is a mathematical equilibrium (NLL gradient = regularization gradient), not capacity-limited. Valid skip.

**Key finding:** The alpha_pos saturation equilibrium is a stable mathematical fixed point, not a local minimum. At log_det/dof = alpha_pos: NLL gradient pushes log_scale higher, regularization gradient pushes toward 0, net gradient ≈ 0. This is architecture-level: the autoregressive affine structure with independent per-step scale parameters cannot escape this attractor under MLE + regularization.

**Molecule-size scaling:** Clear inverse correlation (9-atom molecules ~35% valid, 21-atom aspirin <1%). Each additional atom pair is an independent opportunity for close-collision failure from the ~38% residual compression (exp(-0.48) ≈ 0.62 contraction factor across 8 blocks).

**Postdoc verification:**
- Reviewed final_report.md: technically sound, root cause analysis correct, story validation appropriate
- Reviewed all 5 canonical plots: valid_fraction_comparison, best_results_summary, sweep_comparison, angle_summary, min_pairwise_distance
- Verified pre-merge consistency checks (all passed)
- Confirmed branch follows convention and all 7 commits follow convention

**Source integration:** Completed — src/visualize_hyp003.py added. No .py files in experiment directories.

**Pre-merge checks:**
- [x] No .py files in experiments/
- [x] No __pycache__/
- [x] No TASK_BRIEF.md
- [x] data/ exists with 8 versioned datasets
- [x] results/ folder contains plots and best.pt
- [x] experiment_log.md has hyp_003 entry
- [x] process_log.md has hyp_003 entries with commits section
- [x] synthesis_log.md is append-only
- [x] All commits follow convention
- [x] Branch name follows convention (exp/hyp_003)
- [x] Working tree clean

**PhD execution quality:** CLEAN — no send-backs needed. The failure is an experimental outcome, not an implementation error. Two PhD agents needed due to context exhaustion (Level 2 recovery, seamless).

**W&B runs:**
- SANITY sweep: rccehd8m
- SANITY full: o5naez7a
- HEURISTICS val: o6pnle0k
- HEURISTICS sweep: cmgrp6jo
- HEURISTICS full: 4079op64

**Root cause of failure:** The alpha_pos saturation equilibrium is a deeper version of hyp_002's log_det exploitation. Asymmetric clamping prevents the catastrophic log_det → ∞ failure but replaces it with a stable, bounded saturation at log_det/dof = alpha_pos. No tuning of alpha_pos, reg_weight, LR, or batch_size escapes this equilibrium. Combined with hyp_002, this confirms that TarFlow's autoregressive affine structure is not viable for molecular conformations under MLE training.

---

### 2026-03-02 — hyp_004 synthesis
**Status:** PARTIAL | **Failure level:** None
**Branch:** `exp/hyp_004` | **Merge commit:** `5a982e2` | **Tag:** `hyp_004`

**Experiment:** TarFlow Architectural Ablation + Optimization. Tested three architectural modifications (bidirectional type conditioning, permutation augmentation, positional encodings) on the hyp_003 stabilized baseline, then optimized the best combination with SBG training recipe.

**PhD execution review:**
- Three PhD agents needed (original died after code+configs but before execution; second ran ablation+SANITY+HEURISTICS val; third completed HEURISTICS sweep+full+visualizations+reports). All context transitions via GUPP — .state.json and process_log.md used for cold-start. No work lost.
- OPTIMIZE protocol followed correctly: diagnostic → SANITY (6-config ablation → LR sweep → full) → HEURISTICS (val → 9-run sweep → full) → SCALE (skipped with justification)
- All W&B runs properly tagged and grouped
- Logs maintained (process_log.md, experiment_log.md — both updated)
- No hardcoded reference values; global_std loaded from data
- Source code promoted to src/ in initial implementation (commit 44d4ec9) — no post-experiment integration needed
- One process issue: output directory naming in train.py uses only n_steps+lr (not ema_decay), causing overwrites when multiple ema_decay values share same lr. Only last run's raw output survived per lr. W&B captured all 9 summaries. Documented in sweep_best_practices.md.

**OPTIMIZE angle review:**
1. **SANITY (6-config architectural ablation, 3000 steps each):**
   - D_pos (use_pos_enc=True only): 17.65% mean VF — best by 5ppt over baseline
   - F_bidir_pos: 16.40%, A_baseline: 12.68%, C_perm: 12.60%, B_bidir: 11.80%, E_bidir_perm: 10.92%
   - Key finding: positional encodings help (+5ppt), bidirectional types and permutation augmentation slightly hurt
   - LR sweep (3 runs, D_pos only): lr=5e-5 marginally best at 17.73%. Spread <0.5ppt — no significant LR sensitivity.
   - Full run (10k steps, D_pos, lr=5e-5): 17.48% mean VF. Best checkpoint at step 1000. No improvement from 3k→10k — confirms saturation is not training-budget issue.
   - All 6 configs: loss→0.869, log_det/dof→0.100 by step ~150. Alpha_pos equilibrium confirmed across all architectures.
   - FAILED primary criterion (0/8 molecules ≥ 50%)

2. **HEURISTICS (SBG recipe, Tan et al. 2025, on D_pos config):**
   - Citation verified: SBG recipe published at ICML 2025, designed for normalizing flows on molecular systems.
   - Val run (3000 steps, lr=3e-4, EMA=0.999, bs=512): 17.93% mean VF — promising (>SANITY full 17.48%)
   - 9-run sweep (ema_decay=[0.99,0.999,0.9999] × lr=[1e-4,3e-4,1e-3], bs=512):
     - lr=1e-3 dominates: 29.5% (ema=0.99), 26.1% (ema=0.999), 14.5% (ema=0.9999)
     - lr=1e-4/3e-4: clustered at 15-19%
     - Best: lr=1e-3, ema=0.99 → **29.5% mean VF, ethanol 52.8%** (FIRST MOLECULE >50%)
   - Full run (20k steps, lr=1e-3, ema=0.99): **26.7% mean VF, malonaldehyde 56.6%** (1/8 ≥ 50%)
     - Best checkpoint at step 1000 — same early saturation
     - Different molecule crosses 50% each time (sweep: ethanol; full: malonaldehyde) — stochastic in 500-sample eval
   - FAILED primary criterion (1/8 ≥ 50%, need 4+/8)

3. **SCALE (skipped):**
   - Justified: loss saturates at step ~150 across ALL 6 ablation configs + ALL HEURISTICS configs
   - Not capacity-limited. The alpha_pos equilibrium is a mathematical fixed point.
   - Plan condition ("skip if both SANITY and HEURISTICS saturate by step 150") confirmed.

**Key findings:**
- **Positional encodings are the only beneficial architectural modification.** Bidirectional type conditioning adds no value despite the intuition that full molecular composition knowledge should help. Permutation augmentation hurts — atom ordering in MD17 carries informative structural information (not arbitrary).
- **lr=1e-3 + OneCycleLR + EMA=0.99 is the critical training recipe.** This combination gives +12ppt over the best architectural config alone (+5ppt from pos_enc). The high peak LR allows more aggressive exploration within the alpha_pos constraint. EMA=0.99 (faster tracking) outperforms 0.999 at this training scale.
- **The alpha_pos saturation equilibrium is robust.** All 20+ configurations tested (6 architectures × multiple LRs × multiple EMA decays × multiple training lengths) converge to loss=0.869, log_det/dof=0.100 by step ~150. Best checkpoint always at step 500-1000. The ceiling is ~30% mean VF.
- **First molecule above 50%**: malonaldehyde (56.6% in full run) and ethanol (52.8% in sweep) — both 9-atom molecules. The molecule-size inverse correlation persists: 9-atom ~40-55%, 21-atom aspirin ~6%.

**Postdoc verification:**
- Reviewed final_report.md: technically sound, ablation analysis correct, sweep grid well-chosen
- Verified .state.json: all steps completed/skipped, artifacts listed, machine block populated
- Checked all canonical plots in results/: valid_fraction_comparison.png, experiment_progression.png, min_pairwise_distance.png, loss_curve.png
- Pre-merge consistency checks: all passed (no .py in experiments, no __pycache__, reports present, logs updated, commits follow convention)
- No suspicious values detected — results are internally consistent with hyp_003 findings

**Source integration:** N/A — all new code (BidirectionalTypeEncoder, pos_enc, perm_aug) was promoted to src/ in the initial implementation commit (44d4ec9). No experiment-directory scripts. No Source Integration Directive needed.

**PhD execution quality:** CLEAN — no send-backs needed. Three context exhaustions handled via GUPP without work loss. One minor process issue (output dir naming) logged in sweep_best_practices.md. All 12 commits follow convention.

**W&B runs:**
- Diagnostic: https://wandb.ai/kaityrusnelson1/tnafmol/runs/8s3kfzri
- SANITY full: https://wandb.ai/kaityrusnelson1/tnafmol/runs/k88dxne7
- HEURISTICS val: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ht2xyghi
- HEURISTICS sweep best: https://wandb.ai/kaityrusnelson1/tnafmol/runs/wzsmbdhg
- HEURISTICS full: https://wandb.ai/kaityrusnelson1/tnafmol/runs/z50wvlbl

**Story impact:** The result shifts the narrative from "TarFlow is fundamentally broken" (hyp_003 assessment) to "TarFlow is constrained with a ~30% ceiling." The alpha_pos equilibrium remains the bottleneck, but more performance is accessible within it than previously thought. The experiment plan does not change — DDPM is next (hyp_005) — but the characterization of TarFlow's failure is now more precise.
