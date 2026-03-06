# TNAFMOL — Experiment Log

PhD student-maintained. Append-only record of all experiments.

---

### hyp_001 — MD17 Data Pipeline
**Date:** 2026-02-28
**Branch:** `exp/hyp_001`
**Command:** EXECUTE
**Status:** DONE

Downloaded and preprocessed all 8 MD17 molecules (aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil) into canonical frame representation. Total ~3.6M conformations across all molecules.

**Preprocessing steps:**
1. Download from quantum-machine.org (md17_*.npz format)
2. CoM subtraction (mass-weighted)
3. Kabsch alignment to mean structure per molecule
4. Zero-padding to 21 atoms with attention mask
5. Atom type encoding: H=0, C=1, N=2, O=3
6. Deterministic 80/10/10 split (seed=42)

**Output:** 8 dataset directories in `data/md17_{mol}_v1/`, each with dataset.npz, metadata.json, ref_stats.pt, README.md.

**Verification:** All datasets verified for correct shapes, padding, CoM centering, atom type encoding, and split consistency. Energy and pairwise distance distributions are physically plausible.

---

### hyp_002 — TarFlow OPTIMIZE
**Date:** 2026-03-01
**Branch:** `exp/hyp_002`
**Command:** OPTIMIZE
**Status:** FAILURE (angle budget exhausted)
**Success criterion:** valid_fraction > 0.5 on 5+/8 molecules at T=1.0

**Architecture:** TarFlow (Transformer Autoregressive Normalizing Flow) — 8 alternating forward/reverse autoregressive blocks, d_model=128, n_heads=4, atom type embedding dim=16. SOS token prepended to atom sequence to guarantee valid context for atom 0. Base distribution: N(0,I) over real atom coordinates.

**Angles attempted:**

**Angle 1 — SANITY: shift_only=True (volume-preserving flow)**
- Rationale: diagnostics showed affine blocks chain log_scale to max → z≈0 collapse
- shift_only prevents log_det exploitation; optimal solution = conditional mean predictor
- Results: loss plateau at 0.919 (Gaussian entropy floor), z.std=0.0007 (shift collapse)
- Model learned shift≈x → z≈0 for all inputs. At T=2σ, raw N(0,T²) gives >70% valid on all molecules — model adds nothing above Gaussian baseline
- FAILED: shift collapse is equivalent global minimum of NLL for shift-only

**Angle 2 — HEURISTICS: ActNorm (Kingma & Dhariwal 2018, GLOW)**
- Rationale: ActNorm normalizes per-atom output to N(0,1) with data-dependent init; prevents cumulative scale drift; adds log_det contribution; cites established normalizing flow component
- Re-enabled full affine flow (shift_only=False) + use_actnorm=True
- 5k validation run (lr=3e-4): loss converged to -9.67 (excellent in forward direction)
- Forward pass verified: z ~ N(0,1) properly, total log_det = 428, NLL per dof = -15.38
- Samples: valid_fraction = 0.000 on all 8 molecules. min_pairwise_dist ≈ 0.16 Å (atoms clustered)
- Root cause: model learned negative ActNorm log_scale (≈-0.81 per layer), giving large log_det forward contribution (408 total). In sampling inverse, cumulative contraction ≈ 0.45^8 = 0.0013. Temperature has zero effect on sample diversity (verified T=0.5 to T=50.0)
- FAILED: ActNorm created new collapse mode — same class as affine collapse, different mechanism

**Angle 3 — SCALE: skipped**
- Justification: collapse is architectural, not capacity-limited. Larger model with same unconstrained affine+ActNorm will produce same collapse faster. Skipped as non-applicable.

**Best result:** valid_fraction = 0 on all molecules across all angles. Loss curves show excellent NLL improvement but all due to log_det exploitation rather than data distribution learning.

**Root cause diagnosis:** TarFlow's autoregressive affine coupling objective (maximize log_det) always finds degenerate solutions when any scale DOF is unconstrained. Shift-only eliminates the scale DOF but creates shift collapse. ActNorm adds a new scale DOF that gets exploited. This is a fundamental tension in the architecture design.

**W&B runs:**
- Diagnostic: `hyp_002_diag`
- SANITY val: `hyp_002_sanity_val`
- SANITY val_shift: `hyp_002_sanity_val_val_shift`
- SANITY sweep: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/...
- HEURISTICS val: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ras6geue

---

### hyp_003 — TarFlow Stabilization via Soft Clamping + Soft Equivariance
**Date:** 2026-03-01
**Branch:** `exp/hyp_003`
**Command:** OPTIMIZE
**Status:** FAILURE (angle budget exhausted, primary criterion not met)
**Success criterion:** valid_fraction ≥ 0.5 on ≥ 4/8 molecules

**Architecture:** TarFlow (same as hyp_002) + three stabilization changes:
1. Asymmetric soft scale clamping: _asymmetric_clamp(s, alpha_pos=0.02, alpha_neg=2.0) — replaces tanh, bounds expansion to ≈0.02 per layer
2. Log-det regularization: loss += reg_weight × (log_det_per_dof)² — penalizes deviation from log_det=0
3. Soft equivariance: random SO(3) rotation + CoM noise augmentation during training, unit-variance normalization (global_std=1.2905 Å)

**Diagnostic (500 steps):**
- log_det/dof = 0.78 (stable, not exploding like hyp_002)
- valid_fraction = 0 on all molecules (new collapse mode: alpha_pos saturation)
- Root cause: model sets log_scale ≈ +alpha_pos uniformly across all 8 blocks → log_det/dof = 8 × alpha_pos/π ≈ 0.78 → samples compressed by exp(-0.78 × dof) per atom

**Angles attempted:**

**Angle 1 — SANITY: All three fixes + regularization tuning**
- SANITY val (2000 steps, α=0.05, reg=1.0, lr=3e-4): malonaldehyde 11.6%, ethanol 7.8% — PROMISING
- SANITY sweep (24/30 runs, 3000 steps, Bayesian): best = α=0.02, reg=5, lr=1e-4 → mean VF 17.5%
  - Top results: ethanol 39.3%, malonaldehyde 38.0%, uracil 26.3%, benzene 22.0%
  - W&B sweep: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/rccehd8m
- SANITY full (10000 steps, best sweep config): mean VF 13.1%
  - ethanol 33.0%, malonaldehyde 32.6%, benzene 16.2%, uracil 13.8%
  - Loss flat at 0.8689 from step 300 onward; log_det/dof locked at 0.100; best ckpt at step 500
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/o5naez7a
- FAILED: 0/8 molecules ≥ 50%

**Angle 2 — HEURISTICS: SBG training recipe (Tan et al. 2025)**
- Changes: AdamW betas=(0.9,0.95), OneCycleLR pct_start=0.05, EMA decay=0.999, batch_size=512
- Best SANITY hyperparams (α=0.02, reg=5) retained
- HEURISTICS val (2000 steps, lr=3e-4): mean VF 16.8%, ethanol 41.3%, malonaldehyde 40.7% — PROMISING
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/o6pnle0k
- HEURISTICS sweep (12 runs, 3000 steps): best = bs=512, ema=0.999, lr=1e-3 → mean VF 18.3% (BEST OVERALL)
  - W&B sweep: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/cmgrp6jo
- HEURISTICS full (20000 steps, best sweep config): mean VF 14.3%
  - malonaldehyde 38.0%, ethanol 33.4%, uracil 13.6%, benzene 15.2%
  - Same saturation pattern: loss flat at 0.8689, log_det locked at 0.100, best ckpt at step 2000
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/4079op64
- FAILED: 0/8 molecules ≥ 50%

**Angle 3 — SCALE: SKIPPED**
- Justification: model saturates at step 150 — not capacity-limited. Loss flat from step 150 onward.
  Alpha_pos saturation is a mathematical equilibrium, not a capacity limitation. Larger model would hit the same equilibrium at the same step count.

**Best result:** mean valid fraction 18.3% (HEURISTICS sweep best config at 3000 steps). 0/8 molecules ≥ 50%.

**Root cause diagnosis (deeper):** The NLL gradient pushes log_scale to the upper bound (+alpha_pos) while the log-det regularization gradient pulls toward 0. The two gradients reach a stable fixed point at log_det/dof = alpha_pos regardless of regularization weight, LR, or batch size. The alpha_pos saturation equilibrium is not a local minimum — it is an attractor. No gradient-based optimization can escape it without fundamentally changing the loss formulation.

**Per-molecule pattern:** Clear inverse correlation between molecule size (n_atoms) and valid fraction. 9-atom molecules (ethanol, malonaldehyde): 33-38%. 21-atom molecules (aspirin): <1%. Each additional atom pair is an independent opportunity for close-collision failure due to residual compression.

**Story fit:** CONFLICT — TarFlow is not viable for molecular conformations under standard MLE + regularization. Two consecutive failures confirm the architectural issue. Research story needs to pivot.

---

### hyp_004 — TarFlow Architectural Ablation + Optimization
**Date:** 2026-03-02
**Branch:** `exp/hyp_004`
**Command:** OPTIMIZE
**Hypothesis:** Three architectural modifications (bidirectional type conditioning, permutation augmentation, positional encodings) will improve over hyp_003 baseline (14.3% mean VF).

**Diagnostic:**
- Bidirectional type encoder (BidirectionalTypeEncoder) implemented in src/model.py
- All 3 modifications are independent flags (use_bidir_types, use_perm_aug, use_pos_enc)
- W&B diagnostic run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/8s3kfzri

**Angle 1 — SANITY: 6-config architectural ablation**
- 6 configs tested (3000 steps each): A_baseline, B_bidir, C_perm, D_pos, E_bidir_perm, F_bidir_pos
- Results (mean VF): D_pos 17.65% > F_bidir_pos 16.40% > A_baseline 12.68% > C_perm 12.60% > B_bidir 11.80% > E_bidir_perm 10.92%
- Best: **D_pos** (use_pos_enc=True only) — positional encodings add +5ppt
- Permutation augmentation hurts: atom ordering is informative for MD17. Bidir types slightly hurts.
- ALL configs: loss saturates at 0.869, log_det/dof=0.100 by step ~150 (alpha_pos equilibrium)
- LR sweep (D_pos only): lr=5e-5 marginally best at 17.73% (spread <0.5ppt). Best config: D_pos, lr=5e-5, bs=128.
- SANITY full run (D_pos, 10000 steps, lr=5e-5): mean VF = 17.48% (no improvement over 3000 steps)
  - Best checkpoint: step 1000 (val_loss=0.8176). W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/k88dxne7
  - ethanol 44.2%, malonaldehyde 39.8%, benzene 22.2%, uracil 18.4%, toluene 7.4%, salicylic_acid 5.6%, naphthalene 1.8%, aspirin 0.4%
- FAILED primary criterion (no molecule ≥ 50%)

**Angle 3 — SCALE: SKIPPED**
- Justification: loss saturates at step ~150 across ALL 6 ablation configs — confirmed alpha_pos equilibrium.
  Not capacity-limited. Increasing model size would reach same equilibrium faster.

**Angle 2 — HEURISTICS: SBG training recipe (Tan et al. 2025, ICML)**
- SBG recipe: AdamW betas=(0.9,0.95), OneCycleLR, EMA decay=0.999, batch_size=512
- Applied to D_pos config (best architectural variant)
- HEURISTICS val (3000 steps, lr=3e-4): mean VF 17.93% (+0.45ppt vs SANITY full) — PROMISING
  - Better val_loss: 0.8116 at step 1500 (vs 0.8176 for SANITY full)
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/ht2xyghi
- HEURISTICS sweep (9 runs: ema_decay=[0.99,0.999,0.9999] × lr=[1e-4,3e-4,1e-3], bs=512 fixed):
  - lr=1e-4, ema=0.99: 19.2% | lr=1e-4, ema=0.999: 18.1% | lr=1e-4, ema=0.9999: 15.5%
  - lr=3e-4, ema=0.99: 19.1% | lr=3e-4, ema=0.999: 17.9% | lr=3e-4, ema=0.9999: 15.7%
  - **lr=1e-3, ema=0.99: 29.5%** | lr=1e-3, ema=0.999: 26.1% | lr=1e-3, ema=0.9999: 14.5%
  - Best: lr=1e-3, ema_decay=0.99 → mean VF 29.5%, ethanol 52.8% (FIRST TIME ANY MOLECULE > 50%)
  - W&B best: https://wandb.ai/kaityrusnelson1/tnafmol/runs/wzsmbdhg
  - Key finding: lr=1e-3 (OneCycleLR peak) dominates (+10ppt over lr=1e-4). ema=0.99 optimal for 3000-step runs.
  - Note: output dir naming (n_steps+lr only) caused raw output overwrite for different ema_decay runs.
    W&B captured all 9 run summaries. See sweep_best_practices.md.
- HEURISTICS full run (20000 steps, lr=1e-3, ema_decay=0.99, D_pos, fresh init):
  - Mean VF: **26.7%**, best_val_loss=0.8188 at step 1000 (same early saturation)
  - aspirin 6.6%, benzene 26.6%, ethanol 40.0%, **malonaldehyde 56.6%**, naphthalene 7.8%,
    salicylic_acid 17.8%, toluene 14.6%, uracil 43.6%
  - 1/8 molecules ≥ 50% (malonaldehyde) — FAILS primary criterion (4+/8 required)
  - W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/z50wvlbl

**Best result:** mean VF 29.5% (HEURISTICS sweep best config, 3000 steps). Full run: 26.7%.
  First molecule ever to exceed 50%: ethanol 52.8% (sweep), malonaldehyde 56.6% (full run).
  1/8 molecules ≥ 50% — primary criterion (4+/8) not met.

**Pattern consistent with hyp_003 but improved:** alpha_pos saturation equilibrium persists (loss→0.869,
  log_det/dof→0.100 by step 150). Architectural improvements (pos_enc +5ppt) and training recipe
  (SBG + lr=1e-3 + ema=0.99: +12ppt over SANITY baseline) push performance within the constrained regime.
  The lr=1e-3 + ema=0.99 combination is a key new finding: provides 10× more improvement than architecture alone.
## und_001 — TarFlow Diagnostic Ladder (Phase 3)

**Date:** 2026-03-02
**Branch:** `exp/und_001`
**Type:** Understanding
**Command:** DIAGNOSE

**Goal:** Identify which architectural adaptation from Apple TarFlow to molecular TarFlow causes
performance degradation. Run 6 incremental steps (A-F), 5000 steps each on ethanol (MD17, 9 atoms).

**Steps:**
- **A** (Baseline): Pure Apple TarFlow1D, raw coords, 9 atoms — valid_fraction=**89.1%**, loss=-2.802, logdet/dof=0.122
- **B** (+ Atom type cond): nn.Embedding(4,16) concat — valid_fraction=**92.9%**, loss=-2.772, logdet/dof=0.121
- **C** (+ Padding T=21): 9 real + 12 pad atoms, causal+pad mask — valid_fraction=**2.7%**, loss=-2.801, logdet/dof=0.122
- **D** (+ Noise aug): Gaussian noise sigma=0.05 on real atoms — valid_fraction=**14.3%**, loss=-1.867, logdet/dof=0.088
- **E** (Shared scale): 1 scalar/atom×3 coords (KEY TEST) — valid_fraction=**40.2%**, loss=-1.864, logdet/dof=0.088
- **F** (+ Stabilization): asymmetric clamp alpha_pos=0.1 + log-det reg 0.01 — valid_fraction=**10.4%**, loss=-1.857, logdet/dof=0.087

**Primary finding:** Padding is the failure point (Step C: 89.1% → 2.7%). Shared scale did NOT cause
saturation when normalization was correct. The hyp_002/hyp_003 saturation was caused by two bugs:
(1) self-inclusive causal mask, (2) T*D normalization instead of n_real*D.

**Bugs fixed during Phase 3 debugging:**
1. Attention mask shape: `(B,T,T)` → `(B,1,T,T)` for multi-head broadcast (commit `34fd7dd`)
2. Permutation-aware padding mask: PermutationFlip blocks must receive the flipped mask (commit `901d6c5`)
3. Logdet normalization: T*D → n_real*D to match z² NLL normalization (commit `901d6c5`)

**W&B runs:** https://wandb.ai/kaityrusnelson1/tnafmol (group: `und_001`, prefix: `und_001_phase3_step_*`)

**Story fit:** UPDATES STORY. The primary failure mechanism is different from the original hypothesis.
Padding (not shared scale) causes VF collapse. The correct architecture direction is: reduce or eliminate
padding by working at T=n_real, or use a different approach that doesn't suffer from degenerate
zero-padding tokens in the autoregressive sequence.

---

### und_001 Phase 4 — Ablation Matrix
**Date:** 2026-03-02
**Branch:** `exp/und_001`
**Command:** DIAGNOSE
**Status:** DONE

9 new configs crossing the most impactful Phase 3 factors in a systematic ablation matrix.
All configs: ethanol, 5000 steps, batch_size=256, lr=5e-4, cosine, grad_clip=1.0, seed=42.
Configs run in parallel (2 at a time) on GPUs 5 and 6 (physical CUDA devices).

**Results:**

| Config | Descriptor | Valid % |
|--------|-----------|---------|
| 1 | T=9, no-noise, shared | **93.6%** |
| 2 | T=9, noise, per-dim | **96.2%** |
| 3 | T=9, noise, shared | **95.3%** |
| 4 | T=21, no-noise, shared | 0.9% |
| 5 | T=12 (3 pad), noise, shared | **69.6%** |
| 6 | T=15 (6 pad), noise, shared | **50.4%** |
| 7 | T=21, noise, shared + perm-aug | 2.1% |
| 8 | T=21, noise, shared + SO(3)-aug | 34.8% |
| 9 | T=9, noise, shared + clamp | **93.4%** |

**Key findings:**
1. Best achievable VF: **96.2%** (config 2: T=9, noise, per-dim). All T=9 configs = 93-96%.
2. Padding degrades VF smoothly: ~−96 pp per unit pad_fraction (T=9: 95.3% → T=12: 69.6% → T=15: 50.4% → T=21: 40.2%). NOT a cliff.
3. Permutation augmentation is catastrophically harmful (2.1%) — architecturally incompatible with causal autoregressive flows.
4. SO(3) augmentation modestly harmful in padded regime (−5.4 pp vs baseline).
5. Clamping is neutral without padding (−1.9 pp), harmful only with padding (−29.8 pp with T=21).
6. Noise × shared scale interaction: large positive synergy in padded regime (+39.3 pp joint vs +11.6 pp noise alone).
7. No intervention tested recovers padding-induced VF loss at T=21.

**Architectural conclusion:** The padding gradient imbalance in the log-det objective is the root cause.
An architectural fix is needed (e.g., per-sample n_real normalization, or abandoning the padded sequence format).

**W&B runs:** https://wandb.ai/kaityrusnelson1/tnafmol (group: `und_001`, prefix: `und_001_phase4_config*`)

**Story fit:** CONFIRMS Phase 3 finding. Padding is the primary obstacle. No augmentation or regularization
strategy tested can overcome the fundamental padding gradient imbalance.

---

## Phase 5 — Best Config Validation on All 8 MD17 Molecules

**Date:** 2026-03-02
**Branch:** `exp/und_001`
**Status:** DONE
**Type:** Understanding / DIAGNOSE

**Purpose:** Extend the two best configs from Phase 4 to all 8 MD17 molecules to measure:
(A) architecture ceiling when padding is removed (T=n_real), and
(B) practical multi-molecule performance at T=21 with the best-known padded config.

**Configs run:**
- Config A: T=n_real (no padding), per-dim scale, noise=0.05, atom type embedding, 5000 steps
- Config B: T=21 (padded), shared scale, noise=0.05, atom type embedding, 5000 steps
- 8 molecules × 2 configs = 16 total runs on GPUs 5 (Config A) and 6 (Config B) in parallel

**Results:**
| Molecule | n_real | Config A VF | Config B VF | pad_frac_B |
|----------|--------|-------------|-------------|------------|
| aspirin | 21 | 94.3% | 93.2% | 0.000 |
| naphthalene | 18 | 100.0% | 0.0% | 0.143 |
| salicylic_acid | 16 | 97.8% | 8.1% | 0.238 |
| toluene | 15 | 98.7% | 0.0% | 0.286 |
| benzene | 12 | 100.0% | 2.9% | 0.429 |
| uracil | 12 | 99.2% | 6.9% | 0.429 |
| ethanol | 9 | 96.2% | 40.2% | 0.571 |
| malonaldehyde | 9 | 99.8% | 15.4% | 0.571 |
| **Mean** | — | **98.2%** | **20.8%** | — |

**Key findings:**
- Config A (no padding): 98.2% mean VF across all 8 molecules — architecture ceiling is very high
- Config B (padded, T=21): 20.8% mean VF — beats hyp_003 best (18.3%) with noise + shared scale
- Aspirin Config A ≈ Config B (94.3% vs 93.2%) — expected, aspirin has n_real=21 = T, no padding
- Phase 4 linear model (VF = 95.3% - 96.4%×pad_frac) overestimates for non-ethanol molecules
- Naphthalene and toluene collapse to 0% despite small padding fractions (14%, 29%)

**Plausibility checks:** All pass. Ethanol Config B = 40.2% matches Phase 3 Step E exactly (reproducible). No NaN events.

**W&B runs:** https://wandb.ai/kaityrusnelson1/tnafmol (group: `und_001`, prefix: `und_001_phase5_config*`)
16 runs: 8 Config A (configA_*) + 8 Config B (configB_*) under group und_001

**Story fit:** FITS — confirms padding as primary failure mode. Architecture itself is sound (98.2% ceiling). Config B improvement over hyp_003 is consistent with Phase 3/4 established best practices.

---

## hyp_005 — Padding-Aware Multi-Molecule TarFlow
**Date:** 2026-03-03 | **Branch:** `exp/hyp_005` | **Status:** FAILURE

**Hypothesis:** Fix the two padding corruption channels (PAD token embedding, query zeroing for LayerNorm contamination) to enable multi-molecule TarFlow matching single-molecule performance.

**Code changes:** Fixed causal mask bug (src/model.py), PAD_TOKEN_IDX=4 (src/data.py), query zeroing in TarFlowBlock, Gaussian noise, config integration in src/train.py. 6/6 unit tests pass.

**SANITY angle (4 configs × 1000 steps, ethanol, alpha_pos=1.0):**
| Config | PAD token | Query zeroing | VF | log_det/dof |
|--------|-----------|---------------|-----|-------------|
| A (baseline) | No | No | 0.0% | 7.26 |
| B (PAD token) | Yes | No | 0.0% | 7.3 |
| C (query zero) | No | Yes | 0.0% | 7.3 |
| D (both) | Yes | Yes | 0.0% | 7.3 |
Finding: PAD token and query zeroing have ZERO effect on log-det exploitation. Failure mode is training dynamics, not padding corruption.

**HEURISTICS angle (log_det_reg_weight sweep, 9 configs × 3000 steps):**
- Changed from planned masked LayerNorm to log_det_reg_weight (Andrade et al. 2024) — masked LayerNorm inapplicable since Config D's query zeroing already silences padding from transformer
- Best: reg_weight=2.0, lr=3e-4 → VF=4.7% on ethanol (criterion: >40%). FAILS.
- Pattern: all runs plateau at step 500. Equilibrium log_det/dof ≈ 1/(2*reg_weight).

**SCALE angle:** Skipped — failure is training objective equilibrium, not capacity. Confirmed by hyp_003/004 precedent.

**Best result:** VF=4.7% on ethanol (3000 steps). Target: VF≥50% on ≥4/8 molecules. FAILURE.

**Story conflict:** Padding fixes are correct but insufficient. The actual bottleneck is the SOS+causal architecture's steeper log-det gradient dynamics vs. Apple's output-shift architecture. 10x degradation from single-molecule (44% hyp_004) to multi-molecule (4.7%) unexplained.

**W&B sweep:** https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/kzkja8zy
**W&B val:** lwg95pjk (Config A SANITY), khw1bzkb (HEURISTICS val)

## hyp_006 — Output-Shift TarFlow
**Date:** 2026-03-04 | **Branch:** `exp/hyp_006` | **Status:** FAILURE (angle budget exhausted; primary criterion not met; hypothesis CONFIRMED)

**Hypothesis:** Apple's output-shift mechanism (Zhai et al. 2024) eliminates the log-det exploitation pathway from hyp_005. By using output-shift + self-inclusive causal mask instead of SOS + strictly-causal mask, the model cannot exploit log-det even with alpha_pos=10.0 and no regularization.

**Implementation:** Added `use_output_shift: bool = False` flag to TarFlowBlock and TarFlow. New `_run_transformer_output_shift()` method with self-inclusive causal mask. Output shift: `params = cat([zeros_like(params[:,:1]), params[:,:-1]], dim=1)`. Zero-init both weight and bias of out_proj. 7/7 unit tests pass. All src/ modifications — no experiment-level .py files.

**Diagnostic (hypothesis test, 500 steps, ethanol, alpha_pos=10.0):**
- log_det/dof at step 500: **0.516**
- SOS model would reach 7+ at same step
- **HYPOTHESIS CONFIRMED: output-shift eliminates exploitation pathway**

**SANITY angle (1000 steps, all 8 molecules):**
- Ethanol VF: 13.4% (criterion: >40%). Not met.
- Mean VF: 13.8%. log_det/dof: 0.5-0.6 (stable, no exploitation)
- alpha_pos=1.0 fallback: nearly identical (13.2%)
- Assessment: PROMISING — model learning correctly, insufficient training budget

**HEURISTICS angle (SBG recipe — Tan et al. 2025):**
- Val (lr=1e-3 OneCycleLR, 3k): ethanol VF=15.0%, mean=13.2%
- Sweep A (lr=3e-4 cosine, 5k): ethanol VF=17.0%, mean=16.3%
- Sweep B (lr=5e-4 cosine, 5k): ethanol VF=19.8%, mean=15.1%
- Sweep C (lr=1e-3 cosine, 5k): ethanol VF=24.8%, mean=16.3% **(best overall)**
- VF plateau around 15-25% regardless of lr. Criterion not met.

**SCALE angle (d_model=256, n_blocks=12, 9.6M params, 5k steps):**
- Ethanol VF: 16.2%, mean=13.7%
- Best val loss at step 1000 (1.1675), diverges after
- Larger model overfits — does NOT help vs HEURISTICS
- Promising criterion (>25% ethanol) NOT MET

**Best result:** HEURISTICS C — ethanol VF=24.8%, mean VF=16.3%. Target: VF≥50% on ≥4/8 molecules. **FAILURE.**

**Key finding:** Output-shift architecture is correct and stable. The VF plateau (~25% ceiling) is a remaining open problem — likely caused by overlap (mean min_dist 0.45-0.65 Å vs 0.8 Å threshold) and/or normalization mismatch across molecules.

**W&B runs:** diagnostic=1yd68tmf, sanity=p6voeuas, sanity_alpha1=70775xvm, heur_val=6dn3s9fa, scale_val=paxf84nt

---

## hyp_007 — Padding Isolation + Multi-Molecule OPTIMIZE
**Date:** 2026-03-05 → 2026-03-06
**Branch:** `exp/hyp_007`
**Config:** output-shift TarFlow, n_blocks=8, d_model=128, 8 MD17 molecules, ldr=5.0, lr=3e-4
**Status:** DONE — both success criteria met after HEURISTICS angle

### Phase 1 — Padding Isolation Gate
Ethanol-only TarFlow at max_atoms ∈ {9,12,15,18,21}, 5000 steps each. Verified output-shift makes padding neutral.
- T=9: 34.8%, T=12: 35.2%, T=15: 33.0%, T=18: 31.2%, T=21: 34.8%
- Max drop 4.0pp across all sizes. GATE PASSED.

### Phase 2 SANITY — FAILED
20k steps, all 8 molecules, ldr=0.0. Log-det exploitation detected (log_det/dof → 1.2+). Val loss rose monotonically from step 1000.
- lr=1e-3: ethanol VF=17.6%, mean VF=13.9%
- lr=3e-4: ethanol VF=12.2%, mean VF=11.1%
Root cause: ldr=0 allows log-det exploitation in multi-molecule training.

### Phase 2 HEURISTICS — SUCCESS
Sweep: ldr ∈ {1.0, 5.0} × lr ∈ {1e-3, 3e-4} × n_steps ∈ {20k, 50k}. 6 runs completed.
- Best: ldr=5.0, lr=3e-4, 20k steps → ethanol VF=55.8%, mean VF=34.7%
- ldr=5.0 is critical: all ldr=5.0 runs >50% ethanol VF; ldr=1.0 stays <41%

### Full Run (Best Config)
- Best checkpoint: step 12000/20000, val_loss=1.1902
- Ethanol: 55.8% | Malonaldehyde: 53.2% | Benzene: 42.8% | Uracil: 39.4%
- Toluene: 29.8% | Salicylic_acid: 24.6% | Naphthalene: 22.4% | Aspirin: 9.2%
- Mean VF: 34.7% | Mols >50%: 2/8

### Criteria Assessment
- Ethanol VF = 55.8% > 40% ✓
- Mean VF = 34.7% > 30% ✓
Both criteria met. SCALE skipped.

**Key finding:** ldr=5.0 (same as hyp_003 single-molecule) resolves multi-molecule log-det exploitation. Aspirin (21 atoms, 9.2% VF) is the main outlier — model capacity may be insufficient for the largest molecule in a shared-model multi-molecule setup.

**W&B full run:** https://wandb.ai/kaityrusnelson1/tnafmol/runs/2r296jrf

---

## hyp_008 — Per-Dimension Scale + Architecture Alignment
**Date:** 2026-03-05 | **Branch:** `exp/hyp_008` | **Status:** FAILURE (primary criterion not met)

**Hypothesis:** The 61pp VF gap between model.py and tarflow_apple.py is caused primarily by
shared log_scale (1 scalar per atom) vs per-dimension scale (3 independent scalars per atom).
Fix: add `per_dim_scale=True` flag to TarFlowBlock/TarFlow to match Apple's architecture.

**Implementation:** Added `per_dim_scale` parameter to `TarFlowBlock` (out_proj 4→6 dims,
log_det sum over 3 dims per atom) and `TarFlow` (propagated to all blocks). Added to
`DEFAULT_CONFIG` in train.py. Backward compatible (False by default). All 6 unit tests pass.

**Phase 1 Investigation (single-molecule ethanol, T=9, no padding, cuda:8):**

| Config | n_blocks | ldr | VF | log_det/dof | W&B |
|--------|----------|-----|----|-------------|-----|
| 4b, ldr=0 | 4 | 0.0 | 27.2% | 1.4+ (exploded) | mpx5bh9g |
| 4b, ldr=5 | 4 | 5.0 | 27.4% | 0.09 (stable) | nn0weqoy |
| 8b, ldr=0 | 8 | 0.0 | 39.2% (step 500) | collapsed | pwdbuaf0 |
| 8b, ldr=5 | 8 | 5.0 | 29.0% | 0.09 (stable) | sr581ia3 |

**Phase 1 gate: VF >= 90%. Best achieved: 39.2% (8 blocks, ldr=0, step 500) — collapsed thereafter. FAILED.**

**Re-diagnosis from und_001:**
und_001 Phase 4 already measured per-dim vs shared scale on tarflow_apple.py at T=9:
- per-dim: 96.2% VF | shared: 95.3% VF — **<1pp difference**

The original diagnostic hypothesis was wrong. Per_dim_scale is not the root cause of the
61pp gap. The true architectural differences (model.py vs tarflow_apple.py) are:
- Post-norm vs pre-norm (MEDIUM-HIGH impact per source_comparison.md §8b)
- 1 layer/block vs 2 layers/block (MEDIUM-HIGH per §8a)

Our model.py with per_dim_scale + 8 blocks reaches ~39% VF — almost exactly matching
adaptation ladder Step E (tarflow_apple.py, shared scale, T=21+noise: 40.2% VF). The per_dim_scale
change is absorbed by the remaining architectural gap.

**Phases 2 and 3:** Skipped — Phase 1 never passed.

**Code artifacts:** per_dim_scale implementation committed to src/model.py + src/train.py.
These changes are correct and retained — they align model.py with Apple's scale parameterization
even if they don't resolve the VF gap alone.

**Story fit:** UPDATES STORY. The root cause is architectural (pre-norm, layers_per_block),
not scale parameterization. Next experiment should implement pre-norm + layers_per_block in
model.py and retest single-molecule VF.

---

## hyp_009 — Pre-Norm + Layers Per Block Architectural Alignment
**Date:** 2026-03-05 | **Branch:** `exp/hyp_009` | **Status:** FAILURE (primary criterion not met)

**Hypothesis:** The 56pp VF gap between model.py (TarFlow) and tarflow_apple.py (TarFlow1DMol) is caused
by: (1) post-norm vs pre-norm in attention/FFN layers, (2) 1 attention+FFN layer per block vs 2 layers per
block, (3) dropout=0.1 vs 0.0. Implementing all three should close the gap to within 5pp.

**Implementation:** Added `use_pre_norm` (bool, default False) and `layers_per_block` (int, default 1)
to TarFlowBlock and TarFlow. nn.ModuleList of layers per block. Pre-norm path: LayerNorm before each
sublayer, final_norm after all sublayers. Backward compatible defaults. All 6 unit tests passed.

**Phase 1 Investigation (single-molecule ethanol, T=9, no padding, cuda:8/9):**

| Config | ldr | VF | log_det/dof | W&B |
|--------|-----|----|-------------|-----|
| pre-norm + lpb=2, ldr=0 | 0.0 | 14.4% | 1.5+ exploded | — |
| pre-norm + lpb=2, ldr=5 | 5.0 | 28.4% | 0.09 stable | — |
| post-norm + lpb=1, ldr=5 (baseline) | 5.0 | 29.8% | 0.09 stable | — |
| pre-norm + lpb=2, ldr=1.0 | 1.0 | 34.0% | partial exploitation | — |
| contraction-only (alpha_pos=0.001, ldr=0) | 0.0 | TBD | — | — |

**Phase 1 gate: VF >= 90%. Best achieved: 34.0% (pre-norm, ldr=1.0). FAILED.**

**Key Finding:**
Pre-norm vs post-norm: 1.4pp difference (29.8% vs 28.4%). Not the bottleneck.
Root cause reanalyzed: affine convention. Our convention (y = exp(log_scale)*x + shift) expands
in forward pass — logdet positive → reduces NLL. Apple's convention (z = exp(-xa)*(x-xb)) contracts
in forward → logdet negative → increases NLL. This means our model can trivially reduce loss by
expanding log_scale, independent of learning the distribution.

With ldr=5.0: logdet is over-regularized → model learns nothing useful beyond ~28-34% VF.
With ldr=0.0: model exploits log_det to explosion.
There is no working middle ground with the current affine convention.

**Decision:** Escalate to Postdoc. Phase 1 failed. True root cause is affine convention, not normalization.
Next step: use tarflow_apple.py (TarFlow1DMol) end-to-end — it has the correct contraction convention.

**Story fit:** UPDATES STORY. The architectural alignment of model.py is insufficient. The contraction
vs expansion convention is the dominant factor. Switch to train_phase3.py/tarflow_apple.py directly
for future multi-molecule work (hyp_010).

---

## hyp_010 — TarFlow Apple Multi-Molecule (in progress)
**Date:** 2026-03-05 | **Branch:** `exp/hyp_010` | **Status:** IN_PROGRESS

**Hypothesis:** TarFlow1DMol from tarflow_apple.py (correct contraction convention) achieves >= 90%
VF on single molecules. With the correct padding implementation (seq_length=21, use_padding_mask=True),
it can generalize to multi-molecule training with mean VF > 40%.

**Phase 1 — Ethanol T=9 Sanity Gate:** PASSED
- VF = 95% on ethanol T=9, 5k steps. Reproduces und_001 Phase 4 (96.2%).
- Key fix: use FINAL checkpoint (not val-loss-best) — logdet exploitation makes val_loss diverge,
  but final checkpoint still generates from the correct distribution.

**Phase 2 — T=21 Padding Validation:** PASSED
- Two bugs found and fixed in src/train_phase3.py:

  Bug 1 (sampling): z = randn(n, T=21, 3) filled ALL positions including padding with Gaussian noise.
  In PermutationFlip blocks, padding positions 9-20 appear first (0-11 in permuted space), corrupting
  the autoregressive chain. Fix: zero padding positions in z before reverse(), re-zero between blocks.
  Result: VF improved from 33% to 47.2%.

  Bug 2 (attention): MetaBlockWithCond applied padding KEY masking on top of causal mask. In PermutationFlip,
  padding appears first → all keys 0-11 masked → position 11 (last padding before first real atom) sees
  NO valid context → produces degenerate affine params for position 12 (first real atom). Fix: use causal
  mask ONLY; no key masking. Padding isolation maintained separately by zeroing xa, xb after transformer.
  Result: VF jumped from 47.2% to 93.6%.

- T=9 VF=95%, T=21 VF=93.6%, gap=1.4pp. Phase 2 PASSED (criterion: gap < 10pp, both >= 85%).

**Phase 3 — Multi-molecule OPTIMIZE:**
- 8 molecules at T=21, 20k steps via Slurm (SLURM_JOB_ID=4157 on escher).
- W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/tw349mhw
- Config: channels=256, num_blocks=4, layers_per_block=2, lr=5e-4 cosine, batch_size=256.
- Status: COMPLETE (SLURM_JOB_ID=4157, ~18.5 min wall time on escher).
- W&B: https://wandb.ai/kaityrusnelson1/tnafmol/runs/tw349mhw

**Phase 3 Final Results (final checkpoint, step 20000):**

| Molecule | VF | Min Dist | PW Div |
|---------|-----|----------|--------|
| aspirin | 67.4% | 0.831 | 0.014 |
| benzene | 79.4% | 0.893 | 0.170 |
| ethanol | 64.0% | 0.829 | 0.039 |
| malonaldehyde | 82.6% | 0.923 | 0.039 |
| naphthalene | 81.0% | 0.873 | 0.053 |
| salicylic_acid | 67.4% | 0.841 | 0.026 |
| toluene | 67.4% | 0.823 | 0.044 |
| uracil | 63.6% | 0.817 | 0.051 |
| **Mean** | **71.6%** | 0.854 | 0.048 |

- Ethanol VF = 64.0% > 50%: CRITERION MET
- Mean VF = 71.6% > 40%: CRITERION MET (by 31.6pp)
- All 8 molecules > 50%: exceeded expectation
- HEURISTICS and SCALE skipped — primary criterion met by large margin.

**Story fit:** FITS. Apple TarFlow generalizes across 8 molecules with correct padding implementation.
Major improvement over hyp_007 best (ethanol 55.8% → 64.0%, mean 34.7% → 71.6%, aspirin 9.2% → 67.4%).
