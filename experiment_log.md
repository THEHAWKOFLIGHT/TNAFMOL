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
