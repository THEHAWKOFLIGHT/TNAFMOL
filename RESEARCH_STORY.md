# TNAFMOL — Research Story

## Changelog
- 2026-03-02 PIVOT: hyp_004 changed from DDPM baseline to TarFlow Architectural Ablation + Optimization. Three architectural gaps identified in hyp_003 analysis: (1) causal masking hides future atom types during generation, (2) no permutation augmentation (atoms have arbitrary ordering), (3) no positional encodings. hyp_004 ablates bidirectional type conditioning, permutation augmentation, and positional encodings on top of hyp_003 stabilization baseline, then optimizes best config with SBG recipe. DDPM becomes hyp_005. Experiment plan updated.
- 2026-03-01 hyp_003 FAILURE confirmed. Asymmetric clamping + log-det regularization creates a stable saturation equilibrium at log_det/dof = alpha_pos. Best mean valid fraction 18.3%, 0/8 molecules ≥ 50%. TarFlow is confirmed not viable for molecular conformations. DDPM becomes hyp_004. Open Risks updated.
- 2026-03-01 PIVOT: hyp_003 changed from DDPM baseline to TarFlow stabilization. Instead of moving directly to diffusion, we attempt to fix TarFlow's log_det exploitation using three targeted interventions: asymmetric soft scale clamping (Andrade et al. 2024), log-det regularization penalty, and soft equivariance via SO(3) rotation + CoM noise augmentation with unit-variance normalization (SBG, Tan et al. 2025). If hyp_003 fails, DDPM becomes hyp_004. Experiment plan updated accordingly.
- 2026-03-01 Updated after hyp_002 FAILURE. TarFlow autoregressive affine flow collapses on molecular data due to log_det exploitation. Added findings to Open Risks.
- 2026-02-28 Initial authoring. Research story established from approved spec.

---

## The Question

Can a modern normalizing flow architecture -- specifically a TarFlow-style transformer autoregressive flow -- generate molecular conformations with quality comparable to a DDPM diffusion model? This is an exploratory head-to-head comparison using the MD17 dataset.

## Context and Motivation

Diffusion models (DDPMs) have become the default generative model for molecular conformation generation. They produce high-quality samples but suffer from slow generation (hundreds of reverse diffusion steps) and provide no exact likelihood. Normalizing flows offer exact likelihood computation and single-pass generation, but have historically underperformed diffusion in sample quality for complex distributions.

Recent work on TarFlow (transformer autoregressive flows) showed that flows can match diffusion quality on image benchmarks when given sufficient model capacity and modern architecture (transformers instead of coupling layers). The natural question: does this translate to molecular conformations?

We test this directly. Both models get equivalent design effort, comparable parameter counts, the same data preprocessing, and proper hyperparameter tuning. No reweighting, no rejection sampling -- raw generative quality is the comparison.

## Physical Setting

MD17 provides ~50k DFT-computed molecular conformations for each of 8 small molecules: aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic acid, toluene, and uracil. These are near-equilibrium thermal distributions at 500K, with atom counts ranging from 9 (ethanol) to 21 (aspirin). Each conformation comes with DFT energies and forces.

Both models are trained as a **single multi-molecule model** conditioned on atom types across all 8 molecules. This is a stronger test than per-molecule models -- the model must learn a shared generative process across different molecular topologies.

## Preprocessing: Canonical Frame

Both models share identical preprocessing:

1. **Center of mass subtraction**: x_centered = x - CoM(x) for each conformation
2. **Principal axis alignment**: Kabsch alignment to the mean structure per molecule
3. **Representation**: [N_atoms, 3] Cartesian coordinates, padded to max atom count (21) with attention mask
4. **Conditioning**: one-hot atom types (H, C, N, O) concatenated as conditioning signal
5. **Data split**: 80/10/10 train/val/test per molecule

The canonical frame removes 6 rigid-body DOFs (3 translation + 3 rotation). This means neither model requires SE(3) equivariance -- a significant architectural simplification.

## TarFlow: Transformer Autoregressive Flow

The flow model is a stack of L autoregressive transformer blocks. Each block:

1. Applies masked self-attention over atoms (atom i attends only to atoms 1..i-1)
2. Predicts affine parameters (shift + log-scale) for each atom position
3. Applies the affine autoregressive transform: y_i = s_i * x_i + t_i

The direction of the autoregressive ordering alternates between layers (forward in even layers, reverse in odd layers), following the TarFlow design.

Atom type embeddings are concatenated to position features as conditioning at every layer.

Base distribution: standard Gaussian N(0, I).

Training: exact maximum likelihood via change of variables:
log p(x) = log p_z(f(x)) + log |det(df/dx)|

The autoregressive structure gives a triangular Jacobian with tractable log-determinant.

## DDPM: Diffusion Baseline

The diffusion model uses:
- Same input representation and preprocessing as TarFlow
- Transformer-based denoiser with comparable parameter count
- Standard linear noise schedule, T=1000 steps
- Atom type conditioning via the same embedding scheme
- Training: L_simple (predict noise)
- Generation: full reverse diffusion (no acceleration, no guidance)

## Energy Evaluation

Generated samples need an energy oracle for evaluation. Options:
- **ANI-2x**: pretrained ML potential, handles organic molecules with {H, C, N, O}
- **Lightweight SchNet**: trained on MD17 DFT energies as a shared oracle

The energy oracle is applied identically to both models' outputs.

## Evaluation Metrics (Head-to-Head, Per Molecule)

| Metric | What it measures |
|--------|-----------------|
| Energy Wasserstein distance | Fidelity of generated energy distribution vs. reference |
| Pairwise distance distributions | All-atom pairwise distance histogram comparison |
| Bond length/angle distributions | Per-bond-type structural accuracy |
| RMSD coverage | Fraction of test conformations covered by a nearby generated sample |
| Valid fraction | Fraction with all bond lengths in [0.8, 2.0] Angstrom |
| NLL | Exact for flow; ELBO for DDPM (secondary metric) |

## Key Assumptions

- Canonical frame alignment removes rotation/translation DOFs. Models do NOT need equivariance.
- Padding + attention mask handles variable atom counts (9-21 atoms).
- No reweighting or rejection sampling: raw sample quality is the test.
- MD17 conformations are near-equilibrium (500K thermal distribution).
- Fair comparison requires comparable parameter counts and training compute.

## Open Risks

- **[CONFIRMED — hyp_002 + hyp_003]** TarFlow's autoregressive affine flow with MLE training exploits unconstrained scale DOFs to maximize log_det. hyp_002: three unbounded collapse modes. hyp_003: asymmetric clamping prevents unbounded collapse but creates alpha_pos saturation equilibrium. Best mean valid fraction 18.3%, 0/8 molecules ≥ 50%.
- **[UNDER INVESTIGATION — hyp_004]** Three architectural gaps may contribute to the saturation equilibrium: (1) causal masking hides future atom types, so the model cannot condition on full molecular composition during generation — it builds each atom without knowing what comes next; (2) no permutation augmentation, so the model may overfit to the arbitrary atom ordering in the dataset; (3) no positional encodings, so the model cannot distinguish atom positions within the autoregressive sequence. hyp_004 ablates all three fixes and optimizes the best combination.
- Canonical frame alignment quality depends on the mean structure reference per molecule.
- Variable atom counts + padding may affect flow training differently than diffusion. Masked log-likelihood is essential.
- Energy evaluation requires a reliable oracle; oracle errors could bias the comparison.

## Experiment Plan

1. **hyp_001**: Data pipeline -- download MD17, preprocess all 8 molecules into canonical frame, compute reference statistics.
2. **hyp_002**: TarFlow -- implement the transformer autoregressive flow, train and tune (OPTIMIZE). **RESULT: FAILURE** -- log_det exploitation across 3 collapse modes.
3. **hyp_003**: TarFlow stabilization -- fix log_det exploitation with asymmetric soft clamping (Andrade et al. 2024), log-det regularization, and soft equivariance (SBG, Tan et al. 2025). OPTIMIZE with SANITY/HEURISTICS/SCALE angles. **RESULT: FAILURE** -- alpha_pos saturation equilibrium. Best 18.3% mean valid fraction, 0/8 molecules ≥ 50%.
4. **hyp_004**: TarFlow Architectural Ablation + Optimization -- ablate three architectural fixes (bidirectional type conditioning, permutation augmentation, positional encodings) on top of hyp_003 stabilization baseline, then optimize best combination with SBG training recipe. Primary criterion: valid_fraction >= 0.5 on at least 4/8 molecules.
5. **hyp_005**: DDPM -- implement the diffusion baseline with comparable architecture, train and tune (OPTIMIZE).
6. **hyp_006** (conditional): Head-to-head comparison -- if both TarFlow (hyp_004) and DDPM (hyp_005) produce viable models, compare on all metrics.

## Success Criteria

**Primary**: Both models trained to convergence with proper tuning; head-to-head comparison completed across all metrics for all 8 molecules.

**What "comparable quality" means**: If the flow's valid fraction and energy Wasserstein distance are within 2x of the diffusion model on the majority of molecules, the flow is viable. If the flow is 10x worse on any metric for the majority of molecules, diffusion clearly wins for this application. The space in between requires nuanced interpretation.
