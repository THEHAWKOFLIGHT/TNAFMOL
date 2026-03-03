# TNAFMOL — Research Story

## Changelog
- 2026-03-03 und_001 DONE. TarFlow Diagnostic Ladder identifies padding as the sole failure mechanism. Architecture ceiling: 98.2% mean VF across all 8 molecules (no padding). Multi-molecule padded: 20.8% mean VF. hyp_002/hyp_003 failures traced to two implementation bugs (logdet normalization T*D vs n_real*D, self-inclusive causal mask). Shared scale hypothesis refuted. Alpha_pos equilibrium was a bug artifact, not architectural. TarFlow is viable per-molecule; padding is the multi-molecule bottleneck. Open Risks and Experiment Plan updated.
- 2026-03-02 hyp_004 PARTIAL confirmed. Positional encodings (+5ppt) + SBG recipe with lr=1e-3/ema=0.99 (+12ppt) push TarFlow to 29.5% mean VF (sweep) / 26.7% (full). 1/8 molecules ≥ 50%. Alpha_pos equilibrium persists unchanged. Assessment revised from "fundamentally broken" to "constrained with ~30% ceiling." Open Risks updated. Experiment plan updated with hyp_004 result.
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
- **[REVISED — und_001]** Padding + attention mask was assumed to handle variable atom counts (9-21 atoms). und_001 showed padding catastrophically degrades TarFlow performance. Per-molecule models (T=n_real, no padding) are the viable path for TarFlow. DDPM may handle padding differently — to be tested.
- No reweighting or rejection sampling: raw sample quality is the test.
- MD17 conformations are near-equilibrium (500K thermal distribution).
- Fair comparison requires comparable parameter counts and training compute.

## Open Risks

- **[RESOLVED — und_001]** The alpha_pos saturation equilibrium observed in hyp_002/hyp_003/hyp_004 was caused by two implementation bugs, not architectural limitations: (1) logdet normalization by T*D instead of n_real*D shifted the NLL equilibrium below physical bond lengths, enabling exploitation; (2) SOS token with self-inclusive causal mask created a non-triangular Jacobian (biased NLL). With both bugs fixed (und_001, commit 901d6c5), TarFlow achieves 94-100% VF on all 8 molecules at their natural sizes (T=n_real, no padding). The "~30% ceiling" from hyp_004 was a padding ceiling, not an architectural ceiling.
- **[CONFIRMED — und_001]** Padding is the sole remaining obstacle for multi-molecule TarFlow. When molecules are padded to T=21 for multi-molecule training, VF degrades smoothly with padding fraction (95% at 0% padding to ~0% at 100%). Mean VF at T=21: 20.8%. Aromatic molecules (naphthalene, toluene) collapse to 0% at just 14-29% padding. The mechanism: padding tokens corrupt attention context and create gradient imbalances in the log-det objective. No augmentation or regularization tested resolves this — an architectural solution (padding-free variable-length architecture, or per-molecule models) is required.
- **[RESOLVED — hyp_004]** Of the three architectural gaps investigated: (1) bidirectional type conditioning provides no benefit; (2) permutation augmentation slightly hurts — atom ordering is informative; (3) positional encodings help (+5ppt). Note: hyp_004's improvements were within the buggy alpha_pos equilibrium and are superseded by und_001's bug fixes.
- Canonical frame alignment quality depends on the mean structure reference per molecule.
- Energy evaluation requires a reliable oracle; oracle errors could bias the comparison.

## Experiment Plan

1. **hyp_001**: Data pipeline -- download MD17, preprocess all 8 molecules into canonical frame, compute reference statistics.
2. **hyp_002**: TarFlow -- implement the transformer autoregressive flow, train and tune (OPTIMIZE). **RESULT: FAILURE** -- log_det exploitation across 3 collapse modes. *Note: und_001 revealed this was caused by implementation bugs (logdet normalization + causal mask), not fundamental architecture.*
3. **hyp_003**: TarFlow stabilization -- fix log_det exploitation with asymmetric soft clamping (Andrade et al. 2024), log-det regularization, and soft equivariance (SBG, Tan et al. 2025). OPTIMIZE with SANITY/HEURISTICS/SCALE angles. **RESULT: FAILURE** -- alpha_pos saturation equilibrium. Best 18.3% mean valid fraction, 0/8 molecules ≥ 50%. *Note: und_001 showed the saturation was a bug artifact, not architectural.*
4. **hyp_004**: TarFlow Architectural Ablation + Optimization -- ablate three architectural fixes on top of hyp_003 stabilization baseline. **RESULT: PARTIAL** -- pos_enc (+5ppt) + SBG recipe (+12ppt) push to 29.5% mean VF. *Note: improvements were within the buggy equilibrium; superseded by und_001 bug fixes.*
5. **und_001**: TarFlow Diagnostic Ladder -- systematic 6-phase investigation using Apple's reference TarFlow implementation. **RESULT: DONE** -- architecture achieves 98.2% mean VF without padding. Padding identified as sole failure mechanism for multi-molecule models (20.8% mean VF at T=21). Two bugs discovered and fixed. Shared scale hypothesis refuted.
6. **hyp_005** (next): Per-molecule TarFlow -- train one TarFlow per molecule at T=n_real (no padding). Expected 94-100% VF based on und_001 Config A. This provides the TarFlow baseline for DDPM comparison.
7. **hyp_006**: DDPM -- implement the diffusion baseline with comparable architecture, train and tune (OPTIMIZE).
8. **hyp_007** (conditional): Head-to-head comparison -- TarFlow (per-molecule, hyp_005) vs DDPM (hyp_006) on all metrics.

## Success Criteria

**Primary**: Both models trained to convergence with proper tuning; head-to-head comparison completed across all metrics for all 8 molecules.

**What "comparable quality" means**: If the flow's valid fraction and energy Wasserstein distance are within 2x of the diffusion model on the majority of molecules, the flow is viable. If the flow is 10x worse on any metric for the majority of molecules, diffusion clearly wins for this application. The space in between requires nuanced interpretation.
