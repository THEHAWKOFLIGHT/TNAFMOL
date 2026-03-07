# TNAFMOL — Research Story

## Changelog
- 2026-03-06 hyp_012 DONE. Permutation Reordering for Boltzmann Accuracy. Two-arm OPTIMIZE: Arm A (canonical ordering) 97.7% mean VF at T=1.0 — replicates hyp_011. Arm B (type-sorted + within-group permutation) 0.7% mean VF — catastrophic failure. Type-sorted ordering is physically backwards for TarFlow autoregressive generation (H atoms generated before heavy-atom scaffold they depend on). Combined with within-group permutation preventing stable conditional distributions, the optimization landscape has no stable fixed point near valid geometries. This decisively closes the permutation augmentation direction for TarFlow. The canonical MD17 ordering is necessary for TarFlow's success. Open Risks updated.
- 2026-03-06 Experiment Plan updated: hyp_012 changed from DDPM baseline to Permutation Reordering for Boltzmann Accuracy. Two-arm OPTIMIZE: Arm A (canonical ordering) vs Arm B (type-sorted + within-group permutation augmentation), both at T=1.0. DDPM becomes hyp_013, head-to-head becomes hyp_014.
- 2026-03-06 hyp_011 DONE. Crack MD17 Multi-Molecule TarFlow. Three-phase OPTIMIZE: SANITY showed capacity > budget (384ch/6blk → 83.9% vs 256ch/4blk at 50k → 73.3%). HEURISTICS swept lr/ldr/noise_sigma — noise_sigma=0.03 is the key lever (+5-7pp). Full run at 50k steps → 94.7%. SCALE (512ch/8blk, 50.6M params, 50k steps) → 97.4% at T=1.0. Temperature sweep at T=0.7 → 98.9% mean VF. All 8 molecules above 95.6%. Multi-molecule gap effectively closed — single shared model matches per-molecule ceiling. Experiment Plan updated: TarFlow arm fully optimized, DDPM baseline is next.
- 2026-03-07 hyp_010 DONE. TarFlow Apple Architecture for Multi-Molecule MD17. Uses tarflow_apple.py + TarFlow1DMol directly (bypassing model.py). Phase 1: ethanol T=9 VF=95% (gate passed). Phase 2: padding validation passed (VF gap=1.4pp) after two critical bug fixes — (1) sampling noise at padding positions corrupted PermutationFlip chain, (2) attention key masking starved first real atom of context. Phase 3: all 8 molecules, T=21, 20k steps — mean VF=71.6%, ALL 8 molecules >50%. Best: malonaldehyde 82.6%, naphthalene 81.0%. Aspirin recovered from 9.2% (hyp_007) to 67.4%. No log-det regularization needed (ldr=0). TarFlow arm of the comparison is ready. Open Risks updated.
- 2026-03-06 hyp_009 FAILURE. Architecture Alignment. Pre-norm + layers_per_block=2 gave 14% VF on ethanol T=9 — WORSE than baseline (39%). Incremental patching of model.py exhausted after 4 experiments (hyp_006–hyp_009). PIVOT: hyp_010 uses tarflow_apple.py + TarFlow1DMol directly for multi-molecule training, bypassing model.py entirely. Experiment Plan updated.
- 2026-03-06 hyp_008 FAILURE. Per-Dimension Scale. per_dim_scale implemented correctly (6/6 unit tests), but has <1pp effect on VF (und_001 Phase 4 data: 96.2% vs 95.3%). Phase 1 gate failed: best VF=39.2% on ethanol T=9 (target 90%). True root cause of 61pp gap between model.py and tarflow_apple.py: (1) post-norm vs pre-norm, (2) layers_per_block=1 vs 2. Code retained — aligns with Apple but is not the bottleneck. Experiment Plan updated: hyp_009 should address pre-norm + layers_per_block, not DDPM.
- 2026-03-06 hyp_007 PARTIAL. Padding Isolation + Multi-Molecule OPTIMIZE. Phase 1 CONFIRMED: output-shift makes padding neutral — VF varies only 4pp across T=9,12,15,18,21 for ethanol-only training. Phase 2: SANITY failed (ldr=0 allows log-det exploitation even with output-shift — log_det/dof rose to 1.2+). HEURISTICS: log_det_reg_weight=5.0 is critical — pushes ethanol VF from 17.6% → 55.8%. Best multi-molecule result: ethanol 55.8%, malonaldehyde 53.2%, mean 34.7%, 2/8 molecules above 50%. Primary criterion (4/8 ≥ 50%) not met. Aspirin (21 atoms) at 9.2% — VF inversely correlates with molecule size. Open Risks and Experiment Plan updated.
- 2026-03-04 hyp_006 FAILURE. Output-Shift Multi-Molecule TarFlow. Hypothesis CONFIRMED: output-shift bounds log_det/dof at 0.5-1.3 (vs 7+ for SOS architecture) — log-det exploitation pathway completely eliminated. However, VF criterion NOT MET: best 24.8% ethanol (HEURISTICS C, lr=1e-3, cosine, 5k steps). All 3 angles exhausted (SANITY 13.8%, HEURISTICS best 24.8%, SCALE 16.2% — overfits). Root cause of low VF is atom overlaps (mean min pairwise distance 0.45-0.65 Å vs 0.8 Å threshold), likely due to normalization mismatch between model's coordinate space and physical space. Output-shift is now the correct architectural platform. Open Risks and Experiment Plan updated.
- 2026-03-03 hyp_005 FAILURE. Padding-Aware Multi-Molecule TarFlow. PAD token + query zeroing implemented and verified (6 unit tests pass), but have zero measurable effect on VF — SANITY 2x2 factorial shows identical log_det/dof=7.3 across all 4 configs. Best VF=4.7% (ethanol) with reg_weight=2.0. Story CONFLICT: und_001's prediction that fixing padding corruption would restore multi-molecule VF was wrong — log-det exploitation in SOS+causal architecture is the deeper bottleneck. Padding fixes are necessary but not sufficient. Open Risks and Experiment Plan updated.
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
- **[RESOLVED — hyp_007]** Padding + attention mask was assumed to handle variable atom counts (9-21 atoms). und_001 showed padding catastrophically degrades TarFlow performance with SOS architecture. hyp_005 showed padding fixes have zero effect with SOS+causal. hyp_006 introduced output-shift, eliminating log-det exploitation. hyp_007 Phase 1 CONFIRMED: output-shift makes padding neutral — VF varies only 4pp across T=9 to T=21 for ethanol. Padding is no longer a concern. The remaining VF gap is due to insufficient log-det regularization (ldr=5.0 needed) and molecule-size-dependent capacity limitations.
- No reweighting or rejection sampling: raw sample quality is the test.
- MD17 conformations are near-equilibrium (500K thermal distribution).
- Fair comparison requires comparable parameter counts and training compute.

## Open Risks

- **[RESOLVED — und_001]** The alpha_pos saturation equilibrium observed in hyp_002/hyp_003/hyp_004 was caused by two implementation bugs, not architectural limitations: (1) logdet normalization by T*D instead of n_real*D shifted the NLL equilibrium below physical bond lengths, enabling exploitation; (2) SOS token with self-inclusive causal mask created a non-triangular Jacobian (biased NLL). With both bugs fixed (und_001, commit 901d6c5), TarFlow achieves 94-100% VF on all 8 molecules at their natural sizes (T=n_real, no padding). The "~30% ceiling" from hyp_004 was a padding ceiling, not an architectural ceiling.
- **[RESOLVED — hyp_006]** Output-shift eliminates log-det exploitation. log_det/dof bounded at 0.5-1.3 throughout training with alpha_pos=10.0 and no regularization (vs 7+ for SOS+causal). The architectural hypothesis is confirmed. However, VF remains at 13-25% across all angles — the model generates samples with persistent atom overlaps (mean min pairwise distance 0.45-0.65 Å vs 0.8 Å valid threshold). This is a different problem from log-det exploitation.
- **[RESOLVED — hyp_011]** VF plateau at 13-25% (hyp_006) and molecule-size correlation (hyp_007, 34.7% mean, 2/8 above 50%) were caused by model.py's architectural deficiencies, not fundamental limitations. Using the Apple architecture directly (tarflow_apple.py + TarFlow1DMol) achieves 71.6% mean VF with ALL 8 molecules above 50% (hyp_010). hyp_011 closed the remaining gap: capacity scaling (512ch/8blk, 50.6M params) + noise_sigma=0.03 + ldr=2.0 + temperature T=0.7 → **98.9% mean VF**, all 8 molecules above 95.6%. The multi-molecule gap vs per-molecule ceiling (98.2%) is effectively zero. A single shared model matches dedicated per-molecule models.
- **[RESOLVED — hyp_004, hyp_012]** Permutation augmentation is harmful for TarFlow: hyp_004 showed full random permutation slightly hurts (~2pp), hyp_012 showed type-sorted + within-group permutation catastrophically fails (97.7% → 0.7%). The canonical MD17 atom ordering is necessary — it provides a stable causal structure (heavy atoms before hydrogens) that TarFlow's autoregressive factorization requires. Of hyp_004's other gaps: (1) bidirectional type conditioning provides no benefit; (3) positional encodings help (+5ppt). Note: hyp_004's improvements were within the buggy alpha_pos equilibrium and are superseded by und_001's bug fixes.
- Canonical frame alignment quality depends on the mean structure reference per molecule.
- Energy evaluation requires a reliable oracle; oracle errors could bias the comparison.

## Experiment Plan

1. **hyp_001**: Data pipeline -- download MD17, preprocess all 8 molecules into canonical frame, compute reference statistics.
2. **hyp_002**: TarFlow -- implement the transformer autoregressive flow, train and tune (OPTIMIZE). **RESULT: FAILURE** -- log_det exploitation across 3 collapse modes. *Note: und_001 revealed this was caused by implementation bugs (logdet normalization + causal mask), not fundamental architecture.*
3. **hyp_003**: TarFlow stabilization -- fix log_det exploitation with asymmetric soft clamping (Andrade et al. 2024), log-det regularization, and soft equivariance (SBG, Tan et al. 2025). OPTIMIZE with SANITY/HEURISTICS/SCALE angles. **RESULT: FAILURE** -- alpha_pos saturation equilibrium. Best 18.3% mean valid fraction, 0/8 molecules ≥ 50%. *Note: und_001 showed the saturation was a bug artifact, not architectural.*
4. **hyp_004**: TarFlow Architectural Ablation + Optimization -- ablate three architectural fixes on top of hyp_003 stabilization baseline. **RESULT: PARTIAL** -- pos_enc (+5ppt) + SBG recipe (+12ppt) push to 29.5% mean VF. *Note: improvements were within the buggy equilibrium; superseded by und_001 bug fixes.*
5. **und_001**: TarFlow Diagnostic Ladder -- systematic 6-phase investigation using Apple's reference TarFlow implementation. **RESULT: DONE** -- architecture achieves 98.2% mean VF without padding. Padding identified as sole failure mechanism for multi-molecule models (20.8% mean VF at T=21). Two bugs discovered and fixed. Shared scale hypothesis refuted.
6. **hyp_005**: Padding-Aware Multi-Molecule TarFlow -- tested two padding fixes (PAD token embedding, query zeroing) via 2x2 factorial ablation + reg_weight sweep. **RESULT: FAILURE** -- padding fixes have zero effect on VF; log-det exploitation in SOS+causal architecture is the deeper bottleneck. Best VF=4.7% ethanol. 10x degradation from single-molecule unexplained.
7. **hyp_006**: Output-Shift Multi-Molecule TarFlow — replace SOS+strictly-causal with Apple's output-shift mechanism in src/model.py. **RESULT: FAILURE** — hypothesis CONFIRMED (log_det/dof bounded 0.5-1.3 vs 7+ for SOS), but VF criterion not met (best 24.8% ethanol, 16.3% mean). All 3 angles exhausted. Root cause: atom overlaps in generated samples, possibly due to normalization mismatch. Output-shift is now the correct architectural platform for multi-molecule TarFlow.
8. **hyp_007**: Padding Isolation + Multi-Molecule OPTIMIZE — configurable max_atoms for padding isolation test, then multi-molecule output-shift with sufficient training. **RESULT: PARTIAL** — Phase 1 CONFIRMED padding neutrality (4pp VF variation across 5 padding sizes). Phase 2: ldr=5.0 critical — ethanol 55.8%, malonaldehyde 53.2%, mean 34.7%. 2/8 molecules ≥ 50% (target: 4/8). VF inversely correlates with molecule size.
9. **hyp_008**: Per-Dimension Scale -- add per_dim_scale to model.py (3 independent log_scales per atom, matching Apple). **RESULT: FAILURE** — <1pp effect on VF (und_001 Phase 4 already showed this). True 61pp gap is pre-norm + layers_per_block. Code retained. Phases 2-3 skipped.
10. **hyp_009**: Architecture Alignment -- add pre-norm and layers_per_block to model.py. **RESULT: FAILURE** -- 14% VF on ethanol T=9 (WORSE than baseline 39%). Incremental patching of model.py exhausted after 4 experiments (hyp_006–hyp_009).
11. **hyp_010**: TarFlow Apple Architecture for Multi-Molecule MD17 -- use tarflow_apple.py + TarFlow1DMol directly for multi-molecule joint training. **RESULT: DONE** -- Phase 1: ethanol T=9 VF=95%. Phase 2: padding validation passed (VF gap=1.4pp) after 2 bug fixes (sampling noise at padding + attention key masking). Phase 3: all 8 molecules, T=21, 20k steps — mean VF=71.6%, ALL 8 molecules >50%. Best: malonaldehyde 82.6%. No ldr needed. TarFlow arm ready for head-to-head comparison.
12. **hyp_011**: Crack MD17 Multi-Molecule TarFlow — push VF as high as possible via capacity scaling, HP tuning, and temperature sweep. **RESULT: DONE** — SANITY: capacity > budget (384ch/6blk → 83.9%). HEURISTICS: noise_sigma=0.03 key lever; full run 94.7%. SCALE: 512ch/8blk 50.6M params → 97.4% (T=1.0), 98.9% (T=0.7). All 8 molecules >95.6%. Multi-molecule gap closed. TarFlow arm fully optimized.
13. **hyp_012**: Permutation Reordering for Boltzmann Accuracy — head-to-head test of whether random permutation augmentation within atom-type equivalence classes improves Boltzmann generation quality. Arm A: canonical ordering, Arm B: type-sorted + within-group permutation. Both OPTIMIZE'd at T=1.0. **RESULT: DONE** — Arm A 97.7% mean VF (replicates hyp_011). Arm B 0.7% mean VF — catastrophic failure. Type-sorted ordering physically backwards for autoregressive generation. Permutation augmentation direction closed.
14. **hyp_013**: DDPM diffusion baseline -- implement transformer-based denoiser with comparable parameter count. Train and tune (OPTIMIZE).
15. **hyp_014** (conditional): Head-to-head comparison -- TarFlow vs DDPM on all metrics.

## Success Criteria

**Primary**: Both models trained to convergence with proper tuning; head-to-head comparison completed across all metrics for all 8 molecules.

**What "comparable quality" means**: If the flow's valid fraction and energy Wasserstein distance are within 2x of the diffusion model on the majority of molecules, the flow is viable. If the flow is 10x worse on any metric for the majority of molecules, diffusion clearly wins for this application. The space in between requires nuanced interpretation.
