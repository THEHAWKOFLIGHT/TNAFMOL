# Final Experiment Report — hyp_012: Permutation Reordering for Boltzmann Accuracy
**Status:** DONE
**Branch:** `exp/hyp_012`
**W&B Arm A:** https://wandb.ai/kaityrusnelson1/tnafmol/runs/nqi97n8z
**W&B Arm B:** https://wandb.ai/kaityrusnelson1/tnafmol/runs/9pnnie8w

---

## Experimental Outcome

### Head-to-Head Summary

| Arm | Method | Mean VF (T=1.0) | Molecules > 50% | Status |
|-----|--------|----------------|-----------------|--------|
| **Arm A** | Canonical MD17 ordering | **97.7%** | 8/8 | Replicates hyp_011 |
| **Arm B** | Type-sorted + within-group permutation | **0.7%** | 0/8 | Catastrophic failure |

### Arm A (Canonical) — Per-Molecule Results

| Molecule | VF | pw_div | min_dist_mean |
|----------|-----|--------|---------------|
| aspirin | 90.6% | 0.0070 | 0.911 |
| benzene | 100.0% | 0.0562 | 1.024 |
| ethanol | 96.2% | 0.0099 | 0.956 |
| malonaldehyde | 99.8% | 0.0124 | 1.042 |
| naphthalene | 100.0% | 0.0182 | 1.005 |
| salicylic_acid | 96.2% | 0.0074 | 0.948 |
| toluene | 99.6% | 0.0157 | 0.994 |
| uracil | 99.0% | 0.0112 | 0.985 |
| **Mean** | **97.7%** | | |

Arm A successfully replicates hyp_011 SCALE results (97.4% → 97.7%). Aspirin remains the weakest molecule at 90.6%, consistent with its complexity (n=21 atoms, highest permutation entropy among MD17 set).

### Arm B (permute_within_types) — Per-Molecule Results

| Molecule | VF | pw_div | min_dist_mean |
|----------|-----|--------|---------------|
| aspirin | 0.0% | 0.3124 | 0.220 |
| benzene | 0.0% | 0.5419 | 0.176 |
| ethanol | 5.4% | 0.0874 | 0.490 |
| malonaldehyde | 0.0% | 0.3210 | 0.316 |
| naphthalene | 0.0% | 0.4941 | 0.147 |
| salicylic_acid | 0.0% | 0.3709 | 0.178 |
| toluene | 0.0% | 0.4885 | 0.156 |
| uracil | 0.0% | 0.3774 | 0.274 |
| **Mean** | **0.7%** | | |

Arm B failed catastrophically across all molecules. min_dist_mean values (0.15–0.49 Å) are well below the 0.8 Å validity threshold, indicating generated samples have severe atomic overlaps. The failure is independent of molecule size — even ethanol (n=9, simplest) achieves only 5.4%.

---

## Training Dynamics Analysis

### Loss Trajectory

```
Step    | Arm A loss  | Arm B loss  | A/B ratio
2000    | -1.167      | -0.121      | 9.6×
5000    | -1.52       | -0.19       | ~8×
10000   | -1.51       | -0.40       | ~3.8×
20000   | -1.57       | -0.60       | ~2.6×
35000   | -1.58       | -0.80       | ~2.0×
50000   | -1.65       | -0.88       | ~1.9×
```

Arm B's training loss converged to -0.877 at step 50000 — still nearly 2× less negative than Arm A's -1.652. The NLL gap did not close over training. Arm B's log-determinant magnitudes (mean ~1.35 vs Arm A's ~2.16) are also lower, indicating the flow is not developing the same degree of spatial compression during generation.

### Val Loss Divergence (key diagnostic)

Arm A's val loss exploded after step 1000 (0.74 → 2-4+), consistent with TarFlow's known log-det exploitation behavior. The best checkpoint was saved at step 17000 (val_loss=0.547). Despite the high val loss, the FINAL model (step 50000) evaluates at 97.7% VF — the und_001 convention (evaluate final, not best-val checkpoint) was critical here.

Arm B's val loss diverged even more severely (0.54 → 6-7+ by step 20000), but unlike Arm A, this was not accompanied by actual learning. The log-det exploitation in Arm B is pathological — the model is increasing log-dets without corresponding improvement in generation quality.

---

## Root Cause Analysis: Why Did Arm B Fail?

The hypothesis was that type-sorted + within-group permutation would teach exchangeability within atom types. Instead, it created a pathological training signal.

**The core problem: TarFlow's autoregressive structure requires a stable causal ordering.**

TarFlow1DMol generates atom positions autoregressively: atom i is generated conditioned on atoms 0...(i-1). When atoms are type-sorted (all H first, then C, then O), the model must:

1. Generate all H atoms first — but H atoms are the lightest, most mobile, and most correlated with each other AND with the heavy atoms they're bonded to
2. Then generate C/N/O atoms — conditioned on already-generated H positions

This ordering is physically awkward: H atoms' positions are strongly determined by the C/N/O scaffold, but in type-sorted ordering, the scaffold is generated AFTER the H atoms. The autoregressive conditioning is backwards relative to the physics.

Furthermore, within-group permutation means the model must learn that "H atom 1" could be any of the H atoms — it can't learn a stable conditional distribution for each position. The permutation augmentation that should teach exchangeability instead prevents the model from learning any specific conditional: each H atom's position is randomly assigned to a different "slot" each batch, making the conditional distribution multi-modal and difficult to learn.

**Why hyp_004 (full random permutation) also showed degradation:**
Full random permutation had the same problem but less severe because the permuted ordering was at least consistent with itself (each atom could be any atom), rather than the type-sorted ordering creating a physically backwards causal structure.

**Ethanol's 5.4% (vs 0% for others):**
Ethanol has the simplest structure (n=9: C, C, O, H×6). With only 6 H atoms and 3 heavy atoms, the type-sorted ordering is: H H H H H H C C O. The model appears to have learned a weak correlation structure that sometimes generates valid samples, but the ordering still produces mostly invalid ones.

---

## Project Context

This result strongly confirms the research story's direction: **TarFlow's autoregressive factorization is highly sensitive to atom ordering.** The canonical MD17 ordering (which reflects the physical data collection convention) appears to provide a reasonable causal structure for learning. Permuted orderings that disrupt this causal structure — whether fully random (hyp_004) or type-sorted-within-group (hyp_012) — degrade performance dramatically.

**Implication for future work:** Rather than augmenting the ordering, the correct direction may be to understand WHY the canonical MD17 ordering works well for TarFlow, and whether even better orderings exist. The canonical ordering likely captures some implicit spatial/chemical locality that supports autoregressive factorization.

---

## Story Validation

**This result fits the research story.** The project has consistently found that:
1. Architecture and training setup are critical (hyp_003–009)
2. The Apple/TarFlow1DMol architecture with canonical ordering achieves ~97-99% VF
3. Perturbations to the data representation (even physically motivated ones) can be catastrophically harmful

Arm B's failure is not a surprise given hyp_004's mild degradation from full random permutation. The type-sorted ordering is a more disruptive intervention and its failure confirms that TarFlow's performance is tightly coupled to the specific causal ordering it sees during training.

---

## Figures

### VF Comparison: Arm A vs Arm B
`results/hyp_012_vf_comparison.png`

### Loss Curves: Arm A vs Arm B
`results/hyp_012_loss_comparison.png`

### Arm A Individual Loss Curve and VF Bar
`angles/arm_a_sanity/full/hyp_012_loss_curve.png`
`angles/arm_a_sanity/full/hyp_012_vf_bar.png`

### Arm B Individual Loss Curve and VF Bar
`angles/arm_b_sanity/full/hyp_012_loss_curve.png`
`angles/arm_b_sanity/full/hyp_012_vf_bar.png`

---

## Open Questions

1. **Why does Arm B's val loss diverge so much more severely than Arm A's?** Both exploit log-dets, but Arm B reaches val_loss ~7 while Arm A stays around 2-3. This suggests the log-det exploitation in Arm B is qualitatively different — possibly the flow is learning to map all type-sorted configurations to extreme regions of latent space.

2. **Is the type-sorted ordering itself harmful, or is it the within-group permutation?** A cleaner experiment would test type-sorted ordering WITHOUT within-group permutation as a third arm. This could isolate whether the causal ordering or the stochastic augmentation is the failure mode.

3. **Can canonical ordering be improved?** If type-sorted is worse than canonical, are there orderings that are BETTER than canonical? E.g., ordering by distance from center of mass (spatial locality), or by graph connectivity (chemical bonds).
