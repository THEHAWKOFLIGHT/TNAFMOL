# Diagnostic Report — hyp_012: Permutation Reordering for Boltzmann Accuracy

## Baseline Failure Analysis

**Baseline:** hyp_011 SCALE full run — 512ch, 8 blocks, 50k steps, T=1.0 evaluation.

Per-molecule VF at T=1.0 (from `angles/scale/full/raw/mol_results.pt`):

| Molecule | VF @ T=1.0 | n_atoms | Canonical ordering |
|----------|-----------|---------|-------------------|
| aspirin | 0.896 | 21 | C C C C C C C O O O C C O H H H H H H H H |
| benzene | 1.000 | 12 | C C C C C C H H H H H H |
| ethanol | 0.950 | 9 | C C O H H H H H H |
| malonaldehyde | 0.996 | 9 | C C C O O H H H H |
| naphthalene | 0.998 | 18 | C C C C C C C C C C H H H H H H H H |
| salicylic_acid | 0.968 | 16 | C C C O C C C C O O H H H H H H |
| toluene | 0.998 | 15 | C C C C C C C H H H H H H H H |
| uracil | 0.988 | 12 | C C N C N C O O H H H H |
| **Mean** | **0.974** | — | — |

This is a high-quality baseline — mean VF = 97.4% at T=1.0. The hyp_012 experiment is not a repair task; it is a head-to-head test of whether permutation augmentation within atom type groups can push VF further, particularly for the weaker molecules.

## Observations on Per-Molecule Patterns

**Weakest molecules:** aspirin (0.896), ethanol (0.950), salicylic_acid (0.968).

**Strongest:** benzene (1.000), naphthalene (0.998), toluene (0.998), malonaldehyde (0.996).

**Does ordering explain the gap?**

The canonical atom ordering (from MD17 preprocessing) is not type-sorted. Aspirin has mixed ordering: `C C C C C C C O O O C C O H H H H H H H H` — heavy atoms first (carbon-rich), oxygen cluster in the middle, then a separate C/C/O triplet, then all hydrogens. Salicylic acid has `C C C O C C C C O O H H H H H H` — O interleaved with C.

TarFlow's autoregressive factorization means each atom's distribution is conditioned on all previous atoms. Under the canonical ordering:
- C atoms come first → their distribution must be learned from scratch (no prior context)
- Mixed orderings (like aspirin's middle C C O section) force the model to handle type-switching in the middle of the sequence

With `permute_within_types=True`, the type-sorted ordering ensures:
- All H atoms (type 0) come first, then C (type 1), then N (type 2), then O (type 3)
- Within each group, atoms are randomly permuted per sample → model learns exchangeability

This is directly motivated by TarFlow's sequential nature: a type-homogeneous prefix may be easier to model than the current mixed orderings in aspirin and salicylic_acid.

**Permutation entropy analysis:**
Pearson r(log(within-type permutations), VF) = -0.373. A mild negative correlation: molecules with more within-type permutability tend to have slightly lower VF. This is weak evidence but consistent with the hypothesis that ordering ambiguity hurts the current model.

## Root Cause Assessment

The failures in aspirin (10.4% invalid) and ethanol (5.0% invalid) are likely not a fundamental architecture issue — the model achieves near-perfect VF on many molecules. The root causes are likely a combination of:

1. **Canonical ordering effects**: aspirin has C-group interruptions (the C C O at positions 11-13 breaks the initial C cluster), which may create discontinuities in the autoregressive factorization
2. **Scale of permutation group**: aspirin has 9!×9! within-type permutations not explored during training; the model sees only one canonical ordering
3. **Molecule complexity**: aspirin is the largest molecule (21 atoms, 3 oxygen groups with different chemical environments)

Arm B (permute_within_types=True) directly tests whether teaching the model exchangeability within each type group improves Boltzmann quality.

## Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|------------|-----------|
| SANITY | Yes | The permute_within_types implementation is a new feature — it needs a short validation run to confirm it is working before scaling. This is the first experiment, so SANITY validates the implementation itself. |
| KNOWN HEURISTICS | Not applicable | We are doing a clean head-to-head comparison between canonical (Arm A) and perm-augmented (Arm B). The HEURISTICS phase would apply additional changes; here we want a clean A/B test. |
| SCALE | Not applicable | We are not changing model capacity — the baseline is already at maximum scale (512ch, 8 blocks, 50M params). |

Note: This is an unusual OPTIMIZE structure because we have TWO arms, not one direction of improvement. The SANITY phase serves as the validation run for both arms.

## Proposed Angles (Preliminary)

**Angle 1 — SANITY: implement permute_within_type_groups() and run head-to-head A/B validation**
- Arm A: canonical ordering (same as hyp_011 SCALE)
- Arm B: type-sorted + within-group permutation

Both arms run 5000-step validation runs first, then 50k-step full runs.

Full plan in Plan Sub-report.
