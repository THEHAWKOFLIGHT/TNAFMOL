## [hyp_012] — Permutation Reordering for Boltzmann Accuracy
**Date:** 2026-03-06 | **Type:** Hypothesis | **Tag:** `hyp_012`

### Motivation
TarFlow's autoregressive factorization means atom ordering directly affects the learned distribution. By grouping equivalent atoms and randomly permuting within type groups at each training step, we force the model to learn that same-type atoms are exchangeable — a physical truth that matters for generating true Boltzmann ensembles.

hyp_004 tested FULL random permutation within the buggy architecture and found it slightly hurt. This experiment tests TYPE-SORTED + WITHIN-GROUP permutation with the proven Apple architecture (TarFlow1DMol). Different augmentation, different architecture — results may differ.

### Method
Two-arm OPTIMIZE comparison at T=1.0 (Boltzmann temperature):

**Arm A (canonical ordering):** Train TarFlow1DMol with canonical MD17 atom ordering. `permute=False, permute_within_types=False`. Same setup as hyp_011 SCALE.

**Arm B (perm-augmented):** Same architecture, but atoms sorted by type, then randomly permuted within each type group per training step. `permute=False, permute_within_types=True`.

Both arms use hyp_011 SCALE config as starting baseline: 512ch, 8blk, layers_per_block=2, lr=5e-4, ldr=2.0, noise_sigma=0.03, batch_size=128, 50k steps, cosine+1000 warmup, T=21 seq_length, use_padding_mask=True.

Both arms evaluated at T=1.0 generation temperature.
Each arm gets 3 OPTIMIZE angles (6 total).

### Results
*Pending — experiment in progress.*

### Interpretation
*Pending.*

**Status:** [ ] Fits | [ ] Conflict — escalate to Postdoc | [ ] Inconclusive — reason:
