# OPTIMIZE Plan Sub-report — hyp_012: Permutation Reordering for Boltzmann Accuracy

**Status:** READY_TO_START

## Understanding

This experiment tests whether `permute_within_types=True` (Arm B) improves Boltzmann generation quality compared to the canonical ordering baseline (Arm A), using the proven hyp_011 SCALE architecture (512ch, 8 blocks, 50k steps, T=1.0). The baseline achieves 97.4% mean VF at T=1.0 but aspirin (89.6%) and ethanol (95.0%) still have room for improvement.

The OPTIMIZE structure here is adapted to a head-to-head A/B comparison rather than a single arm improving sequentially. Both arms run SANITY (implementation validation + short run), then FULL (50k steps).

## Diagnostic Summary

Root cause (from diagnostic report):
- aspirin (10.4% invalid) and ethanol (5.0% invalid) are the weakest molecules
- Canonical atom ordering has mixed types (e.g., aspirin: `C C C C C C C O O O C C O H H H H H H H H`)
- TarFlow's autoregressive factorization may overfit to this arbitrary canonical ordering
- Within-type permutation entropy (log of within-type permutation group size) has mild negative correlation with VF (r = -0.373)

The type-sorted + within-group permutation in Arm B teaches the model exchangeability within each chemical class while maintaining a consistent inter-type ordering across all training samples.

## Angle Structure

This experiment uses a single phase: SANITY (validation run → full run) for both arms.

HEURISTICS and SCALE are not applicable:
- This is a clean A/B test; we do not want additional confounds
- The model is already at maximum scale (50M params)
- The "fix" being tested is the implementation itself, not a literature heuristic (it's a research hypothesis)

## Angles

**Angle 1 — SANITY: Head-to-head A/B comparison**

### Arm A (Canonical Ordering Baseline)
- What changes: no change — identical to hyp_011 SCALE config
- Device: cuda:3
- Validation run: 5000 steps
- Promising criterion: VF > 90% (establishes baseline matches hyp_011)
- Full run: 50,000 steps with same config

### Arm B (Permute-Within-Types)
- What changes: `permute_within_types=True` added to training dataset
- Device: cuda:4
- Validation run: 5000 steps
- Promising criterion: VF > 90% (at minimum); meaningful if ≥ Arm A VF at 5k steps
- Full run: 50,000 steps

Both arms use identical hyperparameters except for `permute_within_types`.

Sweep: Not applicable for SANITY phase fixes with no free parameters (permute_within_types is a boolean toggle).

## Config (both arms)

Base config from hyp_011 SCALE full:
```json
{
  "exp_id": "hyp_012",
  "angle": "sanity",
  "stage": "val",
  "command": "OPTIMIZE",
  "seed": 42,
  "seq_length": 21,
  "channels": 512,
  "num_blocks": 8,
  "layers_per_block": 2,
  "head_dim": 64,
  "expansion": 4,
  "use_atom_type_cond": true,
  "atom_type_emb_dim": 16,
  "num_atom_types": 4,
  "use_padding_mask": true,
  "use_shared_scale": false,
  "use_clamp": false,
  "log_det_reg_weight": 2.0,
  "n_steps": 5000,
  "batch_size": 128,
  "lr": 5e-4,
  "lr_schedule": "cosine",
  "warmup_steps": 1000,
  "grad_clip_norm": 1.0,
  "val_interval": 1000,
  "eval_n_samples": 500,
  "weight_decay": 1e-5,
  "noise_sigma": 0.03,
  "augment_train": true,
  "max_atoms": 21,
  "molecules": null,
  "wandb_project": "tnafmol",
  "wandb_group": "hyp_012"
}
```

Arm A adds: `"permute_within_types": false` (explicit)
Arm B adds: `"permute_within_types": true`

## Success Criterion

**Primary:** At 50k steps, Arm B mean VF >= Arm A mean VF (permutation augmentation does not hurt).
**Secondary:** Arm B improves VF for the weakest molecules (aspirin, ethanol) by ≥ 2pp vs Arm A.
**Ideal outcome:** Arm B mean VF > 97.4% (hyp_011 baseline) and aspirin VF > 92%.

## Questions / Concerns

1. Note: at 5000 steps both arms will be undertrained — hyp_011 used 50k. VF at 5k is a directional indicator, not a strong signal.
2. Both arms use fresh random initialization (seed=42).
3. Evaluation is at T=1.0 (Boltzmann temperature), not T=0.7.
4. There is no sweep phase since permute_within_types has no continuous hyperparameter.
