# TNAFMOL — Sweep Best Practices

Accumulated sweep knowledge. Updated as experiments progress.

---

## hyp_003 — TarFlow Stabilization Sweep

**Swept:** alpha_pos [0.02, 0.05, 0.1], log_det_reg_weight [0.5, 1.0, 2.0, 5.0, 10.0], lr [1e-4, 3e-4, 1e-3], 30 runs
**Best:** alpha_pos=0.02, log_det_reg_weight=5.0, lr=1e-4 → mean VF 17.5% at 3000 steps

**Lessons:**
- alpha_pos=0.02 + log_det_reg_weight=5.0 is the stable combination: prevents collapse, avoids overconstrained learning
- Higher alpha_pos (0.1) leads to stronger saturation; lower is always better
- Higher reg_weight (10.0) doesn't help further — the equilibrium is a mathematical attractor, not a constraint that responds linearly to regularization
- lr=1e-4 best for SANITY angle (cosine schedule); higher lr (1e-3) was tested briefly but not fully explored here

**Sweep range recommendation:** For any TarFlow stabilization sweep, start with alpha_pos=[0.01, 0.02, 0.05] and log_det_reg_weight=[2.0, 5.0, 10.0]. Alpha_pos is the dominant parameter.

---

## hyp_003 — HEURISTICS Sweep (SBG recipe)

**Swept:** ema_decay [0.995, 0.999], lr [1e-4, 3e-4, 1e-3], batch_size [256, 512], 12 runs
**Best:** batch_size=512, ema_decay=0.999, lr=1e-3 → mean VF 18.3% at 3000 steps

**Lessons:**
- batch_size=512 consistently better than 256 — larger batches reduce gradient noise in the constrained regime
- ema_decay=0.999 better than 0.995 for 3000 steps when lr=1e-3
- lr=1e-3 with OneCycleLR is important — but ema_decay matters too

**Caution:** The hyp_003 result missed the ema_decay=0.99 case. See hyp_004 for correction.

---

## hyp_004 — SANITY Ablation + LR Sweep

**Swept:** architectural configs (6 variants), lr [5e-5, 1e-4, 2e-4] for best config, batch_size=128
**Best architectural config:** D_pos (use_pos_enc=True only) at 17.65%
**Best LR:** 5e-5 → 17.73% (spread <0.5ppt — LR is not the key lever at this scale)

**Lessons:**
- Positional encodings: clear +5ppt benefit. All other modifications (bidir_types, perm_aug) are neutral or slightly harmful.
- perm_aug hurts: atom ordering in MD17 encodes structural information (e.g., aspirin's atom ordering reflects chemical graph structure). Randomizing it creates a harder problem.
- bidir_types: no benefit. The model is already seeing future atom TYPE positions via the causal masking of coordinates, just not the types themselves. The information gain is minimal.
- For D_pos config: LR sweep is not critical (spread <0.5ppt). Use lr=5e-5 with cosine schedule.

---

## hyp_004 — HEURISTICS Sweep (SBG recipe with ema_decay)

**Swept:** ema_decay [0.99, 0.999, 0.9999], lr [1e-4, 3e-4, 1e-3], batch_size=512 fixed, 9 runs
**Best:** lr=1e-3, ema_decay=0.99 → mean VF 29.5% at 3000 steps (ethanol: 52.8%)

**Lessons:**
1. **lr=1e-3 dominates dramatically**: +10ppt over lr=1e-4/3e-4. The OneCycleLR peak LR is the critical parameter. Previous sweeps missed this by using too-conservative LR ranges.
2. **ema_decay=0.99 (faster) > ema_decay=0.999 > ema_decay=0.9999**: For 3000-step runs, faster EMA tracking (0.99) is better. With ema_decay=0.9999, the EMA model barely converges at 3000 steps — best val_loss only at step 3000 vs step 1000 for ema_decay=0.99.
3. **ema_decay and n_steps are coupled**: For runs up to ~5000 steps, use ema_decay=0.99. For runs >20000 steps, ema_decay=0.999 or 0.9999 becomes appropriate (EMA has time to warm up).
4. **Overwrite bug**: train.py names output dirs by `n_steps` + `lr` only (not ema_decay). Running multiple ema_decay variants with the same lr/n_steps overwrites the directory. Always check output naming before sweeping on EMA parameters.

**Sweep range recommendation for future SBG sweeps:**
- lr: [3e-4, 1e-3, 3e-3] (center around 1e-3, explore higher)
- ema_decay: [0.9, 0.99, 0.999] (for runs ≤5000 steps: center around 0.99)
- batch_size: 512 (no need to sweep; 512 consistently better than 256)
- Fix: betas=(0.9, 0.95), OneCycleLR pct_start=0.05
