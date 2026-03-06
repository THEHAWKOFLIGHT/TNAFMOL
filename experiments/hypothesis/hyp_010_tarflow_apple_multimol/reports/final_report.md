## Final Experiment Report — hyp_010: TarFlow Apple Multi-Molecule
**Status:** DONE
**Branch:** `exp/hyp_010`
**Commits:** [`4b65477` — fix sampling bug], [`85a6f2e` — fix attention mask], [`f8c9dc6` — T=21 config], [`6addf0c` — Phase 2 results], [`b5de8aa` — pre-Phase-3 snapshot], [`c9c3133` — Slurm job log], [`39cc986` — experiment_log updates]

---

### Experimental Outcome

**Phase 1 — Ethanol T=9 Sanity Gate: PASSED**
- Model: TarFlow1DMol from `src/train_phase3.py` (Apple architecture, contraction convention)
- VF = 95.0% on ethanol T=9, 5k steps
- Reproduces und_001 Phase 4 config 2 (96.2% VF) within 1.2pp

**Phase 2 — T=21 Padding Validation: PASSED**
- Initial VF with padding = 33.2% (62pp gap from T=9)
- Two bugs found and fixed in `src/train_phase3.py`:
  1. Sampling bug: Gaussian noise at padding positions corrupted PermutationFlip autoregressive chain. Fix: zero padding positions in `z` before `reverse()`, re-zero between blocks.
  2. Attention key masking bug: padding KEY masking in permuted space starved the first real atom (position 12 in permuted space when flipped) of all context. Fix: causal-only mask in attention; no key masking.
- With both fixes: VF(T=9)=95%, VF(T=21)=93.6%, gap=1.4pp
- Criterion (|gap| < 10pp, both >= 85%): PASSED

**Phase 3 — Multi-Molecule SANITY (all 8 molecules, T=21, 20k steps via Slurm): PASSED**

| Molecule | VF | Min Dist | PW Div |
|---------|-----|----------|--------|
| aspirin | 67.4% | 0.831 | 0.0140 |
| benzene | 79.4% | 0.893 | 0.1702 |
| ethanol | 64.0% | 0.829 | 0.0387 |
| malonaldehyde | 82.6% | 0.923 | 0.0390 |
| naphthalene | 81.0% | 0.873 | 0.0526 |
| salicylic_acid | 67.4% | 0.841 | 0.0258 |
| toluene | 67.4% | 0.823 | 0.0439 |
| uracil | 63.6% | 0.817 | 0.0509 |
| **Mean** | **71.6%** | **0.854** | **0.0481** |

- **Ethanol VF = 64.0% > 50%: CRITERION MET**
- **Mean VF = 71.6% > 40%: CRITERION MET**
- **All 8 molecules have VF > 50%**: well beyond the criterion
- Slurm job: SLURM_JOB_ID=4157 on escher (~18.5 minutes wall time)
- W&B run: https://wandb.ai/kaityrusnelson1/tnafmol/runs/tw349mhw

**Comparison to previous best (hyp_007):**
- hyp_007 (model.py + ldr=5.0, 20k steps): ethanol VF=55.8%, mean VF=34.7%, aspirin VF=9.2%
- hyp_010 (train_phase3.py + both padding fixes, 20k steps): ethanol VF=64.0%, mean VF=71.6%, aspirin VF=67.4%
- Improvement: +8.2pp ethanol, +36.9pp mean, +58.2pp aspirin (elimination of main outlier)

---

### Plots Generated
- `angles/sanity/full/hyp_010_loss_curve.png` — training + val loss curves
- `angles/sanity/full/hyp_010_vf_bar.png` — per-molecule VF bar chart
- `angles/sanity/full/hyp_010_phase3_molecule_breakdown.png` — per-molecule VF + min_dist comparison
- `angles/sanity/full/hyp_010_phase3_training_curves.png` — loss curves + logdet/dof trajectory

---

### Project Context

This experiment established the first multi-molecule TarFlow pipeline using the proven Apple architecture
from und_001. The key contributions are:

1. **Two critical bug fixes in `src/train_phase3.py`** that were blocking T=21 padding-masked training.
   These are now fixed in the canonical source file used by all future experiments.

2. **Validated multi-molecule scalability**: the Apple architecture generalizes from single-molecule
   (95% VF ethanol) to multi-molecule (71.6% mean VF across 8 molecules) at T=21 with padding.

3. **Aspirin VF recovery**: hyp_007's worst molecule (9.2% aspirin VF with old architecture + ldr=5.0)
   improves to 67.4% with the Apple architecture at ldr=0. The logdet regularization in hyp_007 was
   suppressing aspirin learning disproportionately — aspirin is the largest molecule (21 atoms = aspirin
   at maximum, so padding is minimal) and needed the most model flexibility.

---

### Story Validation

**Fits the research story.** hyp_010 demonstrates that the Apple TarFlow architecture (contraction
convention, pre-norm, layers_per_block=2) can successfully handle:
- Multi-molecule training at scale (8 molecules, T=21)
- Padding-masked sequences with correct architecture-aware masking

The experiment also surfaces the importance of correct padding conventions in autoregressive flows:
naive padding (noise at unused positions, key masking that cuts information) is deeply harmful for
PermutationFlip-based flows where padding appears at the start of the generation chain.

---

### Open Questions

1. **Benzene PW divergence is high (0.1702 vs ~0.04 for others)**: benzene has C6 symmetry — the model
   may be generating valid conformations but with rotational phase ambiguity relative to the reference.
   This is not a structural validity failure (VF=79.4%) but a coordinate convention issue. Worth
   investigating if benzene is used as a multi-molecule benchmark.

2. **Ethanol VF lower than single-molecule baseline (64% vs 93.6%)**: the multi-molecule training
   distributes model capacity across 8 molecules. Ethanol is small (9 atoms) and may benefit less from
   a large model trained on diverse chemistry. If ethanol-specific VF matters, single-molecule
   fine-tuning from the multi-molecule checkpoint could recover it.

3. **HEURISTICS sweep (lr × ldr)**: the Phase 3 result already exceeds the success criterion by a large
   margin, so HEURISTICS was skipped per plan. If higher VF is needed on specific molecules (e.g., aspirin
   to >80%), a targeted sweep on ldr and lr could help.
