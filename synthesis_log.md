# TNAFMOL — Synthesis Log

Append-only. All reviews, corrections, project state, and escalations.

---

### 2026-02-28 — Project initialization
Project initialized from approved spec. RESEARCH_STORY.md authored. Git repo created. First experiment: hyp_001 (data pipeline).

### 2026-02-28 — hyp_001 synthesis
**Status:** DONE | **Failure level:** None
**Branch:** `exp/hyp_001` | **Merge commit:** see `git log --oneline` | **Tag:** `hyp_001`

Data pipeline executed cleanly. All 8 MD17 molecules downloaded, preprocessed into canonical frame, reference statistics computed, verification plots generated. No issues encountered.

Source integration: code already in src/ (data.py, preprocess.py, metrics.py, visualize.py). No .py files in experiment directory. Pre-merge checks all passed.

Environment note: PyTorch was broken on this machine (Windows DLL load failure). Fixed by enabling Windows long paths and reinstalling torch 2.10.0+cu126. Required packages installed: h5py, ase, wandb.
