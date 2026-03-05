"""Generate all required visualizations for hyp_007.

Run from project root:
    python3.10 experiments/hypothesis/hyp_007_padding_isolation_multimol/generate_plots.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE = Path("experiments/hypothesis/hyp_007_padding_isolation_multimol")
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MOLECULES = [
    "aspirin", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "salicylic_acid", "toluene", "uracil",
]

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Phase 1 — Padding Isolation (VF vs max_atoms)
# ─────────────────────────────────────────────────────────────
print("Generating Figure 1: Phase 1 Padding Isolation...")

phase1_T = [9, 12, 15, 18, 21]
phase1_vf = []
for T in phase1_T:
    raw = torch.load(
        BASE / f"angles/phase1_padding/val/T{T}/raw/mol_results.pt",
        weights_only=False
    )
    phase1_vf.append(raw["ethanol"]["valid_fraction"])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(phase1_T, [v * 100 for v in phase1_vf], "o-", color="#2196F3", linewidth=2, markersize=8, label="Ethanol VF")
ax.axhline(34.8, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="T=9 baseline (34.8%)")
ax.fill_between(phase1_T, [34.8 - 20] * 5, [34.8 + 20] * 5, alpha=0.08, color="green", label="±20pp tolerance band")
ax.set_xlabel("max_atoms (padding size)")
ax.set_ylabel("Ethanol Valid Fraction (%)")
ax.set_title("hyp_007 Phase 1 — Padding Isolation Test\nEthanol VF vs. max_atoms (5000 steps, output-shift TarFlow)")
ax.set_xticks(phase1_T)
ax.set_ylim(0, 80)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate("GATE PASSED\nMax drop: 4.0pp (T=9→T=18)", xy=(18, 31.2), xytext=(14, 15),
            fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
plt.tight_layout()
fig.savefig(RESULTS_DIR / "hyp_007_phase1_padding_isolation.png", dpi=300)
plt.close()
print(f"  Saved: {RESULTS_DIR}/hyp_007_phase1_padding_isolation.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 2: Sweep Summary — VF by molecule, all runs
# ─────────────────────────────────────────────────────────────
print("Generating Figure 2: Sweep Summary...")

sweep_runs = {
    "SANITY\nlr=1e-3": BASE / "angles/sanity/val/raw/mol_results.pt",
    "SANITY\nlr=3e-4": BASE / "angles/sanity/val/lr3e-4/raw/mol_results.pt",
    "ldr=1.0\nlr=1e-3\n20k": BASE / "angles/heuristics/sweep/runs/run_20000steps_lr1e-3/run_00_ldr1p0_steps20000_lr1e-3/raw/mol_results.pt",
    "ldr=5.0\nlr=1e-3\n20k": BASE / "angles/heuristics/sweep/runs/run_20000steps_lr1e-3/run_01_ldr5p0_steps20000_lr1e-3/raw/mol_results.pt",
    "ldr=1.0\nlr=3e-4\n20k": BASE / "angles/heuristics/sweep/runs/run_20000steps_lr3e-4/run_04_ldr1p0_steps20000_lr3e-4/raw/mol_results.pt",
    "ldr=5.0\nlr=3e-4\n20k★": BASE / "angles/heuristics/sweep/runs/run_20000steps_lr3e-4/run_05_ldr5p0_steps20000_lr3e-4/raw/mol_results.pt",
    "ldr=5.0\nlr=1e-3\n50k": BASE / "angles/heuristics/sweep/runs/run_50000steps_lr1e-3/run_03_ldr5p0_steps50000_lr1e-3/raw/mol_results.pt",
    "ldr=5.0\nlr=3e-4\n50k": BASE / "angles/heuristics/sweep/runs/run_50000steps_lr3e-4/run_07_ldr5p0_steps50000_lr3e-4/raw/mol_results.pt",
    "FULL RUN\n(best cfg)": BASE / "angles/heuristics/full/raw/mol_results.pt",
}

# Build matrix: runs × molecules
run_labels = list(sweep_runs.keys())
n_runs = len(run_labels)
n_mols = len(MOLECULES)
vf_matrix = np.zeros((n_runs, n_mols))

for i, (label, path) in enumerate(sweep_runs.items()):
    r = torch.load(path, weights_only=False)
    for j, mol in enumerate(MOLECULES):
        vf_matrix[i, j] = r[mol]["valid_fraction"] * 100

# Ethanol VF per run
eth_idx = MOLECULES.index("ethanol")
mean_vf = vf_matrix.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: per-molecule heatmap
ax = axes[0]
colors = plt.cm.RdYlGn(vf_matrix / 100)
im = ax.imshow(vf_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks(range(n_mols))
ax.set_xticklabels([m[:4] for m in MOLECULES], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n_runs))
ax.set_yticklabels(run_labels, fontsize=7)
for i in range(n_runs):
    for j in range(n_mols):
        ax.text(j, i, f"{vf_matrix[i, j]:.0f}", ha="center", va="center",
                fontsize=7, color="black" if 20 < vf_matrix[i, j] < 80 else "white")
ax.set_title("Valid Fraction (%) — All Runs × All Molecules\nhyp_007 HEURISTICS sweep", fontsize=10)
plt.colorbar(im, ax=ax, label="VF (%)", fraction=0.03)

# Mark best run
best_run_idx = list(sweep_runs.keys()).index("FULL RUN\n(best cfg)")
ax.axhline(best_run_idx - 0.5, color="blue", linewidth=1.5, linestyle="--")
ax.axhline(best_run_idx + 0.5, color="blue", linewidth=1.5, linestyle="--")

# Right: ethanol VF + mean VF bar chart
ax2 = axes[1]
x = np.arange(n_runs)
w = 0.35
bars1 = ax2.bar(x - w/2, vf_matrix[:, eth_idx], w, label="Ethanol VF", color="#2196F3", alpha=0.85)
bars2 = ax2.bar(x + w/2, mean_vf, w, label="Mean VF (all mols)", color="#FF9800", alpha=0.85)
ax2.axhline(40, color="blue", linestyle="--", linewidth=1.5, label="Ethanol criterion (40%)")
ax2.axhline(30, color="orange", linestyle="--", linewidth=1.5, label="Mean criterion (30%)")
ax2.set_xticks(x)
ax2.set_xticklabels(run_labels, fontsize=7)
ax2.set_ylabel("Valid Fraction (%)")
ax2.set_title("Ethanol VF vs. Mean VF — All Sweep Runs + Full", fontsize=10)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 80)
ax2.grid(True, alpha=0.3, axis="y")
# Highlight winning region
ax2.axvspan(best_run_idx - 0.5, best_run_idx + 0.5, alpha=0.1, color="green", label="Full run")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "hyp_007_sweep_summary.png", dpi=300)
plt.close()
print(f"  Saved: {RESULTS_DIR}/hyp_007_sweep_summary.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 3: Per-Molecule VF — Full Run (bar chart)
# ─────────────────────────────────────────────────────────────
print("Generating Figure 3: Per-Molecule VF (Full Run)...")

full_results = torch.load(
    BASE / "angles/heuristics/full/raw/mol_results.pt",
    weights_only=False
)
mol_vfs = [full_results[mol]["valid_fraction"] * 100 for mol in MOLECULES]
mol_colors = ["#4CAF50" if v >= 40 else "#F44336" for v in mol_vfs]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(MOLECULES, mol_vfs, color=mol_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
ax.axhline(40, color="blue", linestyle="--", linewidth=2, label="Ethanol criterion (40%)")
ax.axhline(30, color="orange", linestyle="--", linewidth=2, label="Mean criterion (30%)")
ax.axhline(sum(mol_vfs) / len(mol_vfs), color="purple", linestyle=":", linewidth=2,
           label=f"Mean VF = {sum(mol_vfs)/len(mol_vfs):.1f}%")

for bar, vf in zip(bars, mol_vfs):
    ax.text(bar.get_x() + bar.get_width()/2, vf + 1.5, f"{vf:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Valid Fraction (%)")
ax.set_title("hyp_007 HEURISTICS Full Run — Per-Molecule Valid Fraction\n"
             "(log_det_reg_weight=5.0, lr=3e-4, 20k steps, best ckpt @ step 12000)")
ax.set_ylim(0, 80)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=15, ha="right")

# Annotate aspirin as outlier
ax.annotate("Aspirin: 9.2%\n(21 atoms, hardest)", xy=(0, 9.2), xytext=(1.5, 30),
            fontsize=8, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))

plt.tight_layout()
fig.savefig(RESULTS_DIR / "hyp_007_per_molecule_vf.png", dpi=300)
plt.close()
print(f"  Saved: {RESULTS_DIR}/hyp_007_per_molecule_vf.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 4: Training Dynamics — Loss Curve (from log)
# ─────────────────────────────────────────────────────────────
print("Generating Figure 4: Training Dynamics (Loss Curve)...")

# Parse training log
log_path = "/tmp/hyp007_heuristics_full.log"
steps, losses, nll_only, log_det_dof, val_losses = [], [], [], [], []
val_steps = []

with open(log_path) as f:
    for line in f:
        if "Step " in line and "/20000" in line and "loss=" in line:
            parts = line.strip().split("|")
            try:
                step = int(parts[0].strip().split("/")[0].replace("Step ", "").strip())
                loss = float([p for p in parts if "loss=" in p][0].split("loss=")[1].strip())
                nll = float([p for p in parts if "nll_only=" in p][0].split("nll_only=")[1].strip())
                ldd = float([p for p in parts if "log_det/dof=" in p][0].split("log_det/dof=")[1].strip())
                steps.append(step)
                losses.append(loss)
                nll_only.append(nll)
                log_det_dof.append(ldd)
            except (IndexError, ValueError):
                pass
        elif "Val loss:" in line:
            try:
                vl = float(line.split("Val loss:")[1].split("|")[0].strip())
                val_losses.append(vl)
                if steps:
                    val_steps.append(steps[-1])
            except (IndexError, ValueError):
                pass

steps = np.array(steps)
losses = np.array(losses)
nll_only = np.array(nll_only)
log_det_dof = np.array(log_det_dof)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# Top: loss curves
ax = axes[0]
ax.plot(steps, losses, alpha=0.4, color="#2196F3", linewidth=0.8, label="Train loss")
# Smooth
if len(steps) > 20:
    from numpy.lib.stride_tricks import sliding_window_view
    w = min(40, len(steps) // 5)
    smooth = np.convolve(losses, np.ones(w)/w, mode="valid")
    smooth_steps = steps[w//2:w//2 + len(smooth)]
    ax.plot(smooth_steps, smooth, color="#1565C0", linewidth=2, label=f"Train loss (smoothed, w={w})")

if val_steps and val_losses:
    ax.plot(val_steps, val_losses, "o-", color="#F44336", linewidth=2, markersize=5, label="Val NLL")
    best_val_idx = np.argmin(val_losses)
    ax.axvline(val_steps[best_val_idx], color="green", linestyle="--", linewidth=1.5,
               label=f"Best val step {val_steps[best_val_idx]} (loss={val_losses[best_val_idx]:.4f})")
    ax.axvline(12000, color="purple", linestyle=":", linewidth=1.5, label="Best ckpt @ step 12000")

ax.set_ylabel("Loss")
ax.set_title("hyp_007 HEURISTICS Full Run — Training Dynamics\n"
             "(log_det_reg_weight=5.0, lr=3e-4, 20k steps cosine schedule)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20000)

# Bottom: log_det/dof
ax2 = axes[1]
ax2.plot(steps, log_det_dof, alpha=0.5, color="#FF9800", linewidth=0.8, label="log_det/dof (train)")
if len(steps) > 20:
    smooth_ldd = np.convolve(log_det_dof, np.ones(w)/w, mode="valid")
    ax2.plot(smooth_steps, smooth_ldd, color="#E65100", linewidth=2, label=f"Smoothed")
ax2.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="0.0 reference")
ax2.set_xlabel("Training Step")
ax2.set_ylabel("log_det / dof")
ax2.set_title("Log-Determinant / DoF — Log-Det Exploitation Check\n"
              "(regularized: stays bounded near ~0.09, no runaway)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 20000)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "hyp_007_training_dynamics.png", dpi=300)
plt.close()
print(f"  Saved: {RESULTS_DIR}/hyp_007_training_dynamics.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 5: Min pairwise distance histograms (ethanol)
# ─────────────────────────────────────────────────────────────
print("Generating Figure 5: Min pairwise distance distribution (ethanol)...")

# Generate fresh samples from best checkpoint for ethanol
sys.path.insert(0, ".")
import torch as T

ckpt_path = BASE / "angles/heuristics/full/best.pt"
ckpt = T.load(ckpt_path, map_location="cpu", weights_only=False)
from src.model import TarFlow
from src.data import MD17Dataset, MAX_ATOMS, PAD_TOKEN_IDX

cfg = ckpt["config"]
n_atom_types = cfg.get("n_atom_types", 5)  # stored in config; default 5 for use_pad_token=True
model = TarFlow(
    n_blocks=cfg.get("n_blocks", 8),
    d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4),
    ffn_mult=cfg.get("ffn_mult", 4),
    atom_type_emb_dim=cfg.get("atom_type_emb_dim", 16),
    n_atom_types=n_atom_types,
    dropout=0.0,
    max_atoms=cfg.get("max_atoms_resolved", MAX_ATOMS),
    alpha_pos=cfg.get("alpha_pos", 10.0),
    alpha_neg=cfg.get("alpha_neg", 10.0),
    shift_only=False,
    use_actnorm=False,
    use_bidir_types=cfg.get("use_bidir_types", True),
    use_pos_enc=cfg.get("use_pos_enc", False),
    zero_padding_queries=cfg.get("zero_padding_queries", True),
    use_output_shift=cfg.get("use_output_shift", True),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load ethanol dataset
global_std_val = cfg.get("global_std", None)
eth_ds = MD17Dataset(
    "data/md17_ethanol_v1",
    split="test",
    augment=False,
    global_std=global_std_val,
    pad_token_idx=PAD_TOKEN_IDX,
    noise_sigma=0.0,
    max_atoms=cfg.get("max_atoms_resolved", MAX_ATOMS),
)

sample0 = eth_ds[0]
atom_types = sample0["atom_types"]
mask = sample0["mask"]

with T.no_grad():
    samples = model.sample(atom_types, mask, n_samples=500)  # (500, max_atoms, 3)

# Denormalize — global_std is stored in cfg (not norm_params)
global_std = cfg.get("global_std", None)
if global_std is not None:
    samples = samples * global_std
    print(f"  Denormalized samples with global_std={global_std:.4f}")

# Compute min pairwise distances for generated samples
def min_pairwise_dist(pos, mask):
    """pos: (N, 3), mask: (N,). Returns min distance among real atoms."""
    real_pos = pos[mask.bool()]
    n = real_pos.shape[0]
    if n < 2:
        return float("inf")
    diffs = real_pos.unsqueeze(0) - real_pos.unsqueeze(1)  # (N, N, 3)
    dists = diffs.norm(dim=-1)
    dists = dists + T.eye(n) * 1e9
    return dists.min().item()

gen_min_dists = []
for i in range(samples.shape[0]):
    gen_min_dists.append(min_pairwise_dist(samples[i], mask))

# Load reference min distances
ref_positions = T.tensor(eth_ds.positions[:500])  # (500, max_atoms, 3)
ref_min_dists = []
for i in range(min(500, ref_positions.shape[0])):
    ref_min_dists.append(min_pairwise_dist(ref_positions[i], mask))

gen_min_dists = np.array(gen_min_dists)
ref_min_dists = np.array(ref_min_dists)

COLLISION_THRESHOLD = 0.8
valid_fraction = (gen_min_dists >= COLLISION_THRESHOLD).mean()

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 2.0, 50)
ax.hist(ref_min_dists, bins=bins, alpha=0.6, color="#FF9800", label=f"Reference (N={len(ref_min_dists)})", density=True)
ax.hist(gen_min_dists, bins=bins, alpha=0.6, color="#2196F3", label=f"Generated (N={len(gen_min_dists)})", density=True)
ax.axvline(COLLISION_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Collision threshold ({COLLISION_THRESHOLD} Å)")
ax.set_xlabel("Min Pairwise Distance (Å)")
ax.set_ylabel("Density")
ax.set_title(f"hyp_007 HEURISTICS Full Run — Ethanol Min Pairwise Distance\n"
             f"Generated vs. Reference | Valid Fraction = {valid_fraction*100:.1f}%")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate(f"VF = {valid_fraction*100:.1f}%\n(samples ≥ {COLLISION_THRESHOLD} Å)",
            xy=(COLLISION_THRESHOLD + 0.05, ax.get_ylim()[1] * 0.7),
            fontsize=10, color="blue", fontweight="bold")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "hyp_007_ethanol_min_dist.png", dpi=300)
plt.close()
print(f"  Saved: {RESULTS_DIR}/hyp_007_ethanol_min_dist.png")
print(f"  Ethanol VF from samples: {valid_fraction*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# FIGURE 6: log_det_reg_weight ablation (key finding)
# ─────────────────────────────────────────────────────────────
print("Generating Figure 6: log_det_reg_weight ablation...")

ldr_runs = {
    "ldr=0\nSANITY\nlr=1e-3": (0.0, 1e-3, 17.6),
    "ldr=0\nSANITY\nlr=3e-4": (0.0, 3e-4, 12.2),
    "ldr=1.0\nlr=1e-3\n20k": (1.0, 1e-3, 36.2),
    "ldr=1.0\nlr=3e-4\n20k": (1.0, 3e-4, 40.4),
    "ldr=5.0\nlr=1e-3\n20k": (5.0, 1e-3, 54.4),
    "ldr=5.0\nlr=3e-4\n20k": (5.0, 3e-4, 55.8),
    "ldr=5.0\nlr=1e-3\n50k": (5.0, 1e-3, 50.0),
    "ldr=5.0\nlr=3e-4\n50k": (5.0, 3e-4, 52.8),
    "FULL\nldr=5.0\nlr=3e-4": (5.0, 3e-4, 55.8),
}

labels = list(ldr_runs.keys())
eth_vfs = [v[2] for v in ldr_runs.values()]
ldrs = [v[0] for v in ldr_runs.values()]

colors = []
for ldr in ldrs:
    if ldr == 0.0:
        colors.append("#F44336")
    elif ldr == 1.0:
        colors.append("#FF9800")
    else:
        colors.append("#4CAF50")

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(labels)), eth_vfs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
ax.axhline(40, color="blue", linestyle="--", linewidth=2, label="Criterion: ethanol VF > 40%")

for bar, vf in zip(bars, eth_vfs):
    ax.text(bar.get_x() + bar.get_width()/2, vf + 1, f"{vf:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#F44336", alpha=0.85, label="ldr=0 (no regularization)"),
    Patch(facecolor="#FF9800", alpha=0.85, label="ldr=1.0"),
    Patch(facecolor="#4CAF50", alpha=0.85, label="ldr=5.0"),
]
ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=2, label="Criterion (40%)")],
          fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Ethanol Valid Fraction (%)")
ax.set_title("hyp_007 — Key Finding: log_det_reg_weight Controls Quality\n"
             "ldr=5.0 pushes all runs above 50% ethanol VF (criterion: >40%)")
ax.set_ylim(0, 80)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "hyp_007_ldr_ablation.png", dpi=300)
plt.close()
print(f"  Saved: {RESULTS_DIR}/hyp_007_ldr_ablation.png")


print("\nAll figures saved to:", RESULTS_DIR)
print("Done.")
