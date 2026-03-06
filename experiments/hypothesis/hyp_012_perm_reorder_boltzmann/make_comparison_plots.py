"""
Hyp_012: Head-to-head comparison plots — Arm A (canonical) vs Arm B (permute_within_types).

Usage:
    python experiments/hypothesis/hyp_012_perm_reorder_boltzmann/make_comparison_plots.py

Reads:
    angles/arm_a_sanity/full/raw/mol_results.pt
    angles/arm_b_sanity/full/raw/mol_results.pt
    angles/arm_a_sanity/full/raw/train_losses.npy  (if available)
    angles/arm_b_sanity/full/raw/train_losses.npy  (if available)

Outputs to results/:
    hyp_012_vf_comparison.png    — side-by-side per-molecule VF bar chart
    hyp_012_loss_comparison.png  — overlaid loss curves for both arms
"""

import os
import sys
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARM_A_RAW = os.path.join(BASE_DIR, "angles", "arm_a_sanity", "full", "raw")
ARM_B_RAW = os.path.join(BASE_DIR, "angles", "arm_b_sanity", "full", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_mol_results(raw_dir, arm_name):
    """Load per-molecule results from raw directory."""
    path = os.path.join(raw_dir, "mol_results.pt")
    if not os.path.exists(path):
        print(f"WARNING: mol_results.pt not found for {arm_name} at {path}")
        return None
    results = torch.load(path)
    print(f"\n{arm_name} mol results:")
    vfs = []
    for mol in sorted(results.keys()):
        vf = results[mol]["valid_fraction"]
        vfs.append(vf)
        print(f"  {mol}: VF={vf:.3f}")
    mean_vf = np.mean(vfs)
    print(f"  Mean VF: {mean_vf:.3f}")
    return results


def load_losses(raw_dir, arm_name):
    """Load training and val loss arrays if available."""
    train_path = os.path.join(raw_dir, "train_losses.npy")
    val_path = os.path.join(raw_dir, "val_losses.npy")
    logdet_path = os.path.join(raw_dir, "logdets.npy")

    train_losses = np.load(train_path) if os.path.exists(train_path) else None
    val_losses = np.load(val_path) if os.path.exists(val_path) else None
    logdets = np.load(logdet_path) if os.path.exists(logdet_path) else None

    if train_losses is not None:
        print(f"{arm_name}: {len(train_losses)} training loss steps loaded")
    return train_losses, val_losses, logdets


def plot_vf_comparison(arm_a_results, arm_b_results, out_dir):
    """Side-by-side per-molecule VF bar chart."""
    molecules = sorted(arm_a_results.keys())
    a_vfs = [arm_a_results[m]["valid_fraction"] for m in molecules]
    b_vfs = [arm_b_results[m]["valid_fraction"] for m in molecules] if arm_b_results else [0.0] * len(molecules)

    a_mean = np.mean(a_vfs)
    b_mean = np.mean(b_vfs)

    x = np.arange(len(molecules))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))

    bars_a = ax.bar(x - width / 2, a_vfs, width, label=f"Arm A: canonical (mean={a_mean:.3f})",
                    color="#2196F3", alpha=0.85, edgecolor="navy")
    bars_b = ax.bar(x + width / 2, b_vfs, width, label=f"Arm B: perm_within_types (mean={b_mean:.3f})",
                    color="#FF9800", alpha=0.85, edgecolor="darkorange")

    ax.axhline(0.9, color="green", linestyle="--", linewidth=1.2, label="90% target")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.0, label="50% threshold")

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valid Fraction at T=1.0", fontsize=12)
    ax.set_title("hyp_012 — Per-Molecule Valid Fraction: Arm A vs Arm B\n(50k steps, SANITY angle, T=1.0)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in molecules], fontsize=10)
    ax.legend(fontsize=10)

    # Annotate bars
    for bar, v in zip(bars_a, a_vfs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="navy")
    for bar, v in zip(bars_b, b_vfs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="darkorange")

    plt.tight_layout()
    path = os.path.join(out_dir, "hyp_012_vf_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved VF comparison plot: {path}")
    return path


def _train_steps(n):
    """Reconstruct step values from train loss log: step 1 first, then every 50."""
    if n == 0:
        return np.array([])
    steps = [1] + list(range(50, 50 * n, 50))
    return np.array(steps[:n])


def plot_loss_comparison(a_train, a_val, a_logdets, b_train, b_val, b_logdets, out_dir,
                         val_interval=1000):
    """Overlaid training loss curves for both arms."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    if a_train is not None:
        a_steps = _train_steps(len(a_train))
        ax.plot(a_steps, a_train, label="Arm A train loss", color="#2196F3", alpha=0.6, linewidth=0.8)
    if b_train is not None:
        b_steps = _train_steps(len(b_train))
        ax.plot(b_steps, b_train, label="Arm B train loss", color="#FF9800", alpha=0.6, linewidth=0.8)
    if a_val is not None:
        a_val_steps = np.arange(1, len(a_val) + 1) * val_interval
        ax.plot(a_val_steps, a_val, label="Arm A val loss", color="#0D47A1", linewidth=1.5, marker="o", markersize=4)
    if b_val is not None:
        b_val_steps = np.arange(1, len(b_val) + 1) * val_interval
        ax.plot(b_val_steps, b_val, label="Arm B val loss", color="#E65100", linewidth=1.5, marker="s", markersize=4)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (NLL + reg)")
    ax.set_title("Training Dynamics — Arm A vs Arm B")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if a_logdets is not None:
        ld_steps = _train_steps(len(a_logdets))
        ax2.plot(ld_steps, a_logdets, label="Arm A logdets", color="#2196F3", alpha=0.7, linewidth=0.8)
    if b_logdets is not None:
        ld_steps = _train_steps(len(b_logdets))
        ax2.plot(ld_steps, b_logdets, label="Arm B logdets", color="#FF9800", alpha=0.7, linewidth=0.8)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Mean |log det|")
    ax2.set_title("Log Determinant Magnitude — Arm A vs Arm B")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("hyp_012 — 50k step full runs, SANITY angle", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "hyp_012_loss_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss comparison plot: {path}")
    return path


def main():
    print("=== hyp_012 Head-to-Head Comparison ===")

    # Load arm results
    arm_a_results = load_mol_results(ARM_A_RAW, "Arm A (canonical)")
    arm_b_results = load_mol_results(ARM_B_RAW, "Arm B (perm_within_types)")

    if arm_a_results is None:
        print("ERROR: Arm A results not found. Have the full runs completed?")
        sys.exit(1)

    # Load loss arrays
    a_train, a_val, a_logdets = load_losses(ARM_A_RAW, "Arm A")
    b_train, b_val, b_logdets = load_losses(ARM_B_RAW, "Arm B")

    # Summary
    a_vfs = [arm_a_results[m]["valid_fraction"] for m in arm_a_results]
    print(f"\n=== SUMMARY ===")
    print(f"Arm A mean VF: {np.mean(a_vfs):.3f}")
    if arm_b_results:
        b_vfs = [arm_b_results[m]["valid_fraction"] for m in arm_b_results]
        print(f"Arm B mean VF: {np.mean(b_vfs):.3f}")
        print(f"Delta (B - A): {np.mean(b_vfs) - np.mean(a_vfs):+.3f}")
        print(f"\nPer-molecule breakdown:")
        for mol in sorted(arm_a_results.keys()):
            a_vf = arm_a_results[mol]["valid_fraction"]
            b_vf = arm_b_results.get(mol, {}).get("valid_fraction", float("nan"))
            delta = b_vf - a_vf
            better = "B wins" if delta > 0.01 else ("tie" if abs(delta) <= 0.01 else "A wins")
            print(f"  {mol:20s}: A={a_vf:.3f}  B={b_vf:.3f}  delta={delta:+.3f}  {better}")

    # Generate plots
    plot_vf_comparison(arm_a_results, arm_b_results, RESULTS_DIR)
    plot_loss_comparison(a_train, a_val, a_logdets, b_train, b_val, b_logdets, RESULTS_DIR)

    print(f"\nAll plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
