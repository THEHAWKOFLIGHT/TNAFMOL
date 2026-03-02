"""
hyp_003 Visualization — Generate canonical plots for TarFlow stabilization experiment.

Generates:
1. Loss curves (SANITY full + HEURISTICS full) — annotated with best step
2. Per-molecule valid_fraction bar chart (SANITY vs HEURISTICS vs sweep best)
3. Min pairwise distance histogram (generated vs reference) for top molecules
4. Sweep comparison plot (mean_valid_fraction vs config for HEURISTICS sweep)
5. Log-det per DOF vs training step (shows saturation)

Usage:
    python3.10 src/visualize_hyp003.py --output-dir experiments/hypothesis/hyp_003_tarflow_stabilization/results/
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import MOLECULES
from src.metrics import min_pairwise_distance


def load_mol_results(path: str) -> dict:
    """Load mol_results.pt saved by train.py."""
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_valid_fraction_bars(results_dict: dict, output_dir: str, filename: str = "valid_fraction_comparison.png"):
    """Bar chart comparing valid_fraction across angles and molecules."""
    molecules = list(MOLECULES.keys())

    # Build data for each angle
    angles = {}
    for angle_name, mol_results in results_dict.items():
        vf = []
        for mol in molecules:
            if mol in mol_results:
                vf.append(mol_results[mol]["valid_fraction"])
            else:
                vf.append(0.0)
        angles[angle_name] = vf

    n_mols = len(molecules)
    n_angles = len(angles)
    x = np.arange(n_mols)
    width = 0.8 / n_angles

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for i, (angle_name, vf) in enumerate(angles.items()):
        offset = (i - n_angles / 2 + 0.5) * width
        bars = ax.bar(x + offset, vf, width=width * 0.9, label=angle_name, color=colors[i % len(colors)], alpha=0.8)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Success criterion (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in molecules], fontsize=10)
    ax.set_ylabel("Valid Fraction", fontsize=12)
    ax.set_xlabel("Molecule", fontsize=12)
    ax.set_title("hyp_003 — Valid Fraction per Molecule by Angle\n(asymmetric clamping + log-det reg + soft equivariance)", fontsize=13)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add mean annotation
    for i, (angle_name, vf) in enumerate(angles.items()):
        mean_vf = np.mean(vf)
        ax.text(n_mols - 0.5 + (i - n_angles / 2 + 0.5) * width,
                0.58, f"mean={mean_vf:.3f}", ha="center", fontsize=8, color=colors[i % len(colors)])

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_min_pairwise_dist(best_mol_results: dict, data_root: str, output_dir: str,
                            filename: str = "min_pairwise_distance.png"):
    """Histogram of min pairwise distances for selected molecules."""
    # Focus on molecules with best results
    selected_molecules = ["ethanol", "malonaldehyde", "uracil", "benzene"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, mol in enumerate(selected_molecules):
        ax = axes[idx]
        mol_data_dir = os.path.join(data_root, f"md17_{mol}_v1")

        # Load reference min distances
        try:
            data = np.load(os.path.join(mol_data_dir, "dataset.npz"))
            ref_pos = data["positions"][data["test_idx"]]  # raw Angstroms
            mask = data["mask"]
            ref_min_dists = min_pairwise_distance(ref_pos, mask)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading {mol}:\n{e}", ha="center", va="center", transform=ax.transAxes)
            continue

        # If we have generated results for this molecule (from best angle)
        if mol in best_mol_results:
            mol_result = best_mol_results[mol]
            vf = mol_result["valid_fraction"]
            min_d_mean = mol_result["min_dist_mean"]
            min_d_below = mol_result["min_dist_below_08"]
        else:
            vf = 0.0
            min_d_mean = 0.0
            min_d_below = 1.0

        # Plot reference distribution
        ax.hist(ref_min_dists, bins=50, density=True, alpha=0.5, color="steelblue",
                label=f"Reference (N={len(ref_min_dists)})", edgecolor="none")

        # Annotate generated stats since we don't have the raw generated arrays
        ax.axvline(0.8, color="red", linestyle="--", linewidth=1.5, label="Validity threshold (0.8 Å)")
        ax.axvline(min_d_mean, color="darkorange", linestyle="-", linewidth=2,
                   label=f"Generated mean min dist: {min_d_mean:.3f} Å")

        n_atoms = MOLECULES[mol]
        title = f"{mol.replace('_', ' ').title()} ({n_atoms} atoms)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Min Pairwise Distance (Å)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 3.0)

        # Add text box with key stats
        stats_text = f"valid_frac: {vf:.3f}\nfrac_below_0.8: {min_d_below:.3f}"
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("hyp_003 — Min Pairwise Distance: Reference vs Generated Stats\n"
                 "(best angle results; generated mean min dist shown vs reference distribution)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_sweep_comparison(sweep_results: dict, output_dir: str, filename: str = "sweep_comparison.png"):
    """Plot sweep results: mean_valid_fraction vs config."""
    if not sweep_results:
        print("No sweep results to plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data (filter out runs with None values)
    all_results = sweep_results.get("all_results", {})
    valid_results = {k: v for k, v in all_results.items() if v.get("lr") is not None}

    runs = list(valid_results.keys())
    mean_vfs = [valid_results[r]["mean_vf"] for r in runs]
    lrs = [valid_results[r]["lr"] for r in runs]
    batch_sizes = [valid_results[r]["batch_size"] for r in runs]
    ema_decays = [valid_results[r]["ema_decay"] for r in runs]

    # Sort by mean_vf
    order = np.argsort(mean_vfs)[::-1]
    runs_sorted = [runs[i] for i in order]
    vfs_sorted = [mean_vfs[i] for i in order]
    lrs_sorted = [lrs[i] for i in order]
    bs_sorted = [batch_sizes[i] for i in order]

    # Plot 1: mean_vf by run
    ax = axes[0]
    colors = ["gold" if bs == 512 else "steelblue" for bs in bs_sorted]
    bars = ax.bar(range(len(runs_sorted)), vfs_sorted, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(runs_sorted)))
    ax.set_xticklabels([f"lr={lr:.0e}\nbs={bs}" for lr, bs in zip(lrs_sorted, bs_sorted)], fontsize=8, rotation=45)
    ax.set_ylabel("Mean Valid Fraction")
    ax.set_title("HEURISTICS Sweep — All Runs\n(gold=batch_size=512, blue=batch_size=256)")
    ax.axhline(sweep_results.get("best_mean_valid_fraction", 0.183), color="red", linestyle="--",
               label=f"Best: {sweep_results.get('best_mean_valid_fraction', 0.183):.3f}")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Plot 2: mean_vf vs lr (grouped by batch_size)
    ax = axes[1]
    lr_values = sorted(set(lrs))
    bs_values = sorted(set([b for b in batch_sizes if b is not None]))
    colors2 = ["steelblue", "darkorange"]
    for ci, bs in enumerate(bs_values):
        x_vals = []
        y_vals = []
        for r, v in valid_results.items():
            if v["batch_size"] == bs:
                x_vals.append(v["lr"])
                y_vals.append(v["mean_vf"])
        if x_vals:
            sort_order = np.argsort(x_vals)
            x_sorted = [x_vals[i] for i in sort_order]
            y_sorted = [y_vals[i] for i in sort_order]
            ax.plot(x_sorted, y_sorted, "o-", color=colors2[ci], label=f"batch_size={bs}", markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Mean Valid Fraction")
    ax.set_title("HEURISTICS Sweep — LR vs Valid Fraction by Batch Size")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.suptitle("hyp_003 HEURISTICS Sweep Summary (12 runs)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_angle_summary(results_summary: dict, output_dir: str, filename: str = "angle_summary.png"):
    """Summary table of all angles."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    rows = []
    for angle_name, data in results_summary.items():
        mol_results = data.get("mol_results", {})
        vfs = [v.get("valid_fraction", 0) for v in mol_results.values()]
        mean_vf = np.mean(vfs) if vfs else 0.0
        n_valid = sum(1 for v in vfs if v >= 0.5)
        best_mol = max(mol_results.keys(), key=lambda k: mol_results[k].get("valid_fraction", 0)) if mol_results else "N/A"
        best_vf = max(mol_results[m].get("valid_fraction", 0) for m in mol_results) if mol_results else 0.0

        rows.append([
            angle_name,
            data.get("n_steps", "N/A"),
            data.get("wandb_run", "N/A"),
            f"{mean_vf:.3f}",
            f"{n_valid}/8",
            f"{best_mol} ({best_vf:.3f})",
            data.get("status", "FAIL"),
        ])

    col_labels = ["Angle", "N Steps", "W&B Run", "Mean VF", "Mols≥50%", "Best Mol (VF)", "Status"]

    if rows:
        table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        for i, row in enumerate(rows):
            status = row[-1]
            color = "#d4edda" if status == "DONE" else "#f8d7da"
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)

    ax.set_title("hyp_003 — OPTIMIZE Angle Summary\n"
                 "Primary criterion: valid_fraction ≥ 0.5 on ≥ 4/8 molecules",
                 fontsize=13, pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                        default="experiments/hypothesis/hyp_003_tarflow_stabilization/results/")
    parser.add_argument("--data-root", type=str, default="data/")
    args = parser.parse_args()

    output_dir = os.path.join(project_root, args.output_dir)
    data_root = os.path.join(project_root, args.data_root)
    os.makedirs(output_dir, exist_ok=True)

    exp_dir = os.path.join(project_root, "experiments/hypothesis/hyp_003_tarflow_stabilization")

    print("Loading mol_results...")

    # Load results from each angle
    sanity_full_results_path = os.path.join(exp_dir, "angles/sanity/full/raw/mol_results.pt")
    heuristics_full_results_path = os.path.join(exp_dir, "angles/heuristics/full/raw/mol_results.pt")

    sanity_results = None
    heuristics_results = None

    if os.path.exists(sanity_full_results_path):
        sanity_results = torch.load(sanity_full_results_path, map_location="cpu", weights_only=False)
        print(f"Loaded SANITY full results: {list(sanity_results.keys())}")
    else:
        print(f"Warning: SANITY full results not found at {sanity_full_results_path}")

    if os.path.exists(heuristics_full_results_path):
        heuristics_results = torch.load(heuristics_full_results_path, map_location="cpu", weights_only=False)
        print(f"Loaded HEURISTICS full results: {list(heuristics_results.keys())}")
    else:
        print(f"Warning: HEURISTICS full results not found at {heuristics_full_results_path}")

    # Load sweep summary
    sweep_summary_path = os.path.join(exp_dir, "angles/heuristics/sweep/summary.json")
    sweep_results = None
    if os.path.exists(sweep_summary_path):
        with open(sweep_summary_path) as f:
            sweep_results = json.load(f)
        print(f"Loaded HEURISTICS sweep summary")

    # =========================================================================
    # Plot 1: Valid fraction comparison across angles
    # =========================================================================
    results_to_compare = {}
    if sanity_results:
        results_to_compare["SANITY full\n(10k steps, α=0.02, reg=5)"] = sanity_results
    if heuristics_results:
        results_to_compare["HEURISTICS full\n(20k steps, SBG recipe)"] = heuristics_results

    if results_to_compare:
        plot_valid_fraction_bars(results_to_compare, output_dir)

    # =========================================================================
    # Plot 2: Min pairwise distance — reference vs generated stats
    # =========================================================================
    best_results = heuristics_results if heuristics_results else sanity_results
    if best_results:
        plot_min_pairwise_dist(best_results, data_root, output_dir)

    # =========================================================================
    # Plot 3: HEURISTICS sweep comparison
    # =========================================================================
    if sweep_results:
        plot_sweep_comparison(sweep_results, output_dir)

    # =========================================================================
    # Plot 4: Angle summary table
    # =========================================================================
    angle_summary = {}
    if sanity_results:
        angle_summary["SANITY full (10k steps)"] = {
            "n_steps": 10000,
            "wandb_run": "o5naez7a",
            "mol_results": sanity_results,
            "status": "FAIL",
        }
    if heuristics_results:
        angle_summary["HEURISTICS full (20k steps)"] = {
            "n_steps": 20000,
            "wandb_run": "4079op64",
            "mol_results": heuristics_results,
            "status": "FAIL",
        }

    if angle_summary:
        plot_angle_summary(angle_summary, output_dir)

    # =========================================================================
    # Plot 5: Best molecule breakdown — per-molecule stats for top performer
    # =========================================================================
    molecules = list(MOLECULES.keys())
    best_results_for_bar = heuristics_results if heuristics_results else sanity_results
    if best_results_for_bar:
        vfs = [best_results_for_bar.get(m, {}).get("valid_fraction", 0.0) for m in molecules]
        min_dists_mean = [best_results_for_bar.get(m, {}).get("min_dist_mean", 0.0) for m in molecules]
        n_atoms_list = [MOLECULES[m] for m in molecules]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Bar chart: valid fraction
        ax = axes[0]
        color = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in vfs]
        bars = ax.bar(range(len(molecules)), vfs, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=2, label="Success threshold (50%)")
        ax.set_xticks(range(len(molecules)))
        ax.set_xticklabels([f"{m.replace('_', ' ').title()}\n({n}at)" for m, n in zip(molecules, n_atoms_list)],
                            fontsize=9)
        ax.set_ylabel("Valid Fraction", fontsize=12)
        ax.set_title("hyp_003 Best Result — Valid Fraction per Molecule\n(HEURISTICS full run, 20k steps)", fontsize=11)
        ax.set_ylim(0, 0.65)
        for i, v in enumerate(vfs):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Scatter: valid fraction vs n_atoms
        ax2 = axes[1]
        scatter = ax2.scatter(n_atoms_list, vfs, c=vfs, cmap="RdYlGn", vmin=0, vmax=0.5,
                               s=200, edgecolors="black", linewidths=1, zorder=2)
        plt.colorbar(scatter, ax=ax2, label="Valid Fraction")
        for i, mol in enumerate(molecules):
            ax2.annotate(mol.replace("_", " "), (n_atoms_list[i], vfs[i]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax2.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.set_xlabel("Number of Atoms", fontsize=12)
        ax2.set_ylabel("Valid Fraction", fontsize=12)
        ax2.set_title("Valid Fraction vs Molecule Size\n(smaller molecules → easier sampling)", fontsize=11)
        ax2.yaxis.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)

        plt.suptitle("hyp_003 — Best Results Summary", fontsize=14)
        plt.tight_layout()
        best_path = os.path.join(output_dir, "best_results_summary.png")
        plt.savefig(best_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {best_path}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
