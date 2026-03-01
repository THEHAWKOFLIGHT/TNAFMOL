"""
TNAFMOL Visualization — Generate verification plots for the data pipeline.

Usage:
    python src/visualize.py --data-root data/ --output-dir experiments/hypothesis/hyp_001_data_pipeline/results/
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import MOLECULES, MAX_ATOMS


def load_molecule_data(data_dir: str):
    """Load processed molecule data."""
    data = np.load(os.path.join(data_dir, "dataset.npz"))
    return {
        "positions": data["positions"],
        "energies": data["energies"],
        "atom_types": data["atom_types"],
        "mask": data["mask"],
        "atomic_numbers": data["atomic_numbers"],
        "train_idx": data["train_idx"],
        "val_idx": data["val_idx"],
        "test_idx": data["test_idx"],
    }


def plot_energy_distributions(data_root: str, output_dir: str):
    """Plot energy distributions for all molecules on a grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (mol, n_atoms) in enumerate(MOLECULES.items()):
        data_dir = os.path.join(data_root, f"md17_{mol}_v1")
        if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
            axes[i].text(0.5, 0.5, f"{mol}\n(not found)", ha="center", va="center")
            continue

        data = load_molecule_data(data_dir)
        E = data["energies"]

        axes[i].hist(E, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        axes[i].set_title(f"{mol.replace('_', ' ').title()} ({n_atoms} atoms)", fontsize=12)
        axes[i].set_xlabel("Energy (kcal/mol)")
        axes[i].set_ylabel("Density")
        axes[i].axvline(E.mean(), color="red", linestyle="--", linewidth=1, label=f"mean={E.mean():.1f}")
        axes[i].legend(fontsize=8)

    plt.suptitle("MD17 Energy Distributions (DFT)", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "energy_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_pairwise_distances(data_root: str, output_dir: str):
    """Plot pairwise distance distributions for all molecules."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (mol, n_atoms) in enumerate(MOLECULES.items()):
        data_dir = os.path.join(data_root, f"md17_{mol}_v1")
        if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
            axes[i].text(0.5, 0.5, f"{mol}\n(not found)", ha="center", va="center")
            continue

        data = load_molecule_data(data_dir)
        pos = data["positions"]
        mask = data["mask"]
        real_idx = np.where(mask > 0.5)[0]

        # Subsample for speed
        n_sub = min(1000, len(pos))
        sub_pos = pos[:n_sub, real_idx, :]

        # Compute pairwise distances
        diff = sub_pos[:, :, np.newaxis, :] - sub_pos[:, np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        triu_i, triu_j = np.triu_indices(len(real_idx), k=1)
        all_dists = dist[:, triu_i, triu_j].flatten()

        axes[i].hist(all_dists, bins=200, range=(0.5, 8.0), density=True,
                     alpha=0.7, color="darkorange", edgecolor="none")
        axes[i].set_title(f"{mol.replace('_', ' ').title()}", fontsize=12)
        axes[i].set_xlabel("Distance (Angstrom)")
        axes[i].set_ylabel("Density")
        axes[i].axvline(0.8, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        axes[i].axvline(2.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.suptitle("MD17 Pairwise Distance Distributions", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "pairwise_distances.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_sample_conformations(data_root: str, output_dir: str):
    """Plot 2D projections of sample conformations for selected molecules."""
    # Select 3 diverse molecules
    selected = ["aspirin", "benzene", "ethanol"]

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    color_map = {0: "white", 1: "gray", 2: "blue", 3: "red"}  # H, C, N, O
    size_map = {0: 30, 1: 80, 2: 80, 3: 80}

    for row, mol in enumerate(selected):
        data_dir = os.path.join(data_root, f"md17_{mol}_v1")
        if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
            continue

        data = load_molecule_data(data_dir)
        pos = data["positions"]
        mask = data["mask"]
        atom_types = data["atom_types"]
        real_idx = np.where(mask > 0.5)[0]

        # Pick 5 random conformations
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(pos), size=5, replace=False)

        for col, si in enumerate(sample_idx):
            ax = axes[row, col]
            real_pos = pos[si, real_idx, :]

            # XY projection
            colors = [color_map.get(int(atom_types[j]), "gray") for j in real_idx]
            sizes = [size_map.get(int(atom_types[j]), 50) for j in real_idx]

            ax.scatter(real_pos[:, 0], real_pos[:, 1], c=colors, s=sizes,
                      edgecolors="black", linewidths=0.5, zorder=2)

            # Draw bonds (distance < 1.8 Angstrom)
            n_real = len(real_idx)
            for a in range(n_real):
                for b in range(a + 1, n_real):
                    d = np.sqrt(((real_pos[a] - real_pos[b]) ** 2).sum())
                    if d < 1.8:
                        ax.plot([real_pos[a, 0], real_pos[b, 0]],
                               [real_pos[a, 1], real_pos[b, 1]],
                               "k-", linewidth=0.8, zorder=1)

            ax.set_aspect("equal")
            ax.set_title(f"{mol} #{si}" if col == 0 else f"#{si}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{mol.replace('_', ' ').title()}", fontsize=12)
            ax.set_xlabel("X (Angstrom)")

    plt.suptitle("Sample Conformations (XY Projection)", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "sample_conformations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_dataset_summary(data_root: str, output_dir: str):
    """Summary statistics table as a figure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.axis("off")

    # Collect data
    rows = []
    for mol, n_atoms in MOLECULES.items():
        data_dir = os.path.join(data_root, f"md17_{mol}_v1")
        if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
            continue
        data = np.load(os.path.join(data_dir, "dataset.npz"))
        E = data["energies"]
        n_samples = len(E)
        train = len(data["train_idx"])
        val = len(data["val_idx"])
        test = len(data["test_idx"])
        rows.append([
            mol.replace("_", " ").title(),
            str(n_atoms),
            str(n_samples),
            str(train),
            str(val),
            str(test),
            f"{E.mean():.1f}",
            f"{E.std():.1f}",
        ])

    col_labels = ["Molecule", "Atoms", "Total", "Train", "Val", "Test", "E mean", "E std"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("MD17 Dataset Summary", fontsize=14, pad=20)
    path = os.path.join(output_dir, "dataset_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/hypothesis/hyp_001_data_pipeline/results/")
    args = parser.parse_args()

    data_root = os.path.join(project_root, args.data_root)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Generating verification plots...")
    plot_energy_distributions(data_root, output_dir)
    plot_pairwise_distances(data_root, output_dir)
    plot_sample_conformations(data_root, output_dir)
    plot_dataset_summary(data_root, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
