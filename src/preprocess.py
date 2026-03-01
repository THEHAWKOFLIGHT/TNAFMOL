"""
TNAFMOL Preprocessing Script — Download and process all 8 MD17 molecules.

Usage:
    python src/preprocess.py --data-root data/ --raw-dir data/raw/

Produces:
    data/md17_{molecule}_v1/ for each molecule, containing:
        dataset.npz, metadata.json, ref_stats.pt, README.md
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import (
    MOLECULES,
    download_all_md17,
    process_molecule,
)


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess MD17 dataset")
    parser.add_argument("--data-root", type=str, default="data/",
                        help="Root directory for processed data")
    parser.add_argument("--raw-dir", type=str, default="data/raw/",
                        help="Directory for raw downloads")
    parser.add_argument("--molecule", type=str, default=None,
                        help="Process a single molecule (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    args = parser.parse_args()

    # Resolve paths relative to project root
    data_root = os.path.join(project_root, args.data_root)
    raw_dir = os.path.join(project_root, args.raw_dir)

    # Download
    print("=" * 60)
    print("Step 1: Download MD17")
    print("=" * 60)
    raw_paths = download_all_md17(raw_dir)

    # Process
    print("\n" + "=" * 60)
    print("Step 2: Process molecules")
    print("=" * 60)

    molecules = [args.molecule] if args.molecule else list(MOLECULES.keys())
    all_metadata = {}

    for mol in molecules:
        if mol not in raw_paths:
            print(f"Skipping {mol}: not downloaded")
            continue

        output_dir = os.path.join(data_root, f"md17_{mol}_v1")
        metadata = process_molecule(mol, raw_paths[mol], output_dir, seed=args.seed)
        all_metadata[mol] = metadata

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Molecule':<20} {'Atoms':<8} {'Samples':<10} {'E mean':<12} {'E std':<10}")
    print("-" * 60)
    for mol, meta in all_metadata.items():
        n_atoms = meta["physics"]["N"]
        n_samples = meta["generation"]["n_samples"]
        e_mean = meta["statistics"]["mean_energy"]
        e_std = meta["statistics"]["std_energy"]
        print(f"{mol:<20} {n_atoms:<8} {n_samples:<10} {e_mean:<12.2f} {e_std:<10.2f}")

    print(f"\nAll datasets saved to {data_root}")


if __name__ == "__main__":
    main()
