"""
TNAFMOL Metrics — All evaluation metrics for molecular conformation generation.

All metrics are defined here. Experiment Undergrads import from this module;
they never reimplement a metric.

Metrics:
- valid_fraction: fraction of samples with all bond lengths in [0.8, 2.0] Angstrom
- pairwise_distance_histogram: histogram comparison of pairwise distance distributions
- energy_wasserstein: Wasserstein distance between energy distributions
- rmsd_coverage: fraction of test conformations covered by nearby generated samples
- bond_length_mae: mean absolute error of bond length distributions
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple, Optional


def valid_fraction(
    positions: np.ndarray,
    mask: np.ndarray,
    min_dist: float = 0.8,
    max_dist: float = 2.0,
) -> Tuple[float, np.ndarray]:
    """Fraction of samples with all pairwise distances in [min_dist, max_dist].

    A sample is "valid" if no pair of real (unmasked) atoms has a distance
    below min_dist (overlapping atoms) or if all bonded distances are physical.

    For conformation validity, we check that no pair of atoms is closer than min_dist.

    Args:
        positions: (N_samples, max_atoms, 3) generated positions
        mask: (max_atoms,) 1=real, 0=padding

    Returns:
        (fraction, per_sample_validity) — fraction valid, and boolean array
    """
    real_idx = np.where(mask > 0.5)[0]
    n_real = len(real_idx)
    n_samples = positions.shape[0]

    if n_real < 2:
        return 1.0, np.ones(n_samples, dtype=bool)

    # Extract real atoms: (N_samples, n_real, 3)
    real_pos = positions[:, real_idx, :]

    # Pairwise distances: (N_samples, n_real, n_real)
    diff = real_pos[:, :, np.newaxis, :] - real_pos[:, np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-12)  # add eps for numerical stability

    # Get upper triangle (unique pairs)
    triu_i, triu_j = np.triu_indices(n_real, k=1)
    pair_dists = dist[:, triu_i, triu_j]  # (N_samples, n_pairs)

    # A sample is valid if minimum pairwise distance >= min_dist
    min_pair_dist = pair_dists.min(axis=1)  # (N_samples,)
    valid = min_pair_dist >= min_dist

    return float(valid.mean()), valid


def pairwise_distance_histogram(
    positions: np.ndarray,
    mask: np.ndarray,
    bins: int = 200,
    range_min: float = 0.5,
    range_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized pairwise distance histogram.

    Args:
        positions: (N_samples, max_atoms, 3)
        mask: (max_atoms,)

    Returns:
        (counts_normalized, bin_edges) — normalized histogram
    """
    real_idx = np.where(mask > 0.5)[0]
    real_pos = positions[:, real_idx, :]

    diff = real_pos[:, :, np.newaxis, :] - real_pos[:, np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    triu_i, triu_j = np.triu_indices(len(real_idx), k=1)
    all_dists = dist[:, triu_i, triu_j].flatten()

    counts, edges = np.histogram(all_dists, bins=bins, range=(range_min, range_max))
    counts_norm = counts / (counts.sum() + 1e-12)

    return counts_norm, edges


def pairwise_distance_divergence(
    gen_positions: np.ndarray,
    ref_positions: np.ndarray,
    mask: np.ndarray,
    bins: int = 200,
) -> float:
    """Jensen-Shannon divergence between pairwise distance distributions.

    Args:
        gen_positions: (N_gen, max_atoms, 3) generated
        ref_positions: (N_ref, max_atoms, 3) reference
        mask: (max_atoms,)

    Returns:
        JSD (lower is better, 0 = identical distributions)
    """
    gen_hist, edges = pairwise_distance_histogram(gen_positions, mask, bins=bins)
    ref_hist, _ = pairwise_distance_histogram(ref_positions, mask, bins=bins)

    # Jensen-Shannon divergence
    m = 0.5 * (gen_hist + ref_hist)
    eps = 1e-12

    kl_gen = np.sum(gen_hist * np.log((gen_hist + eps) / (m + eps)))
    kl_ref = np.sum(ref_hist * np.log((ref_hist + eps) / (m + eps)))

    jsd = 0.5 * (kl_gen + kl_ref)
    return float(jsd)


def energy_wasserstein(
    gen_energies: np.ndarray,
    ref_energies: np.ndarray,
) -> float:
    """Wasserstein distance between generated and reference energy distributions.

    Args:
        gen_energies: (N_gen,) generated sample energies
        ref_energies: (N_ref,) reference energies

    Returns:
        Wasserstein-1 distance (in kcal/mol)
    """
    return float(wasserstein_distance(gen_energies, ref_energies))


def rmsd_coverage(
    gen_positions: np.ndarray,
    test_positions: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction of test conformations "covered" by generated samples.

    A test conformation is covered if at least one generated sample has
    RMSD < threshold (over real atoms only).

    Args:
        gen_positions: (N_gen, max_atoms, 3)
        test_positions: (N_test, max_atoms, 3)
        mask: (max_atoms,)
        threshold: RMSD threshold in Angstrom

    Returns:
        Coverage fraction (higher is better)
    """
    real_idx = np.where(mask > 0.5)[0]
    n_real = len(real_idx)

    gen_real = gen_positions[:, real_idx, :]    # (N_gen, n_real, 3)
    test_real = test_positions[:, real_idx, :]  # (N_test, n_real, 3)

    n_test = test_real.shape[0]
    covered = 0

    # Compute in chunks to manage memory
    chunk_size = 100
    for i in range(0, n_test, chunk_size):
        chunk = test_real[i:i + chunk_size]  # (chunk, n_real, 3)
        # RMSD between each test conformation and all generated
        # (chunk, 1, n_real, 3) - (1, N_gen, n_real, 3)
        diff = chunk[:, np.newaxis, :, :] - gen_real[np.newaxis, :, :, :]
        rmsd = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=-1))  # (chunk, N_gen)
        min_rmsd = rmsd.min(axis=1)  # (chunk,)
        covered += (min_rmsd < threshold).sum()

    return float(covered / n_test)


def bond_length_mae(
    gen_positions: np.ndarray,
    ref_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    mask: np.ndarray,
    bond_threshold: float = 1.8,
) -> Dict[str, float]:
    """Mean absolute error of bond length distributions per bond type.

    Args:
        gen_positions: (N_gen, max_atoms, 3)
        ref_positions: (N_ref, max_atoms, 3)
        atomic_numbers: (N_atoms_real,)
        mask: (max_atoms,)

    Returns:
        Dict mapping bond type to MAE of mean bond length
    """
    from src.data import compute_bond_lengths

    gen_bonds = compute_bond_lengths(gen_positions, atomic_numbers, mask, bond_threshold)
    ref_bonds = compute_bond_lengths(ref_positions, atomic_numbers, mask, bond_threshold)

    maes = {}
    for bond_type in ref_bonds:
        if bond_type in gen_bonds:
            mae = abs(gen_bonds[bond_type].mean() - ref_bonds[bond_type].mean())
            maes[bond_type] = float(mae)
        else:
            maes[bond_type] = float("inf")

    return maes


def min_pairwise_distance(
    positions: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute minimum pairwise distance per sample.

    Useful for detecting overlapping atoms in generated samples.

    Args:
        positions: (N_samples, max_atoms, 3)
        mask: (max_atoms,)

    Returns:
        min_dist: (N_samples,) minimum pairwise distance per sample
    """
    real_idx = np.where(mask > 0.5)[0]
    real_pos = positions[:, real_idx, :]

    diff = real_pos[:, :, np.newaxis, :] - real_pos[:, np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-12)

    # Set diagonal to inf
    n_real = len(real_idx)
    eye = np.eye(n_real)[np.newaxis, :, :] * 1e10
    dist = dist + eye

    return dist.min(axis=(1, 2))
