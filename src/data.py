"""
TNAFMOL Data Pipeline — MD17 Dataset Loading and Preprocessing

Handles:
- MD17 download for all 8 molecules
- Canonical frame preprocessing (CoM subtraction, Kabsch alignment)
- Padding to max atom count with attention mask
- One-hot atom type encoding
- Train/val/test splitting
- Data loading utilities for downstream models
- Soft equivariance augmentation: SO(3) rotation + CoM noise (hyp_003)
- Global std normalization across all molecules (hyp_003)
- Permutation augmentation: random atom ordering per sample (hyp_004)

Conventions:
- Coordinates: Angstroms, CoM-centered, principal-axis-aligned
- Energies: kcal/mol (raw DFT from MD17), no shifting
- Atom types: one-hot [H, C, N, O] → indices {0, 1, 2, 3}
- Padding: zeros for positions, zeros for atom types, mask=0 for padding

hyp_003 additions:
- augment_positions(): random SO(3) rotation + CoM noise per sample
- compute_global_std(): single std over all real-atom positions across all molecules
- MD17Dataset now accepts augment: bool and global_std: Optional[float]
- If global_std is provided, positions are divided by global_std in __init__
  (normalization happens AFTER canonical frame, BEFORE model input)

hyp_004 additions:
- permute_atoms(): randomly permute real atom ordering within each sample
- MD17Dataset now accepts permute: bool (default False)
- Permutation is applied per-sample in __getitem__, after global_std normalization
  but before SO(3) augmentation. Order: load → permute → augment.
  Note: global_std is a scalar that commutes with permutation, so the order is correct.
"""

import os
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation


# =============================================================================
# Soft equivariance augmentation (hyp_003)
# =============================================================================

def augment_positions(
    positions: torch.Tensor,
    mask: torch.Tensor,
    com_noise_std: Optional[float] = None,
) -> torch.Tensor:
    """Apply soft equivariance augmentation: random SO(3) rotation + CoM noise.

    Reference: Tan, H., Tong, A., et al. "Scalable Equilibrium Sampling with
    Sequential Boltzmann Generators," ICML 2025. (Section 3.2: data augmentation)

    Steps per sample:
    1. Random SO(3) rotation: generate 3x3 random Gaussian matrix, QR decompose,
       use Q as rotation (ensure det(Q)=+1 for proper rotation, not reflection).
    2. CoM noise: add N(0, com_noise_std^2) to all real atom positions.
    3. Zero out padding atoms.

    Sanity checks:
    - Rotation is proper: det(Q) = +1. Reflections (det=-1) are excluded. CHECK.
    - Padding atoms are zeroed after augmentation. CHECK.
    - CoM noise adds translation variance; canonical frame is not re-applied
      (soft equivariance, not hard equivariance). This is intentional per SBG.

    Args:
        positions: (N, 3) or (B, N, 3) positions — may include padding
        mask: (N,) or (B, N) 1=real atom, 0=padding
        com_noise_std: std of CoM noise; if None, defaults to 1/sqrt(n_real)

    Returns:
        augmented positions, same shape as input
    """
    if positions.dim() == 2:
        # Single sample: (N, 3) -> handle as batch of 1
        positions = positions.unsqueeze(0)  # (1, N, 3)
        mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
        squeeze_back = True
    else:
        squeeze_back = False

    B, N, _ = positions.shape
    device = positions.device
    dtype = positions.dtype

    # Build mask: (B, N, 1)
    if mask.dim() == 1:
        mask_3d = mask.unsqueeze(0).expand(B, -1).unsqueeze(-1).float()
    else:
        mask_3d = mask.unsqueeze(-1).float()  # (B, N, 1)

    n_real = mask_3d.squeeze(-1).sum(dim=-1)  # (B,)

    # Step 1: Random SO(3) rotation per sample
    # Generate (B, 3, 3) random Gaussian matrices, QR-decompose to get orthogonal Q
    rand_mat = torch.randn(B, 3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(rand_mat)  # Q: (B, 3, 3) orthogonal, R: (B, 3, 3) upper triangular

    # Ensure proper rotation (det=+1): if det(Q) = -1, flip sign of first column
    dets = torch.linalg.det(Q)  # (B,)
    # dets is either +1 or -1 for orthogonal matrices
    # Flip first column where det = -1
    sign = dets.sign().unsqueeze(-1)  # (B, 1)
    Q[:, :, 0] = Q[:, :, 0] * sign  # flip first column of each rotation matrix where det=-1

    # Apply rotation: positions @ Q^T  (Q maps from frame to rotated frame)
    # positions: (B, N, 3), Q: (B, 3, 3)
    # Result: (B, N, 3) @ (B, 3, 3) -> bmm: (B, N, 3)
    augmented = torch.bmm(positions, Q.transpose(1, 2))  # (B, N, 3)

    # Step 2: CoM noise
    # Default: 1/sqrt(n_real) per sample
    if com_noise_std is None:
        # Variable std per sample: (B, 1, 1)
        noise_std = (1.0 / (n_real.clamp(min=1.0).sqrt())).unsqueeze(-1).unsqueeze(-1)
    else:
        noise_std = com_noise_std

    noise = torch.randn(B, 1, 3, device=device, dtype=dtype) * noise_std  # (B, 1, 3)
    augmented = augmented + noise  # broadcast: (B, N, 3)

    # Step 3: Zero out padding
    augmented = augmented * mask_3d

    if squeeze_back:
        augmented = augmented.squeeze(0)

    return augmented


def compute_global_std(
    data_root: str,
    molecules: Optional[List[str]] = None,
    split: str = "train",
) -> float:
    """Compute global position std across all molecules and all real atoms.

    Used for unit-variance normalization in hyp_003. Dividing all positions
    by this value puts the model's input into a ~N(0, 1) scale per coordinate.

    Expected value: ~1.3-1.4 Angstroms (typical MD17 coordinate spread).

    Sanity checks:
    - Only real atoms (mask=1) contribute. Padding zeros do NOT inflate or deflate std. CHECK.
    - Std is a single scalar — same for all molecules, all splits. CHECK.
    - If std < 0.5 or > 3.0, something is wrong — positions may not be canonical. CHECK.

    Args:
        data_root: path to the data directory (e.g., /path/to/tnafmol/data/)
        molecules: list of molecule names; if None, uses all 8
        split: which split to use for computing std (default "train")

    Returns:
        global_std: single float, ~1.3-1.4 Angstroms
    """
    if molecules is None:
        molecules = list(MOLECULES.keys())

    all_positions = []
    for mol in molecules:
        data_dir = os.path.join(data_root, f"md17_{mol}_v1")
        dataset_path = os.path.join(data_dir, "dataset.npz")
        if not os.path.exists(dataset_path):
            print(f"Warning: dataset not found for {mol}, skipping for global_std")
            continue
        data = np.load(dataset_path)
        idx = data[f"{split}_idx"]
        positions = data["positions"][idx]  # (N_split, 21, 3)
        mask = data["mask"]  # (21,)
        n_atoms = int(mask.sum())
        # Extract real atom positions: (N_split, n_atoms, 3)
        real_pos = positions[:, :n_atoms, :]
        all_positions.append(real_pos.reshape(-1))  # flatten all

    if not all_positions:
        raise RuntimeError("No datasets found for computing global_std")

    all_pos = np.concatenate(all_positions)
    std = float(np.std(all_pos))
    print(f"Global std across all molecules ({split} split): {std:.4f} Angstroms")
    return std


# =============================================================================
# Permutation augmentation (hyp_004)
# =============================================================================

def permute_atoms(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly permute real atom ordering. Padding stays at end.

    This augmentation respects the permutation symmetry of atoms: the physical
    state of a molecule does not depend on the arbitrary ordering of atoms in
    the data representation. By presenting random orderings during training,
    the model cannot overfit to the specific atom ordering in the dataset.

    Sanity checks:
    - n_real <= 1: no permutation possible, return unchanged. CHECK.
    - Padding atoms (mask=0) stay at end, untouched. CHECK.
    - Permutation is applied to both positions AND atom_types consistently. CHECK.
    - mask is unchanged (same number of real/padding atoms). CHECK.

    Args:
        positions: (N, 3) single sample positions
        atom_types: (N,) int atom type indices
        mask: (N,) float — 1=real, 0=padding

    Returns:
        permuted positions, atom_types, mask (same shapes, mask unchanged)
    """
    n_real = int(mask.sum().item())
    if n_real <= 1:
        return positions, atom_types, mask

    perm = torch.randperm(n_real, device=positions.device)

    # Clone to avoid in-place mutation of the original tensors
    positions_out = positions.clone()
    atom_types_out = atom_types.clone()

    positions_out[:n_real] = positions[perm]
    atom_types_out[:n_real] = atom_types[perm]

    return positions_out, atom_types_out, mask


# =============================================================================
# Constants
# =============================================================================

# MD17 molecules and their atom counts
MOLECULES = {
    "aspirin": 21,
    "benzene": 12,
    "ethanol": 9,
    "malonaldehyde": 9,
    "naphthalene": 18,
    "salicylic_acid": 16,
    "toluene": 15,
    "uracil": 12,
}

MAX_ATOMS = 21  # aspirin

# Atom type mapping: atomic number -> one-hot index
ATOMIC_NUM_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3}  # H, C, N, O
ATOM_TYPE_NAMES = ["H", "C", "N", "O"]
NUM_ATOM_TYPES = 4

# MD17 download URLs (newer format with ~50k-200k+ conformations per molecule)
MD17_BASE_URL = "http://quantum-machine.org/gdml/data/npz"
MD17_FILENAMES = {
    "aspirin": "md17_aspirin.npz",
    "benzene": "md17_benzene2017.npz",
    "ethanol": "md17_ethanol.npz",
    "malonaldehyde": "md17_malonaldehyde.npz",
    "naphthalene": "md17_naphthalene.npz",
    "salicylic_acid": "md17_salicylic.npz",
    "toluene": "md17_toluene.npz",
    "uracil": "md17_uracil.npz",
}


# =============================================================================
# Download
# =============================================================================

def download_md17(molecule: str, raw_dir: str) -> str:
    """Download a single MD17 molecule npz file.

    Args:
        molecule: Molecule name (key in MOLECULES dict)
        raw_dir: Directory to save raw downloads

    Returns:
        Path to downloaded file
    """
    if molecule not in MD17_FILENAMES:
        raise ValueError(f"Unknown molecule: {molecule}. Choose from {list(MD17_FILENAMES.keys())}")

    os.makedirs(raw_dir, exist_ok=True)
    filename = MD17_FILENAMES[molecule]
    filepath = os.path.join(raw_dir, filename)

    if os.path.exists(filepath):
        print(f"  Already downloaded: {filepath}")
        return filepath

    url = f"{MD17_BASE_URL}/{filename}"
    print(f"  Downloading {molecule} from {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"  Saved to {filepath}")
    return filepath


def download_all_md17(raw_dir: str) -> Dict[str, str]:
    """Download MD17 for all 8 molecules.

    Returns:
        Dict mapping molecule name to file path
    """
    paths = {}
    for mol in MOLECULES:
        print(f"Downloading {mol}...")
        paths[mol] = download_md17(mol, raw_dir)
    return paths


# =============================================================================
# Loading raw data
# =============================================================================

def load_raw_md17(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw MD17 npz file.

    The newer md17_*.npz files have keys: 'z', 'R', 'E', 'F'
    - z: atomic numbers, shape (N_atoms,)
    - R: positions, shape (N_samples, N_atoms, 3) in Angstrom
    - E: energies, shape (N_samples,) in kcal/mol
    - F: forces, shape (N_samples, N_atoms, 3) in kcal/mol/Angstrom

    Some files may use 'nuclear_charges', 'coords', 'energies', 'forces' instead.

    Returns:
        (atomic_numbers, positions, energies, forces)
    """
    data = np.load(filepath)

    # Try newer format first
    if "z" in data:
        z = data["z"]  # (N_atoms,)
        R = data["R"]  # (N_samples, N_atoms, 3)
        E = data["E"]  # (N_samples,) or (N_samples, 1)
        F = data["F"]  # (N_samples, N_atoms, 3)
    elif "nuclear_charges" in data:
        z = data["nuclear_charges"]
        R = data["coords"]
        E = data["energies"]
        F = data["forces"]
    else:
        raise ValueError(f"Unknown MD17 format. Keys: {list(data.keys())}")

    # Ensure correct shapes
    E = E.flatten()  # (N_samples,)
    if R.ndim == 2:
        # Some files may have R as (N_samples * N_atoms, 3)
        n_atoms = len(z)
        R = R.reshape(-1, n_atoms, 3)
        F = F.reshape(-1, n_atoms, 3)

    return z, R, E, F


# =============================================================================
# Canonical frame preprocessing
# =============================================================================

def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute center of mass for a batch of conformations.

    Args:
        positions: (N_samples, N_atoms, 3)
        masses: (N_atoms,)

    Returns:
        com: (N_samples, 3)
    """
    # (N_samples, N_atoms, 3) * (N_atoms, 1) -> sum over atoms -> (N_samples, 3)
    total_mass = masses.sum()
    com = np.einsum("sai,a->si", positions, masses) / total_mass
    return com


def subtract_com(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Center conformations at center of mass.

    Args:
        positions: (N_samples, N_atoms, 3)
        masses: (N_atoms,)

    Returns:
        centered: (N_samples, N_atoms, 3)
    """
    com = center_of_mass(positions, masses)  # (N_samples, 3)
    return positions - com[:, np.newaxis, :]


def compute_mean_structure(positions: np.ndarray) -> np.ndarray:
    """Compute mean structure from CoM-centered conformations.

    Args:
        positions: (N_samples, N_atoms, 3) — should be CoM-centered

    Returns:
        mean_pos: (N_atoms, 3)
    """
    return positions.mean(axis=0)


def kabsch_align(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align conformations to a reference structure using Kabsch algorithm.

    Uses SVD to find optimal rotation minimizing RMSD.

    Args:
        positions: (N_samples, N_atoms, 3) — CoM-centered
        reference: (N_atoms, 3) — CoM-centered reference

    Returns:
        aligned: (N_samples, N_atoms, 3)
    """
    n_samples = positions.shape[0]
    aligned = np.zeros_like(positions)

    for i in range(n_samples):
        # Cross-covariance matrix: H = P^T @ Q
        H = positions[i].T @ reference  # (3, 3)
        U, S, Vt = np.linalg.svd(H)

        # Handle reflection: ensure right-handed rotation
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

        # Optimal rotation: R = V @ sign @ U^T
        R = Vt.T @ sign_matrix @ U.T
        aligned[i] = positions[i] @ R.T

    return aligned


def atomic_masses(atomic_numbers: np.ndarray) -> np.ndarray:
    """Get atomic masses from atomic numbers.

    Args:
        atomic_numbers: (N_atoms,)

    Returns:
        masses: (N_atoms,)
    """
    mass_table = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
    return np.array([mass_table[int(z)] for z in atomic_numbers])


def canonical_frame(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply full canonical frame preprocessing.

    1. Subtract center of mass
    2. Compute mean structure
    3. Kabsch align all conformations to mean structure

    Args:
        positions: (N_samples, N_atoms, 3) raw positions in Angstrom
        atomic_numbers: (N_atoms,)

    Returns:
        aligned: (N_samples, N_atoms, 3) canonical frame positions
        mean_structure: (N_atoms, 3) the reference mean structure
    """
    masses = atomic_masses(atomic_numbers)

    # Step 1: Center of mass subtraction
    centered = subtract_com(positions, masses)

    # Step 2: Compute mean structure
    mean_struct = compute_mean_structure(centered)

    # Step 3: Kabsch align to mean structure
    aligned = kabsch_align(centered, mean_struct)

    # Recompute mean after alignment (should be very close to mean_struct)
    # and re-align once more for stability
    mean_struct_final = compute_mean_structure(aligned)
    aligned = kabsch_align(aligned, mean_struct_final)

    return aligned, mean_struct_final


# =============================================================================
# Padding and encoding
# =============================================================================

def pad_positions(positions: np.ndarray, max_atoms: int = MAX_ATOMS) -> Tuple[np.ndarray, np.ndarray]:
    """Pad positions to max_atoms with zeros and create attention mask.

    Args:
        positions: (N_samples, N_atoms, 3)
        max_atoms: Maximum number of atoms to pad to

    Returns:
        padded: (N_samples, max_atoms, 3) — zero-padded
        mask: (max_atoms,) — 1 for real atoms, 0 for padding
    """
    n_samples, n_atoms, _ = positions.shape
    assert n_atoms <= max_atoms, f"n_atoms={n_atoms} > max_atoms={max_atoms}"

    padded = np.zeros((n_samples, max_atoms, 3), dtype=np.float32)
    padded[:, :n_atoms, :] = positions

    mask = np.zeros(max_atoms, dtype=np.float32)
    mask[:n_atoms] = 1.0

    return padded, mask


def encode_atom_types(atomic_numbers: np.ndarray, max_atoms: int = MAX_ATOMS) -> np.ndarray:
    """Encode atomic numbers as indices and pad.

    Args:
        atomic_numbers: (N_atoms,)
        max_atoms: Maximum number of atoms to pad to

    Returns:
        atom_type_indices: (max_atoms,) — index in {0,1,2,3} for real atoms, 0 for padding
            (padding atoms are distinguished by the mask, not the type index)
    """
    n_atoms = len(atomic_numbers)
    indices = np.zeros(max_atoms, dtype=np.int64)
    for i, z in enumerate(atomic_numbers):
        if int(z) not in ATOMIC_NUM_TO_IDX:
            raise ValueError(f"Unknown atomic number: {z}")
        indices[i] = ATOMIC_NUM_TO_IDX[int(z)]
    return indices


def one_hot_atom_types(atom_type_indices: np.ndarray) -> np.ndarray:
    """Convert atom type indices to one-hot encoding.

    Args:
        atom_type_indices: (max_atoms,) — index in {0,1,2,3}

    Returns:
        one_hot: (max_atoms, NUM_ATOM_TYPES) — one-hot encoded
    """
    one_hot = np.zeros((len(atom_type_indices), NUM_ATOM_TYPES), dtype=np.float32)
    one_hot[np.arange(len(atom_type_indices)), atom_type_indices] = 1.0
    return one_hot


# =============================================================================
# Train/val/test split
# =============================================================================

def split_indices(
    n_samples: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/val/test split indices.

    Args:
        n_samples: Total number of samples
        train_frac, val_frac, test_frac: Split fractions (must sum to 1.0)

    Returns:
        (train_idx, val_idx, test_idx) — disjoint index arrays
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)

    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train:n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val:])

    return train_idx, val_idx, test_idx


# =============================================================================
# Reference statistics
# =============================================================================

def compute_pairwise_distances(positions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances for real atoms (not padding).

    Args:
        positions: (N_samples, max_atoms, 3)
        mask: (max_atoms,) — 1 for real atoms

    Returns:
        distances: flat array of all pairwise distances across all samples
    """
    real_idx = np.where(mask > 0.5)[0]
    n_real = len(real_idx)
    if n_real < 2:
        return np.array([])

    # Extract real atom positions: (N_samples, n_real, 3)
    real_pos = positions[:, real_idx, :]

    # Compute pairwise distances using broadcasting
    # (N_samples, n_real, 1, 3) - (N_samples, 1, n_real, 3)
    diff = real_pos[:, :, np.newaxis, :] - real_pos[:, np.newaxis, :, :]  # (N, n, n, 3)
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # (N, n, n)

    # Extract upper triangle (unique pairs)
    triu_idx = np.triu_indices(n_real, k=1)
    all_distances = dist[:, triu_idx[0], triu_idx[1]]  # (N_samples, n_pairs)

    return all_distances.flatten()


def compute_bond_lengths(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    mask: np.ndarray,
    bond_threshold: float = 1.8,
) -> Dict[str, np.ndarray]:
    """Compute bond lengths for bonded atom pairs.

    Uses a simple distance threshold to identify bonds. This is approximate
    but sufficient for distribution comparison.

    Args:
        positions: (N_samples, max_atoms, 3)
        atomic_numbers: (N_atoms_real,) — original (unpadded) atomic numbers
        mask: (max_atoms,)
        bond_threshold: Maximum distance to consider as bonded (Angstrom)

    Returns:
        Dict mapping bond type string (e.g., "C-H") to array of distances
    """
    real_idx = np.where(mask > 0.5)[0]
    n_real = len(real_idx)
    if n_real < 2:
        return {}

    # Use mean structure to identify bonds
    mean_pos = positions.mean(axis=0)[real_idx]  # (n_real, 3)
    diff = mean_pos[:, np.newaxis, :] - mean_pos[np.newaxis, :, :]
    mean_dist = np.sqrt((diff ** 2).sum(axis=-1))

    bond_lengths = {}
    for i in range(n_real):
        for j in range(i + 1, n_real):
            if mean_dist[i, j] < bond_threshold:
                z_i = int(atomic_numbers[i])
                z_j = int(atomic_numbers[j])
                # Sort for consistent naming
                name_i = {1: "H", 6: "C", 7: "N", 8: "O"}[min(z_i, z_j)]
                name_j = {1: "H", 6: "C", 7: "N", 8: "O"}[max(z_i, z_j)]
                bond_type = f"{name_i}-{name_j}"

                # Compute distance for all samples
                d = np.sqrt(((positions[:, real_idx[i]] - positions[:, real_idx[j]]) ** 2).sum(axis=-1))

                if bond_type not in bond_lengths:
                    bond_lengths[bond_type] = []
                bond_lengths[bond_type].append(d)

    # Concatenate per bond type
    return {k: np.concatenate(v) for k, v in bond_lengths.items()}


def compute_reference_stats(
    positions: np.ndarray,
    energies: np.ndarray,
    atomic_numbers: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compute reference statistics for a molecule dataset.

    Args:
        positions: (N_samples, max_atoms, 3)
        energies: (N_samples,)
        atomic_numbers: (N_atoms_real,) — unpadded
        mask: (max_atoms,)

    Returns:
        Dictionary of reference statistics
    """
    stats = {
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "energy_min": float(energies.min()),
        "energy_max": float(energies.max()),
        "energy_median": float(np.median(energies)),
    }

    # Pairwise distance histogram
    pw_dists = compute_pairwise_distances(positions, mask)
    if len(pw_dists) > 0:
        hist_counts, hist_edges = np.histogram(pw_dists, bins=200, range=(0.5, 10.0))
        stats["pairwise_dist_hist_counts"] = hist_counts
        stats["pairwise_dist_hist_edges"] = hist_edges
        stats["pairwise_dist_mean"] = float(pw_dists.mean())
        stats["pairwise_dist_std"] = float(pw_dists.std())

    # Bond length distributions
    bond_lengths = compute_bond_lengths(positions, atomic_numbers, mask)
    stats["bond_lengths"] = {}
    for bond_type, dists in bond_lengths.items():
        stats["bond_lengths"][bond_type] = {
            "mean": float(dists.mean()),
            "std": float(dists.std()),
            "min": float(dists.min()),
            "max": float(dists.max()),
            "count": int(len(dists)),
        }

    # Position statistics (for normalization reference)
    real_idx = np.where(mask > 0.5)[0]
    real_pos = positions[:, real_idx, :]
    stats["position_mean"] = float(real_pos.mean())
    stats["position_std"] = float(real_pos.std())
    stats["position_min"] = float(real_pos.min())
    stats["position_max"] = float(real_pos.max())

    return stats


# =============================================================================
# Full pipeline
# =============================================================================

def process_molecule(
    molecule: str,
    raw_filepath: str,
    output_dir: str,
    seed: int = 42,
) -> dict:
    """Process a single molecule: load, preprocess, split, compute stats, save.

    Args:
        molecule: Molecule name
        raw_filepath: Path to raw .npz file
        output_dir: Output directory for processed data

    Returns:
        metadata dict
    """
    print(f"\nProcessing {molecule}...")

    # Load raw data
    z, R, E, F = load_raw_md17(raw_filepath)
    n_samples, n_atoms, _ = R.shape
    print(f"  Loaded: {n_samples} conformations, {n_atoms} atoms")
    print(f"  Atomic numbers: {np.unique(z)}")
    print(f"  Energy range: [{E.min():.2f}, {E.max():.2f}] kcal/mol")

    # Canonical frame
    print("  Applying canonical frame...")
    aligned, mean_structure = canonical_frame(R, z)
    print(f"  Position range after alignment: [{aligned.min():.3f}, {aligned.max():.3f}]")

    # Pad
    padded, mask = pad_positions(aligned, MAX_ATOMS)
    atom_type_indices = encode_atom_types(z, MAX_ATOMS)
    atom_type_one_hot = one_hot_atom_types(atom_type_indices)

    # Split
    train_idx, val_idx, test_idx = split_indices(n_samples, seed=seed)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Compute reference statistics (on full dataset)
    print("  Computing reference statistics...")
    ref_stats = compute_reference_stats(padded, E, z, mask)
    ref_stats["mean_structure"] = mean_structure
    ref_stats["n_atoms"] = int(n_atoms)
    ref_stats["n_samples"] = int(n_samples)
    ref_stats["atomic_numbers"] = z.tolist()

    # Add energy histogram
    e_hist_counts, e_hist_edges = np.histogram(E, bins=100)
    ref_stats["energy_hist_counts"] = e_hist_counts
    ref_stats["energy_hist_edges"] = e_hist_edges

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "dataset.npz")
    np.savez_compressed(
        dataset_path,
        positions=padded.astype(np.float32),       # (N_samples, 21, 3)
        energies=E.astype(np.float32),               # (N_samples,)
        atom_types=atom_type_indices,                 # (21,) int64
        atom_types_one_hot=atom_type_one_hot,         # (21, 4) float32
        mask=mask,                                     # (21,) float32
        mean_structure=mean_structure.astype(np.float32),  # (n_atoms, 3)
        atomic_numbers=z,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(f"  Saved dataset to {dataset_path}")

    # Save ref_stats.pt
    ref_stats_path = os.path.join(output_dir, "ref_stats.pt")
    # Convert numpy arrays to lists for torch saving compatibility
    ref_stats_torch = {}
    for k, v in ref_stats.items():
        if isinstance(v, np.ndarray):
            ref_stats_torch[k] = torch.from_numpy(v)
        elif isinstance(v, dict):
            ref_stats_torch[k] = v  # bond_lengths dict stays as-is
        else:
            ref_stats_torch[k] = v
    torch.save(ref_stats_torch, ref_stats_path)
    print(f"  Saved ref_stats to {ref_stats_path}")

    # Compute checksum
    with open(dataset_path, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()

    # Create metadata
    metadata = {
        "name": f"md17_{molecule}_v1",
        "created": "2026-02-28",
        "generator": "src/data.py",
        "git_hash": "",  # filled at commit time
        "physics": {
            "system": "MD17",
            "molecule": molecule,
            "N": int(n_atoms),
            "max_N": MAX_ATOMS,
            "atom_types": [ATOM_TYPE_NAMES[ATOMIC_NUM_TO_IDX[int(zz)]] for zz in z],
            "energy_unit": "kcal/mol",
            "position_unit": "Angstrom",
            "temperature": "500K (thermal MD)",
            "level_of_theory": "DFT (PBE+vdW-TS)",
        },
        "generation": {
            "method": "MD trajectory from MD17 dataset",
            "n_samples": int(n_samples),
            "seed": seed,
        },
        "statistics": {
            "mean_energy": ref_stats["energy_mean"],
            "std_energy": ref_stats["energy_std"],
            "min_energy": ref_stats["energy_min"],
            "max_energy": ref_stats["energy_max"],
            "median_energy": ref_stats["energy_median"],
            "position_mean": ref_stats["position_mean"],
            "position_std": ref_stats["position_std"],
            "shape_positions": list(padded.shape),
            "dtype_positions": "float32",
        },
        "conventions": {
            "coordinate_range": "CoM-centered, Kabsch-aligned",
            "coordinate_representation": "canonical_frame_cartesian",
            "normalization": "none (raw Angstroms after canonical frame)",
            "padding": f"zero-padded to {MAX_ATOMS} atoms",
            "mask_convention": "1=real atom, 0=padding",
            "atom_type_encoding": "index in {H:0, C:1, N:2, O:3}",
        },
        "split": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "method": "random permutation, seed=42",
        },
        "checksum": {
            "algorithm": "md5",
            "value": checksum,
        },
        "ref_stats": "ref_stats.pt",
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    # Create README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# MD17 {molecule.replace('_', ' ').title()} — Preprocessed Dataset\n\n")
        f.write(f"## Overview\n")
        f.write(f"- **Molecule:** {molecule}\n")
        f.write(f"- **Atoms:** {n_atoms} ({', '.join(metadata['physics']['atom_types'])})\n")
        f.write(f"- **Conformations:** {n_samples}\n")
        f.write(f"- **Energy range:** [{E.min():.2f}, {E.max():.2f}] kcal/mol\n")
        f.write(f"- **Split:** train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}\n\n")
        f.write(f"## Preprocessing\n")
        f.write(f"1. Center of mass subtraction\n")
        f.write(f"2. Kabsch alignment to mean structure\n")
        f.write(f"3. Zero-padded to {MAX_ATOMS} atoms with attention mask\n\n")
        f.write(f"## Files\n")
        f.write(f"- `dataset.npz`: positions ({n_samples}, {MAX_ATOMS}, 3), energies ({n_samples},), atom_types ({MAX_ATOMS},), mask ({MAX_ATOMS},), split indices\n")
        f.write(f"- `metadata.json`: generation parameters, statistics, conventions\n")
        f.write(f"- `ref_stats.pt`: reference statistics for evaluation\n\n")
        f.write(f"## Loading\n")
        f.write(f"```python\n")
        f.write(f'data = np.load("dataset.npz")\n')
        f.write(f'positions = data["positions"]      # ({n_samples}, {MAX_ATOMS}, 3) float32\n')
        f.write(f'energies = data["energies"]        # ({n_samples},) float32\n')
        f.write(f'atom_types = data["atom_types"]    # ({MAX_ATOMS},) int64\n')
        f.write(f'mask = data["mask"]                # ({MAX_ATOMS},) float32\n')
        f.write(f'train_idx = data["train_idx"]\n')
        f.write(f'val_idx = data["val_idx"]\n')
        f.write(f'test_idx = data["test_idx"]\n')
        f.write(f"```\n")

    print(f"  Saved README to {readme_path}")

    return metadata


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MD17Dataset(torch.utils.data.Dataset):
    """PyTorch dataset for a single preprocessed MD17 molecule.

    Args:
        data_dir: Path to processed dataset directory (e.g., data/md17_aspirin_v1/)
        split: One of 'train', 'val', 'test'
        augment: If True, apply random SO(3) rotation + CoM noise on each sample (hyp_003)
        global_std: If provided, divide all positions by this scalar for unit-variance
                    normalization (hyp_003). Applied AFTER canonical frame, BEFORE model.
        permute: If True, randomly permute real atom ordering per sample (hyp_004).
                 Applied BEFORE augmentation. Must clone atom_types since they are
                 shared per-molecule (line 899 in original data.py).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augment: bool = False,
        global_std: Optional[float] = None,
        permute: bool = False,
    ):
        assert split in ("train", "val", "test")

        data = np.load(os.path.join(data_dir, "dataset.npz"))
        idx = data[f"{split}_idx"]

        positions = data["positions"][idx]  # (N, 21, 3) float32

        # Apply global std normalization if provided
        # This puts positions in ~N(0, 1) scale per coordinate
        if global_std is not None and global_std > 0:
            positions = positions / global_std

        self.positions = torch.from_numpy(positions)               # (N, 21, 3)
        self.energies = torch.from_numpy(data["energies"][idx])    # (N,)
        self.atom_types = torch.from_numpy(data["atom_types"])     # (21,) — shared
        self.atom_types_one_hot = torch.from_numpy(data["atom_types_one_hot"])  # (21, 4)
        self.mask = torch.from_numpy(data["mask"])                 # (21,)
        self.molecule = os.path.basename(data_dir)
        self.augment = augment
        self.global_std = global_std
        self.permute = permute

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        positions = self.positions[idx]  # (21, 3)

        # Clone atom_types for this sample (shared per-molecule tensor;
        # must clone before permutation to avoid corrupting other samples)
        atom_types = self.atom_types.clone()  # (21,)

        # Apply permutation augmentation (hyp_004) — before SO(3) augmentation
        # Note: global_std normalization is a scalar applied in __init__,
        # which commutes with permutation. Order is correct.
        if self.permute:
            positions, atom_types, _ = permute_atoms(positions, atom_types, self.mask)

        # Apply SO(3) + CoM noise augmentation if enabled (training only)
        if self.augment:
            positions = augment_positions(positions, self.mask)

        return {
            "positions": positions,                  # (21, 3)
            "energy": self.energies[idx],            # scalar
            "atom_types": atom_types,                # (21,) — may be permuted
            "atom_types_one_hot": self.atom_types_one_hot,  # (21, 4) — NOT permuted (not used by model)
            "mask": self.mask,                       # (21,)
        }


class MultiMoleculeDataset(torch.utils.data.Dataset):
    """Combined dataset across all molecules for multi-molecule training.

    Each sample includes a molecule index for conditioning.

    Args:
        data_root: root data directory
        split: 'train', 'val', or 'test'
        molecules: list of molecule names; if None, uses all 8
        augment: if True, apply random SO(3) + CoM noise (training only, hyp_003)
        global_std: if provided, normalize positions by this scalar (hyp_003)
        permute: if True, randomly permute atom ordering per sample (hyp_004)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        molecules: Optional[List[str]] = None,
        augment: bool = False,
        global_std: Optional[float] = None,
        permute: bool = False,
    ):
        if molecules is None:
            molecules = list(MOLECULES.keys())

        self.datasets = []
        self.molecule_indices = []
        self.molecule_names = molecules

        cumulative = 0
        for i, mol in enumerate(molecules):
            data_dir = os.path.join(data_root, f"md17_{mol}_v1")
            if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
                print(f"Warning: dataset not found for {mol} at {data_dir}, skipping")
                continue
            ds = MD17Dataset(data_dir, split, augment=augment, global_std=global_std, permute=permute)
            self.datasets.append(ds)
            self.molecule_indices.extend([i] * len(ds))
            cumulative += len(ds)

        self.molecule_indices = torch.tensor(self.molecule_indices, dtype=torch.long)
        self._cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cumulative_sizes.append(total)

    def __len__(self):
        return len(self.molecule_indices)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        ds_idx = 0
        offset = 0
        for i, size in enumerate(self._cumulative_sizes):
            if idx < size:
                ds_idx = i
                break
            offset = size

        local_idx = idx - offset
        sample = self.datasets[ds_idx][local_idx]
        sample["molecule_idx"] = self.molecule_indices[idx]
        return sample
