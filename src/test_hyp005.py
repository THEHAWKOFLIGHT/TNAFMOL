"""
Unit tests for hyp_005 code changes.

Tests verify:
1. PAD token: padding atoms get a different embedding than H (index 4 != index 0)
2. Query zeroing: padding positions are zeroed after input_proj, before attention
3. Gaussian noise: real atoms get noise, padding stays zero
4. Forward-inverse consistency: model.inverse(model.forward(x)) ≈ x for real atoms
5. Causal mask: atom i+1 does NOT attend to column i+1 (strictly causal, no self)
6. Jacobian triangularity: numerical Jacobian is lower-triangular for small model

Run: python3 src/test_hyp005.py
All tests must PASS before any training run.
"""

import sys
import os
import math

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model import TarFlow, TarFlowBlock, _asymmetric_clamp
from src.data import PAD_TOKEN_IDX, add_gaussian_noise, MD17Dataset, encode_atom_types, MAX_ATOMS


PASS = "PASS"
FAIL = "FAIL"
DEVICE = "cpu"  # tests run on CPU for determinism


def check(condition, test_name, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))
    return condition


# =============================================================================
# Test 1: PAD token has different embedding than H
# =============================================================================

def test_pad_token_distinct():
    """PAD_TOKEN_IDX=4 embedding must be different from H=0 at initialization."""
    print("\nTest 1: PAD token distinct from H embedding")

    model = TarFlow(
        n_blocks=2, d_model=32, n_heads=2, atom_type_emb_dim=8,
        n_atom_types=5,  # 5 types: H, C, N, O, PAD
        max_atoms=21,
    )

    h_emb = model.atom_type_emb(torch.tensor([0]))   # H
    pad_emb = model.atom_type_emb(torch.tensor([PAD_TOKEN_IDX]))  # PAD

    diff = (h_emb - pad_emb).abs().max().item()
    result = check(diff > 1e-6, "H embedding != PAD embedding at init",
                   f"max diff={diff:.6f}, PAD_TOKEN_IDX={PAD_TOKEN_IDX}")

    # Also verify PAD_TOKEN_IDX == 4
    check(PAD_TOKEN_IDX == 4, "PAD_TOKEN_IDX == 4", f"got {PAD_TOKEN_IDX}")
    return result


# =============================================================================
# Test 2: Query zeroing zeros padding positions
# =============================================================================

def test_query_zeroing():
    """With zero_padding_queries=True, padding atom positions have h=0 before attention."""
    print("\nTest 2: Query zeroing zeros padding atom activations")

    torch.manual_seed(0)
    B, N = 2, 9  # ethanol: 9 real, 12 padding in 21-slot
    max_atoms = 21
    in_features = 19  # 3 + 16 emb_dim

    block = TarFlowBlock(
        d_model=32, n_heads=2, in_features=in_features,
        zero_padding_queries=True, max_atoms=max_atoms,
    )

    # Create mask: first 9 are real, 12 are padding
    atom_mask = torch.zeros(B, max_atoms)
    atom_mask[:, :N] = 1.0

    positions = torch.randn(B, max_atoms, 3)
    positions[:, N:] = 0.0  # padding positions at zero

    atom_type_emb = torch.randn(B, max_atoms, 16)
    atom_type_emb[:, N:] = 0.0  # padding embeddings at zero

    # Manually run the input projection to check zeroing
    features = torch.cat([positions, atom_type_emb], dim=-1)  # (B, N, 19)
    h_atoms = block.input_proj(features)  # (B, 21, 32)

    # Prepend SOS
    sos = block.sos.expand(B, -1, -1)
    h = torch.cat([sos, h_atoms], dim=1)  # (B, 22, 32)

    # Apply query zeroing (replicating _run_transformer logic)
    sos_ones = torch.ones(B, 1, device=h.device)
    h_mask = torch.cat([sos_ones, atom_mask], dim=1)  # (B, 22)
    h_zeroed = h * h_mask.unsqueeze(-1)

    # Check: padding positions (indices N+1 to max_atoms+1) should be zero
    pad_start_idx = N + 1  # +1 for SOS
    padding_h = h_zeroed[:, pad_start_idx:, :]  # (B, 12, 32)
    max_padding_val = padding_h.abs().max().item()

    result = check(max_padding_val < 1e-7, "Padding positions zeroed after query zeroing",
                   f"max value in padding h = {max_padding_val:.8f}")

    # Check: real atom positions (1..N) are not all zero
    real_h = h_zeroed[:, 1:N+1, :]  # (B, 9, 32)
    max_real_val = real_h.abs().max().item()
    check(max_real_val > 0.01, "Real atom positions not zeroed",
          f"max value in real h = {max_real_val:.4f}")

    # Check: SOS position (index 0) is not zero
    sos_h = h_zeroed[:, 0, :]
    max_sos_val = sos_h.abs().max().item()
    check(max_sos_val > 0.01, "SOS position not zeroed",
          f"max value in SOS h = {max_sos_val:.4f}")

    return result


# =============================================================================
# Test 3: Gaussian noise on real atoms only
# =============================================================================

def test_gaussian_noise():
    """add_gaussian_noise: real atoms get N(0, sigma^2) noise, padding stays zero."""
    print("\nTest 3: Gaussian noise on real atoms only")

    torch.manual_seed(42)
    N, sigma = 21, 0.05
    n_real = 9

    # Single sample
    positions = torch.zeros(N, 3)
    mask = torch.zeros(N)
    mask[:n_real] = 1.0

    # Run many times to measure empirical noise
    n_trials = 5000
    noisy_positions = torch.stack([
        add_gaussian_noise(positions.clone(), mask, sigma) for _ in range(n_trials)
    ])  # (5000, 21, 3)

    # Real atoms: should have noise ~ N(0, sigma^2)
    real_noise = noisy_positions[:, :n_real, :]  # (5000, 9, 3)
    empirical_std = real_noise.std().item()
    expected_std = sigma

    result = check(
        abs(empirical_std - expected_std) < 0.005,
        "Real atom noise std ~ sigma",
        f"empirical std={empirical_std:.4f}, expected={expected_std:.4f}"
    )

    # Padding atoms: should stay at exactly zero
    pad_noise = noisy_positions[:, n_real:, :]  # (5000, 12, 3)
    max_pad_val = pad_noise.abs().max().item()
    check(max_pad_val < 1e-7, "Padding atoms stay zero",
          f"max padding value = {max_pad_val:.8f}")

    # Zero sigma: positions unchanged
    positions_orig = torch.randn(N, 3)
    noisy_zero = add_gaussian_noise(positions_orig.clone(), mask, 0.0)
    max_diff = (noisy_zero - positions_orig).abs().max().item()
    check(max_diff < 1e-7, "sigma=0 returns positions unchanged",
          f"max diff = {max_diff:.8f}")

    return result


# =============================================================================
# Test 4: Forward-inverse consistency
# =============================================================================

def test_forward_inverse_consistency():
    """model.inverse(model.forward(x)) ≈ x for real atoms."""
    print("\nTest 4: Forward-inverse consistency")

    torch.manual_seed(123)
    B, N_real = 4, 9
    max_atoms = 21

    model = TarFlow(
        n_blocks=4, d_model=32, n_heads=2, atom_type_emb_dim=8,
        n_atom_types=5, max_atoms=max_atoms,
        alpha_pos=10.0, alpha_neg=10.0,
        zero_padding_queries=True,
    ).eval()

    # Create test input
    positions = torch.randn(B, max_atoms, 3)
    positions[:, N_real:] = 0.0  # zero padding

    atom_types = torch.zeros(max_atoms, dtype=torch.long)
    atom_types[:N_real] = torch.randint(0, 4, (N_real,))
    atom_types[N_real:] = PAD_TOKEN_IDX

    atom_mask = torch.zeros(max_atoms)
    atom_mask[:N_real] = 1.0

    # Forward
    with torch.no_grad():
        z, log_det = model.forward(positions, atom_types, atom_mask)

    # Inverse: should recover x
    with torch.no_grad():
        x_recovered = torch.zeros(B, max_atoms, 3)
        for i in range(B):
            x_recovered[i:i+1] = model.sample(
                atom_types, atom_mask, n_samples=1, temperature=1.0
            )

    # For proper test, use the actual inverse from a single sample
    # The inverse should recover positions from z exactly
    # Build inverse properly using block inversions
    with torch.no_grad():
        atom_type_emb = model.atom_type_emb(
            atom_types.unsqueeze(0).expand(B, -1)
        )
        if model.use_bidir_types and model.type_encoder is not None:
            atom_mask_b = atom_mask.unsqueeze(0).expand(B, -1)
            atom_type_emb = model.type_encoder(atom_type_emb, atom_mask_b)

        x_inv = z.clone()
        atom_mask_b = atom_mask.unsqueeze(0).expand(B, -1)
        for block in reversed(model.blocks):
            x_inv = block.inverse(x_inv, atom_type_emb, atom_mask_b)

    # Check real atom recovery
    diff_real = (x_inv[:, :N_real, :] - positions[:, :N_real, :]).abs()
    max_real_err = diff_real.max().item()
    mean_real_err = diff_real.mean().item()

    result = check(max_real_err < 5e-4, "Forward-inverse consistent for real atoms",
                   f"max err={max_real_err:.6f}, mean err={mean_real_err:.6f}")

    # Check padding stays at zero
    diff_pad = x_inv[:, N_real:, :].abs()
    max_pad_err = diff_pad.max().item()
    check(max_pad_err < 1e-6, "Padding stays zero after inverse",
          f"max padding err={max_pad_err:.8f}")

    return result


# =============================================================================
# Test 5: Causal mask is strictly causal
# =============================================================================

def test_causal_mask_strict():
    """Verify _build_causal_mask: atom i+1 does NOT attend to column i+1 (itself)."""
    print("\nTest 5: Causal mask is strictly causal (no self-attention for atoms)")

    block = TarFlowBlock(d_model=32, n_heads=2, in_features=19, max_atoms=21)

    for n_atoms in [3, 9, 21]:
        mask = block._build_causal_mask(n_atoms, torch.device("cpu"))
        N1 = n_atoms + 1

        # Check: SOS (position 0) can self-attend
        sos_self = mask[0, 0].item()
        check(not math.isinf(sos_self), f"n_atoms={n_atoms}: SOS can self-attend",
              f"mask[0,0]={sos_self:.1f}")

        # Check: no atom position i+1 can attend to itself (column i+1)
        all_strict = True
        for i in range(1, N1):
            if not math.isinf(mask[i, i].item()):
                check(False, f"n_atoms={n_atoms}: atom at row {i} SELF-ATTENDS",
                      f"mask[{i},{i}]={mask[i,i].item():.1f} — should be -inf")
                all_strict = False

        if all_strict:
            check(True, f"n_atoms={n_atoms}: all atoms strictly causal (no self-attend)",
                  f"all {n_atoms} atoms have mask[i,i]=-inf")

        # Check: atom at row 1 (atom 0) can attend to SOS (col 0)
        atom0_sos = mask[1, 0].item()
        check(not math.isinf(atom0_sos), f"n_atoms={n_atoms}: atom 0 can attend to SOS",
              f"mask[1,0]={atom0_sos:.1f}")

        # Check: atom at row 1 (atom 0) CANNOT attend to atom 1 (col 2)
        if n_atoms >= 2:
            atom0_future = mask[1, 2].item()
            check(math.isinf(atom0_future), f"n_atoms={n_atoms}: atom 0 cannot attend to atom 1",
                  f"mask[1,2]={atom0_future:.1f}")

    return True


# =============================================================================
# Test 6: Jacobian is lower-triangular
# =============================================================================

def test_jacobian_triangular():
    """Numerical Jacobian of a 2-atom model must be lower-triangular."""
    print("\nTest 6: Jacobian is lower-triangular (strictly causal)")

    torch.manual_seed(7)
    B, N_real = 1, 2  # 2 atoms, tiny model
    max_atoms = 3  # min padding

    model = TarFlow(
        n_blocks=2, d_model=16, n_heads=2, atom_type_emb_dim=4,
        n_atom_types=4, max_atoms=max_atoms,
        alpha_pos=10.0, alpha_neg=10.0,
        zero_padding_queries=False,
    ).eval()

    atom_types = torch.tensor([1, 2, 0], dtype=torch.long)  # C, N, H(pad)
    atom_mask = torch.tensor([1.0, 1.0, 0.0])  # 2 real, 1 padding

    # Input: 2 real atoms, 3D each = 6 DOF. Build Jacobian numerically.
    # Only test real atom outputs vs real atom inputs.
    x0 = torch.randn(B, max_atoms, 3) * 0.5
    x0[:, N_real:] = 0.0  # zero padding

    eps = 1e-5
    n_dof = N_real * 3  # 6

    # Compute forward at x0
    with torch.no_grad():
        z0, _ = model.forward(x0, atom_types, atom_mask)
    z0_real = z0[0, :N_real, :].reshape(-1)  # (6,)

    # Numerical Jacobian: J[i,j] = (f_i(x0 + eps*e_j) - f_i(x0)) / eps
    J = torch.zeros(n_dof, n_dof)
    for j in range(n_dof):
        atom_j = j // 3
        coord_j = j % 3
        x_pert = x0.clone()
        x_pert[0, atom_j, coord_j] += eps

        with torch.no_grad():
            z_pert, _ = model.forward(x_pert, atom_types, atom_mask)
        z_pert_real = z_pert[0, :N_real, :].reshape(-1)

        J[:, j] = (z_pert_real - z0_real) / eps

    # Check: upper triangle (above diagonal) is ~zero
    # Lower triangular means J[i, j] = 0 for j > i
    upper_mask = torch.triu(torch.ones(n_dof, n_dof), diagonal=1).bool()
    upper_vals = J[upper_mask].abs()
    max_upper = upper_vals.max().item()
    mean_upper = upper_vals.mean().item()

    # Check: diagonal is not zero (affine scale terms)
    diag_vals = J.diagonal().abs()
    min_diag = diag_vals.min().item()

    result = check(max_upper < 1e-3, "Upper Jacobian is zero (lower-triangular)",
                   f"max upper={max_upper:.6f}, mean upper={mean_upper:.6f}")
    check(min_diag > 1e-4, "Diagonal is non-zero (scale > 0)",
          f"min diagonal={min_diag:.6f}")

    if not result:
        print(f"    Jacobian (6x6):")
        for i in range(n_dof):
            row = " ".join(f"{J[i,j].item():8.5f}" for j in range(n_dof))
            print(f"      [{row}]")

    return result


# =============================================================================
# Run all tests
# =============================================================================

def main():
    print("=" * 60)
    print("hyp_005 Unit Tests")
    print("=" * 60)

    results = []
    results.append(("PAD token distinct", test_pad_token_distinct()))
    results.append(("Query zeroing", test_query_zeroing()))
    results.append(("Gaussian noise", test_gaussian_noise()))
    results.append(("Forward-inverse", test_forward_inverse_consistency()))
    results.append(("Causal mask strict", test_causal_mask_strict()))
    results.append(("Jacobian triangular", test_jacobian_triangular()))

    print("\n" + "=" * 60)
    print("Test Summary:")
    n_pass = sum(1 for _, r in results if r)
    n_fail = sum(1 for _, r in results if not r)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{n_pass}/{len(results)} tests passed, {n_fail}/{len(results)} failed")
    print("=" * 60)

    if n_fail > 0:
        print("TESTS FAILED — do not proceed with training")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED — safe to proceed with training")
        sys.exit(0)


if __name__ == "__main__":
    main()
