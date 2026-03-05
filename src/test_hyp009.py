"""
Unit tests for hyp_009: pre-norm + layers_per_block architectural changes.

Tests:
1. Backward compat (post-norm, layers_per_block=1) — identical output to reference
2. Pre-norm produces different output from post-norm
3. Forward-inverse consistency (use_pre_norm=True, layers_per_block=1)
4. layers_per_block=2 increases parameter count correctly
5. Forward-inverse consistency (use_pre_norm=True, layers_per_block=2)
6. Jacobian triangularity (CRITICAL — pre-norm must not break autoregressive structure)

Run with:
    cd /home/kai_nelson/the_rig/tnafmol
    python3.10 src/test_hyp009.py
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

from src.model import TarFlow, TarFlowBlock


def make_test_model(use_pre_norm=False, layers_per_block=1, n_blocks=2,
                    d_model=32, n_heads=2, max_atoms=9, use_output_shift=True,
                    dropout=0.0):
    """Create a small test model."""
    return TarFlow(
        n_blocks=n_blocks,
        d_model=d_model,
        n_heads=n_heads,
        ffn_mult=4,
        atom_type_emb_dim=8,
        n_atom_types=4,
        max_atoms=max_atoms,
        dropout=dropout,
        alpha_pos=10.0,
        alpha_neg=10.0,
        shift_only=False,
        use_actnorm=False,
        use_bidir_types=False,
        use_pos_enc=True,
        zero_padding_queries=True,
        use_output_shift=use_output_shift,
        per_dim_scale=True,
        use_pre_norm=use_pre_norm,
        layers_per_block=layers_per_block,
    )


def make_batch(n_atoms=9, batch_size=4, d=3, max_atoms=9, device='cpu'):
    """Create a test batch of molecular positions."""
    positions = torch.randn(batch_size, max_atoms, d, device=device) * 0.5
    atom_types = torch.randint(0, 4, (max_atoms,), device=device)
    # All real atoms (no padding for simplicity)
    atom_mask = torch.ones(max_atoms, device=device)
    # Zero out padding
    if n_atoms < max_atoms:
        atom_mask[n_atoms:] = 0.0
        positions[:, n_atoms:, :] = 0.0

    return positions, atom_types, atom_mask


# =============================================================================
# Test 1: Backward compatibility (post-norm, layers_per_block=1)
# =============================================================================
def test_backward_compat():
    """
    Test that new post-norm, layers_per_block=1 path produces output
    mathematically equivalent to original code.

    We verify this by:
    1. Creating a model with use_pre_norm=False, layers_per_block=1
    2. Checking that the layers[0] structure contains the same operations
       as the original attn, attn_norm, ffn, ffn_norm attributes
    3. Running forward with same input twice (seeded) — output must be identical

    We can't directly compare to "old code" in the same file, so we verify:
    - Two forward passes with same model+seed → same output (deterministic)
    - The model with layers_per_block=1, post-norm has exactly 1 layer in self.layers
    - The layer contains attn, attn_norm, attn_dropout, ffn, ffn_norm
    """
    print("Test 1: Backward compatibility (post-norm, layers_per_block=1)")

    torch.manual_seed(42)
    model = make_test_model(use_pre_norm=False, layers_per_block=1)
    model.eval()

    positions, atom_types, atom_mask = make_batch(n_atoms=9, batch_size=4, max_atoms=9)

    # Check structure
    block = model.blocks[0]
    assert len(block.layers) == 1, f"Expected 1 layer, got {len(block.layers)}"
    assert 'attn' in block.layers[0], "Missing 'attn' in layers[0]"
    assert 'attn_norm' in block.layers[0], "Missing 'attn_norm' in layers[0]"
    assert 'ffn' in block.layers[0], "Missing 'ffn' in layers[0]"
    assert 'ffn_norm' in block.layers[0], "Missing 'ffn_norm' in layers[0]"
    assert not hasattr(block, 'final_norm'), "Should NOT have final_norm for post-norm"

    # Two deterministic forward passes must match
    with torch.no_grad():
        z1, ld1 = model(positions, atom_types, atom_mask)
        z2, ld2 = model(positions, atom_types, atom_mask)

    assert torch.allclose(z1, z2, atol=1e-7), "Forward pass not deterministic!"
    assert torch.allclose(ld1, ld2, atol=1e-7), "Log-det not deterministic!"

    print(f"  PASS — post-norm, layers_per_block=1 is deterministic (max diff: {(z1 - z2).abs().max().item():.2e})")
    return True


# =============================================================================
# Test 2: Pre-norm produces different output from post-norm
# =============================================================================
def test_prenorm_different_output():
    """
    Test that pre-norm produces DIFFERENT output from post-norm.

    The zero-initialized out_proj means both models start as identity transforms.
    To verify the paths differ, we manually reinitialize out_proj to non-zero,
    then check that the same input + same weights (except ordering) produce different output.

    A better structural check: with trained (non-zero) weights, the two paths compute
    different operations:
    - post-norm: h = LayerNorm(h + attn(h))
    - pre-norm: h = h + attn(LayerNorm(h))

    We verify this by:
    1. Creating both models with random (non-zero) out_proj
    2. Copying all shared weights from post-norm → pre-norm
    3. Verifying outputs differ

    Note: even with the same transformer body weights, pre-norm vs post-norm apply
    LayerNorm at different points in the residual stream, producing different results.
    """
    print("Test 2: Pre-norm produces different output from post-norm")

    torch.manual_seed(42)
    model_post = make_test_model(use_pre_norm=False, layers_per_block=1, dropout=0.0)
    model_post.eval()

    torch.manual_seed(42)
    model_pre = make_test_model(use_pre_norm=True, layers_per_block=1, dropout=0.0)

    # Reinitialize out_proj to non-zero (kaiming uniform)
    for m in [model_post, model_pre]:
        for block in m.blocks:
            nn.init.kaiming_uniform_(block.out_proj.weight)
            nn.init.zeros_(block.out_proj.bias)

    # Copy all shared weights from post-norm → pre-norm
    post_sd = model_post.state_dict()
    pre_sd = model_pre.state_dict()
    shared_keys = [k for k in pre_sd.keys() if 'final_norm' not in k]
    for k in shared_keys:
        if k in post_sd:
            pre_sd[k] = post_sd[k]
    model_pre.load_state_dict(pre_sd)
    model_pre.eval()

    positions, atom_types, atom_mask = make_batch(n_atoms=9, batch_size=4, max_atoms=9)

    with torch.no_grad():
        z_post, ld_post = model_post(positions, atom_types, atom_mask)
        z_pre, ld_pre = model_pre(positions, atom_types, atom_mask)

    # They must be different
    max_diff = (z_post - z_pre).abs().max().item()
    ld_diff = (ld_post - ld_pre).abs().max().item()

    # When out_proj is non-zero, the two normalization orders produce different residual streams
    assert max_diff > 1e-4, (
        f"Pre-norm and post-norm produce same output with non-zero weights "
        f"(max_diff={max_diff:.2e})! The two code paths may be identical."
    )
    print(f"  PASS — outputs differ by max={max_diff:.4f}, log_det diff={ld_diff:.4f}")
    return True


# =============================================================================
# Test 3: Forward-inverse consistency (use_pre_norm=True, layers_per_block=1)
# =============================================================================
def test_forward_inverse_prenorm():
    """
    Test that forward-inverse roundtrip recovers the input within tolerance.
    Pre-norm should not break invertibility.
    """
    print("Test 3: Forward-inverse consistency (use_pre_norm=True, layers_per_block=1)")

    torch.manual_seed(42)
    model = make_test_model(use_pre_norm=True, layers_per_block=1, n_blocks=2)
    model.eval()

    positions, atom_types, atom_mask = make_batch(n_atoms=9, batch_size=4, max_atoms=9)

    with torch.no_grad():
        z, log_det = model(positions, atom_types, atom_mask)
        # Inverse: decode z back to x
        atom_types_b = atom_types.unsqueeze(0).expand(4, -1)
        atom_mask_b = atom_mask.unsqueeze(0).expand(4, -1)

        # Run inverse on z
        x_rec = z
        atom_type_emb = model.atom_type_emb(atom_types_b)
        if model.use_bidir_types and model.type_encoder is not None:
            atom_type_emb = model.type_encoder(atom_type_emb, atom_mask_b)

        for block in reversed(model.blocks):
            x_rec = block.inverse(x_rec, atom_type_emb, atom_mask_b)

    # Mask out padding before comparing
    mask_3d = atom_mask.unsqueeze(0).unsqueeze(-1).expand_as(positions)
    pos_masked = positions * mask_3d
    rec_masked = x_rec * mask_3d

    max_err = (pos_masked - rec_masked).abs().max().item()
    mean_err = (pos_masked - rec_masked).abs().mean().item()

    print(f"  max reconstruction error: {max_err:.2e}, mean: {mean_err:.2e}")
    assert max_err < 1e-3, f"Forward-inverse roundtrip error too large: {max_err:.2e}"
    print(f"  PASS — forward-inverse consistent within 1e-3")
    return True


# =============================================================================
# Test 4: layers_per_block=2 increases parameter count correctly
# =============================================================================
def test_param_count_layers_per_block():
    """
    Test that layers_per_block=2 roughly doubles the transformer parameters per block.
    Total params increase by less than 2x (embeddings, in/out proj are shared).
    """
    print("Test 4: layers_per_block=2 increases parameter count correctly")

    torch.manual_seed(42)
    model_1 = make_test_model(use_pre_norm=True, layers_per_block=1, n_blocks=2, d_model=64, n_heads=4)
    torch.manual_seed(42)
    model_2 = make_test_model(use_pre_norm=True, layers_per_block=2, n_blocks=2, d_model=64, n_heads=4)

    p1 = model_1.count_parameters()
    p2 = model_2.count_parameters()

    # Transformer params per block: attn (d_model^2 * 3 + d_model^2) + FFN (d_model * 4*d_model * 2) + norms
    # Should roughly double when layers_per_block goes 1 -> 2 (plus final_norm per block)
    ratio = p2 / p1

    print(f"  layers_per_block=1: {p1:,} params")
    print(f"  layers_per_block=2: {p2:,} params")
    print(f"  ratio: {ratio:.3f}x")

    assert p2 > p1, "layers_per_block=2 should have more params than layers_per_block=1"
    assert ratio < 2.0, f"Ratio {ratio:.3f} >= 2.0: embeddings and proj layers are shared, ratio must be < 2x"
    assert ratio > 1.2, f"Ratio {ratio:.3f} <= 1.2: not enough extra params from extra layer"

    print(f"  PASS — parameter count ratio {ratio:.3f}x (expected 1.2 < ratio < 2.0)")
    return True


# =============================================================================
# Test 5: Forward-inverse consistency (use_pre_norm=True, layers_per_block=2)
# =============================================================================
def test_forward_inverse_prenorm_lpb2():
    """
    Test forward-inverse roundtrip with layers_per_block=2, use_pre_norm=True.
    """
    print("Test 5: Forward-inverse consistency (use_pre_norm=True, layers_per_block=2)")

    torch.manual_seed(42)
    model = make_test_model(use_pre_norm=True, layers_per_block=2, n_blocks=2)
    model.eval()

    positions, atom_types, atom_mask = make_batch(n_atoms=9, batch_size=4, max_atoms=9)

    with torch.no_grad():
        z, log_det = model(positions, atom_types, atom_mask)

        atom_types_b = atom_types.unsqueeze(0).expand(4, -1)
        atom_mask_b = atom_mask.unsqueeze(0).expand(4, -1)

        x_rec = z
        atom_type_emb = model.atom_type_emb(atom_types_b)

        for block in reversed(model.blocks):
            x_rec = block.inverse(x_rec, atom_type_emb, atom_mask_b)

    mask_3d = atom_mask.unsqueeze(0).unsqueeze(-1).expand_as(positions)
    pos_masked = positions * mask_3d
    rec_masked = x_rec * mask_3d

    max_err = (pos_masked - rec_masked).abs().max().item()

    print(f"  max reconstruction error: {max_err:.2e}")
    assert max_err < 1e-3, f"Forward-inverse error too large: {max_err:.2e}"
    print(f"  PASS — forward-inverse consistent with layers_per_block=2 within 1e-3")
    return True


# =============================================================================
# Test 6: Jacobian triangularity (CRITICAL)
# =============================================================================
def test_jacobian_triangularity():
    """
    Test that pre-norm does NOT break autoregressive (lower-triangular) Jacobian structure.

    For a single TarFlowBlock with output_shift:
    - y_i = scale_i * x_i + shift_i, where scale_i, shift_i depend on x_0..x_{i-1}
    - dy_i/dx_i = scale_i (diagonal element)
    - dy_i/dx_j = 0 for j > i (upper-triangular block = 0)

    We verify:
    1. Numerical Jacobian is lower-triangular (upper triangle ≈ 0)
    2. log|det J| from model matches log|det(numerical J)|

    Test uses small model (d_model=16, n_heads=2, n_blocks=2, layers_per_block=2)
    on a 3-atom molecule (max_atoms=3) for tractable Jacobian computation.
    """
    print("Test 6: Jacobian triangularity (CRITICAL)")

    device = 'cpu'
    torch.manual_seed(0)

    # Small model for tractable Jacobian
    n_atoms = 3
    batch_size = 1
    model = TarFlow(
        n_blocks=2,
        d_model=16,
        n_heads=2,
        ffn_mult=2,
        atom_type_emb_dim=4,
        n_atom_types=4,
        max_atoms=n_atoms,
        dropout=0.0,
        alpha_pos=10.0,
        alpha_neg=10.0,
        shift_only=False,
        use_actnorm=False,
        use_bidir_types=False,
        use_pos_enc=True,
        zero_padding_queries=False,
        use_output_shift=True,
        per_dim_scale=True,
        use_pre_norm=True,
        layers_per_block=2,
    ).to(device)
    model.eval()

    # Single molecule, all real atoms, no padding
    atom_types = torch.randint(0, 4, (n_atoms,), device=device)
    atom_mask = torch.ones(n_atoms, device=device)
    positions = torch.randn(batch_size, n_atoms, 3, device=device) * 0.3

    # Flatten position: (1, 3*3) = (1, 9) for Jacobian computation
    positions_flat = positions.view(batch_size, -1)  # (1, 9)

    def forward_flat(x_flat):
        """Forward pass that takes flattened positions."""
        x = x_flat.view(batch_size, n_atoms, 3)
        z, _ = model(x, atom_types, atom_mask)
        return z.view(batch_size, -1)  # (1, 9)

    # Numerical Jacobian via finite differences
    eps = 1e-4
    D = n_atoms * 3  # = 9

    with torch.no_grad():
        z_flat = forward_flat(positions_flat)

    J_numerical = torch.zeros(D, D)
    for j in range(D):
        x_plus = positions_flat.clone()
        x_plus[0, j] += eps
        x_minus = positions_flat.clone()
        x_minus[0, j] -= eps
        with torch.no_grad():
            z_plus = forward_flat(x_plus)
            z_minus = forward_flat(x_minus)
        J_numerical[:, j] = (z_plus - z_minus)[0] / (2 * eps)

    # Model log_det
    with torch.no_grad():
        _, model_log_det = model(positions, atom_types, atom_mask)
    model_log_det = model_log_det[0].item()

    # Numerical log|det J|
    # Note: atoms in sequence order — z_i depends on x_0..x_{i-1}
    # The Jacobian should be lower-triangular in atom-order:
    # Rows 0..2 (z_0 xyz), Cols 0..2 (x_0 xyz) → only diagonal 3x3 block
    # Rows 3..5 (z_1 xyz) depends on x_0 (cols 0..2) and x_1 (diagonal), not x_2
    # Rows 6..8 (z_2 xyz) depends on x_0, x_1, x_2

    # Check upper-triangular structure: for atom i, z_i should NOT depend on x_j where j > i
    # Jacobian sub-blocks: J[i*3:(i+1)*3, j*3:(j+1)*3] for j > i should be ≈ 0
    upper_block_max = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            block = J_numerical[i*3:(i+1)*3, j*3:(j+1)*3]
            upper_block_max = max(upper_block_max, block.abs().max().item())

    print(f"  Upper-triangular block max abs value: {upper_block_max:.2e}")
    assert upper_block_max < 1e-3, (
        f"Upper-triangular block not zero! max={upper_block_max:.2e}. "
        f"Pre-norm broke autoregressive structure!"
    )

    # Check log|det J|: compute log|det(J_numerical)| and compare to model
    det_J = torch.linalg.det(J_numerical).abs()
    numerical_log_det = math.log(max(det_J.item(), 1e-30))

    log_det_diff = abs(model_log_det - numerical_log_det)
    print(f"  Model log_det: {model_log_det:.4f}")
    print(f"  Numerical log|det J|: {numerical_log_det:.4f}")
    print(f"  Difference: {log_det_diff:.4f}")

    # 1% tolerance on log-scale
    rel_tol = 0.01 * max(abs(model_log_det), abs(numerical_log_det), 1.0)
    assert log_det_diff < max(rel_tol, 0.05), (
        f"log_det mismatch: model={model_log_det:.4f}, numerical={numerical_log_det:.4f}, "
        f"diff={log_det_diff:.4f}"
    )

    print(f"  PASS — Jacobian is lower-triangular, log_det matches numerical within tolerance")
    return True


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("hyp_009 Unit Tests: Pre-Norm + Layers Per Block")
    print("=" * 60)

    tests = [
        ("Backward compatibility", test_backward_compat),
        ("Pre-norm produces different output", test_prenorm_different_output),
        ("Forward-inverse consistency (pre-norm, lpb=1)", test_forward_inverse_prenorm),
        ("Parameter count (layers_per_block=2)", test_param_count_layers_per_block),
        ("Forward-inverse consistency (pre-norm, lpb=2)", test_forward_inverse_prenorm_lpb2),
        ("Jacobian triangularity (CRITICAL)", test_jacobian_triangularity),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"  FAIL — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR — {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed > 0:
        print("SOME TESTS FAILED — do not proceed to training!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED — safe to proceed to training")
    print("=" * 60)
