"""
Unit tests for permute_within_type_groups() — hyp_012.

Tests verify:
1. Type-sorted output: after reordering, atom types are H,H,...,C,C,...,N,N,...,O,O,...
2. Padding stays at end: positions[n_real:] remain zeros, atom_types[n_real:] unchanged
3. Conservation: total count of each atom type is preserved
4. Consistency: positions and atom_types are reordered together (no drift)
5. Edge cases: single atom per type, all same type, n_real=1, no N/O atoms
6. Mutually exclusive: permute=True and permute_within_types=True should raise

Run with: python experiments/hypothesis/hyp_012_perm_reorder_boltzmann/test_permute_within_types.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import numpy as np

from src.data import permute_within_type_groups, MD17Dataset, MultiMoleculeDataset


def test_type_sorted_output():
    """After reordering, atom types must be sorted: H(0) first, then C(1), N(2), O(3)."""
    # ethanol: C C O H H H H H H (9 real atoms)
    atom_types_raw = torch.tensor([1, 1, 3, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    positions = torch.randn(9, 3)
    mask = torch.ones(9)

    for trial in range(20):  # Run multiple times to catch randomness issues
        pos_out, at_out, mask_out = permute_within_type_groups(positions, atom_types_raw, mask)
        # First 6 should be H (0), then 2 C (1), then 1 O (3)
        # (H is type 0, C is type 1, N is type 2, O is type 3)
        assert at_out[0].item() == 0, f"Expected H first, got {at_out[0].item()}"
        assert at_out[5].item() == 0, f"Expected H at pos 5, got {at_out[5].item()}"
        assert at_out[6].item() == 1, f"Expected C at pos 6, got {at_out[6].item()}"
        assert at_out[7].item() == 1, f"Expected C at pos 7, got {at_out[7].item()}"
        assert at_out[8].item() == 3, f"Expected O at pos 8, got {at_out[8].item()}"

    print("PASS: test_type_sorted_output")


def test_type_counts_preserved():
    """Total count of each atom type must be the same before and after."""
    # aspirin: C C C C C C C O O O C C O H H H H H H H H (21 atoms)
    # 9 H (type 0), 9 C (type 1), 0 N (type 2), 3 O (type 3)
    atom_types_raw = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.long
    )
    positions = torch.randn(21, 3)
    mask = torch.ones(21)

    for trial in range(10):
        pos_out, at_out, mask_out = permute_within_type_groups(positions, atom_types_raw, mask)
        for t in [0, 1, 2, 3]:
            count_before = (atom_types_raw == t).sum().item()
            count_after = (at_out == t).sum().item()
            assert count_before == count_after, (
                f"Type {t} count changed: {count_before} -> {count_after}"
            )

    print("PASS: test_type_counts_preserved")


def test_padding_stays_at_end():
    """Padding atoms (after n_real) must remain unchanged."""
    # Ethanol (9 real) padded to 21
    n_real = 9
    atom_types_raw = torch.zeros(21, dtype=torch.long)
    atom_types_raw[:9] = torch.tensor([1, 1, 3, 0, 0, 0, 0, 0, 0])  # C C O H H H H H H
    # Padding atom types = 0 (same as H in default config)
    # But we test that positions[n_real:] stay at 0
    positions = torch.zeros(21, 3)
    positions[:9] = torch.randn(9, 3)
    mask = torch.zeros(21)
    mask[:9] = 1.0

    for trial in range(10):
        pos_out, at_out, mask_out = permute_within_type_groups(positions, atom_types_raw, mask)
        # Positions beyond n_real must be zero
        assert (pos_out[n_real:] == 0).all(), (
            f"Padding positions changed: {pos_out[n_real:]}"
        )
        # mask must be unchanged
        assert (mask_out == mask).all(), "mask changed"

    print("PASS: test_padding_stays_at_end")


def test_position_atom_type_consistency():
    """Positions and atom_types must be reordered together (no drift between them)."""
    # Create synthetic: each position uniquely identifies its original atom
    n_real = 9
    atom_types_raw = torch.tensor([1, 1, 3, 0, 0, 0, 0, 0, 0], dtype=torch.long)

    # Positions encode original index: position[i] = [i, 0, 0]
    positions = torch.zeros(9, 3)
    for i in range(9):
        positions[i, 0] = float(i)  # x-coord = original index

    mask = torch.ones(9)

    for trial in range(20):
        pos_out, at_out, mask_out = permute_within_type_groups(
            positions.clone(), atom_types_raw.clone(), mask
        )
        # Verify: at each output position, the atom type matches what was at the original index
        for i in range(n_real):
            orig_idx = int(pos_out[i, 0].item())
            expected_type = atom_types_raw[orig_idx].item()
            actual_type = at_out[i].item()
            assert expected_type == actual_type, (
                f"Position {i}: original atom {orig_idx} has type {expected_type} "
                f"but got atom_type {actual_type}"
            )

    print("PASS: test_position_atom_type_consistency")


def test_within_group_randomization():
    """Within each type group, multiple distinct orderings should appear over trials."""
    # benzene: C C C C C C H H H H H H (12 atoms)
    # 6H (type 0) + 6C (type 1) — both groups have 6 atoms, so randperm should vary
    atom_types_raw = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.long)

    # Unique positions: encode original index in x-coord
    positions = torch.zeros(12, 3)
    for i in range(12):
        positions[i, 0] = float(i)

    mask = torch.ones(12)
    seen_orderings = set()

    for trial in range(50):
        pos_out, at_out, mask_out = permute_within_type_groups(
            positions.clone(), atom_types_raw.clone(), mask
        )
        # Record the ordering of H-group (first 6 output positions)
        h_order = tuple(int(pos_out[i, 0].item()) for i in range(6))
        seen_orderings.add(h_order)

    # With 50 trials on 6! = 720 possible orderings, expect >> 1 distinct ordering
    assert len(seen_orderings) > 1, (
        f"Expected multiple distinct orderings over 50 trials, got {len(seen_orderings)}"
    )
    print(f"PASS: test_within_group_randomization (saw {len(seen_orderings)} distinct H-group orderings)")


def test_single_atom_per_type():
    """Single atom of a type should remain in the correct slot without error."""
    # ethanol: C C O H H H H H H — O appears once
    atom_types_raw = torch.tensor([1, 1, 3, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    positions = torch.randn(9, 3)
    mask = torch.ones(9)

    for trial in range(10):
        pos_out, at_out, _ = permute_within_type_groups(positions.clone(), atom_types_raw.clone(), mask)
        # O must appear exactly once, in the last real-atom slot
        assert at_out[8].item() == 3, f"O should be last real atom, got {at_out[8].item()}"
        assert (at_out == 3).sum().item() == 1, "O count changed"

    print("PASS: test_single_atom_per_type")


def test_n_real_1():
    """n_real=1 should return unchanged tensors."""
    positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    atom_types = torch.tensor([1], dtype=torch.long)
    mask = torch.tensor([1.0])

    pos_out, at_out, mask_out = permute_within_type_groups(positions, atom_types, mask)
    assert (pos_out == positions).all()
    assert (at_out == atom_types).all()

    print("PASS: test_n_real_1")


def test_all_same_type():
    """All atoms of the same type: ordering should be randomized but types unchanged."""
    # benzene carbons only (hypothetically)
    atom_types_raw = torch.ones(6, dtype=torch.long)  # all C
    positions = torch.zeros(6, 3)
    for i in range(6):
        positions[i, 0] = float(i)
    mask = torch.ones(6)

    seen_orderings = set()
    for trial in range(50):
        pos_out, at_out, _ = permute_within_type_groups(
            positions.clone(), atom_types_raw.clone(), mask
        )
        assert (at_out == 1).all(), "All atoms should still be C"
        ordering = tuple(int(pos_out[i, 0].item()) for i in range(6))
        seen_orderings.add(ordering)

    assert len(seen_orderings) > 1, f"Expected multiple orderings, got {len(seen_orderings)}"
    print(f"PASS: test_all_same_type ({len(seen_orderings)} orderings seen)")


def test_no_nitrogen():
    """Molecules without N should work correctly — test with a molecule with known counts."""
    # Ethanol: 6H + 2C + 1O = 9 atoms, no N
    # Canonical: C C O H H H H H H
    # After type-sort: H H H H H H C C O
    scrambled = [1, 1, 3, 0, 0, 0, 0, 0, 0]  # C C O H H H H H H (canonical ethanol)
    atom_types_raw = torch.tensor(scrambled, dtype=torch.long)
    positions = torch.randn(9, 3)
    mask = torch.ones(9)

    for trial in range(10):
        pos_out, at_out, _ = permute_within_type_groups(positions.clone(), atom_types_raw.clone(), mask)
        # First 6 should be H (type 0)
        assert (at_out[:6] == 0).all(), f"First 6 should be H, got {at_out[:6]}"
        # Next 2 should be C (type 1)
        assert (at_out[6:8] == 1).all(), f"Next 2 should be C, got {at_out[6:8]}"
        # Last 1 should be O (type 3)
        assert at_out[8].item() == 3, f"Last should be O, got {at_out[8]}"

    print("PASS: test_no_nitrogen")


def test_mutual_exclusion():
    """permute=True and permute_within_types=True should raise an AssertionError."""
    import os
    data_dir = "data/md17_ethanol_v1"
    if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
        # Try relative to project root
        data_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'md17_ethanol_v1'
        )

    if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
        print("SKIP: test_mutual_exclusion (dataset not found)")
        return

    try:
        ds = MD17Dataset(data_dir, split="train", permute=True, permute_within_types=True)
        assert False, "Expected AssertionError but got none"
    except AssertionError as e:
        assert "mutually exclusive" in str(e).lower(), (
            f"Expected 'mutually exclusive' in error, got: {e}"
        )
    print("PASS: test_mutual_exclusion")


def test_dataset_integration_arm_b():
    """Verify that permute_within_types=True in MD17Dataset produces type-sorted atom_types."""
    import os
    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'data', 'md17_aspirin_v1'
    )

    if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
        print("SKIP: test_dataset_integration_arm_b (dataset not found)")
        return

    ds = MD17Dataset(data_dir, split="train", permute_within_types=True, max_atoms=21)
    n_real = int(ds.mask.sum().item())  # 21 for aspirin

    for trial in range(20):
        sample = ds[trial % len(ds)]
        at = sample["atom_types"][:n_real]

        # Verify type-sorted: no later type appears before an earlier one
        prev_type = -1
        last_type = -1
        for i in range(n_real):
            t = at[i].item()
            # type should be >= previous type (non-decreasing)
            assert t >= last_type, (
                f"Type ordering violated at pos {i}: got type {t} after type {last_type}"
            )
            last_type = t

    print("PASS: test_dataset_integration_arm_b")


def test_dataset_integration_arm_a():
    """Arm A (no permutation) should produce canonical ordering as in dataset."""
    import os
    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'data', 'md17_ethanol_v1'
    )

    if not os.path.exists(os.path.join(data_dir, "dataset.npz")):
        print("SKIP: test_dataset_integration_arm_a (dataset not found)")
        return

    ds = MD17Dataset(data_dir, split="train", permute_within_types=False, permute=False, max_atoms=9)
    n_real = 9

    # Canonical ordering for ethanol: C C O H H H H H H
    expected_types = [1, 1, 3, 0, 0, 0, 0, 0, 0]

    sample = ds[0]
    at = sample["atom_types"][:n_real].tolist()
    assert at == expected_types, f"Arm A: expected {expected_types}, got {at}"

    print("PASS: test_dataset_integration_arm_a")


if __name__ == "__main__":
    print("Running unit tests for permute_within_type_groups()...\n")
    test_type_sorted_output()
    test_type_counts_preserved()
    test_padding_stays_at_end()
    test_position_atom_type_consistency()
    test_within_group_randomization()
    test_single_atom_per_type()
    test_n_real_1()
    test_all_same_type()
    test_no_nitrogen()
    test_mutual_exclusion()
    test_dataset_integration_arm_b()
    test_dataset_integration_arm_a()
    print("\nAll tests PASSED.")
