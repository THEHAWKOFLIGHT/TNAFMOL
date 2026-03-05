"""Quick verification test for max_atoms implementation (hyp_007).

Verifies:
1. max_atoms=9 gives (B, 9, 3) tensors with no excess padding
2. max_atoms=12 gives (B, 12, 3) tensors with 3 padding slots for ethanol (9 real atoms)
3. Forward pass works at each max_atoms value
4. Inverse (sample) works at each max_atoms value
5. Model rejects max_atoms < n_real

Run: python experiments/hypothesis/hyp_007_padding_isolation_multimol/verify_max_atoms.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import numpy as np
from src.data import MD17Dataset, MultiMoleculeDataset, MAX_ATOMS, PAD_TOKEN_IDX
from src.model import TarFlow

device = torch.device("cpu")
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")

# Ethanol has 9 real atoms
ETHANOL_N_REAL = 9

print("=" * 60)
print("hyp_007 max_atoms verification")
print("=" * 60)

# Test 1: max_atoms=9 (no padding)
print("\n--- Test 1: max_atoms=9 (exactly real atoms, no padding) ---")
ds9 = MD17Dataset(
    os.path.join(data_root, "md17_ethanol_v1"),
    split="train", augment=False, global_std=None,
    pad_token_idx=PAD_TOKEN_IDX, noise_sigma=0.0, max_atoms=9
)
s = ds9[0]
assert s["positions"].shape == (9, 3), f"Expected (9,3), got {s['positions'].shape}"
assert s["mask"].shape == (9,), f"Expected (9,), got {s['mask'].shape}"
assert s["atom_types"].shape == (9,), f"Expected (9,), got {s['atom_types'].shape}"
assert s["mask"].sum().item() == 9, f"Expected all 9 real, got {s['mask'].sum().item()}"
print(f"  positions shape: {s['positions'].shape} [PASS]")
print(f"  mask shape: {s['mask'].shape}, sum={s['mask'].sum().item()} [PASS]")
print(f"  atom_types shape: {s['atom_types'].shape} [PASS]")
print(f"  mask = {s['mask'].numpy()} [all 1s, no padding: PASS]")

# Test 2: max_atoms=12 (3 padding slots)
print("\n--- Test 2: max_atoms=12 (9 real + 3 padding) ---")
ds12 = MD17Dataset(
    os.path.join(data_root, "md17_ethanol_v1"),
    split="train", augment=False, global_std=None,
    pad_token_idx=PAD_TOKEN_IDX, noise_sigma=0.0, max_atoms=12
)
s = ds12[0]
assert s["positions"].shape == (12, 3), f"Expected (12,3), got {s['positions'].shape}"
assert s["mask"].shape == (12,), f"Expected (12,), got {s['mask'].shape}"
assert s["mask"].sum().item() == 9, f"Expected 9 real atoms, got {s['mask'].sum().item()}"
assert (s["atom_types"][9:] == PAD_TOKEN_IDX).all(), f"Expected PAD token at positions 9-11"
print(f"  positions shape: {s['positions'].shape} [PASS]")
print(f"  mask[:9]={s['mask'][:9].numpy()} mask[9:]={s['mask'][9:].numpy()} [PASS]")
print(f"  atom_types[9:]={s['atom_types'][9:].numpy()} (PAD_TOKEN={PAD_TOKEN_IDX}) [PASS]")

# Test 3: max_atoms=21 (default — same as before)
print("\n--- Test 3: max_atoms=21 (default, same as MAX_ATOMS) ---")
ds21 = MD17Dataset(
    os.path.join(data_root, "md17_ethanol_v1"),
    split="train", augment=False, global_std=None,
    pad_token_idx=PAD_TOKEN_IDX, noise_sigma=0.0, max_atoms=21
)
s = ds21[0]
assert s["positions"].shape == (21, 3), f"Expected (21,3), got {s['positions'].shape}"
assert s["mask"].sum().item() == 9, f"Expected 9 real atoms, got {s['mask'].sum().item()}"
print(f"  positions shape: {s['positions'].shape} [PASS]")
print(f"  mask real atoms: {s['mask'].sum().item()} [PASS]")

# Test 4: TarFlow forward + sample at max_atoms=9
print("\n--- Test 4: TarFlow forward + sample at max_atoms=9 ---")
model9 = TarFlow(
    n_blocks=4, d_model=64, n_heads=2, ffn_mult=2,
    atom_type_emb_dim=8, n_atom_types=5, dropout=0.0,
    max_atoms=9, alpha_pos=10.0, alpha_neg=10.0,
    shift_only=False, use_actnorm=False, use_bidir_types=True,
    use_pos_enc=False, zero_padding_queries=True, use_output_shift=True,
).to(device)

from torch.utils.data import DataLoader
dl9 = DataLoader(ds9, batch_size=4, shuffle=False)
batch = next(iter(dl9))
pos = batch["positions"].to(device)       # (4, 9, 3)
atypes = batch["atom_types"].to(device)   # (4, 9)
mask = batch["mask"].to(device)           # (4, 9)
assert pos.shape == (4, 9, 3), f"Expected (4,9,3), got {pos.shape}"

z, log_det = model9.forward(pos, atypes, mask)
assert z.shape == (4, 9, 3), f"Expected z (4,9,3), got {z.shape}"
assert torch.isfinite(z).all(), "z contains non-finite values"
print(f"  forward: z shape={z.shape}, finite={torch.isfinite(z).all().item()} [PASS]")

# Sample
atom_types_1d = atypes[0]  # (9,)
mask_1d = mask[0]           # (9,)
samples = model9.sample(atom_types_1d, mask_1d, n_samples=4)
assert samples.shape == (4, 9, 3), f"Expected samples (4,9,3), got {samples.shape}"
assert torch.isfinite(samples).all(), "samples contain non-finite values"
print(f"  sample: shape={samples.shape}, finite={torch.isfinite(samples).all().item()} [PASS]")

# Test 5: TarFlow at max_atoms=12
print("\n--- Test 5: TarFlow forward + sample at max_atoms=12 ---")
model12 = TarFlow(
    n_blocks=4, d_model=64, n_heads=2, ffn_mult=2,
    atom_type_emb_dim=8, n_atom_types=5, dropout=0.0,
    max_atoms=12, alpha_pos=10.0, alpha_neg=10.0,
    shift_only=False, use_actnorm=False, use_bidir_types=True,
    use_pos_enc=False, zero_padding_queries=True, use_output_shift=True,
).to(device)

dl12 = DataLoader(ds12, batch_size=4, shuffle=False)
batch12 = next(iter(dl12))
pos12 = batch12["positions"].to(device)
atypes12 = batch12["atom_types"].to(device)
mask12 = batch12["mask"].to(device)
assert pos12.shape == (4, 12, 3), f"Expected (4,12,3), got {pos12.shape}"

z12, ld12 = model12.forward(pos12, atypes12, mask12)
assert z12.shape == (4, 12, 3), f"Expected z (4,12,3), got {z12.shape}"
assert torch.isfinite(z12).all(), "z12 contains non-finite values"
print(f"  forward: z shape={z12.shape}, finite={torch.isfinite(z12).all().item()} [PASS]")

samples12 = model12.sample(atypes12[0], mask12[0], n_samples=4)
assert samples12.shape == (4, 12, 3), f"Expected (4,12,3), got {samples12.shape}"
assert torch.isfinite(samples12).all(), "samples12 non-finite"
print(f"  sample: shape={samples12.shape}, finite={torch.isfinite(samples12).all().item()} [PASS]")

# Test 6: Loss computation
print("\n--- Test 6: NLL loss at max_atoms=9 and max_atoms=12 ---")
loss9, info9 = model9.nll_loss(pos, atypes, mask, log_det_reg_weight=0.0)
assert torch.isfinite(loss9), f"loss9 non-finite: {loss9}"
print(f"  max_atoms=9: loss={loss9.item():.4f}, finite [PASS]")

loss12, info12 = model12.nll_loss(pos12, atypes12, mask12, log_det_reg_weight=0.0)
assert torch.isfinite(loss12), f"loss12 non-finite: {loss12}"
print(f"  max_atoms=12: loss={loss12.item():.4f}, finite [PASS]")

# Test 7: assert max_atoms < n_real raises
print("\n--- Test 7: max_atoms < n_real raises AssertionError ---")
try:
    bad_ds = MD17Dataset(
        os.path.join(data_root, "md17_ethanol_v1"),
        split="train", augment=False, global_std=None,
        pad_token_idx=PAD_TOKEN_IDX, max_atoms=5  # ethanol has 9 real atoms
    )
    print("  ERROR: should have raised AssertionError [FAIL]")
    sys.exit(1)
except AssertionError as e:
    print(f"  Caught: {e} [PASS]")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
