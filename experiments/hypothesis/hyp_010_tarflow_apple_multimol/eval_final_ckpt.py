"""
Evaluate the FINAL checkpoint (not best val) from Phase 1.

This checks whether the log-det exploitation leads to good or bad samples
when we use the fully-trained model (train loss = -1.06).
"""
import os, sys
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.train_phase3 import TarFlow1DMol
from src.metrics import valid_fraction, min_pairwise_distance

DEVICE = torch.device("cuda:7")
STAGE_DIR = os.path.join(project_root, "experiments/hypothesis/hyp_010_tarflow_apple_multimol/angles/sanity/val")
FINAL_CKPT = os.path.join(STAGE_DIR, "final.pt")
DATA_DIR = os.path.join(project_root, "data/md17_ethanol_v1")

print("Loading final checkpoint...")
ckpt = torch.load(FINAL_CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt["config"]
global_std = ckpt.get("global_std", None)
print(f"  Step: {ckpt['step']}, best_val_loss: {ckpt.get('best_val_loss', 'N/A')}")
print(f"  global_std: {global_std}")

model = TarFlow1DMol(
    in_channels=3,
    seq_length=cfg["seq_length"],
    channels=cfg["channels"],
    num_blocks=cfg["num_blocks"],
    layers_per_block=cfg["layers_per_block"],
    head_dim=cfg.get("head_dim", 64),
    expansion=cfg.get("expansion", 4),
    use_atom_type_cond=cfg.get("use_atom_type_cond", True),
    atom_type_emb_dim=cfg.get("atom_type_emb_dim", 16),
    num_atom_types=cfg.get("num_atom_types", 4),
    use_padding_mask=cfg.get("use_padding_mask", False),
    use_shared_scale=cfg.get("use_shared_scale", False),
    use_clamp=cfg.get("use_clamp", False),
    log_det_reg_weight=cfg.get("log_det_reg_weight", 0.0),
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params)")

# Load ethanol data
data = np.load(os.path.join(DATA_DIR, "dataset.npz"))
atom_types_np = data["atom_types"][:cfg["max_atoms"]]
mask_np = data["mask"][:cfg["max_atoms"]]
n_real = int(mask_np.sum())
print(f"Ethanol: {n_real} real atoms, seq_length={cfg['seq_length']}")

atom_types = torch.from_numpy(atom_types_np).long().unsqueeze(0).expand(500, -1).to(DEVICE)

print("\nGenerating 500 samples from FINAL checkpoint (step 5000)...")
with torch.no_grad():
    samples = model.sample(n=500, device=DEVICE, atom_types=atom_types, temp=1.0)
samples_np = samples.cpu().numpy()

# Denormalize
if global_std:
    samples_np = samples_np * global_std

vf, _ = valid_fraction(samples_np, mask_np)
min_dists = min_pairwise_distance(samples_np, mask_np)
print(f"  VF = {vf:.3f}")
print(f"  Min pairwise dist: mean={min_dists.mean():.3f}, median={np.median(min_dists):.3f}")
print(f"  Fraction below 0.8: {(min_dists < 0.8).mean():.3f}")
print(f"  Fraction below 0.5: {(min_dists < 0.5).mean():.3f}")

# Also check sample scale
print(f"\nSample statistics:")
print(f"  Mean abs: {np.abs(samples_np[:, :n_real]).mean():.3f}")
print(f"  Std: {samples_np[:, :n_real].std():.3f}")
print(f"  Range: [{samples_np[:, :n_real].min():.2f}, {samples_np[:, :n_real].max():.2f}]")
