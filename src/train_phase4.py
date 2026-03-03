"""
train_phase4.py — Phase 4 Ablation Matrix for und_001 TarFlow Diagnostic.

Phase 4 runs 9 new configs filling gaps in the Phase 3 crossing:
  2x2x2 crossing (T × noise × scale), padding sweep, augmentation tests,
  and clamping-without-padding.

KEY FINDINGS FROM PHASE 3 (baseline for comparison):
  Step A: T=9,  no noise, per-dim, no clamp  → 89.1% VF
  Step B: T=9,  no noise, per-dim, no clamp  → 92.9% VF  (+ atom types)
  Step C: T=21, no noise, per-dim, no clamp  → 2.7%  VF  (+ padding — PRIMARY FAILURE)
  Step D: T=21, noise,    per-dim, no clamp  → 14.3% VF
  Step E: T=21, noise,    shared,  no clamp  → 40.2% VF  (shared scale HELPS)
  Step F: T=21, noise,    shared,  clamp     → 10.4% VF  (clamp HURTS)

Configs (all with atom type cond=True, channels=256, 4 blocks, 2 layers/block,
         head_dim=64, batch_size=256, lr=5e-4, cosine, grad_clip=1.0, seed=42):

  Core 2x2x2 crossing (fill Phase 3 gaps, T x noise x scale, no clamping):
  1. T=9,  no noise, shared scale   → shared scale without padding
  2. T=9,  noise,    per-dim scale  → noise effect without padding
  3. T=9,  noise,    shared scale   → best-of-both without padding
  4. T=21, no noise, shared scale   → shared scale helps even without noise?

  Padding sweep (noise=yes, shared scale, no clamp):
  5. T=12, noise, shared  → minimal padding (9 real + 3 pad)
  6. T=15, noise, shared  → moderate padding (9 real + 6 pad)

  Augmentation tests (T=21, noise, shared, no clamp):
  7. + permutation augmentation   (random atom order each batch)
  8. + SO(3) + CoM augmentation   (random rotation each batch)

  Clamping without padding:
  9. T=9, noise, shared, clamp (alpha_pos=0.1, reg=0.01) → our pipeline, no padding

Usage:
  CUDA_VISIBLE_DEVICES=5 python src/train_phase4.py --config 1 --gpu 0
  CUDA_VISIBLE_DEVICES=6 python src/train_phase4.py --config 2 --gpu 0

  # Run all configs sequentially on one GPU
  CUDA_VISIBLE_DEVICES=5 python src/train_phase4.py --config all --gpu 0
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from train_phase3 import (
    TarFlow1DMol, MetaBlockSharedScale, MetaBlockWithCond,
    save_loss_curve, save_pairwise_dist_comparison, _get_git_hash,
    N_REAL_ATOMS, N_PADDED_ATOMS, ATOM_EMB_DIM, NUM_ATOM_TYPES,
    ETHANOL_DATA_DIR,
)
from data import MD17Dataset, augment_positions
import metrics as metrics_module


# ---------------------------------------------------------------------------
# Phase 4 constants
# ---------------------------------------------------------------------------

RESULTS_BASE = 'experiments/understanding/und_001_tarflow_diagnostic/results/phase4'

CONFIG_DESCRIPTORS = {
    1: 'T9_nonoise_shared',
    2: 'T9_noise_perdim',
    3: 'T9_noise_shared',
    4: 'T21_nonoise_shared',
    5: 'T12_noise_shared',
    6: 'T15_noise_shared',
    7: 'T21_noise_shared_permaug',
    8: 'T21_noise_shared_so3aug',
    9: 'T9_noise_shared_clamp',
}

CONFIG_DESCRIPTIONS = {
    1: 'T=9, no noise, shared scale — does shared scale help/hurt without padding?',
    2: 'T=9, noise(σ=0.05), per-dim scale — noise effect without padding',
    3: 'T=9, noise(σ=0.05), shared scale — best-case without padding',
    4: 'T=21, no noise, shared scale — does shared scale help padded without noise?',
    5: 'T=12 (3 pad), noise, shared — minimal padding',
    6: 'T=15 (6 pad), noise, shared — moderate padding',
    7: 'T=21, noise, shared + permutation augmentation',
    8: 'T=21, noise, shared + SO(3) + CoM augmentation',
    9: 'T=9, noise, shared + clamp (alpha_pos=0.1) — no-padding clamping test',
}

# NOTE: For configs 5 and 6, we use non-standard T values.
# We still load the full 21-atom dataset but create a custom mask truncating padding:
# T=12: real atoms 0..8 (9 real) + 3 padding → mask[0:9]=1, mask[9:12]=0, zero everything >=12
# T=15: real atoms 0..8 (9 real) + 6 padding → mask[0:9]=1, mask[9:15]=0, zero everything >=15
# Implementation: we slice the dataset to only the first T atoms and create a custom mask.


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Build configs and models
# ---------------------------------------------------------------------------

def build_config(cfg_id: int, device: torch.device, seed: int, project_root: str) -> tuple:
    """
    Returns (model, cfg, n_real, seq_length, custom_mask_np or None).

    custom_mask_np: if not None, overrides the default mask from dataset.
    """
    base_cfg = {
        'exp_id': 'und_001',
        'phase': 'phase4',
        'config': cfg_id,
        'descriptor': CONFIG_DESCRIPTORS[cfg_id],
        'description': CONFIG_DESCRIPTIONS[cfg_id],
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Model hyperparams (same for all configs)
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training (same for all configs)
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    # Per-config settings
    if cfg_id == 1:
        # T=9, no noise, shared scale
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_REAL_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': False,
               'use_noise': False, 'noise_sigma': 0.0,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': False}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_REAL_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=False, use_shared_scale=True,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_REAL_ATOMS, None

    elif cfg_id == 2:
        # T=9, noise, per-dim scale
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_REAL_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': False,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': False, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': False}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_REAL_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=False, use_shared_scale=False,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_REAL_ATOMS, None

    elif cfg_id == 3:
        # T=9, noise, shared scale
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_REAL_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': False,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': False}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_REAL_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=False, use_shared_scale=True,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_REAL_ATOMS, None

    elif cfg_id == 4:
        # T=21, no noise, shared scale
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_PADDED_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': True,
               'use_noise': False, 'noise_sigma': 0.0,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': False}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_PADDED_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=True, use_shared_scale=True,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_PADDED_ATOMS, None

    elif cfg_id == 5:
        # T=12, noise, shared — minimal padding (9 real + 3 pad)
        T = 12
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': T,
               'use_atom_type_cond': True, 'use_padding_mask': True,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': False,
               'padding_fraction': (T - N_REAL_ATOMS) / T}
        model = TarFlow1DMol(
            in_channels=3, seq_length=T,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=True, use_shared_scale=True,
        ).to(device)
        # Create custom mask: first 9 atoms real, rest padding
        custom_mask = np.zeros(T, dtype=np.float32)
        custom_mask[:N_REAL_ATOMS] = 1.0
        return model, cfg, N_REAL_ATOMS, T, custom_mask

    elif cfg_id == 6:
        # T=15, noise, shared — moderate padding (9 real + 6 pad)
        T = 15
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': T,
               'use_atom_type_cond': True, 'use_padding_mask': True,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': False,
               'padding_fraction': (T - N_REAL_ATOMS) / T}
        model = TarFlow1DMol(
            in_channels=3, seq_length=T,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=True, use_shared_scale=True,
        ).to(device)
        # Create custom mask: first 9 atoms real, rest padding
        custom_mask = np.zeros(T, dtype=np.float32)
        custom_mask[:N_REAL_ATOMS] = 1.0
        return model, cfg, N_REAL_ATOMS, T, custom_mask

    elif cfg_id == 7:
        # T=21, noise, shared + permutation augmentation
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_PADDED_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': True,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': True, 'use_so3_aug': False}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_PADDED_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=True, use_shared_scale=True,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_PADDED_ATOMS, None

    elif cfg_id == 8:
        # T=21, noise, shared + SO(3) + CoM augmentation
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_PADDED_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': True,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': True, 'use_clamp': False,
               'log_det_reg_weight': 0.0,
               'use_perm_aug': False, 'use_so3_aug': True}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_PADDED_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=True, use_shared_scale=True,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_PADDED_ATOMS, None

    elif cfg_id == 9:
        # T=9, noise, shared + clamp (no padding)
        cfg = {**base_cfg,
               'n_real': N_REAL_ATOMS, 'seq_length': N_REAL_ATOMS,
               'use_atom_type_cond': True, 'use_padding_mask': False,
               'use_noise': True, 'noise_sigma': 0.05,
               'use_shared_scale': True, 'use_clamp': True,
               'alpha_pos': 0.1, 'alpha_neg': 2.0,
               'log_det_reg_weight': 0.01,
               'use_perm_aug': False, 'use_so3_aug': False}
        model = TarFlow1DMol(
            in_channels=3, seq_length=N_REAL_ATOMS,
            channels=256, num_blocks=4, layers_per_block=2, head_dim=64,
            use_atom_type_cond=True, atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=False, use_shared_scale=True,
            use_clamp=True, alpha_pos=0.1, alpha_neg=2.0,
            log_det_reg_weight=0.01,
        ).to(device)
        return model, cfg, N_REAL_ATOMS, N_REAL_ATOMS, None

    else:
        raise ValueError(f"Unknown config id: {cfg_id}")


# ---------------------------------------------------------------------------
# Permutation augmentation
# ---------------------------------------------------------------------------

def permute_atoms(
    positions: torch.Tensor,       # (B, T, 3)
    atom_types: torch.Tensor,       # (B, T)
    mask: torch.Tensor,             # (B, T)
    n_real: int,
) -> tuple:
    """
    Randomly permute the order of the first n_real atoms in each batch sample.

    Only permutes the real atom positions (indices 0..n_real-1).
    Padding atoms (indices n_real..T-1) are left in place.
    Atom type indices are permuted consistently with positions.

    Returns:
        (permuted_positions, permuted_atom_types, mask_unchanged)
    """
    B, T, _ = positions.shape
    device = positions.device

    # Generate random permutations of the n_real real atoms for each sample
    perms = torch.stack([
        torch.randperm(n_real, device=device)
        for _ in range(B)
    ])  # (B, n_real)

    # Apply permutation to positions
    pos_out = positions.clone()
    at_out = atom_types.clone()

    # For each sample, reorder the real-atom slice
    # Using gather-based approach for efficiency
    # perms: (B, n_real) → expand to (B, n_real, 3) for gathering
    perm_idx_3d = perms.unsqueeze(-1).expand(-1, -1, 3)  # (B, n_real, 3)
    pos_out[:, :n_real, :] = torch.gather(positions[:, :n_real, :], 1, perm_idx_3d)
    at_out[:, :n_real] = torch.gather(atom_types[:, :n_real], 1, perms)

    return pos_out, at_out, mask


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_phase4_config(
    cfg_id: int,
    cfg: dict,
    device: torch.device,
    exp_dir: Path,
    model: TarFlow1DMol,
    train_set,
    val_set,
    mask_np: np.ndarray,    # the mask for this config's seq_length
    atom_types_np: np.ndarray,  # (21,) full atom types from dataset
    ref_positions_np: np.ndarray,   # (N_val, 21, 3)
    n_real: int,
    seq_length: int,
):
    """
    Unified training loop for Phase 4 configs.

    Key differences from Phase 3 train_step:
    - Supports permutation augmentation (use_perm_aug)
    - Supports SO(3) + CoM augmentation (use_so3_aug)
    - Handles non-standard T (12, 15) by slicing positions to seq_length
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    train_iter = iter(train_loader)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])

    run_name = f"und_001_phase4_config{cfg_id}_{CONFIG_DESCRIPTORS[cfg_id]}"
    wandb.init(
        project='tnafmol',
        name=run_name,
        group='und_001',
        tags=['understanding', 'und_001', 'DIAGNOSE', 'phase4',
              f'config{cfg_id}', CONFIG_DESCRIPTORS[cfg_id], 'ethanol'],
        config=cfg,
        notes=f'Phase 4 Config {cfg_id}: {CONFIG_DESCRIPTIONS[cfg_id]}',
        dir='/tmp',
        reinit=True,
    )
    assert wandb.run is not None, "W&B init failed"
    wandb.log({'setup_check': 1.0}, step=0)
    print(f"W&B run: {wandb.run.url}")

    use_padding = cfg.get('use_padding_mask', False)
    use_atom_type = cfg.get('use_atom_type_cond', True)  # all phase 4 use atom type
    use_noise = cfg.get('use_noise', False)
    noise_sigma = cfg.get('noise_sigma', 0.0)
    use_perm_aug = cfg.get('use_perm_aug', False)
    use_so3_aug = cfg.get('use_so3_aug', False)

    # Prepare static tensors
    # mask_np for this config's seq_length
    mask_tensor = torch.from_numpy(mask_np).to(device)        # (seq_length,)
    at_full_21 = torch.from_numpy(atom_types_np).long().to(device)  # (21,)
    at_for_model = at_full_21[:seq_length]  # (seq_length,) — may be 9, 12, 15, or 21

    losses = []
    logdets_track = []
    best_loss = float('inf')
    best_step = 0
    n_dof = n_real * 3

    model.train()
    for step in range(cfg['steps']):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        positions_full = batch['positions'].to(device)   # (B, 21, 3)
        B = positions_full.size(0)

        # Slice to this config's seq_length
        # For T=9: first 9 atoms (real only, no padding)
        # For T=12, 15: first T atoms (9 real + some padding)
        # For T=21: all atoms
        x = positions_full[:, :seq_length, :]   # (B, seq_length, 3)
        atom_types_batch = at_for_model.unsqueeze(0).expand(B, -1)  # (B, seq_length)

        if use_padding:
            pad_mask = mask_tensor.unsqueeze(0).expand(B, -1)  # (B, seq_length)
        else:
            pad_mask = None

        # Noise augmentation (only on real atoms)
        if use_noise and noise_sigma > 0:
            noise = noise_sigma * torch.randn_like(x)
            if use_padding:
                # Zero noise on padding positions
                noise = noise * mask_tensor.unsqueeze(0).unsqueeze(-1)
            x = x + noise

        # Permutation augmentation: randomly shuffle real atom order
        if use_perm_aug:
            x, atom_types_batch, pad_mask = permute_atoms(x, atom_types_batch, pad_mask, n_real)

        # SO(3) + CoM augmentation: random rotation + CoM noise
        if use_so3_aug:
            # augment_positions expects (B, T, 3) and mask (T,) or (B, T)
            x = augment_positions(x, mask_tensor)

        optimizer.zero_grad()

        z, logdets = model(x, atom_types=atom_types_batch, padding_mask=pad_mask)
        loss, info = model.get_loss(z, logdets, padding_mask=pad_mask)

        if not torch.isfinite(loss):
            print(f"  WARNING: Non-finite loss at step {step}: {loss.item()} — skipping")
            optimizer.zero_grad()
            scheduler.step()
            loss_val = losses[-1] if losses else 0.0
            losses.append(loss_val)
            logdets_track.append(logdets_track[-1] if logdets_track else 0.0)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)
        logdet_per_dof = info['logdets_mean'] / n_dof
        logdets_track.append(logdet_per_dof)

        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
                'n_real': n_real,
                'mask_np': mask_np,
                'atom_types_np': atom_types_np,
            }, exp_dir / 'best.pt')

        if step % 100 == 0 or step == cfg['steps'] - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step:5d} | loss={loss_val:.4f} | logdet/dof={logdet_per_dof:.4f} | lr={lr_now:.2e}")
            wandb.log({
                'train/loss': loss_val,
                'train/nll': info['nll'],
                'train/reg': info['reg'],
                'train/logdet_per_dof': logdet_per_dof,
                'train/logdets_mean': info['logdets_mean'],
                'train/lr': lr_now,
            }, step=step)

        if (step + 1) % 1000 == 0:
            ckpt_path = exp_dir / f'checkpoint_step{step+1}.pt'
            torch.save({'step': step, 'model_state_dict': model.state_dict(),
                        'config': cfg, 'best_loss': best_loss}, ckpt_path)
            ckpts = sorted(exp_dir.glob('checkpoint_step*.pt'),
                           key=lambda p: int(p.stem.split('step')[1]))
            for old in ckpts[:-2]:
                old.unlink()

    # Evaluation
    model.eval()
    print(f"\n  Evaluating config {cfg_id}...")

    with torch.no_grad():
        if use_padding:
            pad_mask_eval = mask_tensor.unsqueeze(0).expand(1000, -1)
            at_eval = at_for_model.unsqueeze(0).expand(1000, -1)
            samples = model.sample(1000, device=device,
                                   atom_types=at_eval, padding_mask=pad_mask_eval)
        else:
            at_eval = at_for_model.unsqueeze(0).expand(1000, -1)
            samples = model.sample(1000, device=device, atom_types=at_eval)

    samples_np = samples.cpu().numpy()   # (1000, seq_length, 3)

    # Embed in (1000, 21, 3) for metrics (metrics expects 21-atom format with mask)
    gen_padded = np.zeros((1000, 21, 3), dtype=np.float32)
    gen_padded[:, :seq_length, :] = samples_np

    # For metrics, always use the 21-atom mask from the dataset (real=9, rest=0)
    full_mask_21 = np.zeros(21, dtype=np.float32)
    full_mask_21[:n_real] = 1.0

    valid_frac, per_sample_valid = metrics_module.valid_fraction(gen_padded, full_mask_21)
    print(f"  Valid fraction: {valid_frac:.4f}")

    final_loss = losses[-1]
    avg_logdet_per_dof = np.mean(logdets_track[-100:]) if logdets_track else 0.0

    print(f"  Final loss: {final_loss:.4f}")
    print(f"  log_det/dof (last 100): {avg_logdet_per_dof:.4f}")

    ref_sub = ref_positions_np[:1000]   # (1000, 21, 3)

    step_label = f'und_001_phase4_config{cfg_id}_{CONFIG_DESCRIPTORS[cfg_id]}'
    loss_path = save_loss_curve(losses, logdets_track, exp_dir, step_label)
    dist_path = save_pairwise_dist_comparison(
        gen_padded, ref_sub, full_mask_21, exp_dir, step_label, valid_frac
    )

    # Save raw outputs
    raw_dir = exp_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)
    np.save(raw_dir / 'generated_positions.npy', gen_padded)
    np.save(raw_dir / 'per_sample_valid.npy', per_sample_valid)
    np.save(raw_dir / 'losses.npy', np.array(losses))
    np.save(raw_dir / 'logdets_track.npy', np.array(logdets_track))

    wandb.log({
        'final/loss_curve': wandb.Image(str(loss_path)),
        'final/pairwise_dist': wandb.Image(str(dist_path)),
        'eval/valid_fraction': valid_frac,
    })
    wandb.run.summary.update({
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': final_loss,
        'valid_fraction': valid_frac,
        'nll_per_dof': final_loss,
        'logdet_per_dof': avg_logdet_per_dof,
        'n_params': cfg.get('n_params', 0),
    })

    artifact = wandb.Artifact(f'und_001_phase4_config{cfg_id}_model', type='model')
    artifact.add_file(str(exp_dir / 'best.pt'))
    wandb.log_artifact(artifact)
    wandb.finish()

    results = {
        'config': cfg_id,
        'descriptor': CONFIG_DESCRIPTORS[cfg_id],
        'description': CONFIG_DESCRIPTIONS[cfg_id],
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': final_loss,
        'valid_fraction': valid_frac,
        'nll_per_dof': final_loss,
        'logdet_per_dof': avg_logdet_per_dof,
        'n_params': cfg.get('n_params', 0),
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Config {cfg_id} complete: valid_frac={valid_frac:.4f}, loss={final_loss:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Phase 4 Ablation Matrix')
    parser.add_argument('--config', type=str, required=True,
                        help='Config to run (1-9) or "all"')
    parser.add_argument('--gpu', type=int, default=0, help='Logical GPU index')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")

    project_root = str(Path(__file__).parent.parent)
    data_dir = str(Path(project_root) / ETHANOL_DATA_DIR)

    print(f"\nLoading ethanol data from {data_dir}...")
    train_set = MD17Dataset(data_dir, split='train')
    val_set = MD17Dataset(data_dir, split='val')
    mask_21 = train_set.mask.numpy()        # (21,) from dataset
    atom_types_21 = train_set.atom_types.numpy()  # (21,) int
    ref_positions_np = val_set.positions.numpy()   # (N_val, 21, 3)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}")
    print(f"  Mask n_real: {int(mask_21.sum())}")

    if args.config == 'all':
        configs_to_run = list(range(1, 10))
    else:
        configs_to_run = [int(args.config)]

    all_results = []

    for cfg_id in configs_to_run:
        print(f"\n{'='*70}")
        print(f"CONFIG {cfg_id}: {CONFIG_DESCRIPTIONS[cfg_id]}")
        print(f"{'='*70}")

        set_seed(args.seed)

        model, cfg, n_real, seq_length, custom_mask = build_config(
            cfg_id, device, args.seed, project_root
        )

        # Determine the mask to use for this config
        if custom_mask is not None:
            # T=12 or T=15: custom mask of length seq_length
            mask_for_cfg = custom_mask
        elif seq_length == N_REAL_ATOMS:
            # No padding: mask is all ones of length 9
            mask_for_cfg = np.ones(N_REAL_ATOMS, dtype=np.float32)
        else:
            # T=21: use dataset mask
            mask_for_cfg = mask_21

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg['n_params'] = n_params
        cfg['device'] = str(device)
        print(f"  Model: {n_params:,} parameters, seq_length={seq_length}")

        exp_dir = Path(project_root) / RESULTS_BASE / f'config_{cfg_id}_{CONFIG_DESCRIPTORS[cfg_id]}'

        results = train_phase4_config(
            cfg_id=cfg_id,
            cfg=cfg,
            device=device,
            exp_dir=exp_dir,
            model=model,
            train_set=train_set,
            val_set=val_set,
            mask_np=mask_for_cfg,
            atom_types_np=atom_types_21,
            ref_positions_np=ref_positions_np,
            n_real=n_real,
            seq_length=seq_length,
        )
        all_results.append(results)

    # Summary
    print(f"\n{'='*80}")
    print(f"PHASE 4 ABLATION MATRIX — SUMMARY")
    print(f"{'='*80}")
    print(f"{'Cfg':<4} {'Descriptor':<30} {'Loss':>8} {'Valid%':>8} {'logd/dof':>10}")
    print(f"{'-'*4} {'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    for r in all_results:
        desc = r['descriptor'][:28]
        print(f"  {r['config']:<2}  {desc:<30} {r['final_loss']:>8.4f} "
              f"{r['valid_fraction']*100:>7.1f}% {r['logdet_per_dof']:>10.4f}")

    return all_results


if __name__ == '__main__':
    main()
