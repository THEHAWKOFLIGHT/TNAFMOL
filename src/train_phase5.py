"""
train_phase5.py — Phase 5: Best Config Validation on All 8 MD17 Molecules.

Runs two configs on all 8 molecules (16 total runs):
  Config A: T=n_real (NO PADDING) — per-dim scale — ceiling test
  Config B: T=21 (PADDED) — shared scale — practical multi-molecule config

Molecules and atom counts:
  aspirin         21 atoms (no padding needed for Config A)
  naphthalene     18 atoms
  salicylic_acid  16 atoms
  toluene         15 atoms
  benzene         12 atoms
  uracil          12 atoms
  ethanol          9 atoms
  malonaldehyde    9 atoms

Usage:
  CUDA_VISIBLE_DEVICES=5 python src/train_phase5.py --molecule ethanol --config A --gpu 0
  CUDA_VISIBLE_DEVICES=6 python src/train_phase5.py --molecule aspirin --config B --gpu 0
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
    ATOM_EMB_DIM, NUM_ATOM_TYPES,
)
from data import MD17Dataset
import metrics as metrics_module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_BASE = 'experiments/understanding/und_001_tarflow_diagnostic/results/phase5'

# Molecule metadata: n_real atoms per molecule
MOLECULE_N_REAL = {
    'aspirin':        21,
    'naphthalene':    18,
    'salicylic_acid': 16,
    'toluene':        15,
    'benzene':        12,
    'uracil':         12,
    'ethanol':         9,
    'malonaldehyde':   9,
}

N_PADDED = 21   # global padding target for Config B

# Data directories (one per molecule)
def get_data_dir(project_root: str, molecule: str) -> str:
    return str(Path(project_root) / f'data/md17_{molecule}_v1')


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Build model for a given molecule and config
# ---------------------------------------------------------------------------

def build_model(
    molecule: str,
    config: str,   # 'A' or 'B'
    device: torch.device,
    seed: int,
    project_root: str,
) -> tuple:
    """
    Returns (model, cfg, n_real, seq_length, mask_np, atom_types_np)

    - model: TarFlow1DMol instance on device
    - cfg: full config dict (for W&B and results.json)
    - n_real: actual number of real atoms
    - seq_length: T used by model (n_real for A, 21 for B)
    - mask_np: (seq_length,) float32 mask for this config
    - atom_types_np: (21,) int atom type indices (always from 21-atom dataset format)
    """
    n_real = MOLECULE_N_REAL[molecule]

    base_cfg = {
        'exp_id': 'und_001',
        'phase': 'phase5',
        'molecule': molecule,
        'config': config,
        'n_real': n_real,
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Model hyperparams (same for all runs)
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training (same for all runs)
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
        # Noise augmentation (both configs)
        'use_noise': True,
        'noise_sigma': 0.05,
        # Common settings
        'use_atom_type_cond': True,
        'use_clamp': False,
        'log_det_reg_weight': 0.0,
        'data_dir': get_data_dir(project_root, molecule),
    }

    if config == 'A':
        # Config A: T=n_real, NO PADDING, per-dim scale (best for T=n_real from Phase 4)
        seq_length = n_real
        use_shared_scale = False
        use_padding_mask = False

        extra = {
            'seq_length': seq_length,
            'use_shared_scale': False,
            'use_padding_mask': False,
            'description': f'Config A (ceiling test): T={n_real}, no padding, per-dim scale, noise=0.05',
            'pad_fraction': 0.0,
        }
        cfg = {**base_cfg, **extra}

        model = TarFlow1DMol(
            in_channels=3,
            seq_length=seq_length,
            channels=256,
            num_blocks=4,
            layers_per_block=2,
            head_dim=64,
            use_atom_type_cond=True,
            atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=False,
            use_shared_scale=False,
        ).to(device)

        # Mask: all ones (no padding)
        mask_np = np.ones(seq_length, dtype=np.float32)

    elif config == 'B':
        # Config B: T=21 (padded), shared scale (best for padded regime from Phase 3/4)
        seq_length = N_PADDED
        use_shared_scale = True
        use_padding_mask = True
        pad_fraction = (N_PADDED - n_real) / N_PADDED

        extra = {
            'seq_length': seq_length,
            'use_shared_scale': True,
            'use_padding_mask': True,
            'description': f'Config B (practical): T=21, padded, shared scale, noise=0.05, pad_frac={pad_fraction:.3f}',
            'pad_fraction': pad_fraction,
        }
        cfg = {**base_cfg, **extra}

        model = TarFlow1DMol(
            in_channels=3,
            seq_length=seq_length,
            channels=256,
            num_blocks=4,
            layers_per_block=2,
            head_dim=64,
            use_atom_type_cond=True,
            atom_type_emb_dim=ATOM_EMB_DIM,
            num_atom_types=NUM_ATOM_TYPES,
            use_padding_mask=True,
            use_shared_scale=True,
        ).to(device)

        # Mask: n_real real atoms, rest padding
        mask_np = np.zeros(N_PADDED, dtype=np.float32)
        mask_np[:n_real] = 1.0

    else:
        raise ValueError(f"Unknown config: {config}. Must be 'A' or 'B'.")

    return model, cfg, n_real, seq_length, mask_np


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_phase5(
    molecule: str,
    config: str,
    device: torch.device,
    project_root: str,
    seed: int = 42,
) -> dict:
    """
    Train Config A or B on a given molecule for 5000 steps.

    Returns results dict with all key metrics.
    """
    set_seed(seed)

    n_real = MOLECULE_N_REAL[molecule]
    data_dir = get_data_dir(project_root, molecule)

    # Load dataset
    print(f"\n  Loading {molecule} data from {data_dir}...")
    train_set = MD17Dataset(data_dir, split='train')
    val_set = MD17Dataset(data_dir, split='val')

    # Get atom types from dataset (always 21-atom format from dataset)
    atom_types_21 = train_set.atom_types.numpy()   # (21,) int
    ref_positions_np = val_set.positions.numpy()   # (N_val, 21, 3)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}, n_real: {n_real}")

    # Build model and config
    model, cfg, n_real_v, seq_length, mask_np = build_model(
        molecule=molecule,
        config=config,
        device=device,
        seed=seed,
        project_root=project_root,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    cfg['device'] = str(device)
    print(f"  Model: {n_params:,} params, seq_length={seq_length}, config={config}")

    # Atom types sliced to seq_length (for Config A: first n_real; Config B: all 21)
    atom_types_for_model = atom_types_21[:seq_length]  # (seq_length,) int

    # Experiment output directory
    exp_dir = Path(project_root) / RESULTS_BASE / f'config_{config.lower()}_{molecule}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    train_iter = iter(train_loader)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])

    # W&B
    run_name = f"und_001_phase5_config{config}_{molecule}"
    wandb.init(
        project='tnafmol',
        name=run_name,
        group='und_001',
        tags=['understanding', 'und_001', 'DIAGNOSE', 'phase5',
              f'config{config}', molecule,
              'no_padding' if config == 'A' else 'padded'],
        config=cfg,
        notes=(
            f"Phase 5 Config {config} on {molecule}: "
            f"{'T=n_real=' + str(n_real) + ', no padding, per-dim scale' if config == 'A' else 'T=21, padded, shared scale'}"
        ),
        dir='/tmp',
        reinit=True,
    )
    assert wandb.run is not None, "W&B init failed"
    wandb.log({'setup_check': 1.0}, step=0)
    print(f"  W&B run: {wandb.run.url}")

    # Static tensors
    mask_tensor = torch.from_numpy(mask_np).to(device)              # (seq_length,)
    at_tensor = torch.from_numpy(atom_types_for_model).long().to(device)  # (seq_length,)

    use_padding = cfg['use_padding_mask']
    use_noise = cfg['use_noise']
    noise_sigma = cfg['noise_sigma']

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

        # Dataset always gives (B, 21, 3) — slice to seq_length
        positions_full = batch['positions'].to(device)   # (B, 21, 3)
        B = positions_full.size(0)

        x = positions_full[:, :seq_length, :]   # (B, seq_length, 3)
        # For Config A (seq_length=n_real): strips padding atoms, leaving only real coords
        # For Config B (seq_length=21): keeps all, mask handles padding in loss

        atom_types_batch = at_tensor.unsqueeze(0).expand(B, -1)  # (B, seq_length)

        if use_padding:
            pad_mask = mask_tensor.unsqueeze(0).expand(B, -1)  # (B, seq_length) float
        else:
            pad_mask = None

        # Noise augmentation on real atoms only
        if use_noise and noise_sigma > 0:
            noise = noise_sigma * torch.randn_like(x)
            if use_padding:
                # Zero noise on padding positions
                noise = noise * mask_tensor.unsqueeze(0).unsqueeze(-1)
            x = x + noise

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
                'seq_length': seq_length,
                'mask_np': mask_np,
                'atom_types_np': atom_types_21,
            }, exp_dir / 'best.pt')

        if step % 200 == 0 or step == cfg['steps'] - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{molecule} {config}] Step {step:5d} | loss={loss_val:.4f} | logdet/dof={logdet_per_dof:.4f} | lr={lr_now:.2e}")
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
    print(f"\n  [{molecule} {config}] Evaluating...")

    with torch.no_grad():
        if use_padding:
            pad_mask_eval = mask_tensor.unsqueeze(0).expand(1000, -1)
            at_eval = at_tensor.unsqueeze(0).expand(1000, -1)
            samples = model.sample(1000, device=device,
                                   atom_types=at_eval, padding_mask=pad_mask_eval)
        else:
            at_eval = at_tensor.unsqueeze(0).expand(1000, -1)
            samples = model.sample(1000, device=device, atom_types=at_eval)

    samples_np = samples.cpu().numpy()   # (1000, seq_length, 3)

    # Embed in (1000, 21, 3) for metrics (metrics expects 21-atom format)
    gen_padded = np.zeros((1000, 21, 3), dtype=np.float32)
    gen_padded[:, :seq_length, :] = samples_np

    # Evaluation mask: always the full 21-atom real/pad mask
    full_mask_21 = np.zeros(21, dtype=np.float32)
    full_mask_21[:n_real] = 1.0

    valid_frac, per_sample_valid = metrics_module.valid_fraction(gen_padded, full_mask_21)
    print(f"  [{molecule} {config}] Valid fraction: {valid_frac:.4f}")

    final_loss = losses[-1]
    avg_logdet_per_dof = float(np.mean(logdets_track[-100:])) if logdets_track else 0.0

    # Plots
    step_label = f'und_001_phase5_config{config}_{molecule}'
    loss_path = save_loss_curve(losses, logdets_track, exp_dir, step_label)

    # Reference for pairwise dist comparison
    ref_sub = ref_positions_np[:1000]  # (1000, 21, 3)
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
        'eval/n_real': n_real,
        'eval/pad_fraction': cfg.get('pad_fraction', 0.0),
    })
    wandb.run.summary.update({
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': final_loss,
        'valid_fraction': valid_frac,
        'logdet_per_dof': avg_logdet_per_dof,
        'n_real': n_real,
        'pad_fraction': cfg.get('pad_fraction', 0.0),
        'n_params': n_params,
        'molecule': molecule,
        'config': config,
    })

    artifact = wandb.Artifact(f'und_001_phase5_config{config}_{molecule}_model', type='model')
    artifact.add_file(str(exp_dir / 'best.pt'))
    wandb.log_artifact(artifact)
    wandb.finish()

    results = {
        'molecule': molecule,
        'config': config,
        'n_real': n_real,
        'seq_length': seq_length,
        'pad_fraction': float(cfg.get('pad_fraction', 0.0)),
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': final_loss,
        'valid_fraction': valid_frac,
        'logdet_per_dof': avg_logdet_per_dof,
        'n_params': n_params,
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  [{molecule} {config}] Done: VF={valid_frac:.4f}, loss={final_loss:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Phase 5: Best Config on All Molecules')
    parser.add_argument('--molecule', type=str, required=True,
                        choices=list(MOLECULE_N_REAL.keys()),
                        help='Molecule to train on')
    parser.add_argument('--config', type=str, required=True,
                        choices=['A', 'B'],
                        help='Config A (no padding, per-dim) or B (T=21, padded, shared)')
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

    print(f"\nPhase 5: Config {args.config} on {args.molecule}")
    print(f"  n_real={MOLECULE_N_REAL[args.molecule]}, T={'n_real' if args.config == 'A' else 21}")

    results = train_phase5(
        molecule=args.molecule,
        config=args.config,
        device=device,
        project_root=project_root,
        seed=args.seed,
    )

    print(f"\nFinal result: VF={results['valid_fraction']:.4f}, loss={results['final_loss']:.4f}")


if __name__ == '__main__':
    main()
