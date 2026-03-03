"""
train_ladder.py — Unified training script for the TarFlow Diagnostic Ladder (und_001).

Handles all benchmark levels:
  Level 0: 2D 8-mode Gaussian mixture
  Level 1: MNIST
  Level 2: CIFAR-10
  Level 3: Molecular adaptation (used in Phase 3 of und_001)

Usage:
  python src/train_ladder.py --level 0 --gpu 5 --exp_dir experiments/understanding/und_001_tarflow_diagnostic/results/phase2/level0_2d_gaussian
  python src/train_ladder.py --level 1 --gpu 6 --exp_dir experiments/understanding/und_001_tarflow_diagnostic/results/phase2/level1_mnist
  python src/train_ladder.py --level 2 --gpu 7 --exp_dir experiments/understanding/und_001_tarflow_diagnostic/results/phase2/level2_cifar10
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from tarflow_apple import TarFlowApple, TarFlow1D


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Data: Level 0 — 2D 8-mode Gaussian mixture
# ---------------------------------------------------------------------------

def make_gaussian_mixture_dataset(
    n_modes: int = 8,
    radius: float = 5.0,
    sigma: float = 0.5,
    n_samples: int = 100_000,
    seed: int = 42,
) -> torch.Tensor:
    """Generate 2D 8-mode Gaussian ring mixture."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    samples_per_mode = n_samples // n_modes
    all_samples = []
    for c in centers:
        pts = rng.normal(c, sigma, size=(samples_per_mode, 2))
        all_samples.append(pts)
    data = np.concatenate(all_samples, axis=0).astype(np.float32)
    np.random.shuffle(data)
    return torch.from_numpy(data)


class GaussianMixtureDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return (B, 1, 2) shape — single "patch" of 2 dims
        return self.data[idx].unsqueeze(0)  # (1, 2)


# ---------------------------------------------------------------------------
# Data: Level 1 — MNIST
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int, num_workers: int = 4):
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # normalize to [-1, 1]
    ])
    train_set = torchvision.datasets.MNIST(
        root='/tmp/mnist_data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root='/tmp/mnist_data', train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Data: Level 2 — CIFAR-10
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int, num_workers: int = 4):
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize to [-1, 1]
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10_data', train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10_data', train=False, download=True, transform=transform_test
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Bits per dim computation
# ---------------------------------------------------------------------------

def compute_bits_per_dim(
    model, data_loader, device, noise_sigma: float, n_dims: int
) -> float:
    """
    Compute bits/dim on a dataset.
    bits/dim = (NLL_nats + noise_correction) / log(2)
    where NLL_nats = model loss and noise_correction adjusts for dequantization noise.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            if noise_sigma > 0:
                x = x + noise_sigma * torch.randn_like(x)
            if hasattr(model, 'patchify'):
                z, _, logdets = model(x)
            else:
                # 1D model
                z, logdets = model(x)
            loss = model.get_loss(z, logdets)
            total_loss += loss.item()
            n_batches += 1
    avg_nll = total_loss / n_batches
    # Convert from nats/dim to bits/dim
    bits_per_dim = avg_nll / np.log(2)
    return bits_per_dim


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def save_loss_curve(losses: list[float], exp_dir: Path, exp_id: str, log_scale: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = np.arange(len(losses))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, linewidth=0.8, color='steelblue', alpha=0.7)

    # Smoothed version
    if len(losses) > 50:
        window = max(len(losses) // 50, 10)
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                linewidth=2, color='darkblue', label=f'Smoothed (w={window})')
        ax.legend()

    ax.set_xlabel('Training Step')
    ax.set_ylabel('NLL Loss')
    ax.set_title(f'{exp_id} — Training Loss')
    ax.grid(True, alpha=0.3)
    if log_scale and min(losses) > 0:
        ax.set_yscale('log')

    # Annotate best loss
    best_step = int(np.argmin(losses))
    best_val = losses[best_step]
    ax.axvline(best_step, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.annotate(f'Best: {best_val:.4f}\n@step {best_step}',
                xy=(best_step, best_val),
                xytext=(best_step + len(losses) * 0.02, best_val),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    plt.tight_layout()
    save_path = exp_dir / 'loss_curve.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_gaussian_mixture_samples(model, device, exp_dir: Path, exp_id: str,
                                   n_samples: int = 2000, n_modes: int = 8,
                                   radius: float = 5.0, sigma: float = 0.5):
    """Scatter plot of generated samples vs true Gaussian mode centers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples, device=device, temp=1.0)  # (B, 1, 2)
        samples = samples.squeeze(1).cpu().numpy()  # (B, 2)

    # True mode centers
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: generated samples
    ax = axes[0]
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.3, color='steelblue', label='Generated')
    ax.scatter(centers[:, 0], centers[:, 1], s=100, color='red', marker='*', zorder=5, label='Mode centers')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title('Generated Samples vs. True Modes')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Right: reference samples from true distribution
    rng = np.random.default_rng(0)
    ref_samples = []
    for c in centers:
        pts = rng.normal(c, sigma, size=(n_samples // n_modes, 2))
        ref_samples.append(pts)
    ref = np.concatenate(ref_samples, axis=0)

    ax = axes[1]
    ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.3, color='orange', label='Reference')
    ax.scatter(centers[:, 0], centers[:, 1], s=100, color='red', marker='*', zorder=5, label='Mode centers')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title('Reference Distribution')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle(f'{exp_id} — 2D 8-mode Gaussian')
    plt.tight_layout()
    save_path = exp_dir / 'samples.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_image_samples(model, device, exp_dir: Path, exp_id: str,
                       n_cols: int = 8, n_rows: int = 8, temp: float = 1.0,
                       img_channels: int = 1):
    """Generate image samples and save as grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = n_cols * n_rows
    model.eval()
    with torch.no_grad():
        samples = model.sample(n, device=device, temp=temp)  # (N, C, H, W)
        samples = samples.cpu().numpy()

    # Denormalize from [-1, 1] to [0, 1]
    samples = np.clip((samples + 1) / 2, 0, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    for i, ax in enumerate(axes.flat):
        if img_channels == 1:
            ax.imshow(samples[i, 0], cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(np.transpose(samples[i], (1, 2, 0)), vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle(f'{exp_id} — Generated Samples (T={temp})', y=1.01)
    plt.tight_layout()
    save_path = exp_dir / 'samples.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Training: Level 0 — 2D Gaussian
# ---------------------------------------------------------------------------

def train_level0(args, cfg, device, exp_dir: Path):
    """Train on 2D 8-mode Gaussian mixture."""
    print(f"\n{'='*60}")
    print(f"LEVEL 0: 2D 8-mode Gaussian Mixture")
    print(f"{'='*60}")

    # Data
    data = make_gaussian_mixture_dataset(
        n_modes=cfg['n_modes'], radius=5.0, sigma=0.5, n_samples=100_000, seed=cfg['seed']
    )
    dataset = GaussianMixtureDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    loader_iter = iter(loader)

    # Model: TarFlow1D with seq_length=1, in_channels=2
    model = TarFlow1D(
        in_channels=2,
        seq_length=1,
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=min(64, cfg['channels']),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    cfg['n_params'] = n_params

    # Optimizer + LR schedule
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])

    # W&B
    wandb.init(
        project='tnafmol',
        name=f"und_001_level0_gaussian",
        group='und_001',
        tags=['understanding', 'und_001', 'DIAGNOSE', 'level0', 'gaussian'],
        config=cfg,
        notes='Level 0: 2D 8-mode Gaussian mixture. Apple TarFlow1D. Sanity check that architecture works on simple data.',
        dir='/tmp',
    )
    assert wandb.run is not None, "W&B init failed"
    # Log test metric to confirm logging
    wandb.log({'setup_check': 1.0}, step=0)
    print(f"W&B run: {wandb.run.url}")

    losses = []
    best_loss = float('inf')
    best_step = 0

    model.train()
    for step in range(cfg['steps']):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x = batch.to(device)  # (B, 1, 2)

        # No noise augmentation for level 0 (clean 2D data)
        optimizer.zero_grad()
        z, logdets = model(x)
        loss = model.get_loss(z, logdets)

        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, exp_dir / 'best.pt')

        if step % 100 == 0 or step == cfg['steps'] - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Step {step:5d} | loss={loss_val:.4f} | lr={lr_now:.2e}")
            wandb.log({
                'train/loss': loss_val,
                'train/lr': lr_now,
                'train/step': step,
            }, step=step)

        # Save periodic checkpoint every 1000 steps
        if (step + 1) % 1000 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, exp_dir / f'checkpoint_step{step+1}.pt')

    # Final evaluation: generate samples and check mode coverage
    model.eval()
    with torch.no_grad():
        samples = model.sample(2000, device=device, temp=1.0)
        samples_np = samples.squeeze(1).cpu().numpy()

    # Check mode coverage: fraction of samples within 2*sigma of any mode center
    angles = np.linspace(0, 2 * np.pi, cfg['n_modes'], endpoint=False)
    centers = np.stack([5.0 * np.cos(angles), 5.0 * np.sin(angles)], axis=1)
    dists = np.min(
        np.sqrt(((samples_np[:, None] - centers[None]) ** 2).sum(-1)), axis=1
    )  # (N,) min distance to any mode
    frac_near_mode = (dists < 2.0).mean()
    print(f"\nMode coverage (within 2.0 of center): {frac_near_mode:.3f}")

    # Visualizations
    loss_path = save_loss_curve(losses, exp_dir, 'und_001_level0')
    samples_path = save_gaussian_mixture_samples(
        model, device, exp_dir, 'und_001_level0',
        n_samples=2000, n_modes=cfg['n_modes']
    )
    print(f"Saved: {loss_path}, {samples_path}")

    # W&B log images and summary
    wandb.log({
        'final/loss_curve': wandb.Image(str(loss_path)),
        'final/samples': wandb.Image(str(samples_path)),
    })
    wandb.run.summary.update({
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': losses[-1],
        'mode_coverage': float(frac_near_mode),
        'n_params': n_params,
    })

    # Log checkpoint artifact
    artifact = wandb.Artifact('und_001_level0_model', type='model')
    artifact.add_file(str(exp_dir / 'best.pt'))
    wandb.log_artifact(artifact)

    wandb.finish()

    results = {
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': losses[-1],
        'mode_coverage': float(frac_near_mode),
        'n_params': n_params,
    }
    return results, losses


# ---------------------------------------------------------------------------
# Training: Level 1 — MNIST
# ---------------------------------------------------------------------------

def train_level1(args, cfg, device, exp_dir: Path):
    """Train on MNIST."""
    print(f"\n{'='*60}")
    print(f"LEVEL 1: MNIST")
    print(f"{'='*60}")

    train_loader, test_loader = get_mnist_loaders(cfg['batch_size'])

    model = TarFlowApple(
        in_channels=1,
        img_size=28,
        patch_size=cfg['patch_size'],
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    cfg['n_params'] = n_params

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])

    wandb.init(
        project='tnafmol',
        name='und_001_level1_mnist',
        group='und_001',
        tags=['understanding', 'und_001', 'DIAGNOSE', 'level1', 'mnist'],
        config=cfg,
        notes='Level 1: MNIST 1x28x28. Apple TarFlowApple. Verify architecture works on image data.',
        dir='/tmp',
    )
    assert wandb.run is not None, "W&B init failed"
    wandb.log({'setup_check': 1.0}, step=0)
    print(f"W&B run: {wandb.run.url}")

    losses = []
    best_loss = float('inf')
    best_step = 0
    train_iter = iter(train_loader)

    model.train()
    for step in range(cfg['steps']):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, _ = next(train_iter)

        x = x.to(device)

        # Gaussian noise augmentation (Apple default: sigma=0.05)
        x = x + cfg['noise_sigma'] * torch.randn_like(x)

        optimizer.zero_grad()
        z, _, logdets = model(x)
        loss = model.get_loss(z, logdets)

        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, exp_dir / 'best.pt')

        if step % 200 == 0 or step == cfg['steps'] - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Step {step:6d} | loss={loss_val:.4f} | lr={lr_now:.2e}")
            wandb.log({
                'train/loss': loss_val,
                'train/lr': lr_now,
                'train/step': step,
            }, step=step)

        if (step + 1) % 2000 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, exp_dir / f'checkpoint_step{step+1}.pt')
            # Keep only last 2 periodic checkpoints
            ckpts = sorted(exp_dir.glob('checkpoint_step*.pt'), key=lambda p: int(p.stem.split('step')[1]))
            for old in ckpts[:-2]:
                old.unlink()

    # Compute bits/dim on test set
    n_dims = 1 * 28 * 28  # MNIST dimensions
    bpd = compute_bits_per_dim(model, test_loader, device, cfg['noise_sigma'], n_dims)
    print(f"\nBits/dim (test): {bpd:.4f}")
    (exp_dir / 'bits_per_dim.txt').write_text(f"Bits per dim (test set): {bpd:.4f}\n")

    # Generate and save visualizations
    model.eval()
    loss_path = save_loss_curve(losses, exp_dir, 'und_001_level1')
    samples_path = save_image_samples(
        model, device, exp_dir, 'und_001_level1',
        n_cols=8, n_rows=8, temp=1.0, img_channels=1
    )
    print(f"Saved: {loss_path}, {samples_path}")

    wandb.log({
        'final/loss_curve': wandb.Image(str(loss_path)),
        'final/samples': wandb.Image(str(samples_path)),
        'final/bits_per_dim': bpd,
    })
    wandb.run.summary.update({
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': losses[-1],
        'bits_per_dim': bpd,
        'n_params': n_params,
    })

    artifact = wandb.Artifact('und_001_level1_model', type='model')
    artifact.add_file(str(exp_dir / 'best.pt'))
    wandb.log_artifact(artifact)

    wandb.finish()

    results = {
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': losses[-1],
        'bits_per_dim': bpd,
        'n_params': n_params,
    }
    return results, losses


# ---------------------------------------------------------------------------
# Training: Level 2 — CIFAR-10
# ---------------------------------------------------------------------------

def train_level2(args, cfg, device, exp_dir: Path):
    """Train on CIFAR-10."""
    print(f"\n{'='*60}")
    print(f"LEVEL 2: CIFAR-10")
    print(f"{'='*60}")

    train_loader, test_loader = get_cifar10_loaders(cfg['batch_size'])

    model = TarFlowApple(
        in_channels=3,
        img_size=32,
        patch_size=cfg['patch_size'],
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    cfg['n_params'] = n_params

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])

    wandb.init(
        project='tnafmol',
        name='und_001_level2_cifar10',
        group='und_001',
        tags=['understanding', 'und_001', 'DIAGNOSE', 'level2', 'cifar10'],
        config=cfg,
        notes='Level 2: CIFAR-10 3x32x32. Apple TarFlowApple. Verify architecture works on color images.',
        dir='/tmp',
    )
    assert wandb.run is not None, "W&B init failed"
    wandb.log({'setup_check': 1.0}, step=0)
    print(f"W&B run: {wandb.run.url}")

    losses = []
    best_loss = float('inf')
    best_step = 0
    train_iter = iter(train_loader)

    model.train()
    for step in range(cfg['steps']):
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, _ = next(train_iter)

        x = x.to(device)
        x = x + cfg['noise_sigma'] * torch.randn_like(x)

        optimizer.zero_grad()
        z, _, logdets = model(x)
        loss = model.get_loss(z, logdets)

        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, exp_dir / 'best.pt')

        if step % 500 == 0 or step == cfg['steps'] - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Step {step:6d} | loss={loss_val:.4f} | lr={lr_now:.2e}")
            wandb.log({
                'train/loss': loss_val,
                'train/lr': lr_now,
                'train/step': step,
            }, step=step)

        if (step + 1) % 5000 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, exp_dir / f'checkpoint_step{step+1}.pt')
            ckpts = sorted(exp_dir.glob('checkpoint_step*.pt'), key=lambda p: int(p.stem.split('step')[1]))
            for old in ckpts[:-2]:
                old.unlink()

    # Compute bits/dim on test set
    n_dims = 3 * 32 * 32
    bpd = compute_bits_per_dim(model, test_loader, device, cfg['noise_sigma'], n_dims)
    print(f"\nBits/dim (test): {bpd:.4f}")
    (exp_dir / 'bits_per_dim.txt').write_text(f"Bits per dim (test set): {bpd:.4f}\n")

    model.eval()
    loss_path = save_loss_curve(losses, exp_dir, 'und_001_level2')
    samples_path = save_image_samples(
        model, device, exp_dir, 'und_001_level2',
        n_cols=8, n_rows=8, temp=1.0, img_channels=3
    )
    print(f"Saved: {loss_path}, {samples_path}")

    wandb.log({
        'final/loss_curve': wandb.Image(str(loss_path)),
        'final/samples': wandb.Image(str(samples_path)),
        'final/bits_per_dim': bpd,
    })
    wandb.run.summary.update({
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': losses[-1],
        'bits_per_dim': bpd,
        'n_params': n_params,
    })

    artifact = wandb.Artifact('und_001_level2_model', type='model')
    artifact.add_file(str(exp_dir / 'best.pt'))
    wandb.log_artifact(artifact)

    wandb.finish()

    results = {
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': losses[-1],
        'bits_per_dim': bpd,
        'n_params': n_params,
    }
    return results, losses


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='TarFlow Diagnostic Ladder Training')

    # Required
    parser.add_argument('--level', type=int, required=True,
                        choices=[0, 1, 2, 3],
                        help='Benchmark level: 0=2d_gaussian, 1=mnist, 2=cifar10, 3=molecular')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU index (CUDA_VISIBLE_DEVICES index)')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Output directory for results')

    # Training
    parser.add_argument('--steps', type=int, default=None,
                        help='Training steps (default: level-specific)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: level-specific)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: level-specific)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_sigma', type=float, default=None,
                        help='Gaussian noise augmentation sigma (default: level-specific)')

    # Model architecture
    parser.add_argument('--channels', type=int, default=None)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--layers_per_block', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=4)

    # Level 0 specific
    parser.add_argument('--n_modes', type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed before anything else
    set_seed(args.seed)

    # Device setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")

    # Create output directory
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Git hash for reproducibility
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).parent.parent)
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    # Build level-specific config with defaults
    if args.level == 0:
        # 2D Gaussian
        cfg = {
            'level': 0,
            'name': '2d_gaussian',
            'exp_id': 'und_001',
            'command': 'DIAGNOSE',
            'seed': args.seed,
            'device': str(device),
            'git_hash': git_hash,
            # Data
            'n_modes': args.n_modes,
            # Model
            'in_channels': 2,
            'seq_length': 1,
            'channels': args.channels or 64,
            'num_blocks': args.num_blocks or 4,
            'layers_per_block': args.layers_per_block or 2,
            # Training
            'steps': args.steps or 5000,
            'batch_size': args.batch_size or 256,
            'lr': args.lr or 1e-3,
            'noise_sigma': args.noise_sigma if args.noise_sigma is not None else 0.0,
            'grad_clip_norm': 1.0,
            'lr_schedule': 'cosine',
        }
    elif args.level == 1:
        # MNIST
        cfg = {
            'level': 1,
            'name': 'mnist',
            'exp_id': 'und_001',
            'command': 'DIAGNOSE',
            'seed': args.seed,
            'device': str(device),
            'git_hash': git_hash,
            # Data
            'in_channels': 1,
            'img_size': 28,
            'patch_size': args.patch_size or 4,
            # Model
            'channels': args.channels or 256,
            'num_blocks': args.num_blocks or 4,
            'layers_per_block': args.layers_per_block or 4,
            # Training
            'steps': args.steps or 20000,
            'batch_size': args.batch_size or 128,
            'lr': args.lr or 5e-4,
            'noise_sigma': args.noise_sigma if args.noise_sigma is not None else 0.05,
            'grad_clip_norm': 1.0,
            'lr_schedule': 'cosine',
        }
    elif args.level == 2:
        # CIFAR-10
        cfg = {
            'level': 2,
            'name': 'cifar10',
            'exp_id': 'und_001',
            'command': 'DIAGNOSE',
            'seed': args.seed,
            'device': str(device),
            'git_hash': git_hash,
            # Data
            'in_channels': 3,
            'img_size': 32,
            'patch_size': args.patch_size or 4,
            # Model
            'channels': args.channels or 768,
            'num_blocks': args.num_blocks or 8,
            'layers_per_block': args.layers_per_block or 4,
            # Training
            'steps': args.steps or 50000,
            'batch_size': args.batch_size or 64,
            'lr': args.lr or 3e-4,
            'noise_sigma': args.noise_sigma if args.noise_sigma is not None else 0.05,
            'grad_clip_norm': 1.0,
            'lr_schedule': 'cosine',
        }
    elif args.level == 3:
        raise NotImplementedError("Level 3 (molecular adaptation) is implemented in Phase 3.")
    else:
        raise ValueError(f"Unknown level: {args.level}")

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\nConfig:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Pre-run git check: working tree must be clean on src/
    try:
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain', 'src/'],
            cwd=str(Path(__file__).parent.parent)
        ).decode().strip()
        if dirty:
            print(f"WARNING: Uncommitted changes in src/:\n{dirty}")
    except Exception:
        pass

    # Run training
    start_time = time.time()

    if args.level == 0:
        results, losses = train_level0(args, cfg, device, exp_dir)
    elif args.level == 1:
        results, losses = train_level1(args, cfg, device, exp_dir)
    elif args.level == 2:
        results, losses = train_level2(args, cfg, device, exp_dir)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Results: {results}")

    # Save final results JSON
    results['elapsed_min'] = elapsed / 60
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save losses array
    np.save(exp_dir / 'losses.npy', np.array(losses))

    print(f"\nAll outputs saved to: {exp_dir}")
    return results


if __name__ == '__main__':
    main()
