"""
train_apple.py — Training script for TarFlow1DMol (Apple architecture).

Uses TarFlow1DMol from src/train_phase3.py directly — the proven architecture
from und_001 that achieves 96-98% VF per-molecule on ethanol.

This script replaces the incremental-patching approach of hyp_006 through hyp_009
with the proven architecture used end-to-end for multi-molecule MD17 training.

Usage:
    python src/train_apple.py --config config.json [--device cuda:7]

Key differences from src/train.py (model.py's TarFlow):
    - Model: TarFlow1DMol from train_phase3.py (not model.py TarFlow)
    - Atom types: integer indices passed directly (not pre-embedded)
    - Normalization: external (divide by global_std before forward, multiply after sample)
    - Loss: model.get_loss(z, logdets, padding_mask) handles per-real-atom normalization
    - Sampling: model.sample(n, device, atom_types, padding_mask, temp)
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import (
    MD17Dataset, MultiMoleculeDataset, MOLECULES, MAX_ATOMS,
    compute_global_std, PAD_TOKEN_IDX,
)
from src.metrics import valid_fraction, pairwise_distance_divergence, min_pairwise_distance
from src.train_phase3 import TarFlow1DMol


# =============================================================================
# Default config
# =============================================================================

DEFAULT_CONFIG = {
    # Identity
    "exp_id": "hyp_010",
    "angle": "sanity",
    "stage": "val",
    "command": "OPTIMIZE",

    # Reproducibility
    "seed": 42,
    "device": "cuda:7",

    # Model (TarFlow1DMol)
    "seq_length": 9,       # 9 for ethanol-only (no padding), 21 for multi-mol
    "channels": 256,
    "num_blocks": 4,
    "layers_per_block": 2,
    "head_dim": 64,
    "expansion": 4,
    "use_atom_type_cond": True,
    "atom_type_emb_dim": 16,
    "num_atom_types": 4,
    "use_padding_mask": False,   # True for T=21 runs (multi-mol or padded ethanol)
    "use_shared_scale": False,   # Always False — per-dim scale is proven better
    "use_clamp": False,
    "log_det_reg_weight": 0.0,   # 5.0 proven critical in hyp_007 for multi-mol

    # Training
    "n_steps": 5000,
    "batch_size": 256,
    "lr": 5e-4,
    "lr_schedule": "cosine",
    "warmup_steps": 500,
    "grad_clip_norm": 1.0,
    "val_interval": 200,
    "eval_n_samples": 500,
    "weight_decay": 1e-5,

    # Data
    "data_root": "data/",
    "molecules": ["ethanol"],   # None = all 8
    "max_atoms": 9,             # 9 for ethanol-only, 21 for multi-mol
    "noise_sigma": 0.05,        # Gaussian noise augmentation on real atoms
    "augment_train": True,      # SO(3) rotation + CoM noise

    # W&B
    "wandb_project": "tnafmol",
    "wandb_group": "hyp_010",
    "wandb_tags": ["hypothesis", "hyp_010", "OPTIMIZE"],
    "wandb_notes": "",

    # Output
    "output_dir": "experiments/hypothesis/hyp_010_tarflow_apple_multimol",
}


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root
        ).decode().strip()
        return result[:8]
    except Exception:
        return "unknown"


def get_cosine_schedule(optimizer, n_steps: int, warmup_steps: int):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Per-molecule evaluation
# =============================================================================

def evaluate_molecule(
    model: TarFlow1DMol,
    data_dir: str,
    n_samples: int,
    device: torch.device,
    global_std: Optional[float] = None,
    max_atoms: Optional[int] = None,
    use_padding_mask: bool = False,
    temperature: float = 1.0,
) -> Dict:
    """Evaluate the model on a single molecule.

    Args:
        model: trained TarFlow1DMol model
        data_dir: path to processed molecule dataset (e.g. data/md17_ethanol_v1/)
        n_samples: number of samples to generate
        device: compute device
        global_std: if provided, multiply samples by this to denormalize
        max_atoms: effective sequence length (must match model.seq_length)
        use_padding_mask: if True, use padding mask during sampling
        temperature: sampling temperature

    Returns:
        dict with valid_fraction, pairwise_dist_divergence, min_dist stats
    """
    data = np.load(os.path.join(data_dir, "dataset.npz"))

    # Get molecule info
    raw_atom_types = data["atom_types"]   # (21,) int64
    mask_np = data["mask"]                # (21,) float32
    n_real = int(mask_np.sum())

    # Resolve effective max_atoms for this evaluation
    effective_max = max_atoms if max_atoms is not None else int(mask_np.sum())
    assert effective_max >= n_real, (
        f"max_atoms={effective_max} < n_real={n_real} for {os.path.basename(data_dir)}"
    )

    # Truncate to effective_max
    atom_types_eval = raw_atom_types[:effective_max]
    mask_np_eval = mask_np[:effective_max]

    # Build tensors for sampling
    # atom_types: (1, T) — single molecule, broadcast to (n_samples, T) in model.sample
    atom_types_t = torch.from_numpy(atom_types_eval).long().unsqueeze(0).to(device)  # (1, T)
    atom_types_t = atom_types_t.expand(n_samples, -1)  # (n_samples, T)

    padding_mask_t = None
    if use_padding_mask:
        mask_float = torch.from_numpy(mask_np_eval).float().unsqueeze(0).to(device)  # (1, T)
        padding_mask_t = mask_float.expand(n_samples, -1)  # (n_samples, T)

    # Reference positions (test set, raw Angstroms)
    test_idx = data["test_idx"]
    ref_positions = data["positions"][test_idx][:, :effective_max, :]  # (N_test, max_atoms, 3)

    # Generate samples
    model.eval()
    with torch.no_grad():
        samples = model.sample(
            n=n_samples,
            device=device,
            atom_types=atom_types_t,
            padding_mask=padding_mask_t,
            temp=temperature,
        )  # (n_samples, T, 3) — in normalized space

    samples_np = samples.cpu().numpy()  # (n_samples, T, 3)

    # Denormalize: model generates in normalized space, metrics need raw Angstroms
    if global_std is not None and global_std > 0:
        samples_np = samples_np * global_std

    # Metrics (over real atoms only — mask_np_eval has 1 for real atoms)
    vf, _ = valid_fraction(samples_np, mask_np_eval)
    pw_div = pairwise_distance_divergence(samples_np, ref_positions, mask_np_eval)
    min_dists = min_pairwise_distance(samples_np, mask_np_eval)

    return {
        "valid_fraction": float(vf),
        "pairwise_dist_divergence": float(pw_div),
        "min_dist_mean": float(min_dists.mean()),
        "min_dist_below_08": float((min_dists < 0.8).mean()),
        "n_samples": n_samples,
    }


# =============================================================================
# Visualization
# =============================================================================

def save_loss_curve(
    steps: List[int],
    train_losses: List[float],
    val_losses: List[float],
    val_steps: List[int],
    logdets: List[float],
    stage_dir: str,
    exp_id: str,
):
    """Save annotated training loss curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(steps, train_losses, label="train loss", alpha=0.7, linewidth=0.8)
    if val_losses:
        ax.plot(val_steps, val_losses, label="val loss", linewidth=1.5)
    if val_losses:
        best_idx = int(np.argmin(val_losses))
        best_step = val_steps[best_idx]
        best_val = val_losses[best_idx]
        ax.axvline(best_step, color="red", linestyle="--", alpha=0.6, label=f"best val: {best_val:.4f} @ {best_step}")
    ax.set_xlabel("Step")
    ax.set_ylabel("NLL Loss")
    ax.set_title(f"{exp_id} — Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(steps, logdets, label="logdets mean", color="green", alpha=0.7, linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Log-Determinant")
    ax2.set_title(f"{exp_id} — Log-Determinant Track")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(stage_dir, f"{exp_id}_loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def save_vf_bar_chart(
    mol_results: Dict,
    stage_dir: str,
    exp_id: str,
):
    """Save per-molecule valid fraction bar chart."""
    molecules = list(mol_results.keys())
    vf_vals = [mol_results[mol]["valid_fraction"] for mol in molecules]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if v >= 0.9 else ("orange" if v >= 0.5 else "red") for v in vf_vals]
    bars = ax.bar(molecules, vf_vals, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0.9, color="green", linestyle="--", label="90% target")
    ax.axhline(0.5, color="orange", linestyle="--", label="50% threshold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valid Fraction")
    ax.set_title(f"{exp_id} — Per-Molecule Valid Fraction")
    ax.legend()
    for bar, v in zip(bars, vf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(stage_dir, f"{exp_id}_vf_bar.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# =============================================================================
# Main training function
# =============================================================================

def train(cfg: dict):
    """Main training function for TarFlow1DMol."""
    # Setup
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    # Git hash for reproducibility
    cfg["git_hash"] = get_git_hash()

    # Resolve paths
    data_root = os.path.join(project_root, cfg["data_root"])
    output_dir = os.path.join(project_root, cfg["output_dir"])

    # Determine output subdirectory
    angle = cfg.get("angle", "sanity")
    stage = cfg.get("stage", "val")
    stage_dir = os.path.join(output_dir, "angles", angle, stage)
    os.makedirs(stage_dir, exist_ok=True)
    raw_dir = os.path.join(stage_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Molecules list
    molecules = cfg.get("molecules") or list(MOLECULES.keys())

    # Global std normalization
    global_std = None
    if cfg.get("normalize_to_unit_var", True):
        global_std = compute_global_std(data_root, molecules, split="train")
        cfg["global_std"] = global_std
        print(f"Global std for normalization: {global_std:.4f} Angstroms")

    # Effective seq_length / max_atoms
    seq_length = cfg.get("seq_length", 9)
    max_atoms = cfg.get("max_atoms", seq_length)
    # For TarFlow1DMol, seq_length IS max_atoms
    assert seq_length == max_atoms, (
        f"seq_length={seq_length} must equal max_atoms={max_atoms} for TarFlow1DMol"
    )
    cfg["max_atoms"] = max_atoms
    cfg["seq_length"] = seq_length
    print(f"seq_length/max_atoms: {seq_length}")

    # W&B initialization
    if wandb.run is None:
        exp_id = cfg.get("exp_id", "hyp_010")
        run_name = f"{exp_id}_{angle}_{stage}"
        if cfg.get("run_name_suffix"):
            run_name = run_name + "_" + cfg["run_name_suffix"]

        tags = list(cfg.get("wandb_tags", []))
        for t in [angle, stage]:
            if t not in tags:
                tags.append(t)

        wandb.init(
            project=cfg["wandb_project"],
            name=run_name,
            group=cfg["wandb_group"],
            tags=tags,
            notes=cfg.get("wandb_notes", ""),
            config=cfg,
            entity="kaityrusnelson1",
        )
    else:
        wandb.config.update(cfg, allow_val_change=True)

    assert wandb.run is not None, "W&B init failed"
    print(f"W&B run: {wandb.run.url}")

    # Log a test metric to confirm logging is active
    wandb.log({"init_check": 1.0}, step=0)

    # Build TarFlow1DMol model
    model = TarFlow1DMol(
        in_channels=3,
        seq_length=seq_length,
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
    ).to(device)

    n_params = count_parameters(model)
    cfg["n_params"] = n_params
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # Dataset
    noise_sigma = cfg.get("noise_sigma", 0.05)
    augment_train = cfg.get("augment_train", True)

    # Note: global_std normalization is applied inside MD17Dataset when global_std is provided.
    # The dataset divides positions by global_std in __init__.
    # So positions from the dataloader are already in normalized space.
    train_ds = MultiMoleculeDataset(
        data_root,
        split="train",
        molecules=molecules,
        augment=augment_train,
        global_std=global_std,
        permute=False,         # No permutation for Apple architecture (sequence-order matters)
        pad_token_idx=0,       # Standard H=0 for real atom types; 0 for padding is fine with TarFlow1DMol
        noise_sigma=noise_sigma,
        max_atoms=max_atoms,
    )
    val_ds = MultiMoleculeDataset(
        data_root,
        split="val",
        molecules=molecules,
        augment=False,
        global_std=global_std,
        permute=False,
        pad_token_idx=0,
        noise_sigma=0.0,       # No noise for validation
        max_atoms=max_atoms,
    )

    print(f"Train size: {len(train_ds):,}, Val size: {len(val_ds):,}")
    print(f"Augment train: {augment_train}, Noise sigma: {noise_sigma}, Global std: {global_std}")
    print(f"use_padding_mask: {cfg.get('use_padding_mask', False)}, use_atom_type_cond: {cfg.get('use_atom_type_cond', True)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    # LR schedule: cosine with warmup
    scheduler = get_cosine_schedule(
        optimizer,
        cfg["n_steps"],
        cfg.get("warmup_steps", 500),
    )

    # Training state
    best_val_loss = float("inf")
    best_step = 0
    checkpoint_path = os.path.join(stage_dir, "best.pt")
    final_checkpoint_path = os.path.join(stage_dir, "final.pt")

    # Tracking for plotting
    train_steps_log = []
    train_losses_log = []
    logdets_log = []
    val_steps_log = []
    val_losses_log = []

    use_padding_mask = cfg.get("use_padding_mask", False)

    step = 0
    train_iter = iter(train_loader)
    model.train()

    print(f"\nStarting training for {cfg['n_steps']} steps...")
    t_start = time.time()

    while step < cfg["n_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Positions are already normalized (dataset divides by global_std in __init__)
        positions = batch["positions"].to(device)    # (B, T, 3) — normalized
        atom_types = batch["atom_types"].to(device)  # (B, T) — integer indices
        atom_mask = batch["mask"].to(device)         # (B, T) — float 1=real, 0=pad

        # Padding mask for TarFlow1DMol: float (B, T), or None if not using padding
        padding_mask = atom_mask if use_padding_mask else None

        optimizer.zero_grad()

        # Forward pass: TarFlow1DMol takes (x, atom_types, padding_mask)
        z, logdets = model(positions, atom_types=atom_types, padding_mask=padding_mask)

        # Loss: NLL per real atom DOF
        loss, info = model.get_loss(z, logdets, padding_mask=padding_mask)

        # Finite loss check
        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

        optimizer.step()
        scheduler.step()

        step += 1

        # Log to stdout and W&B
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg["lr"]
            print(
                f"Step {step}/{cfg['n_steps']} | loss={loss.item():.4f} | "
                f"nll={info['nll']:.4f} | reg={info['reg']:.4f} | "
                f"logdets={info['logdets_mean']:.3f} | "
                f"grad={grad_norm:.3f} | lr={lr:.2e} | t={elapsed:.1f}s"
            )
            wandb.log({
                "train/loss": loss.item(),
                "train/nll": info["nll"],
                "train/reg": info["reg"],
                "train/logdets_mean": info["logdets_mean"],
                "train/grad_norm": grad_norm,
                "train/lr": lr,
            }, step=step)

            train_steps_log.append(step)
            train_losses_log.append(loss.item())
            logdets_log.append(info["logdets_mean"])

        # Validation
        if step % cfg["val_interval"] == 0 or step == cfg["n_steps"]:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vpos = vbatch["positions"].to(device)
                    vatypes = vbatch["atom_types"].to(device)
                    vmask = vbatch["mask"].to(device)
                    vpad_mask = vmask if use_padding_mask else None
                    vz, vlogdets = model(vpos, atom_types=vatypes, padding_mask=vpad_mask)
                    vloss, _ = model.get_loss(vz, vlogdets, padding_mask=vpad_mask)
                    val_losses.append(vloss.item())

            val_loss = float(np.mean(val_losses))
            print(f"  Val loss: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=step)
            val_steps_log.append(step)
            val_losses_log.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
                    "global_std": global_std,
                    "git_hash": cfg["git_hash"],
                }, checkpoint_path)
                print(f"  Saved best checkpoint at step {step} (val_loss={val_loss:.4f})")

            model.train()

    # Save final checkpoint
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "global_std": global_std,
        "git_hash": cfg["git_hash"],
    }, final_checkpoint_path)

    # Load best checkpoint for final evaluation
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\nLoaded best checkpoint from step {best_step} (val_loss={best_val_loss:.4f})")

    # Per-molecule evaluation
    print("\nEvaluating per molecule...")
    mol_results = {}
    for mol in molecules:
        mol_data_dir = os.path.join(data_root, f"md17_{mol}_v1")
        if not os.path.exists(os.path.join(mol_data_dir, "dataset.npz")):
            print(f"  Skipping {mol} (dataset not found)")
            continue

        print(f"  Evaluating {mol}...")
        with torch.no_grad():
            result = evaluate_molecule(
                model,
                mol_data_dir,
                n_samples=cfg.get("eval_n_samples", 500),
                device=device,
                global_std=global_std,
                max_atoms=max_atoms,
                use_padding_mask=use_padding_mask,
                temperature=1.0,
            )
        mol_results[mol] = result

        print(
            f"    {mol}: vf={result['valid_fraction']:.3f}, "
            f"pw_div={result['pairwise_dist_divergence']:.4f}, "
            f"min_dist_mean={result['min_dist_mean']:.3f}"
        )

        wandb.log({
            f"eval/{mol}/valid_fraction": result["valid_fraction"],
            f"eval/{mol}/pairwise_dist_divergence": result["pairwise_dist_divergence"],
            f"eval/{mol}/min_dist_mean": result["min_dist_mean"],
            f"eval/{mol}/min_dist_below_08": result["min_dist_below_08"],
        }, step=step)

    # Summary metrics
    valid_fracs = [r["valid_fraction"] for r in mol_results.values()]
    mean_valid = float(np.mean(valid_fracs)) if valid_fracs else 0.0
    n_valid_majority = sum(1 for vf in valid_fracs if vf > 0.5)
    print(f"\nSummary:")
    print(f"  Mean valid fraction: {mean_valid:.3f}")
    print(f"  Molecules with VF > 0.5: {n_valid_majority}/{len(molecules)}")

    # Save raw results
    results_path = os.path.join(raw_dir, "mol_results.pt")
    torch.save(mol_results, results_path)

    np.save(os.path.join(raw_dir, "train_losses.npy"), np.array(train_losses_log))
    np.save(os.path.join(raw_dir, "logdets.npy"), np.array(logdets_log))
    np.save(os.path.join(raw_dir, "val_losses.npy"), np.array(val_losses_log))

    # Save config
    config_path = os.path.join(stage_dir, "config.json")
    with open(config_path, "w") as f:
        cfg_save = {k: v for k, v in cfg.items()}
        json.dump(cfg_save, f, indent=2)

    # Visualizations
    loss_plot = save_loss_curve(
        train_steps_log, train_losses_log, val_losses_log, val_steps_log, logdets_log,
        stage_dir, cfg.get("exp_id", "hyp_010"),
    )
    wandb.log({"plot/loss_curve": wandb.Image(loss_plot)}, step=step)

    if len(mol_results) >= 1:
        vf_plot = save_vf_bar_chart(mol_results, stage_dir, cfg.get("exp_id", "hyp_010"))
        wandb.log({"plot/vf_bar": wandb.Image(vf_plot)}, step=step)

    # W&B summary
    wandb.run.summary.update({
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "mean_valid_fraction": mean_valid,
        "n_molecules_valid_majority": n_valid_majority,
        "mol_results": {m: r["valid_fraction"] for m, r in mol_results.items()},
        "checkpoint_path": checkpoint_path,
        "global_std": global_std,
    })

    # Log model artifact
    exp_id = cfg.get("exp_id", "hyp_010")
    ckpt_artifact = wandb.Artifact(f"{exp_id}_{angle}_{stage}_model", type="model")
    ckpt_artifact.add_file(checkpoint_path)
    wandb.log_artifact(ckpt_artifact)

    wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} at step {best_step}")
    return mol_results


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TarFlow1DMol (Apple architecture)")
    parser.add_argument("--config", type=str, default=None, help="JSON config file path (relative to project root)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--angle", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--molecules", type=str, default=None, help="Comma-separated molecule list, or 'all'")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--use-padding-mask", action="store_true", default=None)
    parser.add_argument("--log-det-reg-weight", type=float, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-name-suffix", type=str, default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)

    if args.config:
        config_path = os.path.join(project_root, args.config)
        with open(config_path) as f:
            cfg.update(json.load(f))

    # CLI overrides
    if args.device:
        cfg["device"] = args.device
    if args.n_steps is not None:
        cfg["n_steps"] = args.n_steps
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.angle is not None:
        cfg["angle"] = args.angle
    if args.stage is not None:
        cfg["stage"] = args.stage
    if args.molecules is not None:
        if args.molecules == "all":
            cfg["molecules"] = None
        else:
            cfg["molecules"] = [m.strip() for m in args.molecules.split(",")]
    if args.seq_length is not None:
        cfg["seq_length"] = args.seq_length
    if args.max_atoms is not None:
        cfg["max_atoms"] = args.max_atoms
    if args.use_padding_mask:
        cfg["use_padding_mask"] = True
    if args.log_det_reg_weight is not None:
        cfg["log_det_reg_weight"] = args.log_det_reg_weight
    if args.channels is not None:
        cfg["channels"] = args.channels
    if args.num_blocks is not None:
        cfg["num_blocks"] = args.num_blocks
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.run_name_suffix is not None:
        cfg["run_name_suffix"] = args.run_name_suffix

    train(cfg)


if __name__ == "__main__":
    main()
