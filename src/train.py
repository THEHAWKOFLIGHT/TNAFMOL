"""
TNAFMOL Training — TarFlow training loop with MLE loss and W&B logging.

Usage:
    python3.10 src/train.py --config config.json [--device cuda:0]

The script:
1. Loads the multi-molecule dataset
2. Trains TarFlow with NLL loss
3. Logs to W&B (project: tnafmol)
4. Evaluates per-molecule metrics at the end
5. Saves checkpoint and all output arrays
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
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import MD17Dataset, MultiMoleculeDataset, MOLECULES, MAX_ATOMS
from src.metrics import valid_fraction, energy_wasserstein, pairwise_distance_divergence, min_pairwise_distance
from src.model import TarFlow


# =============================================================================
# Default config
# =============================================================================

DEFAULT_CONFIG = {
    # Identity
    "exp_id": "hyp_002",
    "angle": "diagnostic",
    "stage": "diag",
    "command": "OPTIMIZE",

    # Reproducibility
    "seed": 42,
    "device": "cuda:8",

    # Model
    "n_blocks": 8,
    "d_model": 128,
    "n_heads": 4,
    "ffn_mult": 4,
    "atom_type_emb_dim": 16,
    "dropout": 0.1,
    "log_scale_max": 0.5,  # limits |log_scale| per block to tanh*log_scale_max
    "shift_only": True,   # if True: shift-only flow (no scale), prevents collapse

    # Training
    "n_steps": 5000,
    "batch_size": 128,
    "lr": 3e-4,
    "lr_schedule": "cosine",
    "warmup_steps": 500,
    "grad_clip_norm": 1.0,
    "val_interval": 200,
    "eval_n_samples": 500,

    # Data
    "data_root": "data/",
    "molecules": None,  # None = all 8

    # W&B
    "wandb_project": "tnafmol",
    "wandb_group": "hyp_002",
    "wandb_tags": ["hypothesis", "hyp_002", "OPTIMIZE"],
    "wandb_notes": "Diagnostic run to understand baseline TarFlow behavior",

    # Output
    "output_dir": "experiments/hypothesis/hyp_002_tarflow",
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


# =============================================================================
# Dataset fingerprint
# =============================================================================

def compute_dataset_fingerprint(data_root: str, mol: str) -> str:
    """Compute a short fingerprint of the dataset for reproducibility logging."""
    try:
        dataset_path = os.path.join(project_root, data_root, f"md17_{mol}_v1", "dataset.npz")
        data = np.load(dataset_path)
        fp = hashlib.md5(data["positions"].tobytes()).hexdigest()[:8]
        return fp
    except Exception:
        return "unknown"


# =============================================================================
# Per-molecule evaluation
# =============================================================================

def evaluate_molecule(
    model: TarFlow,
    data_dir: str,
    n_samples: int,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict:
    """Evaluate the model on a single molecule.

    Args:
        model: trained TarFlow model
        data_dir: path to processed molecule dataset
        n_samples: number of samples to generate
        device: compute device

    Returns:
        dict with valid_fraction, energy_wasserstein, pairwise_distance_divergence
    """
    import torch
    from torch.utils.data import DataLoader as DL

    data = np.load(os.path.join(data_dir, "dataset.npz"))
    ref_stats = torch.load(os.path.join(data_dir, "ref_stats.pt"), weights_only=False)

    # Get molecule info
    atom_types = torch.from_numpy(data["atom_types"]).to(device)
    mask = torch.from_numpy(data["mask"]).to(device)
    test_idx = data["test_idx"]
    ref_positions = data["positions"][test_idx]  # (N_test, 21, 3)
    ref_energies = data["energies"][test_idx]

    # Generate samples
    model.eval()
    samples = model.sample(atom_types, mask, n_samples=n_samples, temperature=temperature)
    samples_np = samples.cpu().numpy()  # (n_samples, 21, 3)

    # Metrics
    mask_np = mask.cpu().numpy()
    vf, _ = valid_fraction(samples_np, mask_np)

    # For energy, we need to approximate using ref stats
    # We don't have an energy function, so we use pairwise distance divergence as proxy
    pw_div = pairwise_distance_divergence(samples_np, ref_positions, mask_np)

    # Min pairwise distances (for validity diagnosis)
    min_dists = min_pairwise_distance(samples_np, mask_np)

    return {
        "valid_fraction": vf,
        "pairwise_dist_divergence": pw_div,
        "min_dist_mean": float(min_dists.mean()),
        "min_dist_below_08": float((min_dists < 0.8).mean()),
        "n_samples": n_samples,
    }


# =============================================================================
# Main training function
# =============================================================================

def train(cfg: dict):
    """Main training function."""
    # Setup
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    # Git hash
    cfg["git_hash"] = get_git_hash()

    # Resolve paths
    data_root = os.path.join(project_root, cfg["data_root"])
    output_dir = os.path.join(project_root, cfg["output_dir"])

    # Determine output subdirectory based on angle/stage
    stage_dir = os.path.join(output_dir, "angles", cfg["angle"], cfg["stage"])
    os.makedirs(stage_dir, exist_ok=True)
    raw_dir = os.path.join(stage_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # W&B init
    run_name = f"hyp_002_{cfg['angle']}_{cfg['stage']}"
    if cfg.get("run_name_suffix"):
        run_name = run_name + "_" + cfg["run_name_suffix"]

    tags = list(cfg.get("wandb_tags", []))
    if cfg["angle"] not in tags:
        tags.append(cfg["angle"])
    if cfg["stage"] not in tags:
        tags.append(cfg["stage"])

    run = wandb.init(
        project=cfg["wandb_project"],
        name=run_name,
        group=cfg["wandb_group"],
        tags=tags,
        notes=cfg.get("wandb_notes", ""),
        config=cfg,
    )

    assert wandb.run is not None, "W&B init failed"
    print(f"W&B run: {wandb.run.url}")

    # Log test metric to confirm logging
    wandb.log({"init_check": 1.0}, step=0)

    # Model
    model = TarFlow(
        n_blocks=cfg["n_blocks"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ffn_mult=cfg["ffn_mult"],
        atom_type_emb_dim=cfg["atom_type_emb_dim"],
        dropout=cfg["dropout"],
        max_atoms=MAX_ATOMS,
        log_scale_max=cfg.get("log_scale_max", 0.5),
        shift_only=cfg.get("shift_only", False),
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # Dataset
    molecules = cfg.get("molecules") or list(MOLECULES.keys())
    train_ds = MultiMoleculeDataset(data_root, split="train", molecules=molecules)
    val_ds = MultiMoleculeDataset(data_root, split="val", molecules=molecules)

    print(f"Train size: {len(train_ds):,}, Val size: {len(val_ds):,}")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = get_cosine_schedule(optimizer, cfg["n_steps"], cfg["warmup_steps"])

    # Training loop
    best_val_loss = float("inf")
    best_step = 0
    checkpoint_path = os.path.join(stage_dir, "best.pt")
    final_checkpoint_path = os.path.join(stage_dir, "final.pt")

    step = 0
    train_iter = iter(train_loader)
    model.train()

    print(f"\nStarting training for {cfg['n_steps']} steps...")
    t_start = time.time()

    while step < cfg["n_steps"]:
        # Get batch (cycling through loader)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        positions = batch["positions"].to(device)  # (B, 21, 3)
        atom_types = batch["atom_types"].to(device)  # (B, 21)
        atom_mask = batch["mask"].to(device)  # (B, 21)

        # If atom_types/mask are per-dataset (not per-sample), they come as (B, 21)
        # from the multi-molecule dataset. This is correct.

        # Forward + loss
        optimizer.zero_grad()
        loss, info = model.nll_loss(positions, atom_types, atom_mask)

        # Backward
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

        optimizer.step()
        scheduler.step()

        step += 1

        # Log
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step}/{cfg['n_steps']} | loss={loss.item():.4f} | "
                f"log_det={info['total_log_det']:.2f} | "
                f"grad_norm={grad_norm:.3f} | lr={lr:.2e} | t={elapsed:.1f}s"
            )
            wandb.log({
                "train/loss": loss.item(),
                "train/nll": info["nll"],
                "train/log_det": info["total_log_det"],
                "train/log_pz": info["log_pz"],
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/n_real_mean": info["n_real_mean"],
            }, step=step)

        # Validation
        if step % cfg["val_interval"] == 0 or step == cfg["n_steps"]:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vpos = vbatch["positions"].to(device)
                    vatypes = vbatch["atom_types"].to(device)
                    vmask = vbatch["mask"].to(device)
                    vloss, _ = model.nll_loss(vpos, vatypes, vmask)
                    val_losses.append(vloss.item())

            val_loss = np.mean(val_losses)
            print(f"  Val loss: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
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
        "git_hash": cfg["git_hash"],
    }, final_checkpoint_path)

    # Load best checkpoint for evaluation
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
                model, mol_data_dir,
                n_samples=cfg.get("eval_n_samples", 500),
                device=device,
            )
        mol_results[mol] = result

        print(f"    valid_fraction={result['valid_fraction']:.3f}, "
              f"pw_div={result['pairwise_dist_divergence']:.4f}, "
              f"min_dist_mean={result['min_dist_mean']:.3f}")

        # Log molecule-level metrics
        wandb.log({
            f"eval/{mol}/valid_fraction": result["valid_fraction"],
            f"eval/{mol}/pairwise_dist_divergence": result["pairwise_dist_divergence"],
            f"eval/{mol}/min_dist_mean": result["min_dist_mean"],
            f"eval/{mol}/min_dist_below_08": result["min_dist_below_08"],
        }, step=step)

    # Summary metrics
    valid_fracs = [r["valid_fraction"] for r in mol_results.values()]
    n_valid_majority = sum(1 for vf in valid_fracs if vf > 0.5)
    mean_valid = np.mean(valid_fracs) if valid_fracs else 0.0

    print(f"\nSummary:")
    print(f"  Mean valid fraction: {mean_valid:.3f}")
    print(f"  Molecules with VF > 0.5: {n_valid_majority}/{len(molecules)}")

    # Save raw results
    results_path = os.path.join(raw_dir, "mol_results.pt")
    torch.save(mol_results, results_path)

    # Save config
    config_path = os.path.join(stage_dir, "config.json")
    with open(config_path, "w") as f:
        # Convert non-serializable values
        cfg_save = {k: v for k, v in cfg.items()}
        json.dump(cfg_save, f, indent=2)

    # W&B summary
    wandb.run.summary.update({
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "final_train_loss": loss.item(),
        "mean_valid_fraction": mean_valid,
        "n_molecules_valid": n_valid_majority,
        "mol_results": mol_results,
        "checkpoint_path": checkpoint_path,
        "results_path": results_path,
    })

    # Log artifacts
    ckpt_artifact = wandb.Artifact(f"hyp_002_{cfg['angle']}_{cfg['stage']}_model", type="model")
    ckpt_artifact.add_file(checkpoint_path)
    wandb.log_artifact(ckpt_artifact)

    wandb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} at step {best_step}")
    return mol_results


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--angle", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
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

    train(cfg)


if __name__ == "__main__":
    main()
