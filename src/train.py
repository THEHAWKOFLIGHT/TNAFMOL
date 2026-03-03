"""
TNAFMOL Training — TarFlow training loop with MLE loss and W&B logging.

Usage:
    python3.10 src/train.py --config config.json [--device cuda:0]

The script:
1. Loads the multi-molecule dataset (with optional augmentation + normalization)
2. Trains TarFlow with NLL loss + optional log-det regularization
3. Logs to W&B (project: tnafmol)
4. Evaluates per-molecule metrics at the end
5. Saves checkpoint and all output arrays

hyp_003 changes:
- compute_global_std() called to normalize positions to unit variance
- augment=True for train dataset, augment=False for val/test
- alpha_pos/alpha_neg instead of log_scale_max for asymmetric clamping
- log_det_reg_weight parameter passed to nll_loss
- OneCycleLR schedule option
- AdamW betas config option
- EMA model support (for HEURISTICS angle)
- evaluate_molecule() accepts global_std for denormalization
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

from src.data import MD17Dataset, MultiMoleculeDataset, MOLECULES, MAX_ATOMS, compute_global_std, PAD_TOKEN_IDX
from src.metrics import valid_fraction, energy_wasserstein, pairwise_distance_divergence, min_pairwise_distance
from src.model import TarFlow, EMAModel, BidirectionalTypeEncoder


# =============================================================================
# Default config
# =============================================================================

DEFAULT_CONFIG = {
    # Identity
    "exp_id": "hyp_004",
    "angle": "sanity",
    "stage": "diag",
    "command": "OPTIMIZE",

    # Reproducibility
    "seed": 42,
    "device": "cuda:0",

    # Model
    "n_blocks": 8,
    "d_model": 128,
    "n_heads": 4,
    "ffn_mult": 4,
    "atom_type_emb_dim": 16,
    "dropout": 0.1,
    "alpha_pos": 0.02,       # asymmetric clamp: expansion bound per layer (hyp_003 best)
    "alpha_neg": 2.0,        # asymmetric clamp: contraction bound per layer
    "shift_only": False,     # use full affine flow (not shift-only)
    "use_actnorm": False,    # no ActNorm (collapsed in hyp_002)

    # hyp_004 architectural flags (all default False for backward compatibility)
    "use_bidir_types": False,    # bidirectional type conditioning
    "use_pos_enc": False,        # learned positional encodings per block
    "use_perm_aug": False,       # permutation augmentation on atoms

    # hyp_005 padding-aware flags (all default False/0.0 for backward compatibility)
    "use_pad_token": False,      # use PAD_TOKEN_IDX=4 for padding (requires n_atom_types=5)
    "zero_padding_queries": False,  # zero padding atom queries before attention
    "noise_sigma": 0.0,          # per-coord Gaussian noise std on real atoms (0.0 = off)

    # Training
    "n_steps": 500,          # default: diagnostic run
    "batch_size": 128,
    "lr": 1e-4,
    "lr_schedule": "cosine", # "cosine" or "onecycle"
    "warmup_steps": 500,
    "grad_clip_norm": 1.0,
    "val_interval": 200,
    "eval_n_samples": 500,
    "log_det_reg_weight": 5.0,  # penalty weight (hyp_003 best)
    "betas": [0.9, 0.999],   # AdamW betas (can be changed for HEURISTICS)
    "weight_decay": 1e-5,    # AdamW weight decay

    # EMA (for HEURISTICS angle)
    "use_ema": False,
    "ema_decay": 0.999,

    # Data augmentation and normalization
    "augment_train": True,              # random SO(3) + CoM noise on train
    "normalize_to_unit_var": True,      # divide by global std

    # Data
    "data_root": "data/",
    "molecules": None,  # None = all 8

    # W&B
    "wandb_project": "tnafmol",
    "wandb_group": "hyp_004",
    "wandb_tags": ["hypothesis", "hyp_004", "OPTIMIZE"],
    "wandb_notes": "",

    # Output
    "output_dir": "experiments/hypothesis/hyp_004_tarflow_arch_ablation",
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
    global_std: Optional[float] = None,
    pad_token_idx: int = 0,
) -> Dict:
    """Evaluate the model on a single molecule.

    Args:
        model: trained TarFlow model
        data_dir: path to processed molecule dataset
        n_samples: number of samples to generate
        device: compute device
        temperature: sampling temperature
        global_std: if provided, multiply samples by this to denormalize
                    (model generates in normalized space, metrics need Angstroms)
        pad_token_idx: index used for padding atoms in atom_types (hyp_005 default 0)

    Returns:
        dict with valid_fraction, energy_wasserstein, pairwise_distance_divergence
    """
    data = np.load(os.path.join(data_dir, "dataset.npz"))
    ref_stats = torch.load(os.path.join(data_dir, "ref_stats.pt"), weights_only=False)

    # Get molecule info — apply correct pad_token_idx to padding positions
    raw_atom_types = data["atom_types"]  # (21,) int64
    mask_np = data["mask"]               # (21,) float32
    n_real = int(mask_np.sum())
    if pad_token_idx != 0:
        atom_types_eval = raw_atom_types.copy()
        atom_types_eval[n_real:] = pad_token_idx
    else:
        atom_types_eval = raw_atom_types
    atom_types = torch.from_numpy(atom_types_eval).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    test_idx = data["test_idx"]
    ref_positions = data["positions"][test_idx]  # (N_test, 21, 3) — raw Angstroms
    ref_energies = data["energies"][test_idx]

    # Generate samples — model generates in normalized space if global_std was used
    model.eval()
    samples = model.sample(atom_types, mask, n_samples=n_samples, temperature=temperature)
    samples_np = samples.cpu().numpy()  # (n_samples, 21, 3)

    # Denormalize: model generates in normalized space, metrics need raw Angstroms
    if global_std is not None and global_std > 0:
        samples_np = samples_np * global_std

    # Metrics — use mask_np already defined above (avoids redundant cpu().numpy() call)
    vf, _ = valid_fraction(samples_np, mask_np)

    # Pairwise distance divergence (proxy for structural quality)
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
    stage_subdir = cfg["stage"]
    if cfg["stage"] == "sweep":
        lr_str = f"lr{cfg['lr']:.0e}".replace("-0", "-").replace("+0", "")
        stage_subdir = f"sweep/runs/run_{cfg['n_steps']}steps_{lr_str}"
    # val_subdir: when multiple val configs share the same stage name, use this
    # to create unique output subdirectories (e.g., "config_a", "config_b")
    if cfg.get("val_subdir"):
        stage_subdir = os.path.join(stage_subdir, cfg["val_subdir"])
    stage_dir = os.path.join(output_dir, "angles", cfg["angle"], stage_subdir)
    os.makedirs(stage_dir, exist_ok=True)
    raw_dir = os.path.join(stage_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Compute global std for normalization
    molecules = cfg.get("molecules") or list(MOLECULES.keys())
    global_std = None
    if cfg.get("normalize_to_unit_var", False):
        global_std = compute_global_std(data_root, molecules, split="train")
        cfg["global_std"] = global_std
        print(f"Global std for normalization: {global_std:.4f} Angstroms")

    # W&B: if a run is already active (called from sweep agent), reuse it
    # Otherwise, init a new run.
    if wandb.run is None:
        exp_id = cfg.get("exp_id", "hyp_004")
        run_name = f"{exp_id}_{cfg['angle']}_{cfg['stage']}"
        if cfg.get("run_name_suffix"):
            run_name = run_name + "_" + cfg["run_name_suffix"]

        tags = list(cfg.get("wandb_tags", []))
        if cfg["angle"] not in tags:
            tags.append(cfg["angle"])
        if cfg["stage"] not in tags:
            tags.append(cfg["stage"])

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
        # Already running inside a sweep agent — update config with any new keys
        wandb.config.update(cfg, allow_val_change=True)

    assert wandb.run is not None, "W&B init failed"
    print(f"W&B run: {wandb.run.url}")

    # Log test metric to confirm logging
    wandb.log({"init_check": 1.0}, step=0)

    # Model — includes hyp_004 and hyp_005 architectural flags
    # n_atom_types: 5 when use_pad_token=True (adds separate PAD embedding at index 4)
    use_pad_token = cfg.get("use_pad_token", False)
    n_atom_types = 5 if use_pad_token else 4

    model = TarFlow(
        n_blocks=cfg["n_blocks"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ffn_mult=cfg["ffn_mult"],
        atom_type_emb_dim=cfg["atom_type_emb_dim"],
        n_atom_types=n_atom_types,
        dropout=cfg["dropout"],
        max_atoms=MAX_ATOMS,
        alpha_pos=cfg.get("alpha_pos", 0.02),
        alpha_neg=cfg.get("alpha_neg", 2.0),
        shift_only=cfg.get("shift_only", False),
        use_actnorm=cfg.get("use_actnorm", False),
        use_bidir_types=cfg.get("use_bidir_types", False),
        use_pos_enc=cfg.get("use_pos_enc", False),
        zero_padding_queries=cfg.get("zero_padding_queries", False),
    ).to(device)
    cfg["n_atom_types"] = n_atom_types  # log to wandb config

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # EMA setup (HEURISTICS angle)
    ema = None
    if cfg.get("use_ema", False):
        ema = EMAModel(model, decay=cfg.get("ema_decay", 0.999))
        print(f"EMA enabled with decay={cfg.get('ema_decay', 0.999)}")

    # Dataset — with augmentation, normalization, permutation, and hyp_005 options
    augment_train = cfg.get("augment_train", False)
    use_perm_aug = cfg.get("use_perm_aug", False)
    pad_token_idx = PAD_TOKEN_IDX if use_pad_token else 0
    noise_sigma = cfg.get("noise_sigma", 0.0)

    train_ds = MultiMoleculeDataset(
        data_root, split="train", molecules=molecules,
        augment=augment_train, global_std=global_std,
        permute=use_perm_aug,
        pad_token_idx=pad_token_idx,
        noise_sigma=noise_sigma,
    )
    val_ds = MultiMoleculeDataset(
        data_root, split="val", molecules=molecules,
        augment=False, global_std=global_std,
        permute=False,  # no permutation for validation
        pad_token_idx=pad_token_idx,
        noise_sigma=0.0,  # no noise for validation (evaluate clean data)
    )

    print(f"Train size: {len(train_ds):,}, Val size: {len(val_ds):,}")
    print(f"Augment train: {augment_train}, Permute: {use_perm_aug}, Global std: {global_std}")
    print(f"Pad token: {use_pad_token} (idx={pad_token_idx}), Noise sigma: {noise_sigma}, Zero pad queries: {cfg.get('zero_padding_queries', False)}")

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
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    weight_decay = cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=betas, weight_decay=weight_decay
    )

    # LR Schedule
    lr_schedule = cfg.get("lr_schedule", "cosine")
    if lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr"],
            total_steps=cfg["n_steps"],
            pct_start=0.05,  # 5% warmup
            anneal_strategy="cos",
        )
    else:
        # Default: cosine with warmup
        scheduler = get_cosine_schedule(optimizer, cfg["n_steps"], cfg.get("warmup_steps", 500))

    # Log-det regularization weight
    log_det_reg_weight = cfg.get("log_det_reg_weight", 0.0)

    # Training loop
    best_val_loss = float("inf")
    best_step = 0
    checkpoint_path = os.path.join(stage_dir, "best.pt")
    final_checkpoint_path = os.path.join(stage_dir, "final.pt")

    step = 0
    train_iter = iter(train_loader)
    model.train()

    print(f"\nStarting training for {cfg['n_steps']} steps...")
    print(f"log_det_reg_weight={log_det_reg_weight}, alpha_pos={cfg.get('alpha_pos', 0.1)}, alpha_neg={cfg.get('alpha_neg', 2.0)}")
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

        # Forward + loss
        optimizer.zero_grad()
        loss, info = model.nll_loss(
            positions, atom_types, atom_mask,
            log_det_reg_weight=log_det_reg_weight
        )

        # Sanity check — finite loss
        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss.item()}"

        # Backward
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

        optimizer.step()
        scheduler.step()

        # EMA update
        if ema is not None:
            ema.update()

        step += 1

        # Log
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else cfg["lr"]
            log_det_per_dof = info.get("log_det_per_dof", 0.0)
            print(
                f"Step {step}/{cfg['n_steps']} | loss={loss.item():.4f} | "
                f"nll_only={info.get('nll_loss_only', 0.0):.4f} | "
                f"log_det/dof={log_det_per_dof:.3f} | "
                f"log_det_penalty={info.get('log_det_penalty', 0.0):.4f} | "
                f"grad_norm={grad_norm:.3f} | lr={lr:.2e} | t={elapsed:.1f}s"
            )
            wandb.log({
                "train/loss": loss.item(),
                "train/nll": info["nll"],
                "train/nll_per_dof": info["nll_per_dof"],
                "train/nll_loss_only": info.get("nll_loss_only", info["nll_per_dof"]),
                "train/log_det": info["total_log_det"],
                "train/log_det_per_dof": info.get("log_det_per_dof", 0.0),
                "train/log_det_penalty": info.get("log_det_penalty", 0.0),
                "train/log_pz": info["log_pz"],
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/n_real_mean": info["n_real_mean"],
            }, step=step)

        # Validation
        if step % cfg["val_interval"] == 0 or step == cfg["n_steps"]:
            # Use EMA model for validation if enabled
            if ema is not None:
                ema.apply_shadow()

            model.eval()
            val_losses = []
            val_log_det_per_dofs = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vpos = vbatch["positions"].to(device)
                    vatypes = vbatch["atom_types"].to(device)
                    vmask = vbatch["mask"].to(device)
                    vloss, vinfo = model.nll_loss(vpos, vatypes, vmask, log_det_reg_weight=0.0)
                    val_losses.append(vloss.item())
                    val_log_det_per_dofs.append(vinfo.get("log_det_per_dof", 0.0))

            val_loss = np.mean(val_losses)
            val_log_det_per_dof = np.mean(val_log_det_per_dofs)
            print(f"  Val loss: {val_loss:.4f} | Val log_det/dof: {val_log_det_per_dof:.3f}")
            wandb.log({
                "val/loss": val_loss,
                "val/log_det_per_dof": val_log_det_per_dof,
            }, step=step)

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
                    "global_std": global_std,
                }, checkpoint_path)
                print(f"  Saved best checkpoint at step {step} (val_loss={val_loss:.4f})")

            if ema is not None:
                ema.restore()

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
        "global_std": global_std,
    }, final_checkpoint_path)

    # Load best checkpoint for evaluation
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Apply EMA weights for final evaluation
    if ema is not None:
        ema.apply_shadow()

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
                global_std=global_std,
                pad_token_idx=pad_token_idx,
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

    if ema is not None:
        ema.restore()

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
        cfg_save = dict(cfg)
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
        "global_std": global_std,
    })

    # Log artifacts
    exp_id = cfg.get("exp_id", "hyp_004")
    angle = cfg.get("angle", "sanity")
    stage = cfg.get("stage", "diag")
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
