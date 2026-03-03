"""
hyp_005 HEURISTICS sweep: log_det_reg_weight x lr grid (3x3 = 9 runs, 3000 steps each, ethanol).

Config D base: PAD token + query zeroing + alpha_pos=1.0 + noise_sigma=0.05.
Sweep: log_det_reg_weight in [0.5, 1.0, 2.0] x lr in [1e-4, 3e-4, 5e-4].

Usage:
    CUDA_VISIBLE_DEVICES=2 python3.10 experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/heuristics/sweep/run_sweep.py

This initializes a W&B sweep and launches a single agent that runs all 9 configurations.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, project_root)

import wandb
from src.train import train

# Base config (Config D)
BASE_CFG = {
    "exp_id": "hyp_005",
    "angle": "heuristics",
    "stage": "sweep",
    "command": "OPTIMIZE",
    "seed": 42,
    "n_blocks": 8,
    "d_model": 128,
    "n_heads": 4,
    "ffn_mult": 4,
    "atom_type_emb_dim": 16,
    "dropout": 0.1,
    "alpha_pos": 1.0,
    "alpha_neg": 10.0,
    "shift_only": False,
    "use_actnorm": False,
    "use_bidir_types": True,
    "use_pos_enc": False,
    "use_pad_token": True,
    "zero_padding_queries": True,
    "noise_sigma": 0.05,
    "n_steps": 3000,
    "batch_size": 128,
    "lr_schedule": "cosine",
    "warmup_steps": 200,
    "grad_clip_norm": 1.0,
    "val_interval": 500,
    "eval_n_samples": 300,
    "betas": [0.9, 0.999],
    "weight_decay": 1e-5,
    "use_ema": False,
    "augment_train": True,
    "normalize_to_unit_var": True,
    "use_perm_aug": False,
    "data_root": "data/",
    "molecules": ["ethanol"],
    "wandb_project": "tnafmol",
    "wandb_group": "hyp_005",
    "wandb_tags": ["hypothesis", "hyp_005", "OPTIMIZE", "heuristics", "sweep"],
    "output_dir": "experiments/hypothesis/hyp_005_padding_aware_tarflow",
}

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "eval/ethanol/valid_fraction", "goal": "maximize"},
    "parameters": {
        "log_det_reg_weight": {"values": [0.5, 1.0, 2.0]},
        "lr": {"values": [1e-4, 3e-4, 5e-4]},
    },
    "name": "hyp_005_heuristics_sweep",
    "run_cap": 9,
}


def sweep_train():
    """Train function for W&B sweep agent."""
    wandb.init()
    cfg = dict(BASE_CFG)
    cfg.update(wandb.config)
    # Set device — will be overridden by CUDA_VISIBLE_DEVICES
    cfg["device"] = "cuda:0"
    cfg["wandb_notes"] = (
        f"Sweep run: log_det_reg_weight={cfg['log_det_reg_weight']}, lr={cfg['lr']}. "
        f"Config D (PAD token + query zeroing). 3000 steps, ethanol."
    )
    train(cfg)


if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="tnafmol")
    print(f"Sweep URL: https://wandb.ai/kaityrusnelson1/tnafmol/sweeps/{sweep_id}")
    print(f"Sweep ID: {sweep_id}")

    # Log sweep ID for process_log
    with open(os.path.join(project_root, "experiments/hypothesis/hyp_005_padding_aware_tarflow/angles/heuristics/sweep/sweep_id.txt"), "w") as f:
        f.write(sweep_id + "\n")

    wandb.agent(sweep_id, function=sweep_train, project="tnafmol", count=9)
