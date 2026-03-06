"""
eval_temp_sweep.py — Temperature sweep evaluation on Phase 2 checkpoint.

Disposable eval script (experiment directory — cleaned up during source integration).

Loads the Phase 2 final checkpoint (heuristics full run) and evaluates
at different sampling temperatures to find the optimal temperature.

Usage:
    python eval_temp_sweep.py --checkpoint path/to/final.pt --device cuda:5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.train_apple import evaluate_molecule
from src.train_phase3 import TarFlow1DMol
from src.data import MOLECULES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/hypothesis/hyp_011_crack_md17_multimol/angles/heuristics/full/final.pt")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="experiments/hypothesis/hyp_011_crack_md17_multimol/angles/scale/temp_sweep_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    cfg = ckpt["config"]
    global_std = ckpt["global_std"]

    print(f"Checkpoint config: channels={cfg['channels']}, num_blocks={cfg['num_blocks']}, "
          f"seq_length={cfg['seq_length']}")
    print(f"Global std: {global_std:.4f}")

    # Reconstruct model from config
    model = TarFlow1DMol(
        in_channels=3,
        seq_length=cfg["seq_length"],
        channels=cfg["channels"],
        num_blocks=cfg["num_blocks"],
        layers_per_block=cfg.get("layers_per_block", 2),
        head_dim=cfg.get("head_dim", 64),
        expansion=cfg.get("expansion", 4),
        use_atom_type_cond=cfg.get("use_atom_type_cond", True),
        atom_type_emb_dim=cfg.get("atom_type_emb_dim", 16),
        num_atom_types=cfg.get("num_atom_types", 4),
        use_padding_mask=cfg.get("use_padding_mask", True),
        use_shared_scale=cfg.get("use_shared_scale", False),
        use_clamp=cfg.get("use_clamp", False),
        log_det_reg_weight=cfg.get("log_det_reg_weight", 2.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # Temperature sweep
    temperatures = [0.7, 0.8, 0.9, 0.95, 1.0]
    data_root = project_root / "data"

    results = {}

    print(f"\n{'='*70}")
    print(f"Temperature Sweep — Phase 2 Checkpoint ({cfg['channels']}ch, {cfg['num_blocks']}blk)")
    print(f"{'='*70}")

    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        mol_results = {}
        vfs = []

        for mol_name in MOLECULES:
            data_dir = str(data_root / f"md17_{mol_name}_v1")
            try:
                res = evaluate_molecule(
                    model=model,
                    data_dir=data_dir,
                    n_samples=args.n_samples,
                    device=device,
                    global_std=global_std,
                    max_atoms=cfg["seq_length"],
                    use_padding_mask=cfg.get("use_padding_mask", True),
                    temperature=temp,
                )
                vf = res["valid_fraction"]
                vfs.append(vf)
                mol_results[mol_name] = res
                print(f"  {mol_name:20s}: VF={vf*100:.1f}%")
            except Exception as e:
                print(f"  {mol_name:20s}: ERROR — {e}")
                mol_results[mol_name] = {"error": str(e)}

        mean_vf = float(np.mean(vfs)) if vfs else 0.0
        results[str(temp)] = {
            "temperature": temp,
            "mean_vf": mean_vf,
            "molecules": mol_results,
        }
        print(f"  {'MEAN':20s}: VF={mean_vf*100:.1f}%")

    # Summary table
    print(f"\n{'='*70}")
    print("TEMPERATURE SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Temp':>8} | {'Mean VF':>8} | {'aspirin':>8} | {'benzene':>8} | {'ethanol':>8} | {'malonaldehyde':>14} | {'naphthalene':>12} | {'salicylic':>10} | {'toluene':>8} | {'uracil':>8}")
    print("-" * 120)

    best_temp = None
    best_mean_vf = -1
    for temp_str, res in results.items():
        temp = res["temperature"]
        mean_vf = res["mean_vf"]
        mols = res["molecules"]
        vf_strs = []
        for mol in MOLECULES:
            if mol in mols and "valid_fraction" in mols[mol]:
                vf_strs.append(f"{mols[mol]['valid_fraction']*100:.1f}%")
            else:
                vf_strs.append("ERR")

        row = f"{temp:>8} | {mean_vf*100:>7.1f}% | " + " | ".join(f"{v:>7}" for v in vf_strs[:4])
        # add remaining
        remaining = " | ".join(f"{v:>10}" for v in vf_strs[4:])
        print(f"{temp:>8.2f} | {mean_vf*100:>7.1f}% | {' | '.join(f'{v:>8}' for v in vf_strs)}")
        if mean_vf > best_mean_vf:
            best_mean_vf = mean_vf
            best_temp = temp

    print(f"\nBest temperature: {best_temp} (Mean VF = {best_mean_vf*100:.1f}%)")
    results["best_temperature"] = best_temp
    results["best_mean_vf"] = best_mean_vf

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
