"""
Diagnostic runs for hyp_009 Phase 1 investigation.
Runs two comparison configs sequentially on cuda:9:
1. Post-norm baseline (ldr=5.0): same as hyp_008 arch, confirms pre-norm effect
2. Pre-norm ldr=1.0: tests if lower ldr allows meaningful log_det and improves VF
"""
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

exp_dir = os.path.dirname(os.path.abspath(__file__))
diag_dir = os.path.join(exp_dir, "angles/sanity/diag")

from src.train import train

configs = [
    os.path.join(diag_dir, "config_postnorm_ldr5.json"),
    os.path.join(diag_dir, "config_prenorm_ldr1.json"),
]

for config_path in configs:
    print(f"\n{'='*60}")
    print(f"Running config: {os.path.basename(config_path)}")
    print(f"{'='*60}")
    with open(config_path) as f:
        cfg = json.load(f)
    train(cfg)
