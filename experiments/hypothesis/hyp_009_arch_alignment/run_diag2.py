"""
Diagnostic run 2 for hyp_009: contraction-only test.
alpha_pos=0.001 effectively forces log_scale to be non-positive,
mirroring Apple's convention (contraction in forward, expansion in sampling).
This prevents log_det exploitation without needing ldr regularization.
"""
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

exp_dir = os.path.dirname(os.path.abspath(__file__))
diag_dir = os.path.join(exp_dir, "angles/sanity/diag")

from src.train import train

config_path = os.path.join(diag_dir, "config_alpha_pos0.json")
print(f"Running: {os.path.basename(config_path)}")
with open(config_path) as f:
    cfg = json.load(f)
train(cfg)
