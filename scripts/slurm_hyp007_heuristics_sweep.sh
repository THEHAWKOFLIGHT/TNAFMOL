#!/bin/bash
#SBATCH --job-name=hyp007_heuristics_sweep
#SBATCH --output=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/heuristics/sweep/runs/slurm_%A_%a.out
#SBATCH --error=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_007_padding_isolation_multimol/angles/heuristics/sweep/runs/slurm_%A_%a.err
#SBATCH --array=0-7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# hyp_007 HEURISTICS sweep — log_det_reg_weight x n_steps x lr
# 8 array jobs, each runs one sweep config
# Sweep targets: log-det exploitation prevention via log_det_reg_weight > 0

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"

# Load conda (use eval form for bash compat in non-interactive shells)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/conda/etc/profile.d/conda.sh" ]; then
    . "$HOME/conda/etc/profile.d/conda.sh"
fi

# Activate environment
conda activate tnafmol 2>/dev/null && echo "conda env activated" || echo "conda activate failed — using system python3.10"

PYTHON=$(which python 2>/dev/null || which python3.10 2>/dev/null || echo "python3.10")
echo "Python: $PYTHON"
$PYTHON --version

# Config file indexed by array task ID
CONFIG="experiments/hypothesis/hyp_007_padding_isolation_multimol/config/sweep_runs/run_$(printf '%02d' $SLURM_ARRAY_TASK_ID).json"

echo "Using config: $CONFIG"

cd /home/kai_nelson/the_rig/tnafmol

$PYTHON src/train.py --config "$CONFIG"

echo "Job $SLURM_ARRAY_TASK_ID complete"
