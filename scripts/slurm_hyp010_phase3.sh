#!/bin/bash
#SBATCH --job-name=hyp010_phase3_multimol
#SBATCH --output=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_010_tarflow_apple_multimol/angles/sanity/full/slurm_%j.out
#SBATCH --error=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_010_tarflow_apple_multimol/angles/sanity/full/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# hyp_010 Phase 3 SANITY: all 8 molecules at T=21, 20k steps
# Tests whether Apple TarFlow with padding fixes can generalize across molecules
# Success: VF > 50% on ethanol AND mean VF > 40%

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"

# Load conda (use eval form for bash compat in non-interactive shells)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/conda/etc/profile.d/conda.sh" ]; then
    . "$HOME/conda/etc/profile.d/conda.sh"
fi

# Activate environment
conda activate tnafmol 2>/dev/null && echo "conda env activated" || echo "conda activate failed — using system python"

PYTHON=$(which python 2>/dev/null || which python3 2>/dev/null || echo "python3")
echo "Python: $PYTHON"
$PYTHON --version

CONFIG="experiments/hypothesis/hyp_010_tarflow_apple_multimol/angles/sanity/full/config.json"
echo "Config: $CONFIG"

cd /home/kai_nelson/the_rig/tnafmol

$PYTHON src/train_apple.py --config "$CONFIG"

EXIT_CODE=$?
echo "Job complete with exit code: $EXIT_CODE"
echo "End time: $(date)"
exit $EXIT_CODE
