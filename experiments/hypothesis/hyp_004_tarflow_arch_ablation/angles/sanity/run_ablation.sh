#!/bin/bash
#SBATCH --job-name=hyp004_abl
#SBATCH --output=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/slurm_%A_%a.out
#SBATCH --error=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/slurm_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-5

# Map array index to config name
CONFIGS=(A_baseline B_bidir C_perm D_pos E_bidir_perm F_bidir_pos)
CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONFIG_DIR="/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/ablation_configs"
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_NAME}.json"

echo "Running ablation config: ${CONFIG_NAME}"
echo "Config file: ${CONFIG_FILE}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $(hostname)"
echo "Date: $(date)"

cd /home/kai_nelson/the_rig/tnafmol

python3.10 src/train.py --config ${CONFIG_FILE} --device cuda:0

echo "Completed config ${CONFIG_NAME} at $(date)"
