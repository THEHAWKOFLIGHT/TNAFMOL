#!/bin/bash
#SBATCH --job-name=hyp004_swp
#SBATCH --output=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/slurm_sweep_%A_%a.out
#SBATCH --error=/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/slurm_sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-5

CONFIGS=(sweep_pos_lr5e-5_bs128 sweep_pos_lr5e-5_bs256 sweep_pos_lr1e-4_bs128 sweep_pos_lr1e-4_bs256 sweep_pos_lr2e-4_bs128 sweep_pos_lr2e-4_bs256)
CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONFIG_DIR="/home/kai_nelson/the_rig/tnafmol/experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/sanity/ablation_configs"
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_NAME}.json"

echo "Running sweep config: ${CONFIG_NAME}"
echo "Config file: ${CONFIG_FILE}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

cd /home/kai_nelson/the_rig/tnafmol

python3.10 src/train.py --config ${CONFIG_FILE} --device cuda:0

echo "Completed config ${CONFIG_NAME} at $(date)"
