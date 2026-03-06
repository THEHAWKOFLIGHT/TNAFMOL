#!/bin/bash
# hyp_011 Phase 2 HEURISTICS Sweep Script
# 27 configs across 6 GPUs (0, 3, 4, 5, 6, 7) in 5 batches
# Each run: 20k steps, ~37 min on A6000
# Total: ~5 batches × 37 min = ~3 hours
#
# Usage: bash experiments/hypothesis/hyp_011_crack_md17_multimol/run_sweep.sh
# Run from project root: /home/kai_nelson/the_rig/tnafmol/

PYTHON=/home/kai_nelson/ObservableAlignment/ENTER/envs/obs_env/bin/python
SCRIPT=src/train_apple.py
CONFIG_DIR=experiments/hypothesis/hyp_011_crack_md17_multimol/config/sweep
LOG_DIR=experiments/hypothesis/hyp_011_crack_md17_multimol/angles/heuristics/sweep/logs
GPUS=(0 3 4 5 6 7)

mkdir -p "$LOG_DIR"

# All 27 configs in order
CONFIGS=(
    "$CONFIG_DIR/sweep_lr1e-4_ldr0p0_ns003.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr0p0_ns005.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr0p0_ns01.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr2p0_ns003.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr2p0_ns005.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr2p0_ns01.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr5p0_ns003.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr5p0_ns005.json"
    "$CONFIG_DIR/sweep_lr1e-4_ldr5p0_ns01.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr0p0_ns003.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr0p0_ns005.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr0p0_ns01.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr2p0_ns003.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr2p0_ns005.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr2p0_ns01.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr5p0_ns003.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr5p0_ns005.json"
    "$CONFIG_DIR/sweep_lr3e-4_ldr5p0_ns01.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr0p0_ns003.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr0p0_ns005.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr0p0_ns01.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr2p0_ns003.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr2p0_ns005.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr2p0_ns01.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr5p0_ns003.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr5p0_ns005.json"
    "$CONFIG_DIR/sweep_lr5e-4_ldr5p0_ns01.json"
)

N_CONFIGS=${#CONFIGS[@]}
N_GPUS=${#GPUS[@]}
echo "Launching $N_CONFIGS configs across $N_GPUS GPUs in batches..."

# Iterate in batches of N_GPUS
for ((batch_start=0; batch_start<N_CONFIGS; batch_start+=N_GPUS)); do
    batch_end=$((batch_start + N_GPUS - 1))
    if [ $batch_end -ge $N_CONFIGS ]; then
        batch_end=$((N_CONFIGS - 1))
    fi
    batch_num=$((batch_start / N_GPUS + 1))
    echo ""
    echo "=== Batch $batch_num: configs $((batch_start+1)) to $((batch_end+1)) ==="

    gpu_idx=0
    pids=()
    for ((i=batch_start; i<=batch_end; i++)); do
        config="${CONFIGS[$i]}"
        gpu="${GPUS[$gpu_idx]}"
        log_name=$(basename "$config" .json)
        log_file="$LOG_DIR/${log_name}.log"

        echo "  Config $((i+1))/27: $log_name -> GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT --config "$config" --device cuda:0 \
            > "$log_file" 2>&1 &
        pids+=($!)
        gpu_idx=$((gpu_idx + 1))
    done

    echo "  Waiting for batch $batch_num to complete (PIDs: ${pids[*]})..."
    for pid in "${pids[@]}"; do
        wait $pid
        status=$?
        if [ $status -ne 0 ]; then
            echo "  WARNING: PID $pid exited with status $status"
        fi
    done
    echo "  Batch $batch_num complete."
done

echo ""
echo "All $N_CONFIGS sweep runs complete."
echo "Results in: experiments/hypothesis/hyp_011_crack_md17_multimol/angles/heuristics/sweep/runs/"
