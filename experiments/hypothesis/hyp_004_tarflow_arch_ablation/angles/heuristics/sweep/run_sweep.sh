#!/bin/bash
# Run all HEURISTICS sweep configs sequentially
set -e

echo 'Starting run_3000steps_lr1e-4_ema99...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-4_ema99/config.json --device cuda:0
echo 'Done run_3000steps_lr1e-4_ema99'

echo 'Starting run_3000steps_lr1e-4_ema999...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-4_ema999/config.json --device cuda:0
echo 'Done run_3000steps_lr1e-4_ema999'

echo 'Starting run_3000steps_lr1e-4_ema9999...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-4_ema9999/config.json --device cuda:0
echo 'Done run_3000steps_lr1e-4_ema9999'

echo 'Starting run_3000steps_lr3e-4_ema99...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr3e-4_ema99/config.json --device cuda:0
echo 'Done run_3000steps_lr3e-4_ema99'

echo 'Starting run_3000steps_lr3e-4_ema999...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr3e-4_ema999/config.json --device cuda:0
echo 'Done run_3000steps_lr3e-4_ema999'

echo 'Starting run_3000steps_lr3e-4_ema9999...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr3e-4_ema9999/config.json --device cuda:0
echo 'Done run_3000steps_lr3e-4_ema9999'

echo 'Starting run_3000steps_lr1e-3_ema99...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-3_ema99/config.json --device cuda:0
echo 'Done run_3000steps_lr1e-3_ema99'

echo 'Starting run_3000steps_lr1e-3_ema999...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-3_ema999/config.json --device cuda:0
echo 'Done run_3000steps_lr1e-3_ema999'

echo 'Starting run_3000steps_lr1e-3_ema9999...'
python3.10 src/train.py --config experiments/hypothesis/hyp_004_tarflow_arch_ablation/angles/heuristics/sweep/runs/run_3000steps_lr1e-3_ema9999/config.json --device cuda:0
echo 'Done run_3000steps_lr1e-3_ema9999'
