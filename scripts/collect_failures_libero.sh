#!/usr/bin/env bash
# Rolls out the frozen OpenVLA-SFT from random initial placements to collect failure episodes for the world model.
set -euo pipefail
cd "$(dirname "$0")/.."
ROLLOUTS=${ROLLOUTS:-rollouts/vanilla}
TRIALS=${TRIALS:-20}  # per task; export_world_model_data.py keeps at most 10 failures per task

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
    --task_suite_name=$SUITE --use_reasoner=false --random_init_states \
    --num_trials_per_task=$TRIALS --save_videos=false --save_rollouts_dir=$ROLLOUTS
done
