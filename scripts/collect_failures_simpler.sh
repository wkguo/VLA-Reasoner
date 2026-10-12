#!/usr/bin/env bash
# Step 0 (SimplerEnv): roll out the frozen Octo-Small / SpatialVLA to collect failure episodes for world-model
# fine-tuning. The WidowX tasks have a finite set of initial configurations, so these rollouts share initial
# configurations with the evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
ROLLOUTS=${ROLLOUTS:-rollouts/vanilla}
POLICY=${POLICY:-octo}  # octo | spatialvla (run each in its own environment)
EPISODES=${EPISODES:-20}

for TASK in widowx_stack_cube widowx_spoon_on_towel widowx_carrot_on_plate widowx_put_eggplant_in_basket; do
  python experiments/eval_simpler.py --flagfile=experiments/configs/simpler_${POLICY}.flags \
    --task_name=$TASK --use_reasoner=false --num_eval_episodes=$EPISODES --seed=100 \
    --save_videos=false --save_rollouts_dir=$ROLLOUTS/$POLICY
done
