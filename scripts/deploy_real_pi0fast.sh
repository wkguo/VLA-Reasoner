#!/usr/bin/env bash
# Real robot (Table II): pi0-FAST served by openpi, with VLA-Reasoner. Start the policy server first, e.g. in
# the openpi repo:  uv run scripts/serve_policy.py policy:checkpoint --policy.config=<your pi0_fast config> \
#                     --policy.dir=<your fine-tuned checkpoint>
# ROBOT must point to your implementation of vla_reasoner.deploy.Robot ("package.module:Class").
set -euo pipefail
cd "$(dirname "$0")/.."
ROBOT=${ROBOT:?package.module:Class of your robot driver}
TASK=${TASK:-"stack the green cup on the purple cup"}
CKPT=${CKPT:-checkpoints/real}
IVG=${IVG:?path to iVideoGPT}

python experiments/deploy_real.py --flagfile=experiments/configs/real_pi0fast.flags \
  --robot=$ROBOT --instruction="$TASK" --num_episodes=20 --host=${HOST:-localhost} --port=${PORT:-8000} \
  --use_reasoner=${USE_REASONER:-true} \
  --kde_path=$CKPT/kde_pi0fast_c5.npz --value_path=$CKPT/value.pt \
  --ivideogpt_root=$IVG --wm_tokenizer=$CKPT/wm/tokenizer --wm_transformer=$CKPT/wm/transformer \
  --wm_config=$IVG/configs/llama/config_medium.json --save_dir=rollouts/real
