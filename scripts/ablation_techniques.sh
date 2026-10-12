#!/usr/bin/env bash
# Fig. 6 (LIBERO-Spatial): (left) Gaussian-noise sampling instead of KDE Top-k;
# (right) iVideoGPT token reward head instead of the image value function.
# The token-reward variant expects a transformer checkpoint whose `reward_linear` head was trained on the
# Sec. III-D progress labels; pass it as WM_REWARD (see README, "Implementation details").
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints}
IVG=${IVG:?path to iVideoGPT}
SUITE=libero_spatial
WM=${WM:-$(ls -d $CKPT/wm/$SUITE/log_trm/*/checkpoints/checkpoint_* | sort -V | tail -n 1)}
TOK=$(ls -d $CKPT/wm/$SUITE/log_vqgan/*/checkpoint-*/unwrapped_model | sort -V | tail -n 1)
COMMON="--flagfile=experiments/configs/libero_openvla.flags --task_suite_name=$SUITE --use_reasoner=true \
  --kde_path=$CKPT/$SUITE/kde.npz --ivideogpt_root=$IVG --wm_tokenizer=$TOK \
  --wm_config=$IVG/configs/llama/config_medium.json"

# Sampling criterion: Gaussian noise around the VLA action (std = NOISE x per-dimension dataset std).
NOISE=${NOISE:-0.1}
python experiments/eval_libero.py $COMMON --wm_transformer=$WM --value_path=$CKPT/$SUITE/value.pt \
  --sampler=gaussian --noise_std=$NOISE

# Value estimation: token reward head of the world model.
if [ -n "${WM_REWARD:-}" ]; then
  python experiments/eval_libero.py $COMMON --wm_transformer=$WM_REWARD --value=token_reward
else
  echo "Skipping the token-reward ablation: set WM_REWARD to a transformer checkpoint whose reward_linear head was trained on progress labels."
fi
