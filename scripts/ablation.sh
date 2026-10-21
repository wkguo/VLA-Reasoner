#!/usr/bin/env bash
# Ablations on LIBERO: injection strength alpha, and Gaussian-noise sampling instead of KDE Top-k.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints}
IVG=${IVG:?path to iVideoGPT}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

for SUITE in $SUITES; do
  WM=$(ls -d $CKPT/wm/$SUITE/log_trm/*/checkpoints/checkpoint_* | sort -V | tail -n 1)
  TOK=$(ls -d $CKPT/wm/$SUITE/log_vqgan/*/checkpoint-*/unwrapped_model | sort -V | tail -n 1)
  COMMON="--flagfile=experiments/configs/libero_openvla.flags --task_suite_name=$SUITE --use_reasoner=true
    --kde_path=$CKPT/$SUITE/kde.npz --value_path=$CKPT/$SUITE/value.pt --ivideogpt_root=$IVG
    --wm_tokenizer=$TOK --wm_transformer=$WM --wm_config=$IVG/configs/llama/config_medium.json"
  for ALPHA in 1.0 0.8 0.6 0.4 0.2; do
    python experiments/eval_libero.py $COMMON --alpha=$ALPHA
  done
  python experiments/eval_libero.py $COMMON --sampler=gaussian --noise_std=0.1
done
