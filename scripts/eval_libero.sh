#!/usr/bin/env bash
# LIBERO (Table I top): OpenVLA-SFT with and without VLA-Reasoner, 500 episodes per suite.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CKPT=${CKPT:-checkpoints}
IVG=${IVG:?path to iVideoGPT}

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
    --task_suite_name=$SUITE --use_reasoner=false

  WM=$(ls -d $CKPT/wm/$SUITE/log_trm/*/checkpoints/checkpoint_* | sort -V | tail -n 1)
  TOK=$(ls -d $CKPT/wm/$SUITE/log_vqgan/*/checkpoint-*/unwrapped_model | sort -V | tail -n 1)
  python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
    --task_suite_name=$SUITE --use_reasoner=true \
    --kde_path=$CKPT/$SUITE/kde.npz --value_path=$CKPT/$SUITE/value.pt \
    --ivideogpt_root=$IVG --wm_tokenizer=$TOK --wm_transformer=$WM \
    --wm_config=$IVG/configs/llama/config_medium.json
done
