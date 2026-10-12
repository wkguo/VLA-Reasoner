#!/usr/bin/env bash
# Fig. 5: injection strength alpha in {1.0, 0.8, 0.6, 0.4, 0.2} (alpha = 1.0 is the vanilla VLA).
# The paper reports this grid on all four LIBERO suites; set SUITES to choose.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints}
IVG=${IVG:?path to iVideoGPT}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

for SUITE in $SUITES; do
  WM=$(ls -d $CKPT/wm/$SUITE/log_trm/*/checkpoints/checkpoint_* | sort -V | tail -n 1)
  TOK=$(ls -d $CKPT/wm/$SUITE/log_vqgan/*/checkpoint-*/unwrapped_model | sort -V | tail -n 1)
  for ALPHA in 1.0 0.8 0.6 0.4 0.2; do
    python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
      --task_suite_name=$SUITE --use_reasoner=true --alpha=$ALPHA \
      --kde_path=$CKPT/$SUITE/kde.npz --value_path=$CKPT/$SUITE/value.pt \
      --ivideogpt_root=$IVG --wm_tokenizer=$TOK --wm_transformer=$WM \
      --wm_config=$IVG/configs/llama/config_medium.json
  done
done
