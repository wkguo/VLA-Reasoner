#!/usr/bin/env bash
# SimplerEnv WidowX (Table I bottom): Octo-Small with and without VLA-Reasoner, 100 episodes per task.
# Environment: nakamotoo/SimplerEnv + nakamotoo/octo (same as V-GPS), plus torch for the reasoner.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CKPT=${CKPT:-checkpoints}
IVG=${IVG:?path to iVideoGPT}
WM_DIR=$CKPT/wm/bridge_octo
WM=$(ls -d $WM_DIR/log_trm/*/checkpoints/checkpoint_* | sort -V | tail -n 1)
TOK=$(ls -d $WM_DIR/log_vqgan/*/checkpoint-*/unwrapped_model | sort -V | tail -n 1)

for TASK in widowx_stack_cube widowx_spoon_on_towel widowx_carrot_on_plate widowx_put_eggplant_in_basket; do
  python experiments/eval_simpler.py --flagfile=experiments/configs/simpler_octo.flags \
    --task_name=$TASK --seed=0 --use_reasoner=false
  python experiments/eval_simpler.py --flagfile=experiments/configs/simpler_octo.flags \
    --task_name=$TASK --seed=0 --use_reasoner=true \
    --kde_path=$CKPT/bridge/kde_octo.npz --value_path=$CKPT/bridge/value.pt \
    --ivideogpt_root=$IVG --wm_tokenizer=$TOK --wm_transformer=$WM \
    --wm_config=$IVG/configs/llama/config_medium.json
done
