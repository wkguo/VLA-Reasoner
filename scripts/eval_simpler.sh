#!/usr/bin/env bash
# SimplerEnv WidowX evaluation of Octo-Small (POLICY=octo) or SpatialVLA (POLICY=spatialvla), with and without VLA-Reasoner.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
POLICY=${POLICY:-octo}
CKPT=${CKPT:-checkpoints}
IVG=${IVG:?path to iVideoGPT}
case $POLICY in
  octo) WM_DIR=$CKPT/wm/bridge_octo; KDE=$CKPT/bridge/kde_octo.npz ;;
  spatialvla) WM_DIR=$CKPT/wm/bridge_spatialvla_c4; KDE=$CKPT/bridge/kde_spatialvla_c4.npz ;;
esac
WM=$(ls -d $WM_DIR/log_trm/*/checkpoints/checkpoint_* | sort -V | tail -n 1)
TOK=$(ls -d $WM_DIR/log_vqgan/*/checkpoint-*/unwrapped_model | sort -V | tail -n 1)

for TASK in widowx_stack_cube widowx_spoon_on_towel widowx_carrot_on_plate widowx_put_eggplant_in_basket; do
  python experiments/eval_simpler.py --flagfile=experiments/configs/simpler_$POLICY.flags \
    --task_name=$TASK --use_reasoner=false
  python experiments/eval_simpler.py --flagfile=experiments/configs/simpler_$POLICY.flags \
    --task_name=$TASK --use_reasoner=true --kde_path=$KDE --value_path=$CKPT/bridge/value.pt \
    --ivideogpt_root=$IVG --wm_tokenizer=$TOK --wm_transformer=$WM \
    --wm_config=$IVG/configs/llama/config_medium.json
done
