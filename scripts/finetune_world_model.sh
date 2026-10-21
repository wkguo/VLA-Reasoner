#!/usr/bin/env bash
# Fine-tunes iVideoGPT in two stages (tokenizer, then action-conditioned transformer); set TOKENIZER to run stage 2.
set -euo pipefail
cd "$(dirname "$0")/.."
IVG=${IVG:?path to iVideoGPT}
PRETRAINED=${PRETRAINED:?path to pretrained iVideoGPT checkpoints}
DATA=${DATA:?world-model data root}
ACTION_DIM=${ACTION_DIM:-7}
NAME=${NAME:-vla_reasoner}
OUT=${OUT:-checkpoints/wm/$(basename $DATA)}
TOKENIZER=${TOKENIZER:-}  # set to skip stage 1, e.g. $OUT/log_vqgan/<run>/checkpoint-200000/unwrapped_model

if [ -z "$TOKENIZER" ]; then
  accelerate launch experiments/ivideogpt_launch.py --ivideogpt_root=$IVG --script=train_tokenizer.py \
    --dataset_name=$NAME -- \
    --exp_name ${NAME}_tokenizer --output_dir $OUT/log_vqgan --seed 0 --mixed_precision bf16 \
    --model_type ctx_vqgan --train_batch_size 16 --gradient_accumulation_steps 1 --disc_start 1000005 \
    --oxe_data_mixes_type $NAME --dataset_path $DATA --resolution 64 --dataloader_num_workers 16 \
    --rand_select --video_stepsize 1 --segment_horizon 12 --segment_length 8 --context_length 2 \
    --pretrained_model_name_or_path $PRETRAINED/ivideogpt-oxe-64-act-free/tokenizer \
    --max_train_steps 200005
  echo "Stage 1 done. Re-run with TOKENIZER=<run dir>/checkpoint-<step>/unwrapped_model for stage 2."
  exit 0
fi

I3D=$IVG/pretrained_models/i3d/i3d_torchscript.pt
if [ ! -f "$I3D" ]; then
  echo "Missing $I3D (needed by iVideoGPT's train_gpt.py); see the README." >&2
  exit 1
fi

# --config_name stays relative: train_gpt.py reads the model type from the path.
accelerate launch experiments/ivideogpt_launch.py --ivideogpt_root=$IVG --script=train_gpt.py \
  --dataset_name=$NAME -- \
  --exp_name ${NAME}_transformer --output_dir $OUT/log_trm --seed 0 --mixed_precision bf16 \
  --vqgan_type ctx_vqgan --pretrained_model_name_or_path $TOKENIZER \
  --config_name configs/llama/config_medium.json --load_internal_llm \
  --action_conditioned --action_dim $ACTION_DIM \
  --pretrained_transformer_path $PRETRAINED/ivideogpt-oxe-64-act-free-medium/transformer \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --lr_scheduler_type cosine \
  --oxe_data_mixes_type $NAME --dataset_path $DATA --resolution 64 --dataloader_num_workers 16 \
  --video_stepsize 1 --segment_length 12 --context_length 2 \
  --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd --max_train_steps 100005
