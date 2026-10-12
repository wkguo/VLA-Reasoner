#!/usr/bin/env bash
# Step 3b: fine-tune iVideoGPT (tokenizer, then action-conditioned transformer) with the official training
# scripts, following iVideoGPT's scripts/finetune/*-64-act-cond.sh recipes. The world model is the 114M tokenizer
# + the 436M "medium" transformer.
#   IVG:        clone of github.com/thuml/iVideoGPT (environment: requirements/ivideogpt.txt)
#   PRETRAINED: folder with HF downloads thuml/ivideogpt-oxe-64-act-free and thuml/ivideogpt-oxe-64-act-free-medium
#   DATA:       --output_dir of export_world_model_data.py;  ACTION_DIM: printed by that script
# Stage 2 (train_gpt.py) builds iVideoGPT's video-metric evaluator, which loads the I3D TorchScript model from
# $IVG/pretrained_models/i3d/i3d_torchscript.pt. Download it once (link from iVideoGPT's README):
#   mkdir -p $IVG/pretrained_models/i3d && wget -O $IVG/pretrained_models/i3d/i3d_torchscript.pt \
#     "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
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
  echo "Missing $I3D (needed by iVideoGPT's train_gpt.py). Download it as described at the top of this script." >&2
  exit 1
fi

# --config_name stays relative: the launcher runs train_gpt.py from $IVG, which reads the model type from the path.
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
# Use the last <run dir>/checkpoints/checkpoint_<step> directory as --wm_transformer
# (its model.safetensors holds the HeadModelWithAction weights) and $IVG/configs/llama/config_medium.json as --wm_config.
