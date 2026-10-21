#!/usr/bin/env bash
# Builds the KDE, value function and world-model data for the WidowX tasks from BridgeData V2.
set -euo pipefail
cd "$(dirname "$0")/.."
BRIDGE=${BRIDGE:?set BRIDGE to the TFDS directory containing bridge_dataset/1.0.0}
OUT=${OUT:-checkpoints}
ROLLOUTS=${ROLLOUTS:-rollouts/vanilla}
WM_DATA=${WM_DATA:-data/wm}

python experiments/train_value.py --data_format=bridge_rlds --data_path=$BRIDGE --output=$OUT/bridge/value.pt

python experiments/fit_kde.py --data_format=bridge_rlds --data_path=$BRIDGE --action_convention=octo \
  --chunk_size=1 --output=$OUT/bridge/kde_octo.npz
python experiments/export_world_model_data.py --data_format=bridge_rlds --data_path=$BRIDGE \
  --action_convention=octo --chunk_size=1 --resolution=64 --failure_path=$ROLLOUTS/octo \
  --output_dir=$WM_DATA/bridge_octo --dataset_name=vla_reasoner

python experiments/fit_kde.py --data_format=bridge_rlds --data_path=$BRIDGE --action_convention=bridge_orig \
  --chunk_size=4 --output=$OUT/bridge/kde_spatialvla_c4.npz
python experiments/export_world_model_data.py --data_format=bridge_rlds --data_path=$BRIDGE \
  --action_convention=bridge_orig --chunk_size=4 --resolution=64 --failure_path=$ROLLOUTS/spatialvla \
  --output_dir=$WM_DATA/bridge_spatialvla_c4 --dataset_name=vla_reasoner
