#!/usr/bin/env bash
# Builds the KDE, value function and world-model data for each LIBERO suite (LIBERO_DATA has one folder per suite).
set -euo pipefail
cd "$(dirname "$0")/.."
LIBERO_DATA=${LIBERO_DATA:?set LIBERO_DATA to the folder containing libero_spatial/, libero_object/, ...}
OUT=${OUT:-checkpoints}
ROLLOUTS=${ROLLOUTS:-rollouts/vanilla}
WM_DATA=${WM_DATA:-data/wm}

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  python experiments/fit_kde.py --data_format=libero_hdf5 --data_path=$LIBERO_DATA/$SUITE \
    --action_convention=openvla --chunk_size=1 --output=$OUT/$SUITE/kde.npz
  python experiments/train_value.py --data_format=libero_hdf5 --data_path=$LIBERO_DATA/$SUITE \
    --output=$OUT/$SUITE/value.pt
  python experiments/export_world_model_data.py --data_format=libero_hdf5 --data_path=$LIBERO_DATA/$SUITE \
    --action_convention=openvla --failure_path=$ROLLOUTS/$SUITE --chunk_size=1 --resolution=64 \
    --output_dir=$WM_DATA/$SUITE --dataset_name=vla_reasoner
done
