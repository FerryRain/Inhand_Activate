#!/bin/bash

GPUS=$1
CHECKPOINT_PATH="runs/final/z-ps.pth"

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

# ================= config =================

NUM_ENVS=64
TARGET_SAMPLES=10000
SAMPLES_PER_FILE=10000
REWARD_THRESHOLD=600

SAVE_NAME="runs/rollout/teacher_data_z_ps_filtered.npz"

# =======================================

echo "Starting Filtered Data Collection..."
echo "Threshold: > ${REWARD_THRESHOLD}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train_distillation.py \
    headless=True \
    test=True \
    checkpoint=${CHECKPOINT_PATH} \
    task=AllegroArmMOAR \
    task.env.numEnvs=${NUM_ENVS} \
    task.env.objSet=C \
    task.env.axis=z \
    task.env.observationType=partial_stack \
    task.env.legacy_obs=True \
    task.env.ablation_mode=no-pc \
    experiment=z-ps \
    train.params.config.user_prefix=z-ps \
    \
    +collect_data=True \
    +samples_per_file=${SAMPLES_PER_FILE} \
    +collect_samples=${TARGET_SAMPLES} \
    +reward_threshold=${REWARD_THRESHOLD} \
    +save_name=${SAVE_NAME} \
    \
    ${EXTRA_ARGS}