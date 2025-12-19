#!/bin/bash
GPUS=$1

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/distillation/distill_from_rollouts.py \
  --rollout-file ./runs/rollout/z_ps/ \
  --use-transformer \
  --seq-len 30 \
  --qpos-dim 16 \
  --batch-size 16 \
  --epochs 200 \
  --normalize