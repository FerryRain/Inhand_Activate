#!/bin/bash
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

GPUS=$1

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task=AllegroArmMOAR \
task.env.objSet=C task.env.axis=z \
task.env.numEnvs=4096 \
task.env.observationType=partial_stack \
task.env.legacy_obs=True \
task.env.ablation_mode=no-pc \
+train.algo=DemonTrain \
+train.demon_path="./runs/final/z-ps.pth" \
+train.ppo.distill=True \
+train.ppo.is_demon=True \
+train.ppo.learning_rate=1e-3 \
+train.ppo.input_mode=proprio \
+train.ppo.priv_info=False \
+train.ppo.proprio_mode=True \
+train.ppo.proprio_len=30 \
+train.ppo.use_l1=True \
+train.ppo.enable_latent_loss=False \
train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
+train.ppo.output_name=AllegroArmMOAR/student_distill \
experiment=z-ps-student \
wandb_activate=False \
${EXTRA_ARGS}