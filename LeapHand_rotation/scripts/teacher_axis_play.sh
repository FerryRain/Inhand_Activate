GPUS=$1

CHECKPOINT_PATH="runs/final/y-ps.pth"
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py \
    headless=False \
    test=True \
    checkpoint=${CHECKPOINT_PATH} \
    task=AllegroArmMOAR \
    task.env.numEnvs=16 \
    task.env.objSet=C \
    task.env.axis=y \
    task.env.observationType=partial_stack \
    task.env.legacy_obs=True \
    task.env.ablation_mode=no-pc \
    experiment=y-ps \
    train.params.config.user_prefix=y-ps \
${EXTRA_ARGS}

#headless=False test=True  checkpoint=runs/final/z-ps.pth   task=AllegroArmMOAR     task.env.numEnvs=1     task.env.objSet=C     task.env.axis=z     task.env.observationType=partial_stack     task.env.legacy_obs=True     task.env.ablation_mode=no-pc     experiment=z-ps     train.params.config.user_prefix=z-ps 