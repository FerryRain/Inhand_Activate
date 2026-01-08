python ./isaacgymenvs/finetune_mlp.py \
  --data-path ./isaacgymenvs/collected_trajectories/z \
  --teacher-ckpt ./runs/final/z-ps.pth \
  --stack-num 4 \
  --epochs 50