#!/bin/bash

# Run learner for ManiSkill environment
# For PickCube-v1 with RGB+State observation

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
python async_drq_sim_maniskill.py "$@" \
    --learner \
    --exp_name=place_sphere_100_demo_fixed_dense \
    --env PlaceSphere-v1 \
    --obs_mode rgb+state \
    --control_mode pd_ee_delta_pose \
    --seed 0 \
    --batch_size 256 \
    --encoder_type resnet-pretrained \
    --robot_uids panda_wristcam \
    --demo_path /home/caiyishuai/workspace/maniskill-ws/serl_data/mani_skill_place_dense_100.pkl 
    # --debug
    # --env PushCube-v1 \ #StackCube-v1
    # --debug

