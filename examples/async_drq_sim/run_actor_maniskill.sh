#!/bin/bash

# Run actor for ManiSkill environment
# For PickCube-v1 with RGB+State observation

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_drq_sim_maniskill.py "$@" \
    --actor \
    --exp_name=place_sphere_100_demo \
    --env PlaceSphere-v1 \
    --control_mode pd_ee_delta_pose \
    --obs_mode rgb+state \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --robot_uids panda_wristcam
    # --obs_mode rgb+state \ PickCube-v1
