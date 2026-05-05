#!/bin/bash

# Run actor for ManiSkill environment
# For PickCube-v1 with RGB+State observation

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# python async_drq_sim_maniskill_support_potential_reward.py "$@" \
python async_drq_sim_maniskill_ty.py "$@" \
    --actor \
    --exp_name=pull_cube_old_20demo_sparse \
    --env PullCube-v1 \
    --control_mode pd_ee_delta_pose \
    --obs_mode rgb+state \
    --seed 0 \
    --random_steps 1000 \
    --max_episode_steps 100 \
    --encoder_type resnet-pretrained \
    --reward_mode sparse \
    --robot_uids panda_wristcam \
    --adaptive_tau_enabled=True 

    # --reward_mode normalized_dense \sparse