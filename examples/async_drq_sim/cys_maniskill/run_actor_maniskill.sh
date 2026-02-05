#!/bin/bash

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
# For ManiSkill/Sapien, use CPU rendering to avoid Vulkan initialization issues
# Set to cuda:0 if you have CUDA and want GPU rendering
export MS_RENDER_DEVICE=cpu
# Only set DISPLAY if rendering is needed (when --render flag is used)
# export DISPLAY=:0

# Optional: Set IP address if learner is on a different machine
# export LEARNER_IP=localhost  # or the IP address of the learner machine
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# Run actor
python async_drq_sim_maniskill.py "$@" \
    --actor \
    # --env=PickCube-v1 \
    # --obs_mode=rgb+state \
    # --control_mode=pd_ee_delta_pose \
    # --robot_uids=panda \
    # --exp_name=pick_cube_maniskill \
    # --seed=42 \
    # --random_steps=300 \
    # --encoder_type=resnet-pretrained \
    # --max_steps=1000000 \
    # --steps_per_update=30 \
    # --log_period=10 \
    # --eval_period=2000 \
    # --eval_n_trajs=5 \
    # --ip=localhost
    # --render
