#!/bin/bash

# Run learner for ManiSkill environment
# For PickCube-v1 with RGB+State observation
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda-12.6

# Avoid "Illegal instruction" on some CPUs when JAX compiles (use conservative CPU backend)
export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"
# export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
# export MUJOCO_GL=egl && \
export JAX_PLATFORM_NAME=cuda && \
# Avoid JAX CUDA detection bug (cuda_nvcc.__file__ is None); use system CUDA
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}"
# export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# Option 2: Use default wandb login (if already logged in system)

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# python async_drq_sim_maniskill_support_potential_reward.py"$@" \
# python async_drq_sim_maniskill_ty.py "$@" \
#     --learner \
#     --exp_name=push_cube_old_20demo \
#     --env PushCube-v1 \
#     --obs_mode rgb+state \
#     --control_mode pd_ee_delta_pose \
#     --seed 0 \
#     --batch_size 128 \
#     --encoder_type resnet-pretrained \
#     --robot_uids panda_wristcam \
#     --reward_mode normalized_dense \
#    --adaptive_tau_enabled=True \
#    --demo_path /home/caiyishuai/workspace/maniskill-ws/serl_data/mani_skill_push_cube_normalized_dense_pbr_no_clip_100_fixed_complete.pkl

python async_drq_sim_maniskill_ty.py "$@" \
    --learner \
    --exp_name=pull_cube_old_20demo_sparse \
    --env PullCube-v1 \
    --obs_mode rgb+state \
    --control_mode pd_ee_delta_pose \
    --seed 0 \
    --batch_size 128 \
    --encoder_type resnet-pretrained \
    --robot_uids panda_wristcam \
    --reward_mode sparse \
   --adaptive_tau_enabled=True \
   --demo_path /home/caiyishuai/workspace/maniskill-ws/serl_data/mani_skill_pull_cube_sparse_no_clip_100_fixed_complete.pkl