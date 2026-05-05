#!/bin/bash
# Run ACTOR: ReFlow residual RL for PushCube-v1
#
# Prerequisites:
#   1. Learner is already running (run_learner.sh)
#   2. ReFlow checkpoint exists at $REFLOW_CKPT
#
# Usage:
#   REFLOW_CKPT=runs/PushCube-v1__reflow__42__xxx/checkpoints/best_success_once.pt \
#   bash examples/reflow_res_rl/run_actor.sh

set -e

REFLOW_CKPT="${REFLOW_CKPT:-/home/caiyishuai/workspace/maniskill-ws/runs/PushCube-v1__reflow__42__1773947269/checkpoints/best_success_once.pt}"

export DISPLAY=:1
export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export WANDB_API_KEY="${WANDB_API_KEY}"

python "$(dirname "$0")/async_drq_sim.py" \
    --actor \
    --env PushCube-v1 \
    --obs_mode "rgb+state" \
    --control_mode pd_ee_delta_pose \
    --robot_uids panda_wristcam \
    --reward_mode normalized_dense \
    --max_episode_steps 200 \
    --use_residual True \
    --reflow_ckpt "${REFLOW_CKPT}" \
    --reflow_obs_horizon 2 \
    --reflow_pred_horizon 4 \
    --reflow_act_horizon 4 \
    --alpha_scale 0.1 \
    --encoder_type resnet-pretrained \
    --random_steps 300 \
    --steps_per_update 30 \
    --max_steps 1000000 \
    --eval_period 2000 \
    --eval_n_trajs 5 \
    --potential_reward_shaping True \
    --server_port 5490 \
    --broadcast_port 5491 \
    --seed 42 \
    "$@"
