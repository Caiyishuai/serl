#!/bin/bash
# Run LEARNER: ReFlow residual RL for PushCube-v1
#
# Usage:
#   bash examples/reflow_res_rl/run_learner.sh
#
# With residual demos (optional, improves sample efficiency):
#   DEMO_PATH=examples/reflow_res_rl/demo_data/pushcube_residual.pkl \
#   bash examples/reflow_res_rl/run_learner.sh

set -e

DEMO_PATH="${DEMO_PATH:-}"   # leave empty to train without demos

export DISPLAY=:1
export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4
export WANDB_API_KEY="${WANDB_API_KEY}"

DEMO_ARGS=""
if [ -n "$DEMO_PATH" ]; then
    DEMO_ARGS="--demo_path ${DEMO_PATH} --demo_batch_size 32"
fi

python "$(dirname "$0")/async_drq_sim.py" \
    --learner \
    --env PushCube-v1 \
    --obs_mode "rgb+state" \
    --control_mode pd_ee_delta_pose \
    --robot_uids panda_wristcam \
    --reward_mode normalized_dense \
    --max_episode_steps 200 \
    --use_residual True \
    --reflow_ckpt /home/caiyishuai/workspace/maniskill-ws/runs/PushCube-v1__reflow__42__1773947269/checkpoints/best_success_once.pt \
    --encoder_type resnet-pretrained \
    --exp_name pushcube_reflow_residual \
    --batch_size 256 \
    --random_steps 300 \
    --training_starts 300 \
    --steps_per_update 30 \
    --critic_actor_ratio 4 \
    --max_steps 1000000 \
    --replay_buffer_capacity 200000 \
    --log_period 10 \
    --eval_period 2000 \
    --potential_reward_shaping True \
    --checkpoint_path /tmp/serl_ckpt/pushcube_reflow_residual \
    --checkpoint_period 5000 \
    --server_port 5490 \
    --broadcast_port 5491 \
    --seed 42 \
    $DEMO_ARGS \
    "$@"
