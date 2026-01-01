# Auto-kill old processes using port 5488 before starting
PORT=5488
OLD_PIDS=$(lsof -ti:${PORT} 2>/dev/null)
if [ -n "$OLD_PIDS" ]; then
    echo "Found old processes on port ${PORT}: $OLD_PIDS"
    echo "Killing old processes..."
    kill $OLD_PIDS 2>/dev/null
    sleep 2
    # Force kill if still running
    REMAINING=$(lsof -ti:${PORT} 2>/dev/null)
    if [ -n "$REMAINING" ]; then
        kill -9 $REMAINING 2>/dev/null
        echo "Force killed remaining processes: $REMAINING"
    fi
    echo "Port ${PORT} is now free"
fi

# export CUDA_VISIBLE_DEVICES=4,5,6,7 && \
# export CUDA_VISIBLE_DEVICES=4 && \
export DISPLAY=:0 && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
export MUJOCO_GL=egl && \
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# Option 2: Use default wandb login (if already logged in system)
# python async_drq_sim.py "$@" \
#     --learner \
#     --render   \
#     --exp_name=baseline_dense_reward_with_demo_40cm_bt=128 \
#     --seed 0 \
#     --training_starts 1000 \
#     --critic_actor_ratio 4 \
#     --encoder_type resnet-pretrained \
#     --demo_path success_trajs_20_40cm_dense_reward.pkl \
    # --debug # wandb is disabled when debug

python async_drq_sim.py "$@" \
    --learner \
    --render   \
    --exp_name=baseline_01reward_no_demo_40cm_bt=128 \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path success_trajs_20_40cm_dense_reward.pkl \
    # --debug # wandb is disabled when debug

    
# python async_drq_sim_new_reward.py "$@" \
#     --learner \
#     --render   \
#     --exp_name=drq_demo_mew+01reward_bt=128 \
#     --seed 0 \
#     --training_starts 1000 \
#     --critic_actor_ratio 4 \
#     --encoder_type resnet-pretrained \
#     --demo_path franka_lift_cube_image_20_trajs_new_reward2.pkl \
#     # --debug # wandb is disabled when debug

