# Auto-kill old processes using port 5488 before starting
# PORT=5488
# OLD_PIDS=$(lsof -ti:${PORT} 2>/dev/null)
# if [ -n "$OLD_PIDS" ]; then
#     echo "Found old processes on port ${PORT}: $OLD_PIDS"
#     echo "Killing old processes..."
#     kill $OLD_PIDS 2>/dev/null
#     sleep 2
#     # Force kill if still running
#     REMAINING=$(lsof -ti:${PORT} 2>/dev/null)
#     if [ -n "$REMAINING" ]; then
#         kill -9 $REMAINING 2>/dev/null
#         echo "Force killed remaining processes: $REMAINING"
#     fi
#     echo "Port ${PORT} is now free"
# fi

# export CUDA_VISIBLE_DEVICES=4,5,6,7 && \
# export CUDA_VISIBLE_DEVICES=4 && \
# export DISPLAY=:0 && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
# export MUJOCO_GL=egl && \
export JAX_PLATFORM_NAME=cuda && \
# Avoid JAX CUDA detection bug (cuda_nvcc.__file__ is None); use system CUDA
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}"
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

# python async_drq_sim.py "$@" \
#     --learner \
#     --exp_name=pick_cube_sparse_reward_demo_bt=128_no_render \
#     --seed 0 \
#     --training_starts 1000 \
#     --critic_actor_ratio 4 \
#     --encoder_type resnet-pretrained \
#     --demo_path success_trajs_20_01.pkl \

    # --render   \
    # --debug

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda-12.6

# Avoid "Illegal instruction" on some CPUs when JAX compiles (use conservative CPU backend)
export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"

python async_drq_sim_ori.py "$@" \
    --learner \
    --exp_name=stack_cube \
    --seed 0 \
    --training_starts 100 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path serl_data/mani_skill_stack_cube_20.pkl \
    --debug
    # --exp_name=pick_cube \
    # --demo_path franka_lift_cube_image_20_trajs.pkl \


    
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

