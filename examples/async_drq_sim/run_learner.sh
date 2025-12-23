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
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
export MUJOCO_GL=egl && \
# export JAX_PLATFORM_NAME=gpu && \
# export CUDA_HOME=/usr/local/cuda && \
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} && \
# export JAX_TRACEBACK_FILTERING=off && \
# Configure wandb for this script only (temporary, won't affect system defaults)
# Option 1: Set WANDB_API_KEY here (temporary, only for this script run)
# Get API key from: https://wandb.ai/authorize (login with 1274775234@qq.com / caiyishuai)
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# Option 2: Use default wandb login (if already logged in system)
python async_drq_sim.py "$@" \
    --learner \
    --render   \
    --exp_name=drq_demo_dense_bt=128 \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --demo_path franka_lift_cube_image_20_trajs.pkl \
    # --debug # wandb is disabled when debug
