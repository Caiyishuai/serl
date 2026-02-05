#!/bin/bash

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

# Set environment variables for headless mode (no rendering needed for learner)
# Unset DISPLAY to avoid rendering issues
unset DISPLAY

# JAX memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6

# For ManiSkill/Sapien, use CPU rendering for headless mode to avoid CUDA conflicts
export MS_RENDER_DEVICE=cpu

# Memory allocator settings to avoid double free errors
export MALLOC_CHECK_=0
export MALLOC_PERTURB_=0

# Set PyTorch to use CPU only (avoids CUDA conflicts with JAX)
# This is done in Python code, but we can also set it here as a fallback
export TORCH_USE_CUDA_DSA=0

# Optional: Set WANDB API key if needed
# export WANDB_API_KEY=your_wandb_api_key_here
export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e && \
# Run learner
python async_drq_sim_maniskill.py "$@" \
    --learner \
    # --env=PickCube-v1 \
    # --obs_mode=rgb+state \
    # --control_mode=pd_ee_delta_pose \
    # --robot_uids=panda \
    # --exp_name=pick_cube_maniskill \
    # --seed=42 \
    # --training_starts=300 \
    # --critic_actor_ratio=4 \
    # --encoder_type=resnet-pretrained \
    # --batch_size=256 \
    # --max_steps=1000000 \
    # --replay_buffer_capacity=200000 \
    # --random_steps=300 \
    # --steps_per_update=30 \
    # --log_period=10 \
    # --eval_period=2000 \
    # --eval_n_trajs=5
    # --demo_path=path_to_demo.pkl \
    # --checkpoint_period=10000 \
    # --checkpoint_path=./checkpoints \
    # --debug
