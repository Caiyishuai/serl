# export CUDA_VISIBLE_DEVICES=4,5,6,7 && \
# # export CUDA_VISIBLE_DEVICES=4 && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export MUJOCO_GL=glfw && \
export DISPLAY=:0 && \
# export HF_ENDPOINT=https://hf-mirror.com && \
# export JAX_PLATFORM_NAME=gpu && \
# export CUDA_HOME=/usr/local/cuda && \
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} && \
# export JAX_TRACEBACK_FILTERING=off && \

# xxx
python async_drq_sim.py "$@" \
    --actor \
    --exp_name=baseline_01reward_no_demo_40cm_bt=128 \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --render
    # --debug



# export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN=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
# python async_drq_sim_new_reward.py "$@" \
#     --actor \
#     --exp_name=drq_demo_mew+01reward_bt=128 \
#     --seed 0 \
#     --random_steps 1000 \
#     --encoder_type resnet-pretrained \
#     --render
    # --debug
