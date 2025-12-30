import os
import sys
import pickle as pkl
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import gymnasium as gym  # Use gymnasium (new API)

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# Import franka_sim to register environments with gymnasium
import franka_sim

# Fix for JAX CUDA detection issue - must be set before any JAX imports
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

# Workaround for cuda_nvcc.__file__ being None
try:
    import types
    # Import nvidia package if available
    try:
        from nvidia import cuda_nvcc
    except ImportError:
        # Create nvidia package if it doesn't exist
        if 'nvidia' not in sys.modules:
            nvidia = types.ModuleType('nvidia')
            sys.modules['nvidia'] = nvidia
        else:
            nvidia = sys.modules['nvidia']
            
        cuda_nvcc = types.ModuleType('cuda_nvcc')
        if 'nvidia.cuda_nvcc' not in sys.modules:
            sys.modules['nvidia.cuda_nvcc'] = cuda_nvcc
        nvidia.cuda_nvcc = cuda_nvcc

    if not hasattr(cuda_nvcc, '__file__') or cuda_nvcc.__file__ is None:
        cuda_nvcc.__file__ = '/tmp/cuda_nvcc_workaround.py'
except ImportError:
    pass

# Set HF Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"

try:
    from reward_model.reward1223 import load_pytorch_model
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'reward_model'))
    from reward1223 import load_pytorch_model

def process_image(img, device):
    """Process image for reward model."""
    if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    
    return torch.tensor(img, device=device)

def process_state(s, device):
    """Process state for reward model."""
    if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] == 1:
        s = s[0]
    return torch.tensor(s, device=device, dtype=torch.float32)

def main():
    # Configuration
    input_path = "examples/async_drq_sim/success_trajs_100.pkl"
    output_pkl_path = "examples/async_drq_sim/franka_lift_cube_image_20_trajs_new_reward3.pkl"
    output_plot_path = "examples/async_drq_sim/avg_reward_curve.png"
    reward_model_path = "examples/async_drq_sim/reward_model/reward_model3.pt"
    
    # 开关: 是否将环境的原始reward累加到reward_model的输出上
    # 最终reward = reward_model输出 + 环境reward
    add_env_reward = True
    # add_reward_model_reward = False
    env_name = "PandaPickCubeVision-v0"
    
    # Check paths
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    if not os.path.exists(reward_model_path):
        print(f"Error: {reward_model_path} not found.")
        return

    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path, "rb") as f:
        all_trajs = pkl.load(f)
    
    # Take first 20 trajectories
    trajs = all_trajs[:20]
    print(f"Processing {len(trajs)} trajectories.")

    # Load reward model
    print(f"Loading reward model from {reward_model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    reward_model = load_pytorch_model(reward_model_path, device=device)
    reward_model.eval()
    
    window_size = 3
    all_traj_rewards = []
    
    print("Relabeling trajectories...")
    
    env = None
    if add_env_reward:
        # 先检查轨迹中是否已有保存的reward
        has_saved_rewards = all('rewards' in traj for traj in trajs)
        
        if not has_saved_rewards:
            # 只有在需要重新计算环境reward时才创建环境
            print(f"Creating environment {env_name} to compute rewards...")
            try:
                env = gym.make(env_name)
                print(f"Environment {env_name} created successfully.")
            except Exception as e:
                print(f"Warning: Could not create environment {env_name}: {e}")
                print("Will try to use saved rewards from trajectories if available.")
                add_env_reward = False
        else:
            print("Using saved rewards from trajectories (no need to create environment).")
            print("Saved rewards are the original environment rewards.")

    for traj_idx, traj in enumerate(tqdm(trajs)):
        observations = traj['observations']
        actions = traj.get('actions', [])
        traj_rewards = []
        
        # 如果开启了环境奖励，准备环境以便计算环境奖励
        # （如果轨迹数据中已包含原始奖励，则无需执行环境；否则需要重置环境并执行动作）
        if add_env_reward and env is not None:
            env.reset()
            # 如果轨迹中保存了物体的初始位置信息，将其应用到环境中
            if 'initial_object_pos' in traj and traj['initial_object_pos'] is not None:
                init_pos = traj['initial_object_pos']
                try:
                    # 使用 unwrapped 访问底层 Mujoco 数据并设置物体位置
                    # 假设物体关节名称为 "block"
                    env.unwrapped.data.jnt("block").qpos[:3] = init_pos
                    import mujoco
                    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
                except Exception as e:
                    print(f"Warning: Could not set initial object position for trajectory {traj_idx}: {e}")
        
        # Queues
        base_img_queue = deque(maxlen=window_size)
        hand_img_queue = deque(maxlen=window_size)
        state_queue = deque(maxlen=window_size)
        episode_first_frame = True
        
        for i, obs in enumerate(observations):
            curr_base = process_image(obs['front'], device)
            curr_hand = process_image(obs['wrist'], device)
            curr_state = process_state(obs['state'], device)
            
            if episode_first_frame:
                base_img_queue.clear()
                hand_img_queue.clear()
                state_queue.clear()
                for _ in range(window_size):
                    base_img_queue.append(curr_base)
                    hand_img_queue.append(curr_hand)
                    state_queue.append(curr_state)
                episode_first_frame = False
            else:
                base_img_queue.append(curr_base)
                hand_img_queue.append(curr_hand)
                state_queue.append(curr_state)
            
            # Predict
            base_batch = torch.stack(list(base_img_queue)).unsqueeze(0)
            hand_batch = torch.stack(list(hand_img_queue)).unsqueeze(0)
            state_batch = torch.stack(list(state_queue)).unsqueeze(0)
            
            with torch.no_grad():
                reward = reward_model.predict_reward_bounded({
                    "base_img": base_batch,
                    "hand_img": hand_batch,
                    "state": state_batch
                })
            
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item()
            else:
                reward_val = float(reward)
            
            # 添加环境奖励到 reward_val
            if add_env_reward:
                # 优先使用轨迹中已保存的原始环境奖励
                if 'rewards' in traj and i < len(traj['rewards']):
                    env_reward = float(traj['rewards'][i])
                    reward_val += env_reward
                # 如果没有保存的奖励且环境已创建，通过执行动作重新计算
                elif env is not None and i < len(actions):
                    action = actions[i]
                    _, env_reward, _, _, _ = env.step(action)
                    reward_val += float(env_reward)
                
            traj_rewards.append(reward_val)
        
        # Update rewards in trajectory
        traj['rewards'] = np.array(traj_rewards, dtype=np.float32)
        all_traj_rewards.append(traj_rewards)

    # Save to new pickle file
    print(f"Saving relabeled data to {output_pkl_path}...")
    with open(output_pkl_path, "wb") as f:
        pkl.dump(trajs, f)
    
    # Plotting
    print("Generating plot...")
    
    # Determine max length
    max_len = max(len(r) for r in all_traj_rewards)
    
    # Create array with NaNs
    reward_matrix = np.full((len(trajs), max_len), np.nan)
    
    for i, r in enumerate(all_traj_rewards):
        reward_matrix[i, :len(r)] = r
        
    # Compute mean and std ignoring NaNs
    means = np.nanmean(reward_matrix, axis=0)
    stds = np.nanstd(reward_matrix, axis=0)
    steps = np.arange(max_len)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, label='Average Reward', color='blue')
    plt.fill_between(steps, means - stds, means + stds, alpha=0.2, color='blue', label='Std Dev')
    
    title = f'Average Reward over Time (First {len(trajs)} Trajectories)'
    if add_env_reward:
        title += ' [Model + Env Reward]'
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")
    
    print("All tasks complete.")

if __name__ == "__main__":
    main()

