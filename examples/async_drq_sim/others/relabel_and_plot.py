import os
import sys
import pickle as pkl
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

# Set HF Mirror to ensure connectivity
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Uncomment and set your HF token here to access gated models (like dinov3)
os.environ["HF_TOKEN"] = "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"

# Fix for JAX CUDA detection issue - must be set before any JAX imports
# This prevents the cuda_nvcc.__file__ NoneType error
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

# Workaround for cuda_nvcc.__file__ being None (namespace package issue)
# JAX imports cuda_nvcc from nvidia package, so we need to patch nvidia.cuda_nvcc
try:
    import sys
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

    # If module exists but __file__ is None, set it to a dummy value
    if not hasattr(cuda_nvcc, '__file__') or cuda_nvcc.__file__ is None:
        cuda_nvcc.__file__ = '/tmp/cuda_nvcc_workaround.py'
except ImportError:
    pass

# Move imports that depend on HF environment variables after setting them
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
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    # Normalize
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
    reward_model_path = "examples/async_drq_sim/reward_model/reward_model3.pt"
    plot_output_path = "examples/async_drq_sim/avg_reward_curve.png"
    
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
    for traj_idx, traj in enumerate(tqdm(trajs)):
        observations = traj['observations']
        traj_rewards = []
        
        # Queues
        base_img_queue = deque(maxlen=window_size)
        hand_img_queue = deque(maxlen=window_size)
        state_queue = deque(maxlen=window_size)
        episode_first_frame = True
        
        # We need to process each step
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
                
            traj_rewards.append(reward_val)
        
        # Update rewards in trajectory (optional, but requested "relabel")
        # Since 'rewards' in pkl was list of scalars or array, we overwrite
        # Note: Original rewards list might be length N-1 or N depending on implementation
        # Here we have N rewards for N observations.
        traj['rewards'] = np.array(traj_rewards, dtype=np.float32)
        all_traj_rewards.append(traj_rewards)

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
    
    plt.title(f'Average Reward over Time (First {len(trajs)} Trajectories)')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(plot_output_path)
    print(f"Plot saved to {plot_output_path}")

if __name__ == "__main__":
    main()

