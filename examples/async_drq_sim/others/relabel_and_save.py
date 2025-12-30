import os
import sys
import pickle as pkl
import numpy as np
import torch
import cv2
from collections import deque
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

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
    output_path = "examples/async_drq_sim/franka_lift_cube_image_20_trajs_new_reward.pkl"
    reward_model_path = "examples/async_drq_sim/reward_model/reward_model3.pt"
    
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
    
    print("Relabeling trajectories...")
    for traj_idx, traj in enumerate(tqdm(trajs)):
        observations = traj['observations']
        traj_rewards = []
        
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
                
            traj_rewards.append(reward_val)
        
        # Update rewards in trajectory
        # Convert to numpy array of float32 to match expected format
        traj['rewards'] = np.array(traj_rewards, dtype=np.float32)

    # Save to new pickle file
    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pkl.dump(trajs, f)
    print("Done.")

if __name__ == "__main__":
    main()

