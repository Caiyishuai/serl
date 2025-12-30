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

try:
    from reward_model.reward1223 import load_pytorch_model
except ImportError:
    # If running from examples/async_drq_sim/ directly
    sys.path.append(os.path.join(os.getcwd(), 'reward_model'))
    from reward1223 import load_pytorch_model

def process_image(img, device):
    """Process image for reward model."""
    if isinstance(img, np.ndarray) and img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Resize to 224x224 as expected by the model
    img = cv2.resize(img, (224, 224))
    
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return torch.tensor(img, device=device)

def process_state(s, device):
    """Process state for reward model."""
    if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] == 1:
        s = s[0]
    return torch.tensor(s, device=device, dtype=torch.float32)

def main():
    # Configuration
    data_path = "franka_lift_cube_image_20_trajs.pkl"
    output_path = "franka_lift_cube_image_20_trajs_relabeled.pkl"
    reward_model_path = "reward_model/reward_model3.pt"
    
    # Check files
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
    if not os.path.exists(reward_model_path):
        print(f"Error: Reward model {reward_model_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        transitions = pkl.load(f)
    
    print(f"Loaded {len(transitions)} transitions.")
    
    # Determine trajectory length
    # Assuming 20 trajectories as per filename
    num_trajs = 20
    if len(transitions) % num_trajs != 0:
        print(f"Warning: Number of transitions {len(transitions)} is not divisible by {num_trajs} trajectories.")
        # Try to infer from dones? (Found 0 dones in inspection)
        # Fallback to simple division
    
    traj_len = len(transitions) // num_trajs
    print(f"Assuming {num_trajs} trajectories with {traj_len} steps each.")

    # Load reward model
    print(f"Loading reward model from {reward_model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    reward_model = load_pytorch_model(reward_model_path, device=device)
    reward_model.eval()
    
    window_size = 3
    
    # Statistics
    old_rewards = []
    new_rewards = []
    
    print("Relabeling data...")
    for traj_idx in tqdm(range(num_trajs), desc="Trajectories"):
        start_idx = traj_idx * traj_len
        end_idx = start_idx + traj_len
        
        # Initialize queues for this trajectory
        base_img_queue = deque(maxlen=window_size)
        hand_img_queue = deque(maxlen=window_size)
        state_queue = deque(maxlen=window_size)
        episode_first_frame = True
        
        for i in range(start_idx, end_idx):
            transition = transitions[i]
            obs = transition['observations']
            
            # Extract observations
            curr_base = process_image(obs['front'], device)
            curr_hand = process_image(obs['wrist'], device)
            curr_state = process_state(obs['state'], device)
            
            # Update queues
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
            
            # Prepare batch for inference (Batch size 1, Sequence length 3)
            # shape: [1, 3, C, H, W]
            base_img_batch = torch.stack(list(base_img_queue)).unsqueeze(0)
            hand_img_batch = torch.stack(list(hand_img_queue)).unsqueeze(0)
            # shape: [1, 3, D]
            state_batch = torch.stack(list(state_queue)).unsqueeze(0)
            
            # Predict reward
            with torch.no_grad():
                reward = reward_model.predict_reward_bounded({
                    "base_img": base_img_batch,
                    "hand_img": hand_img_batch,
                    "state": state_batch
                })
            
            # Convert to scalar
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item()
            else:
                reward_val = float(reward)
            
            # Save stats
            if isinstance(transition['rewards'], (int, float, np.number)):
                old_rewards.append(float(transition['rewards']))
            else:
                # Handle array case
                old_rewards.append(float(np.mean(transition['rewards'])) if np.size(transition['rewards']) > 0 else 0.0)
                
            new_rewards.append(reward_val)
            
            # Update transition
            # Store as scalar float32 or array depending on original format?
            # Original format was scalar float32 or array scalar. 
            # Let's match typical format: np.array(r, dtype=np.float32)
            transition['rewards'] = np.array(reward_val, dtype=np.float32)

    # Print statistics
    print("\nRelabeling complete.")
    print(f"Old rewards mean: {np.mean(old_rewards):.4f}, std: {np.std(old_rewards):.4f}")
    print(f"New rewards mean: {np.mean(new_rewards):.4f}, std: {np.std(new_rewards):.4f}")
    
    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pkl.dump(transitions, f)
    print("Done.")

if __name__ == "__main__":
    main()

