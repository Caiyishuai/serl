import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Error: 'lerobot' library not found. Please install it using `pip install lerobot`.")
    exit(1)

def load_pkl_data(pkl_path):
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trajectories.")
    return data

def convert_serl_to_lerobot(pkl_path, repo_id, root=None, fps=10, robot_type="panda"):
    data = load_pkl_data(pkl_path)
    
    if not data:
        print("No data found in pickle file.")
        return

    # Define features based on the first frame of the first trajectory
    first_traj = data[0]
    first_obs = first_traj['observations'][0]
    first_action = first_traj['actions'][0]
    
    # Inspect shapes to define features
    # Images in SERL pkl are often (1, H, W, C), need to squeeze
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (first_obs['state'].squeeze().shape),
            "names": ["joint_pos"] # Simplified name
        },
        "action": {
            "dtype": "float32",
            "shape": (first_action.shape),
            "names": ["joint_vel"] # Simplified name, modify if position/torque
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        },
        "observation.images.front": {
             "dtype": "video",
             "shape": first_obs['front'].squeeze().shape,
             "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
             "dtype": "video",
             "shape": first_obs['wrist'].squeeze().shape,
             "names": ["height", "width", "channels"],
        }
    } 

    # Create the dataset instance
    # Note: We need to define the directory where the dataset will be created.
    if root is None:
        root = Path(f"data/{repo_id}")
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True, # Encode images as videos
        root=root
    )

    print("Converting trajectories...")
    for episode_idx, traj in enumerate(tqdm(data)):
        observations = traj['observations']
        actions = traj['actions']
        dones = traj['dones']
        
        num_frames = len(observations)
        
        for i in range(num_frames):
            # Process observations
            obs = observations[i]
            
            # Squeeze (1, H, W, C) to (H, W, C)
            front_img = obs['front'].squeeze()
            wrist_img = obs['wrist'].squeeze()
            state = obs['state'].squeeze().astype(np.float32)
            
            action = actions[i].astype(np.float32)
            
            # Create frame dictionary
            frame = {
                "observation.images.front": front_img,
                "observation.images.wrist": wrist_img,
                "observation.state": state,
                "action": action,
                "next.done": np.array([bool(dones[i])]), # Expecting numpy array for bool
                "task": "franka_lift_cube",  # Add dummy task since we don't have task descriptions
            }
            
            dataset.add_frame(frame)
            
        # Signal end of episode
        dataset.save_episode()
            
    print("Consolidating dataset...")
    dataset.finalize()
    print(f"Dataset saved to {root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SERL pickle to LeRobot dataset")
    parser.add_argument("--pkl_path", type=str, default="examples/async_drq_sim/success_trajs_100.pkl", help="Path to input pickle file")
    parser.add_argument("--repo_id", type=str, default="serl/custom_dataset", help="Hugging Face repo ID or local dir name")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--robot_type", type=str, default="panda", help="Robot type")
    
    args = parser.parse_args()
    
    convert_serl_to_lerobot(
        pkl_path=args.pkl_path,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type
    )

# python pkl_to_lerobot.py --pkl_path examples/async_drq_sim/success_trajs_100.pkl --repo_id serl/franka_lift_cube_success_trajs_100_lerobot
# rm -rf data/serl/franka_lift_cube_success_trajs_100_lerobot