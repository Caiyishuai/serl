import pickle
import numpy as np
import os

def inspect_pkl(abs_file_path):
    # Resolve absolute path relative to this script file
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # abs_file_path = os.path.join(script_dir, os.path.basename(file_path))
    
    print(f"Loading {abs_file_path}...")
    
    if not os.path.exists(abs_file_path):
        print(f"Error: File {abs_file_path} not found.")
        return

    with open(abs_file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Data type: {type(data)}")
    
    if isinstance(data, list):
        print(f"Number of trajectories: {len(data)}")
        if len(data) == 0:
            print("Data is empty.")
            return
        
        # Inspect the first trajectory
        traj = data[0]
        print("\n--- Structure of the first trajectory ---")
        print(f"Keys in trajectory: {list(traj.keys())}")
        
        # Inspect Observations
        if 'observations' in traj:
            obs = traj['observations']
            print("\n[Observations]")
            print(f"Type: {type(obs)}")
            if isinstance(obs, dict):
                print("Keys and Shapes:")
                for key, value in obs.items():
                    if hasattr(value, 'shape'):
                        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  - {key}: type={type(value)}")
            elif hasattr(obs, 'shape'):
                print(f"Shape: {obs.shape}, dtype={obs.dtype}")
            else:
                print(f"Value: {obs}")

        # Inspect Actions
        if 'actions' in traj:
            actions = traj['actions']
            print("\n[Actions]")
            print(f"Type: {type(actions)}")
            if hasattr(actions, 'shape'):
                print(f"Shape: {actions.shape}, dtype={actions.dtype}")
                print(f"Range: min={np.min(actions)}, max={np.max(actions)}")
                if len(actions) > 0:
                    print(f"Sample action (first step): {actions[0]}")
            else:
                print(f"Value: {actions}")
                
        # Inspect other keys briefly
        print("\n[Other Keys]")
        for key in traj.keys():
            if key not in ['observations', 'actions']:
                val = traj[key]
                if hasattr(val, 'shape'):
                     print(f"  - {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                     print(f"  - {key}: type={type(val)}")

    else:
        print("Data is not a list, please manually inspect.")

if __name__ == "__main__":
    # filename only, path resolution handled inside function
    file_path = "./examples/async_drq_sim/franka_lift_cube_image_20_trajs.pkl"
    inspect_pkl(file_path)
