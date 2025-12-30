import gymnasium as gym
import numpy as np
import pickle as pkl
from tqdm import tqdm
import copy
import sys
import os
import cv2

# Ensure we can import from local modules
# We need to add the parent directory of the franka_sim package to sys.path
# The structure is serl/franka_sim/franka_sim
# so we add serl/franka_sim to path.
sys.path.insert(0, os.path.join(os.getcwd(), "franka_sim"))

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv

def generate_demos():
    # Initialize environment with image observations enabled
    # Action scale is [0.1, 1] by default which means:
    # Position delta = action * 0.1
    # Gripper delta = action * 1.0
    # Use rgb_array to avoid EGL/GLFW context conflicts (EGLError)
    # We will manually visualize using cv2.imshow
    env = PandaPickCubeGymEnv(
        render_mode="rgb_array", 
        image_obs=True,
        control_dt=0.02, 
    )

    transitions = []
    success_count = 0
    target_success = 100
    
    print(f"Generating {target_success} successful trajectories...")
    pbar = tqdm(total=target_success)
    
    while success_count < target_success:
        obs, _ = env.reset()
        episode_transitions = []
        
        # Access internal data to cheat for motion planning
        data = env.unwrapped.data
        
        # Get block position (x, y, z)
        # We target slightly above the block center Z for the grasp
        block_pos = data.sensor("block_pos").data.copy()
        
        # Define waypoints (stages)
        # Stage 0: Hover above block
        hover_pos = block_pos.copy()
        hover_pos[2] += 0.02
        
        # Stage 1: Move down to block
        grasp_pos = block_pos.copy()
        # block_z is usually around 0.02 (half size). 
        # Pinch site needs to be around there. 
        # Let's target the exact block center returned by sensor.
        
        # Stage 2: Lift
        lift_pos = block_pos.copy()
        lift_pos[2] += 0.6
        
        # Stages list: (Target Position, Gripper Value, Wait Steps)
        # Gripper: -1 for open, 1 for close -> NO.
        # Env logic: new_g = old_g + action. clipped [0, 1].
        # We assume 0.0 is CLOSED (width 0), 1.0 is OPEN (width max).
        # We want to CLOSE to grasp.
        
        stages = [
            {"target": hover_pos, "gripper": 1.0, "wait": 10}, # Ensure open
            {"target": grasp_pos, "gripper": 1.0, "wait": 10}, # Go down open
            {"target": grasp_pos, "gripper": 0.0, "wait": 30}, # Close gripper and wait
            {"target": lift_pos, "gripper": 0.0, "wait": 20},  # Lift closed
        ]
        
        current_stage_idx = 0
        wait_counter = 0
        
        # Max steps per episode
        for step in range(500):
            # Get current TCP pos from observation
            tcp_pos = obs["state"]["panda/tcp_pos"]
            # gripper_pos might be a 0-dim array or 1-dim array
            gripper_pos = obs["state"]["panda/gripper_pos"]
            if gripper_pos.shape == (1,):
                gripper_pos = gripper_pos[0]
            # If 0-dim, already scalar
            
            if current_stage_idx >= len(stages):
                break
                
            stage = stages[current_stage_idx]
            target = stage["target"]
            gripper_target = stage["gripper"]
            wait_steps = stage["wait"]
            
            # Calculate position error
            pos_error = target - tcp_pos
            dist = np.linalg.norm(pos_error)
            
            # Check if we reached the target (within tolerance)
            # Tolerance: 1cm for position
            # For gripper: check if close to target
            gripper_error = gripper_target - gripper_pos
            
            # Check transition conditions
            pos_reached = dist < 0.01
            gripper_reached = abs(gripper_error) < 0.1
            
            # If we are in a 'wait' phase (gripper change or settle), just count
            # But we only start waiting if we are at the target
            if pos_reached: 
                 # If this stage is mainly for gripper change, we also check gripper
                 if wait_counter < wait_steps:
                     wait_counter += 1
                 else:
                     # Only proceed if gripper is also roughly there (unless we don't care)
                     if gripper_reached or wait_steps > 0: # If we waited, we assume done
                         current_stage_idx += 1
                         wait_counter = 0
                         if current_stage_idx >= len(stages):
                             break
                         stage = stages[current_stage_idx]
                         target = stage["target"]
                         gripper_target = stage["gripper"]
                         wait_steps = stage["wait"]
            
            # Calculate action
            # Position control
            action_pos = (target - tcp_pos) / 0.1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            
            # Gripper control
            # Target is absolute [0, 1]. Env takes delta.
            # action = target - current
            action_gripper = gripper_target - gripper_pos
            
            action = np.concatenate([action_pos, [action_gripper]])
            
            next_obs, rew, done, truncated, info = env.step(action)
            
            # Debug info (print every 50 steps or if important)
            if step % 50 == 0:
                print(f"Step {step}: Stage {current_stage_idx}, Dist {dist:.3f}, Gripper {gripper_pos:.2f} -> {gripper_target}, Z {tcp_pos[2]:.3f}/{block_pos[2]:.3f}")
            
            transition = {
                "observations": obs,
                "actions": action,
                "next_observations": next_obs,
                "rewards": rew,
                "masks": 1.0 - float(done),
                "dones": done,
                "infos": info # Save info which might contain 'succeed'
            }
            episode_transitions.append(transition)
            
            obs = next_obs
            
            # Manual visualization since we are in rgb_array mode
            if "images" in obs and "front" in obs["images"]:
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(obs["images"]["front"], cv2.COLOR_RGB2BGR)
                cv2.imshow("Panda Sim", img_bgr)
                cv2.waitKey(1)

            if done:
                print(f"Episode finished. Success: {info.get('succeed', False)}")
                break
        
        # Check success
        # The environment has a binary reward if block is lifted
        # Or we can check block height manually
        # env._z_success is the threshold
        final_block_z = env.unwrapped.data.sensor("block_pos").data[2]
        success_threshold = env.unwrapped._z_success
        
        if final_block_z >= success_threshold:
            print("Success!")
            transitions.extend(episode_transitions)
            success_count += 1
            pbar.update(1)
        else:
            print(f"Failed. Final block z: {final_block_z}, Threshold: {success_threshold}")
            pass
            
    pbar.close()
    
    # Save to file
    output_file = "generated_100_trajs.pkl"
    print(f"Saving {len(transitions)} transitions to {output_file}...")
    with open(output_file, "wb") as f:
        pkl.dump(transitions, f)
    print("Done.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_demos()

