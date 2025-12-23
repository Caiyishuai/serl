#!/usr/bin/env python3

import os
import sys
import types

# Fix for JAX CUDA detection issue - must be set before any JAX imports
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

try:
    from nvidia import cuda_nvcc
    # If module exists but __file__ is None, set it to a dummy value
    if not hasattr(cuda_nvcc, '__file__') or cuda_nvcc.__file__ is None:
        cuda_nvcc.__file__ = '/tmp/cuda_nvcc_workaround.py'
except ImportError:
    # Create nvidia package and cuda_nvcc module if they don't exist
    if 'nvidia' not in sys.modules:
        nvidia = types.ModuleType('nvidia')
        sys.modules['nvidia'] = nvidia
    else:
        nvidia = sys.modules['nvidia']
    
    cuda_nvcc = types.ModuleType('cuda_nvcc')
    cuda_nvcc.__file__ = '/tmp/cuda_nvcc_workaround.py'
    nvidia.cuda_nvcc = cuda_nvcc
    sys.modules['nvidia.cuda_nvcc'] = cuda_nvcc

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from absl import app, flags
from flax.training import checkpoints
import pickle as pkl
from tqdm import tqdm

import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.launcher import make_drq_agent
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("checkpoint_path", "examples/async_drq_sim/checkpoints/checkpoint_562000", "Path to the checkpoint directory.")
flags.DEFINE_string("reward_type", "binary", "Reward type: 'dense' or 'binary'.")
flags.DEFINE_boolean("save_trajs", True, "Whether to save trajectories.")
flags.DEFINE_boolean("save_success_only", False, "Whether to save only successful trajectories.")
flags.DEFINE_integer("num_episodes", 100, "Number of episodes to evaluate.")
flags.DEFINE_string("output_path", os.path.join(os.path.dirname(__file__), "success_trajs_{}.pkl"), "Path to save the collected data. Use {} as placeholder for num_episodes.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_boolean("render", True, "Whether to render the environment.")
flags.DEFINE_boolean("save_video", True, "Whether to save the evaluation video.")
flags.DEFINE_string("video_path", os.path.join(os.path.dirname(__file__), "eval_video.mp4"), "Path to save the video.")
flags.DEFINE_boolean("pause_on_success", True, "Whether to pause for 0.5s on success.")
flags.DEFINE_integer("target_success_count", 100, "Target number of successful trajectories to collect.")

def main(_):
    if not FLAGS.checkpoint_path:
        # Fallback if flag is somehow None
        FLAGS.checkpoint_path = "examples/async_drq_sim/checkpoints/checkpoint_562000"

    # Use GPU if available
    if jax.default_backend() == "cpu":
        print("Warning: JAX is using CPU. Inference might be slow.")

    # Create Environment
    # Note: We pass reward_type via kwargs which gym.make passes to the env constructor
    print(f"Creating environment {FLAGS.env} with reward_type={FLAGS.reward_type}")
    
    # Manually register if needed - forcing import and registration
    try:
        env_spec = gym.spec(FLAGS.env)
    except gym.error.NameNotFound:
        print(f"Environment {FLAGS.env} not found. Attempting explicit registration...")
        import franka_sim
        from gymnasium.envs.registration import register
        
        # Explicitly register again to be sure
        if FLAGS.env == "PandaPickCubeVision-v0":
            register(
                id="PandaPickCubeVision-v0",
                entry_point="franka_sim.envs:PandaPickCubeGymEnv",
                max_episode_steps=100,
                kwargs={"image_obs": True},
            )
            print("Registered PandaPickCubeVision-v0")
    
    # Force rgb_array for internal rendering logic to work correctly
    env = gym.make(FLAGS.env, reward_type=FLAGS.reward_type, render_mode="rgb_array")
    
    # Wrappers
    if FLAGS.env == "PandaPickCubeVision-v0":
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    
    # Initialize Agent (needed to restore checkpoint into)
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, key = jax.random.split(rng)
    
    print("Initializing agent...")
    agent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # Restore Checkpoint
    ckpt_path = os.path.abspath(FLAGS.checkpoint_path)
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # We restore the 'state' part of the agent
    # If ckpt_path is a file, verify it exists. If directory, restore_checkpoint handles it.
    restored_state = checkpoints.restore_checkpoint(ckpt_path, target=agent.state)
    agent = agent.replace(state=restored_state)
    
    # Evaluation Loop
    collected_trajs = []
    success_count = 0
    video_writer = None
    
    # Calculate max episodes as 1.5 * target_success_count, but at least num_episodes
    max_episodes = max(FLAGS.num_episodes, int(FLAGS.target_success_count * 1.5))
    
    print(f"Starting evaluation... Target successes: {FLAGS.target_success_count}. Max episodes: {max_episodes}")
    
    # We use a while loop or modify the range to stop when target is reached
    pbar = tqdm(total=FLAGS.target_success_count)
    episode_idx = 0
    
    while success_count < FLAGS.target_success_count and episode_idx < max_episodes:
        episode_idx += 1
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Trajectory storage
        traj = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'infos': []
        }
        
        episode_success = False
        
        while not (done or truncated):
            rng, key = jax.random.split(rng)
            
            # Sample action (deterministic)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                argmax=True
            )
            action = np.asarray(jax.device_get(actions))
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Record transition
            traj['observations'].append(obs)
            traj['actions'].append(action)
            traj['rewards'].append(reward)
            traj['next_observations'].append(next_obs)
            traj['dones'].append(done or truncated)
            traj['infos'].append(info)
            
            # Check success
            # If reward_type is binary, reward=1.0 means success
            # If dense, we rely on the same check if needed, but here assuming binary logic
            if FLAGS.reward_type == "binary" and reward == 1.0:
                episode_success = True
            
            obs = next_obs
            
            # Visualization and Video Saving
            img_bgr = None
            if FLAGS.render or FLAGS.save_video:
                if "images" in obs and "front" in obs["images"]:
                    img = obs["images"]["front"]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = env.render()
                    if isinstance(img, list): img = img[0]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if FLAGS.render and img_bgr is not None:
                cv2.imshow("Evaluation", img_bgr)
                cv2.waitKey(1)
            
            if FLAGS.save_video and img_bgr is not None:
                if video_writer is None:
                    height, width, layers = img_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(FLAGS.video_path, fourcc, 20.0, (width, height))
                video_writer.write(img_bgr)
                
                # If paused on success, write the same frame multiple times to pause in video too
                if FLAGS.pause_on_success and FLAGS.reward_type == "binary" and reward == 1.0:
                    # Pause for 0.5s = 10 frames at 20fps
                    for _ in range(10):
                        video_writer.write(img_bgr)

            if FLAGS.pause_on_success and FLAGS.reward_type == "binary" and reward == 1.0 and FLAGS.render:
                cv2.waitKey(500) # 500ms = 0.5s

        if episode_success:
            success_count += 1
            pbar.update(1)
            
        if FLAGS.save_trajs:
            # Always saving success only logic as requested by "只保存成功数据" implies filter
            # Or strict adherence to FLAGS.save_success_only? 
            # User said "只保存成功数据", so let's enforce it or check the flag. 
            # Assuming user wants to enforce save_success_only behavior based on query.
            if episode_success:
                collected_trajs.append(traj)
    
    pbar.close()

    print(f"\nEvaluation Complete.")
    print(f"Success Rate: {success_count}/{episode_idx} ({success_count/episode_idx*100:.2f}%)")
    
    if FLAGS.save_trajs and collected_trajs:
        # Format the output path with the number of collected successful trajectories
        output_path = FLAGS.output_path.format(success_count)
        with open(output_path, 'wb') as f:
            pkl.dump(collected_trajs, f)
        print(f"Saved {len(collected_trajs)} trajectories to {output_path}")
    elif FLAGS.save_trajs:
        print("No trajectories saved (none met criteria).")
        
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {FLAGS.video_path}")
        
    env.close()
    if FLAGS.render:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(main)

