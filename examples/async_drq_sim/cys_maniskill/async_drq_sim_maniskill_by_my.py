#!/usr/bin/env python3

import os
import sys
import types

# Fix for JAX CUDA detection issue - must be set before any JAX imports
# This prevents the cuda_nvcc.__file__ NoneType error
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

# Workaround for cuda_nvcc.__file__ being None (namespace package issue)
# JAX imports cuda_nvcc from nvidia package, so we need to patch nvidia.cuda_nvcc
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


import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import cv2
import os

# Set PyTorch to use CPU to avoid CUDA conflicts with JAX
# This must be done before importing ManiSkill (which uses PyTorch)
# Set environment variable to prevent PyTorch from initializing CUDA
os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
try:
    import torch
    # Force PyTorch to use CPU only to avoid CUDA memory conflicts with JAX
    # Set default tensor type to CPU
    if hasattr(torch, 'set_default_tensor_type'):
        torch.set_default_tensor_type('torch.FloatTensor')
    # Disable CUDA for PyTorch if it's available
    if torch.cuda.is_available():
        # Don't initialize CUDA context for PyTorch
        torch.backends.cuda.matmul.allow_tf32 = False
except ImportError:
    pass

from typing import Any, Dict, Optional
import pickle as pkl
import gymnasium as gym
import mani_skill.envs
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

# Suppress socket warnings from agentlace and absl
import logging
import warnings

# Suppress warnings from agentlace selector_events
logging.getLogger('agentlace').setLevel(logging.ERROR)
# Suppress specific warnings about socket.send() from selector_events
logging.getLogger('agentlace.trainer').setLevel(logging.ERROR)

# Suppress absl logging warnings (including socket.send() warnings)
# Set absl logging level to ERROR to suppress WARNING messages
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Also suppress Python warnings
warnings.filterwarnings('ignore')

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PickCube-v1", "Name of environment.")
flags.DEFINE_string("obs_mode", "rgb+state", "Observation mode for ManiSkill environment.")
flags.DEFINE_string("control_mode", "pd_ee_delta_pose", "Control mode for ManiSkill environment.")
flags.DEFINE_string("robot_uids", "panda", "Robot UIDs for ManiSkill environment.") # panda_wristcam
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################
# Custom wrapper to convert PyTorch tensors to numpy arrays
class ConvertTensorWrapper(gym.Wrapper):
    """
    Wrapper to convert PyTorch tensors to numpy arrays in step and reset returns.
    This is needed because ManiSkill may return PyTorch tensors, but gymnasium
    wrappers expect numpy arrays.
    """
    
    def _convert_to_numpy(self, value):
        """Convert PyTorch tensor or JAX array to numpy."""
        try:
            import torch
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    value = value.cpu()
                value = value.numpy()
        except ImportError:
            pass
        
        # Convert JAX arrays
        if hasattr(value, '__jax_array__') or str(type(value)).startswith("<class 'jax"):
            value = np.asarray(value)
        
        return value
    
    def _convert_obs_to_numpy(self, obs):
        """Recursively convert all tensors in observation to numpy."""
        if isinstance(obs, dict):
            return {k: self._convert_obs_to_numpy(v) for k, v in obs.items()}
        else:
            return self._convert_to_numpy(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Convert observation to numpy
        obs = self._convert_obs_to_numpy(obs)
        
        # Convert all return values to numpy
        reward = self._convert_to_numpy(reward)
        terminated = self._convert_to_numpy(terminated)
        truncated = self._convert_to_numpy(truncated)
        
        # Convert terminated/truncated to Python bool if scalar
        if isinstance(terminated, np.ndarray) and terminated.ndim == 0:
            terminated = bool(terminated.item())
        if isinstance(truncated, np.ndarray) and truncated.ndim == 0:
            truncated = bool(truncated.item())
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Convert observation to numpy
        obs = self._convert_obs_to_numpy(obs)
        return obs, info


##############################################################################
# Custom wrapper to handle multi-channel images by keeping only first 3 channels
class TrimImageChannelsWrapper(gym.ObservationWrapper):
    """
    Wrapper to trim multi-channel images to 3 channels (RGB).
    This is needed because ResNet encoder expects 3-channel RGB images.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Check if observation space is a dict
        if not isinstance(self.observation_space, gym.spaces.Dict):
            return
        
        # Update observation space for image keys with >3 channels
        new_spaces = {}
        self.trim_keys = []  # Track which keys need trimming
        
        for key, space in self.observation_space.spaces.items():
            if key == "state":
                new_spaces[key] = space
            elif isinstance(space, gym.spaces.Box) and len(space.shape) >= 3:
                # Check if it's an image with more than 3 channels
                if space.shape[-1] > 3:
                    # Trim to 3 channels
                    new_shape = space.shape[:-1] + (3,)
                    # Handle low and high bounds
                    try:
                        if isinstance(space.low, np.ndarray) and space.low.ndim > 0:
                            new_low = space.low[..., :3]
                        else:
                            new_low = space.low
                        if isinstance(space.high, np.ndarray) and space.high.ndim > 0:
                            new_high = space.high[..., :3]
                        else:
                            new_high = space.high
                    except (IndexError, TypeError):
                        new_low = space.low
                        new_high = space.high
                    
                    new_spaces[key] = gym.spaces.Box(
                        low=new_low,
                        high=new_high,
                        shape=new_shape,
                        dtype=space.dtype
                    )
                    self.trim_keys.append(key)
                else:
                    new_spaces[key] = space
            else:
                new_spaces[key] = space
        
        self.observation_space = gym.spaces.Dict(new_spaces)
    
    def observation(self, obs):
        if not isinstance(obs, dict):
            return obs
        
        new_obs = {}
        
        for key, value in obs.items():
            if key in self.trim_keys:
                # Trim to first 3 channels
                # Handle PyTorch tensors (ManiSkill may return CUDA tensors)
                try:
                    import torch
                    if isinstance(value, torch.Tensor):
                        # Move to CPU and convert to numpy
                        if value.is_cuda:
                            value = value.cpu()
                        value = value.numpy()
                except ImportError:
                    pass
                
                # Handle numpy arrays and jax arrays
                if isinstance(value, (np.ndarray, jnp.ndarray)):
                    new_obs[key] = value[..., :3]
                else:
                    # Fallback: try to convert and trim
                    value_array = np.asarray(value)
                    new_obs[key] = value_array[..., :3]
            else:
                new_obs[key] = value
        
        return new_obs


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # Get image keys from observation space
    if hasattr(env.observation_space, 'keys'):
        image_keys = [key for key in env.observation_space.keys() if key != "state"]
    else:
        image_keys = []

    # Create evaluation environment
    eval_env = gym.make(
        FLAGS.env,
        obs_mode=FLAGS.obs_mode,
        control_mode=FLAGS.control_mode,
        robot_uids=FLAGS.robot_uids,
    )
    
    # Apply wrappers if needed based on observation mode
    if FLAGS.obs_mode in ["rgbd", "rgb", "rgb+state"]:
        try:
            from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
            eval_env = FlattenRGBDObservationWrapper(
                eval_env,
                rgb=True if FLAGS.obs_mode in ["rgbd", "rgb", "rgb+state"] else False,
                depth=True if FLAGS.obs_mode == "rgbd" else False,
                state=True if FLAGS.obs_mode == "rgb+state" else False
            )
            # Trim multi-channel images to 3 channels for ResNet encoder compatibility
            eval_env = TrimImageChannelsWrapper(eval_env)
            # Convert PyTorch tensors to numpy before ChunkingWrapper
            eval_env = ConvertTensorWrapper(eval_env)
            eval_env = ChunkingWrapper(eval_env, obs_horizon=1, act_exec_horizon=None)
        except ImportError:
            eval_env = SERLObsWrapper(eval_env)
            # Convert PyTorch tensors to numpy before ChunkingWrapper
            eval_env = ConvertTensorWrapper(eval_env)
            eval_env = ChunkingWrapper(eval_env, obs_horizon=1, act_exec_horizon=None)
    elif FLAGS.obs_mode in ["state_dict", "state"]:
        eval_env = gym.wrappers.FlattenObservation(eval_env)
        # Convert PyTorch tensors to numpy
        eval_env = ConvertTensorWrapper(eval_env)
    
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))
                # Remove batch dimension if present (from ChunkingWrapper)
                if len(actions.shape) > 1 and actions.shape[0] == 1:
                    actions = actions[0]

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            
            # Convert PyTorch tensors to numpy/Python types if needed
            try:
                import torch
                if isinstance(reward, torch.Tensor):
                    reward = reward.cpu().numpy() if reward.is_cuda else reward.numpy()
                if isinstance(done, torch.Tensor):
                    done = done.cpu().numpy() if done.is_cuda else done.numpy()
                    # Convert to Python bool if scalar
                    if done.ndim == 0:
                        done = bool(done.item())
                if isinstance(truncated, torch.Tensor):
                    truncated = truncated.cpu().numpy() if truncated.is_cuda else truncated.numpy()
                    # Convert to Python bool if scalar
                    if truncated.ndim == 0:
                        truncated = bool(truncated.item())
            except ImportError:
                pass
            
            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            
            # Convert done to numpy for mask calculation
            done_np = np.asarray(done, dtype=np.float32) if not isinstance(done, bool) else float(done)
            
            running_return += reward
            
            # MemoryEfficientReplayBuffer expects observations with time dimension (T, H, W, C)
            # ChunkingWrapper with obs_horizon=1 returns (1, H, W, C), which is correct
            # The replay buffer's insert() method will extract the last time step using [-1]
            # However, there seems to be a bug where the time dimension is not properly removed
            # So we need to ensure observations have the correct shape before insertion
            # Get pixel_keys from data_store if available, otherwise use image_keys from env
            pixel_keys = getattr(data_store, 'pixel_keys', image_keys)
            
            # Ensure observations have the correct shape for pixel_keys
            # The insert method expects (T, H, W, C) and will use [-1] to get (H, W, C)
            # So we pass observations as-is: (1, H, W, C) for ChunkingWrapper with obs_horizon=1
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done_np,
                dones=bool(done) or bool(truncated),
            )
            data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                # Wrap policy function to convert JAX arrays to numpy
                def policy_fn_wrapper(obs):
                    action = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=jax.random.PRNGKey(0),
                        argmax=True,
                    )
                    # Convert JAX array to numpy and remove batch dimension if present
                    action = np.asarray(jax.device_get(action))
                    if len(action.shape) > 1 and action.shape[0] == 1:
                        action = action[0]
                    return action
                
                evaluate_info = evaluate(
                    policy_fn=policy_fn_wrapper,
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(
    rng,
    agent: DrQAgent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        single_buffer_batch_size = FLAGS.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(
            FLAGS.env,
            obs_mode=FLAGS.obs_mode,
            control_mode=FLAGS.control_mode,
            robot_uids=FLAGS.robot_uids,
            render_mode="human"
        )
    else:
        env = gym.make(
            FLAGS.env,
            obs_mode=FLAGS.obs_mode,
            control_mode=FLAGS.control_mode,
            robot_uids=FLAGS.robot_uids,
        )

    # Apply wrappers if needed based on observation mode
    if FLAGS.obs_mode in ["rgbd", "rgb", "rgb+state"]:
        # For visual observations, we need to use ManiSkill's FlattenRGBDObservationWrapper
        # instead of SERL's wrapper
        try:
            from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
            env = FlattenRGBDObservationWrapper(
                env,
                rgb=True if FLAGS.obs_mode in ["rgbd", "rgb", "rgb+state"] else False,
                depth=True if FLAGS.obs_mode == "rgbd" else False,
                state=True if FLAGS.obs_mode == "rgb+state" else False
            )
            # Trim multi-channel images to 3 channels for ResNet encoder compatibility
            env = TrimImageChannelsWrapper(env)
            # Convert PyTorch tensors to numpy before ChunkingWrapper
            env = ConvertTensorWrapper(env)
            # Then apply SERL wrappers if observations are in the right format
            # env = SERLObsWrapper(env)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        except ImportError:
            # Fallback to SERL wrappers for non-ManiSkill environments
            env = SERLObsWrapper(env)
            # Convert PyTorch tensors to numpy before ChunkingWrapper
            env = ConvertTensorWrapper(env)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    elif FLAGS.obs_mode == "state_dict":
        # For state_dict, flatten the observation
        env = gym.wrappers.FlattenObservation(env)
    elif FLAGS.obs_mode == "state":
        # For state observation, just flatten if needed
        # ManiSkill state obs may have batch dimension, flatten it
        env = gym.wrappers.FlattenObservation(env)

    # Determine image keys based on observation space
    if hasattr(env.observation_space, 'keys'):
        image_keys = [key for key in env.observation_space.keys() if key != "state"]
    else:
        image_keys = []

    rng, sampling_rng = jax.random.split(rng)
    
    # Sample observations and actions for agent initialization
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    
    # Ensure sample_action has the same batch dimension as sample_obs
    # ChunkingWrapper with obs_horizon=1 adds a time/batch dimension to observations
    # We need to check if observations have this dimension and add it to actions if needed
    if isinstance(sample_obs, dict):
        # Check if any observation has a batch/time dimension
        has_batch_dim = False
        for key, value in sample_obs.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                # ChunkingWrapper with obs_horizon=1 creates shape like (1, H, W, C) or (1, D)
                if value.shape[0] == 1:
                    has_batch_dim = True
                    break
        
        # If observations have batch dimension but action doesn't, add it
        if has_batch_dim:
            sample_action = np.asarray(sample_action)
            if len(sample_action.shape) == 1:
                sample_action = np.expand_dims(sample_action, axis=0)
    
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )

        print_green("replay buffer created")
        print_green(f"replay_buffer size: {len(replay_buffer)}")

        # if demo data is provided, load it into the demo buffer
        # in the learner node, we support 2 ways to load demo data:
        # 1. load from pickle file; 2. load from tf rlds data
        if FLAGS.demo_path or FLAGS.preload_rlds_path:

            def preload_data_transform(data, metadata) -> Optional[Dict[str, Any]]:
                # NOTE: Create your own custom data transform function here if you
                # are loading this via with --preload_rlds_path with tf rlds data
                # This default does nothing
                return data

            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                type="memory_efficient_replay_buffer",
                image_keys=image_keys,
                preload_rlds_path=FLAGS.preload_rlds_path,
                preload_data_transform=preload_data_transform,
            )

            if FLAGS.demo_path:
                # Check if the file exists
                if not os.path.exists(FLAGS.demo_path):
                    raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

                with open(FLAGS.demo_path, "rb") as f:
                    trajs = pkl.load(f)
                    for traj in trajs:
                        demo_buffer.insert(traj)

            print(f"demo buffer size: {len(demo_buffer)}")
        else:
            demo_buffer = None

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,  # None if no demo data is provided
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
