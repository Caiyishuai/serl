#!/usr/bin/env python3

import os
import sys
import types

from jax._src.api import F

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

from typing import Any, Dict, Optional
import pickle as pkl
import gymnasium as gym
import mani_skill.envs
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from collections import defaultdict

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate, flatten, add_to
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches

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

flags.DEFINE_string("env", "PlaceSphere-v1", "Name of environment.")
flags.DEFINE_string("obs_mode", "rgb+state", "Observation mode for ManiSkill environment.")
flags.DEFINE_string("control_mode", "pd_ee_delta_pose", "Control mode for ManiSkill environment.")
flags.DEFINE_string("robot_uids", "panda_wristcam", "Robot UIDs for ManiSkill environment.")
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
# flags.DEFINE_string("demo_path", "/home/caiyishuai/workspace/maniskill-ws/serl_data/mani_skill_place_sphere_100.pkl", "Path to the demo data.")
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


def _get_success_from_info(info: dict) -> float:
    """从 env info 中取出 success (0/1)，支持 ManiSkill 等嵌套键。"""
    flat = flatten(info)
    for key in ("success", "eval_success", "final.success", "final.eval_success"):
        if key in flat:
            v = flat[key]
            if hasattr(v, "item"):
                return float(v.item()) if getattr(v, "ndim", 1) == 0 else float(v)
            return float(bool(v))
    return 0.0


def evaluate_with_episode_success(policy_fn, env, num_episodes: int) -> Dict[str, float]:
    """与 evaluate() 相同，但额外返回按「轨迹数」平均的 episode_success_rate（0/0.2/.../1.0）。"""
    stats = defaultdict(list)
    episode_successes = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))
        episode_successes.append(_get_success_from_info(info))
    for k, v in stats.items():
        stats[k] = np.mean(v)
    stats["episode_success_rate"] = float(np.mean(episode_successes))
    return stats


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

    # Create evaluation environment
    import gymnasium
    eval_env = gymnasium.make(
        FLAGS.env,
        obs_mode=FLAGS.obs_mode,
        control_mode=FLAGS.control_mode,
        robot_uids=FLAGS.robot_uids,
    )
    
    # ManiSkill multi-camera wrapper
    import torch
    from gymnasium.spaces import Box as GymBox
    from gymnasium.spaces import Dict as GymDict
    
    class ManiSkillMultiCameraWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            sample_obs, _ = env.reset()
            
            obs_dict = {}
            if "state" in sample_obs:
                state_data = sample_obs["state"]
                if torch.is_tensor(state_data):
                    state_data = state_data.cpu().numpy()
                state_shape = tuple(state_data.reshape(-1).shape)
                obs_dict["state"] = GymBox(
                    low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
                )
            
            if "sensor_data" in sample_obs:
                for cam_name, cam_data in sample_obs["sensor_data"].items():
                    if "rgb" in cam_data:
                        rgb_data = cam_data["rgb"]
                        if torch.is_tensor(rgb_data):
                            rgb_data = rgb_data.cpu().numpy()
                        if rgb_data.ndim == 4 and rgb_data.shape[0] == 1:
                            rgb_data = rgb_data[0]
                        rgb_shape = tuple(rgb_data.shape)
                        obs_dict[cam_name] = GymBox(
                            low=0, high=255, shape=rgb_shape, dtype=np.uint8
                        )
            
            self.observation_space = GymDict(obs_dict)
            env.reset()
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._convert_obs(obs), info
        
        def step(self, action):
            # Convert jax array to numpy if needed
            if hasattr(action, '__array__'):
                action = np.asarray(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Convert torch tensors to numpy
            if torch.is_tensor(reward):
                reward = reward.cpu().item() if reward.numel() == 1 else reward.cpu().numpy()
            if torch.is_tensor(terminated):
                terminated = terminated.cpu().item() if terminated.numel() == 1 else terminated.cpu().numpy()
            if torch.is_tensor(truncated):
                truncated = truncated.cpu().item() if truncated.numel() == 1 else truncated.cpu().numpy()
            
            return self._convert_obs(obs), float(reward), bool(terminated), bool(truncated), info
        
        def _convert_obs(self, obs):
            obs_dict = {}
            if "state" in obs:
                state_data = obs["state"]
                if torch.is_tensor(state_data):
                    state_data = state_data.cpu().numpy()
                obs_dict["state"] = state_data.reshape(-1).astype(np.float32)
            
            if "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    if "rgb" in cam_data:
                        rgb_data = cam_data["rgb"]
                        if torch.is_tensor(rgb_data):
                            rgb_data = rgb_data.cpu().numpy()
                        if rgb_data.ndim == 4 and rgb_data.shape[0] == 1:
                            rgb_data = rgb_data[0]
                        obs_dict[cam_name] = rgb_data.astype(np.uint8)
            return obs_dict
    
    eval_env = ManiSkillMultiCameraWrapper(eval_env)
    eval_env = ChunkingWrapper(eval_env, obs_horizon=1, act_exec_horizon=None)
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

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            
            # Convert torch tensors to numpy if needed
            if hasattr(done, 'cpu'):  # Check if it's a torch tensor
                done = done.cpu().item() if done.numel() == 1 else done.cpu().numpy()
            if hasattr(truncated, 'cpu'):
                truncated = truncated.cpu().item() if truncated.numel() == 1 else truncated.cpu().numpy()
            if hasattr(reward, 'cpu'):
                reward = reward.cpu().item() if reward.numel() == 1 else reward.cpu().numpy()
            
            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - float(done),
                dones=done or truncated,
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
                evaluate_info = evaluate_with_episode_success(
                    policy_fn=partial(agent.sample_actions, argmax=True),
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
        project="maniskill_serl",
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
    import gymnasium
    if FLAGS.render:
        env = gymnasium.make(
            FLAGS.env,
            obs_mode=FLAGS.obs_mode,
            control_mode=FLAGS.control_mode,
            robot_uids=FLAGS.robot_uids,
            render_mode="human"
        )
    else:
        env = gymnasium.make(
            FLAGS.env,
            obs_mode=FLAGS.obs_mode,
            control_mode=FLAGS.control_mode,
            robot_uids=FLAGS.robot_uids,
        )
    # envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # ManiSkill multi-camera wrapper
    import torch
    from gymnasium.spaces import Box as GymBox
    from gymnasium.spaces import Dict as GymDict
    
    class ManiSkillMultiCameraWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            sample_obs, _ = env.reset()
            
            obs_dict = {}
            if "state" in sample_obs:
                state_data = sample_obs["state"]
                if torch.is_tensor(state_data):
                    state_data = state_data.cpu().numpy()
                state_shape = tuple(state_data.reshape(-1).shape)
                obs_dict["state"] = GymBox(
                    low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
                )
            
            if "sensor_data" in sample_obs:
                for cam_name, cam_data in sample_obs["sensor_data"].items():
                    if "rgb" in cam_data:
                        rgb_data = cam_data["rgb"]
                        if torch.is_tensor(rgb_data):
                            rgb_data = rgb_data.cpu().numpy()
                        if rgb_data.ndim == 4 and rgb_data.shape[0] == 1:
                            rgb_data = rgb_data[0]
                        rgb_shape = tuple(rgb_data.shape)
                        obs_dict[cam_name] = GymBox(
                            low=0, high=255, shape=rgb_shape, dtype=np.uint8
                        )
            
            self.observation_space = GymDict(obs_dict)
            env.reset()
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._convert_obs(obs), info
        
        def step(self, action):
            # Convert jax array to numpy if needed
            if hasattr(action, '__array__'):
                action = np.asarray(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Convert torch tensors to numpy
            if torch.is_tensor(reward):
                reward = reward.cpu().item() if reward.numel() == 1 else reward.cpu().numpy()
            if torch.is_tensor(terminated):
                terminated = terminated.cpu().item() if terminated.numel() == 1 else terminated.cpu().numpy()
            if torch.is_tensor(truncated):
                truncated = truncated.cpu().item() if truncated.numel() == 1 else truncated.cpu().numpy()
            
            return self._convert_obs(obs), float(reward), bool(terminated), bool(truncated), info
        
        def _convert_obs(self, obs):
            obs_dict = {}
            if "state" in obs:
                state_data = obs["state"]
                if torch.is_tensor(state_data):
                    state_data = state_data.cpu().numpy()
                obs_dict["state"] = state_data.reshape(-1).astype(np.float32)
            
            if "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    if "rgb" in cam_data:
                        rgb_data = cam_data["rgb"]
                        if torch.is_tensor(rgb_data):
                            rgb_data = rgb_data.cpu().numpy()
                        if rgb_data.ndim == 4 and rgb_data.shape[0] == 1:
                            rgb_data = rgb_data[0]
                        obs_dict[cam_name] = rgb_data.astype(np.uint8)
            return obs_dict
    
    env = ManiSkillMultiCameraWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    # Determine image keys based on observation space
    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
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
                    transitions = pkl.load(f)
                # pkl 格式：list，每个元素是一条 transition（dict: observations, next_observations, actions, rewards, masks, dones）
                for transition in transitions:
                    demo_buffer.insert(transition)

            print_green(f"demo buffer size: {len(demo_buffer)}")
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
