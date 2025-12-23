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

from typing import Any, Dict, Optional
import pickle as pkl
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
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

import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 128, "Batch size.") #256
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
flags.DEFINE_integer("port", 5488, "Port number for trainer server.")
flags.DEFINE_integer("broadcast_port", 5489, "Broadcast port number for trainer server.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 2000, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", "checkpoints", "Path to save checkpoints.")

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


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port, broadcast_port=FLAGS.broadcast_port),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = gym.make(FLAGS.env)
    if FLAGS.env == "PandaPickCubeVision-v0":
        eval_env = SERLObsWrapper(eval_env)
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
            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
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
                evaluate_info = evaluate(
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
    # Configure wandb for this program only (doesn't change system defaults)
    # WANDB_API_KEY can be set in run_learner.sh temporarily (only affects this script)
    # The entity parameter in wandb.init() only affects this run, not system defaults
    wandb_logger = make_wandb_logger(
        project="cys_drq_sim",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
        # entity=None,  # Only affects this run, doesn't change system defaults
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
    server = TrainerServer(
        make_trainer_config(port_number=FLAGS.port, broadcast_port=FLAGS.broadcast_port),
        request_callback=stats_callback
    )
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
    # Increase queue_size for multi-GPU training to avoid GPU starvation
    # queue_size should be at least 2 * num_devices for better throughput
    # Increased queue_size for better data prefetching
    # For single GPU, use larger queue to avoid GPU starvation
    queue_size = max(32, 8 * num_devices)
    
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
            queue_size=queue_size,
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        queue_size=queue_size,
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
        # Only block when we actually need to publish to avoid unnecessary synchronization
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            with timer.context("publish_network"):
                agent = jax.block_until_ready(agent)
                server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            timer_stats = timer.get_average_times()
            wandb_logger.log({"timer": timer_stats}, step=update_steps)
            # Print detailed timing info to console for debugging
            if update_steps % (FLAGS.log_period * 10) == 0:  # Print every 100 steps
                print_green("\n" + "=" * 80)
                print_green("PERFORMANCE STATS (averaged over last period):")
                for key, value in timer_stats.items():
                    print_green(f"  {key}: {value*1000:.2f}ms")
                total_time = sum(timer_stats.values())
                print_green(f"  Total per step: {total_time*1000:.2f}ms ({1/total_time:.2f} it/s)")
                print_green("=" * 80 + "\n")

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            # Ensure the directory exists
            ckpt_path = os.path.abspath(FLAGS.checkpoint_path)
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path, exist_ok=True)
            
            checkpoints.save_checkpoint(
                ckpt_path, agent.state, step=update_steps, keep=10, overwrite=True
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    # Print device information
    print_green("=" * 80)
    print_green("DEVICE INFORMATION:")
    print_green(f"JAX Platform: {jax.default_backend()}")
    print_green(f"Number of devices: {num_devices}")
    print_green(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (using all available GPUs)')}")
    
    # Check if using GPU or CPU
    device_types = [str(d.device_kind) for d in devices]
    # Check if it's GPU - device_kind contains GPU model name or device string contains 'gpu'/'cuda'
    is_gpu = any(
        'gpu' in str(d).lower() or 
        'nvidia' in dt.lower() or 
        'cuda' in str(d).lower() or
        dt not in ['cpu', 'tpu']
        for d, dt in zip(devices, device_types)
    )
    
    if is_gpu:
        print_green(f"✓ Using GPU devices: {device_types}")
        for i, device in enumerate(devices):
            print_green(f"  Device {i}: {device} (ID: {device.id})")
    else:
        print_green("⚠ WARNING: Using CPU devices (GPU not detected)!")
        print_green(f"  Device types: {device_types}")
        print_green("  Check CUDA installation and JAX_PLATFORM_NAME setting")
    
    print_green("=" * 80)
    
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env)

    if FLAGS.env == "PandaPickCube-v0":
        env = gym.wrappers.FlattenObservation(env)
    if FLAGS.env == "PandaPickCubeVision-v0":
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

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
