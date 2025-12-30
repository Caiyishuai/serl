#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
# from flax.training import checkpoints
import cv2
import os
os.environ["MUJOCO_GL"] = "egl"
from typing import Any, Dict, Optional, Tuple
import pickle as pkl
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics


from serl_launcher.agents.continuous.drq import DrQAgent # 需要转为 torch版本
from serl_launcher.common.evaluation import evaluate # 需要转为 torch版本
from serl_launcher.utils.timer_utils import Timer

from serl_launcher.wrappers.chunking import ChunkingWrapper # 需要转为 torch版本
# from serl_launcher_torch_torch.serl_launcher_torch.wrappers.chunking_torch import ChunkingWrapper as ChunkingWrapperTorch

from serl_launcher.utils.train_utils import concat_batches # 需要转为 torch版本

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

import franka_sim
import os
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", "visual_demo_jax_non-distributed", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 0, "Random seed.") #42
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 64, "Batch size.") #256 128
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.") #4

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 100, "Sample random actions for this many steps.") #300
flags.DEFINE_integer("training_starts", 100, "Training starts after this step.") #300
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.") # 30

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", True, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
# flags.DEFINE_string("demo_path",os.path.join(DIR_PATH, "./franka_demo_data_3_trajs.pkl"), "Path to the demo data.")
flags.DEFINE_string("demo_path",os.path.join(DIR_PATH, "../franka_lift_cube_image_20_trajs.pkl"), "Path to the demo data.")
# flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
# flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean(
    "debug", True, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

# 使用单个设备
device = jax.devices()[0]

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def _set_mujoco_gl_backend(enable_render: bool):
    """Ensure MuJoCo uses GLFW for human viewer, EGL otherwise."""
    desired_backend = "glfw" if enable_render else "egl"
    if os.environ.get("MUJOCO_GL") != desired_backend:
        os.environ["MUJOCO_GL"] = desired_backend


def make_env(enable_render: bool) -> Tuple[gym.Env, str]:
    """Create env with optional human rendering fallback to rgb_array."""
    _set_mujoco_gl_backend(enable_render)
    requested_mode = "human" if enable_render else "rgb_array"
    render_mode = requested_mode
    try:
        env = gym.make(FLAGS.env, render_mode=requested_mode)
    except Exception as err:
        if not enable_render:
            raise
        print(
            f"Warning: failed to create env with render_mode='{requested_mode}': {err}. "
            "Falling back to 'rgb_array'."
        )
        _set_mujoco_gl_backend(False)
        env = gym.make(FLAGS.env, render_mode="rgb_array")
        render_mode = "rgb_array"

    if FLAGS.env == "PandaPickCube-v0":
        env = gym.wrappers.FlattenObservation(env)
    if FLAGS.env == "PandaPickCubeVision-v0":
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    return env, render_mode


def main(_):
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env, env_render_mode = make_env(FLAGS.render)
    print_green(f"Environment render_mode: {env_render_mode}")

    image_keys = [key for key in env.observation_space.keys() if key != "state"] #['front', 'wrist']

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # put agent on device
    agent: DrQAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), device
    )

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
    if FLAGS.demo_path or FLAGS.preload_rlds_path:
        def preload_data_transform(data, metadata) -> Optional[Dict[str, Any]]:
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
            if not os.path.exists(FLAGS.demo_path):
                raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

            with open(FLAGS.demo_path, "rb") as f:
                trajs = pkl.load(f)
                for traj in trajs:
                    demo_buffer.insert(traj)

        print_green(f"demo buffer size: {len(demo_buffer)}") #2001
    else:
        demo_buffer = None

    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # Main training loop
    update_steps = 0
    timer = Timer()

    # First fill the buffer with random actions
    print_green("Filling buffer with random actions...")
    obs, _ = env.reset()
    for _ in range(FLAGS.training_starts):
        actions = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(actions)
        reward = np.asarray(reward, dtype=np.float32)
        transition = dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=reward,
            masks=1.0 - done,
            dones=done or truncated,
        )
        replay_buffer.insert(transition)
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()

    print_green(f"Initial buffer filled with {len(replay_buffer)} transitions")

    # Create data iterators
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        single_buffer_batch_size = FLAGS.batch_size // 2
        # 创建一个迭代器，每次调用 next(iterator) 时返回一个批次的数据
        demo_iterator = demo_buffer.get_iterator( 
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=device,
        )

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=device,
    )

    # Training loop
    obs, _ = env.reset()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        # Actor phase: collect experience
        with timer.context("actor"):
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

            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )
            '''
            transition
            dict_keys(['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones'])
            transition['observations']    
            'front': array([[[[ 36,  61,  85],   (1,128,128,3)  (1,7) 
                            [ 36,  61,  85],
                            [ 36,  61,  85],
                            ...
                            [ 40,  81, 122]]]], dtype=uint8) 
            '''
            replay_buffer.insert(transition) 
            obs = next_obs
            if done or truncated:
                obs, _ = env.reset()

        # Learner phase: update the agent (only every steps_per_update steps)
        # 每 30 步才训练一次一次
        # if step > 0 and step % FLAGS.steps_per_update == 0:
        #     for _ in range(FLAGS.steps_per_update):
                # print(f"Training at step {step}")
        with timer.context("learner"):
            # run n-1 critic updates and 1 critic + actor update
            for critic_step in range(FLAGS.critic_actor_ratio - 1):
                batch = next(replay_iterator)
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)
                agent, critics_info = agent.update_critics(batch)

            batch = next(replay_iterator)
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # Logging and evaluation
        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                eval_env, _ = make_env(FLAGS.render)
                eval_env = RecordEpisodeStatistics(eval_env)
                
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
                if wandb_logger:
                    wandb_logger.log({"eval": evaluate_info}, step=update_steps)

        # Only log training info if we've actually trained
        if step > 0 and step % FLAGS.steps_per_update == 0:
            if step % FLAGS.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=update_steps)
                wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        # if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
        #     assert FLAGS.checkpoint_path is not None
        #     checkpoints.save_checkpoint(
        #         FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
        #     )

        update_steps += 1


if __name__ == "__main__":
    app.run(main)
