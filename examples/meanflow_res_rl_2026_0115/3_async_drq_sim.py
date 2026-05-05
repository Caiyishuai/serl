#!/usr/bin/env python3
"""
async_drq_sim.py — ManiFlow + Residual RL (SERL DrQ) for ManiSkill.

Architecture
------------
Base policy  : ManiFlow (DiTX) + TimmObsEncoder  (local PyTorch, loaded from checkpoint)
Residual RL  : DrQAgent (JAX)  learns a small delta correction on top of the base
Final action : a_base + a_delta * alpha_scale

Distributed : actor process + learner process communicating via AgentLace.

Usage
-----
# Terminal 1 — learner
python examples/meanflow_res_rl_2026_0115/3_async_drq_sim.py --learner \
    --env PushCube-v1 --exp_name pushcube_meanflow_res

# Terminal 2 — actor
python examples/meanflow_res_rl_2026_0115/3_async_drq_sim.py --actor \
    --env PushCube-v1 \
    --meanflow_ckpt runs/PushCube-v1__meanflow__42__xxx/checkpoints/best_success_once.pt

Reuses from examples/openpi/async_drq_sim.py:
    ManiSkillMultiCameraWrapper, extract_eef_from_env,
    PotentialBasedRewardWrapper, print_green,
    _adapt_demo_for_chunking, _validate_demo_vs_env,
    evaluate_with_episode_success, _trajectory_frames_to_mp4
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers import RecordEpisodeStatistics
import gymnasium as gym
import mani_skill.envs  # noqa: F401

# ── serl / agentlace ──────────────────────────────────────────────────────────
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate_with_trajectories
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_replay_buffer,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.wrappers.chunking import ChunkingWrapper

# ── Patch gymnasium import used by openpi (gymnasium ≥1.0 moved the module) ──
import sys as _sys
import gymnasium.wrappers as _gw
if "gymnasium.wrappers.record_episode_statistics" not in _sys.modules:
    import types as _types
    _mod = _types.ModuleType("gymnasium.wrappers.record_episode_statistics")
    _mod.RecordEpisodeStatistics = _gw.RecordEpisodeStatistics
    _sys.modules["gymnasium.wrappers.record_episode_statistics"] = _mod

# ── reuse helpers from openpi example ────────────────────────────────────────
_SERL_ROOT = Path(__file__).resolve().parents[2]
if str(_SERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERL_ROOT))

# pi_link lives inside examples/openpi/ — add openpi dir so 'pi_link.xxx' resolves
_OPENPI_DIR = _SERL_ROOT / "examples" / "openpi"
if str(_OPENPI_DIR) not in _sys.path:
    _sys.path.insert(0, str(_OPENPI_DIR))

from examples.openpi.async_drq_sim import (  # noqa: E402
    ManiSkillMultiCameraWrapper,
    PotentialBasedRewardWrapper,
    _adapt_demo_for_chunking,
    _get_success_from_info,
    _trajectory_frames_to_mp4,
    _validate_demo_vs_env,
    evaluate_with_episode_success,
    extract_eef_from_env,
    print_green,
)
from examples.openpi.env_wrappers import AddPolicyActionWrapper  # noqa: E402

# ── local ManiFlow base policy ────────────────────────────────────────────────
from examples.meanflow_res_rl_2026_0115.meanflow_policy import ManiFlowPolicy  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Flags
# ─────────────────────────────────────────────────────────────────────────────
# Most flags (env, obs_mode, seed, batch_size, learner, actor, etc.) are already
# registered by `examples.openpi.async_drq_sim` when we import helpers from it.
# Only define flags that are unique to this meanflow_res_rl script.
FLAGS = flags.FLAGS

# ManiFlow base policy (unique to this script)
flags.DEFINE_string("meanflow_ckpt", None,
                    "Path to ManiFlow checkpoint (.pt). Required when --actor.")
flags.DEFINE_integer("meanflow_obs_horizon", 2, "obs_horizon used during ManiFlow training.")
flags.DEFINE_integer("meanflow_pred_horizon", 16, "pred_horizon used during ManiFlow training.")
flags.DEFINE_integer("meanflow_act_horizon", 8, "act_horizon: action chunk size.")
flags.DEFINE_integer("meanflow_num_inference_steps", 10, "Number of denoising steps.")
flags.DEFINE_string("meanflow_backbone", "resnet18", "Backbone for TimmObsEncoder.")
flags.DEFINE_integer("meanflow_n_layer", 4, "DiTX num layers.")
flags.DEFINE_integer("meanflow_n_head", 4, "DiTX num heads.")
flags.DEFINE_integer("meanflow_n_emb", 256, "DiTX embedding dim.")
flags.DEFINE_integer("meanflow_img_size", 128, "Camera image size.")

# use_residual and alpha_scale are already registered by openpi; no need to redefine.

num_devices = len(jax.devices())
sharding = jax.sharding.PositionalSharding(jax.devices())


# ─────────────────────────────────────────────────────────────────────────────
# Env creation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_env(render_mode=None):
    kwargs = dict(
        obs_mode=FLAGS.obs_mode,
        control_mode=FLAGS.control_mode,
        robot_uids=FLAGS.robot_uids,
        reward_mode=FLAGS.reward_mode,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(FLAGS.env, **kwargs)


def _wrap_env(env, meanflow_policy: Optional[ManiFlowPolicy] = None):
    """
    ManiSkill → (PBRS) → SERL flat obs → (Residual) → ChunkingWrapper

    meanflow_policy: if provided, wraps env with AddPolicyActionWrapper using
                     the ManiFlow model as base policy.
    """
    if FLAGS.potential_reward_shaping:
        env = PotentialBasedRewardWrapper(env)
        print_green("PotentialBasedRewardWrapper enabled")

    env = ManiSkillMultiCameraWrapper(env)
    _env_ref = env  # keep reference for eef extraction

    if meanflow_policy is not None:
        def _base_policy(obs: dict) -> np.ndarray:
            """
            Called by AddPolicyActionWrapper on each obs.
            obs: SERL flat dict from ManiSkillMultiCameraWrapper
                 {state:(N,), hand_camera:(H,W,3), base_camera:(H,W,3)}
            Returns (act_horizon, act_dim) action chunk.
            """
            # Replace flat state with 8-dim eef (consistent with ManiFlow training)
            meanflow_obs = {
                "state":     extract_eef_from_env(_env_ref),
                "img_wrist": obs["hand_camera"],
                "img_third": obs["base_camera"],
            }
            return meanflow_policy.step(meanflow_obs)

        # Hook meanflow_policy.reset() into env.reset via wrapper
        env = _ManiFlowResetWrapper(env, meanflow_policy)

        env = AddPolicyActionWrapper(
            env=env,
            base_policy=_base_policy,
            clip_action=False,
            allow_none_delta=True,
            prefetch_base_action=True,
            obs_mapping=None,     # key remapping done inside _base_policy
            alpha_scale=FLAGS.alpha_scale,
        )
        # AddPolicyActionWrapper is a @dataclass (not gym.Env); adapt it so
        # ChunkingWrapper (which asserts isinstance(env, gym.Env)) is satisfied.
        env = _GymEnvAdapter(env)
        print_green(f"AddPolicyActionWrapper (ManiFlow base policy, alpha={FLAGS.alpha_scale})")

    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    return env


class _ManiFlowResetWrapper(gym.Wrapper):
    """gym.Wrapper that calls meanflow_policy.reset() whenever env.reset() is called."""

    def __init__(self, env: gym.Env, meanflow_policy: ManiFlowPolicy):
        super().__init__(env)
        self._policy = meanflow_policy

    def reset(self, **kwargs):
        self._policy.reset()
        return super().reset(**kwargs)


class _GymEnvAdapter(gym.Env):
    """Adapts a duck-typed env (e.g. AddPolicyActionWrapper @dataclass) to gymnasium.Env.

    gymnasium ≥1.0 Wrapper.__init__ asserts isinstance(env, gym.Env).
    This adapter satisfies that assertion without modifying AddPolicyActionWrapper.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.observation_space = wrapped.observation_space
        self.action_space = wrapped.action_space
        self.metadata = getattr(wrapped, "metadata", {})
        self.render_mode = getattr(wrapped, "render_mode", None)

    def step(self, action):
        return self._wrapped.step(action)

    def reset(self, **kwargs):
        return self._wrapped.reset(**kwargs)

    def render(self):
        return self._wrapped.render() if hasattr(self._wrapped, "render") else None

    def close(self):
        return self._wrapped.close()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# ─────────────────────────────────────────────────────────────────────────────
# Actor
# ─────────────────────────────────────────────────────────────────────────────
def actor(agent: DrQAgent, data_store, env, sampling_rng):
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(
            port_number=FLAGS.server_port,
            broadcast_port=FLAGS.broadcast_port,
        ),
        data_store,
        wait_for_server=True,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = _wrap_env(
        _make_env(),
        ManiFlowPolicy(
            ckpt_path=FLAGS.meanflow_ckpt,
            obs_horizon=FLAGS.meanflow_obs_horizon,
            pred_horizon=FLAGS.meanflow_pred_horizon,
            act_horizon=FLAGS.meanflow_act_horizon,
            num_inference_steps=FLAGS.meanflow_num_inference_steps,
            backbone=FLAGS.meanflow_backbone,
            n_layer=FLAGS.meanflow_n_layer,
            n_head=FLAGS.meanflow_n_head,
            n_emb=FLAGS.meanflow_n_emb,
            img_size=FLAGS.meanflow_img_size,
        ) if FLAGS.use_residual else None,
    )
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False
    timer = Timer()
    running_return = 0.0
    total_transitions = 0

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

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            for attr in ("done", "truncated", "reward"):
                val = locals().get(attr, reward)
                if hasattr(val, "cpu"):
                    locals()[attr] = val.cpu().item() if val.numel() == 1 else val.cpu().numpy()

            reward    = float(np.asarray(reward))
            running_return += reward

            if FLAGS.ignore_terminations:
                mask_val  = 1.0 - float(truncated)
                done_val  = bool(truncated)
                should_reset = truncated
            else:
                mask_val  = 1.0 - float(done or truncated)
                done_val  = bool(done or truncated)
                should_reset = done or truncated

            data_store.insert(dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=np.float32(reward),
                masks=np.float32(mask_val),
                dones=done_val,
            ))
            total_transitions += 1
            obs = next_obs
            if should_reset:
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                if FLAGS.save_eval_video:
                    eval_info, trajs = evaluate_with_trajectories(
                        policy_fn=partial(agent.sample_actions, argmax=True),
                        env=eval_env,
                        num_episodes=FLAGS.eval_n_trajs,
                    )
                    episode_successes = [
                        _get_success_from_info(traj["info"][-1]) if traj["info"] else 0.0
                        for traj in trajs
                    ]
                    eval_info["success"] = float(np.mean(episode_successes))
                    image_keys = [k for k in eval_env.observation_space.keys() if k != "state"]
                    os.makedirs(FLAGS.eval_video_dir, exist_ok=True)
                    for ep_idx, traj in enumerate(trajs):
                        _trajectory_frames_to_mp4(
                            traj, image_keys,
                            os.path.join(FLAGS.eval_video_dir, f"step_{step}_ep{ep_idx}.mp4"),
                        )
                else:
                    eval_info = evaluate_with_episode_success(
                        policy_fn=partial(agent.sample_actions, argmax=True),
                        env=eval_env,
                        num_episodes=FLAGS.eval_n_trajs,
                    )
            client.request("send-stats", {
                "eval": eval_info,
                "actor_total_transitions": total_transitions,
            })

        timer.tock("total")
        if step % FLAGS.log_period == 0:
            client.request("send-stats", {
                "timer": timer.get_average_times(),
                "actor_total_transitions": total_transitions,
            })


# ─────────────────────────────────────────────────────────────────────────────
# Learner
# ─────────────────────────────────────────────────────────────────────────────
def learner(
    rng,
    agent: DrQAgent,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    _exp = (FLAGS.exp_name or FLAGS.env) + ("_residual" if FLAGS.use_residual else "")
    wandb_logger = make_wandb_logger(
        project="maniskill_serl_meanflow",
        description=_exp,
        debug=FLAGS.debug,
    )

    update_steps = 0
    actor_stats = {"last_total_transitions": 0, "total_at_training_start": None}

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats"
        if "actor_total_transitions" in payload:
            actor_stats["last_total_transitions"] = payload["actor_total_transitions"]
            if actor_stats["total_at_training_start"] is None:
                actor_stats["total_at_training_start"] = payload["actor_total_transitions"]
        if wandb_logger:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(
        make_trainer_config(
            port_number=FLAGS.server_port,
            broadcast_port=FLAGS.broadcast_port,
        ),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Wait for buffer to fill
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling replay buffer",
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("Sent initial network to actor")

    # Demo mixing
    if demo_buffer is not None and FLAGS.demo_batch_size > 0 and len(demo_buffer) > 0:
        demo_bs = min(FLAGS.demo_batch_size, FLAGS.batch_size - 1)
        online_bs = FLAGS.batch_size - demo_bs
        demo_iter = demo_buffer.get_iterator(
            sample_args={"batch_size": demo_bs, "pack_obs_and_next_obs": True},
            device=sharding.replicate(),
        )
        print_green(f"Demo mixing: {demo_bs} demo + {online_bs} online per batch")
    else:
        online_bs = FLAGS.batch_size
        demo_iter = None

    replay_iter = replay_buffer.get_iterator(
        sample_args={"batch_size": online_bs, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )

    timer = Timer()
    total_grad_updates = 0
    pbar = tqdm.tqdm(total=FLAGS.replay_buffer_capacity, initial=len(replay_buffer),
                     desc="replay buffer")

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        for _ in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iter)
                if demo_iter:
                    batch = concat_batches(batch, next(demo_iter), axis=0)
            with timer.context("train_critics"):
                agent, _ = agent.update_critics(batch)

        with timer.context("train"):
            batch = next(replay_iter)
            if demo_iter:
                batch = concat_batches(batch, next(demo_iter), axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        total_grad_updates += FLAGS.critic_actor_ratio

        if step > 0 and step % FLAGS.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
            current_total = actor_stats["last_total_transitions"]
            baseline = actor_stats["total_at_training_start"] or 0
            new_trans = max(0, current_total - baseline)
            wandb_logger.log({
                "utd/cumulative": total_grad_updates / new_trans if new_trans > 0 else 0,
                "utd/replay_buffer_size": len(replay_buffer),
            }, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            if FLAGS.checkpoint_path:
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state,
                    step=update_steps, keep=20, overwrite=True,
                )

        pbar.update(len(replay_buffer) - pbar.n)
        update_steps += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(_):
    assert FLAGS.batch_size % num_devices == 0

    print_green(f"env={FLAGS.env}  control_mode={FLAGS.control_mode}  "
                f"reward_mode={FLAGS.reward_mode}  robot_uids={FLAGS.robot_uids}")
    print_green(f"use_residual={FLAGS.use_residual}  alpha_scale={FLAGS.alpha_scale}")
    if FLAGS.use_residual:
        assert FLAGS.meanflow_ckpt, "--meanflow_ckpt is required when --use_residual"
        print_green(f"meanflow_ckpt={FLAGS.meanflow_ckpt}")

    rng = jax.random.PRNGKey(FLAGS.seed)

    # Build env (actor wraps with ManiFlow policy; learner uses plain env)
    meanflow_policy: Optional[ManiFlowPolicy] = None
    if FLAGS.actor and FLAGS.use_residual:
        meanflow_policy = ManiFlowPolicy(
            ckpt_path=FLAGS.meanflow_ckpt,
            obs_horizon=FLAGS.meanflow_obs_horizon,
            pred_horizon=FLAGS.meanflow_pred_horizon,
            act_horizon=FLAGS.meanflow_act_horizon,
            num_inference_steps=FLAGS.meanflow_num_inference_steps,
            backbone=FLAGS.meanflow_backbone,
            n_layer=FLAGS.meanflow_n_layer,
            n_head=FLAGS.meanflow_n_head,
            n_emb=FLAGS.meanflow_n_emb,
            img_size=FLAGS.meanflow_img_size,
        )

    env = _wrap_env(_make_env(render_mode="human" if FLAGS.render else None), meanflow_policy)
    image_keys = [k for k in env.observation_space.keys() if k != "state"]

    print("Observation space:", env.observation_space)
    print("Action space:     ", env.action_space)

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
        adaptive_tau_enabled=FLAGS.adaptive_tau_enabled,
    )
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
        )
        print_green(f"Replay buffer created (capacity={FLAGS.replay_buffer_capacity})")

        demo_buffer = None
        if FLAGS.demo_path or FLAGS.preload_rlds_path:
            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                type="memory_efficient_replay_buffer",
                image_keys=image_keys,
                preload_rlds_path=FLAGS.preload_rlds_path,
                preload_data_transform=lambda data, _: data,
            )
            if FLAGS.demo_path:
                if not os.path.exists(FLAGS.demo_path):
                    raise FileNotFoundError(f"Demo file not found: {FLAGS.demo_path}")
                with open(FLAGS.demo_path, "rb") as f:
                    trajs = pkl.load(f)
                print_green(f"Loading {len(trajs)} demo transitions from {FLAGS.demo_path}")
                if trajs:
                    errors = _validate_demo_vs_env(
                        _adapt_demo_for_chunking(trajs[0]),
                        env.observation_space, env.action_space,
                    )
                    if errors:
                        print("\n".join(["[ERROR] Demo shape mismatch:"] + errors))
                        raise ValueError("Demo shape mismatch — see above.")
                for traj in trajs:
                    demo_buffer.insert(_adapt_demo_for_chunking(traj))
            print_green(f"Demo buffer: {len(demo_buffer)} transitions")

        print_green("Starting learner …")
        learner(sampling_rng, agent, replay_buffer, demo_buffer)

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)
        print_green("Starting actor …")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise ValueError("Pass --learner or --actor")


if __name__ == "__main__":
    app.run(main)
