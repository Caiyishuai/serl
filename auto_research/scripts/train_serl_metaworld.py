#!/usr/bin/env python3
"""Single-process online SERL/RLPD trainer for MetaWorld v3.

This is the environment-step counterpart to the offline matrix script.  It
supports all eight benchmark tasks, dense or sparse environment rewards, an
optional 50/50 demonstration replay buffer, and fixed/adaptive target-network
tau.  Evaluation always reports MetaWorld ``info["success"]`` independently of
the training reward.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the tested local SAC construction and its lightweight tensorflow /
# agentlace stubs before importing serl_launcher.
from train_serl_offline_maniskill import make_sac_agent  # noqa: E402

from metaworld_common import TASK_SPECS, make_env, success_from_info  # noqa: E402
from serl_launcher.data.data_store import ReplayBufferDataStore  # noqa: E402
from serl_launcher.utils.train_utils import concat_batches  # noqa: E402


def _insert_stacked(path: Path, buffer: ReplayBufferDataStore) -> int:
    with path.open("rb") as file:
        data = pickle.load(file)
    required = {"observations", "next_observations", "actions", "rewards", "masks", "dones"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    count = len(data["rewards"])
    for index in range(count):
        buffer.insert(
            {
                "observations": np.asarray(data["observations"][index], dtype=np.float32),
                "next_observations": np.asarray(data["next_observations"][index], dtype=np.float32),
                "actions": np.asarray(data["actions"][index], dtype=np.float32),
                "rewards": np.float32(data["rewards"][index]),
                "masks": np.float32(data["masks"][index]),
                "dones": bool(data["dones"][index]),
            }
        )
    return count


def _training_reward(mode: str, dense_reward: float, info: dict) -> float:
    if mode == "dense":
        return float(dense_reward)
    if mode == "sparse":
        return float(success_from_info(info))
    raise ValueError(mode)


def evaluate(
    agent,
    task: str,
    reward_mode: str,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> tuple[float, float, float]:
    success_values, dense_returns, train_returns = [], [], []
    for episode in range(episodes):
        env = make_env(task, seed=seed + episode, render=False)
        observation, _ = env.reset(seed=seed + episode)
        dense_return = 0.0
        train_return = 0.0
        succeeded = False
        try:
            for _ in range(max_episode_steps):
                action = agent.sample_actions(observations=jax.device_put(observation), argmax=True)
                action = np.asarray(jax.device_get(action), dtype=np.float32)
                observation, dense_reward, terminated, truncated, info = env.step(action)
                succeeded = succeeded or success_from_info(info)
                dense_return += float(dense_reward)
                train_return += _training_reward(reward_mode, dense_reward, info)
                if terminated or truncated or succeeded:
                    break
        finally:
            env.close()
        success_values.append(float(succeeded))
        dense_returns.append(dense_return)
        train_returns.append(train_return)
    return float(np.mean(success_values)), float(np.mean(dense_returns)), float(np.mean(train_returns))


def _save_agent(agent, checkpoint_dir: Path, step: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"agent_step_{step:09d}.msgpack"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(agent))
    temporary.replace(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online SERL/RLPD for eight MetaWorld tasks")
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--demo-path", type=Path, default=None)
    parser.add_argument("--adaptive-tau", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--random-steps", type=int, default=5_000)
    parser.add_argument("--training-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd-ratio", type=int, default=4)
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit-min", type=float, default=0.0)
    parser.add_argument("--log-period", type=int, default=1_000)
    parser.add_argument("--eval-period", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size % 2 != 0 and args.demo_path:
        raise ValueError("--batch-size must be even with a demo buffer")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    checkpoint_dir = args.output_dir / "checkpoints"

    env = make_env(args.task, seed=args.seed, render=False)
    observation, _ = env.reset(seed=args.seed)
    observation = np.asarray(observation, dtype=np.float32)
    sample_action = np.asarray(env.action_space.sample(), dtype=np.float32)

    agent = make_sac_agent(
        seed=args.seed,
        sample_obs=observation,
        sample_action=sample_action,
        adaptive_tau_enabled=args.adaptive_tau,
    )
    online_buffer = ReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=args.buffer_capacity,
    )

    demo_buffer = None
    if args.demo_path:
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=args.buffer_capacity,
        )
        count = _insert_stacked(args.demo_path, demo_buffer)
        print(f"[demo] loaded {count} transitions from {args.demo_path}")

    online_batch_size = args.batch_size // 2 if demo_buffer is not None else args.batch_size
    online_iterator = online_buffer.get_iterator(sample_args={"batch_size": online_batch_size})
    demo_iterator = (
        demo_buffer.get_iterator(sample_args={"batch_size": args.batch_size // 2})
        if demo_buffer is not None
        else None
    )

    rng = jax.random.PRNGKey(args.seed)
    episode_return = 0.0
    episode_dense_return = 0.0
    episode_length = 0
    episode_success = False
    episodes = 0
    updates = 0
    started = time.time()
    last_info = {}

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "step",
                "episodes",
                "updates",
                "train_return",
                "dense_return",
                "episode_success",
                "eval_success_rate",
                "eval_dense_return",
                "eval_train_return",
                "critic_loss",
                "actor_loss",
                "tau",
            ],
        )
        writer.writeheader()

        for step in range(1, args.max_steps + 1):
            if args.time_limit_min > 0 and (time.time() - started) / 60.0 >= args.time_limit_min:
                print(f"[stop] wall-clock limit at environment step {step}")
                break

            if step <= args.random_steps:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                rng, action_key = jax.random.split(rng)
                action = agent.sample_actions(
                    observations=jax.device_put(observation),
                    seed=action_key,
                    argmax=False,
                )
                action = np.asarray(jax.device_get(action), dtype=np.float32)

            next_observation, dense_reward, terminated, truncated, info = env.step(action)
            next_observation = np.asarray(next_observation, dtype=np.float32)
            episode_length += 1
            episode_success = episode_success or success_from_info(info)
            horizon = episode_length >= args.max_episode_steps
            done = bool(terminated or truncated or horizon or episode_success)
            reward = _training_reward(args.reward_mode, dense_reward, info)

            online_buffer.insert(
                {
                    "observations": observation,
                    "next_observations": next_observation,
                    "actions": action,
                    "rewards": np.float32(reward),
                    "masks": np.float32(0.0 if done else 1.0),
                    "dones": done,
                }
            )
            observation = next_observation
            episode_return += reward
            episode_dense_return += float(dense_reward)

            if step >= args.training_starts and len(online_buffer) >= online_batch_size:
                batch = next(online_iterator)
                if demo_iterator is not None:
                    batch = concat_batches(batch, next(demo_iterator), axis=0)
                agent, last_info = agent.update_high_utd(batch, utd_ratio=args.utd_ratio)
                updates += 1

            completed = None
            if done:
                completed = (episode_return, episode_dense_return, float(episode_success))
                episodes += 1
                observation, _ = env.reset(seed=args.seed + episodes)
                observation = np.asarray(observation, dtype=np.float32)
                episode_return = 0.0
                episode_dense_return = 0.0
                episode_length = 0
                episode_success = False

            should_eval = step == 1 or step % args.eval_period == 0
            eval_success = eval_dense = eval_train = float("nan")
            if should_eval:
                eval_success, eval_dense, eval_train = evaluate(
                    agent,
                    args.task,
                    args.reward_mode,
                    args.eval_episodes,
                    args.max_episode_steps,
                    args.seed + 10_000_000 + step,
                )

            if completed is not None or should_eval or step % args.log_period == 0:
                critic = (
                    float(np.asarray(last_info["critic"]["critic_loss"]))
                    if last_info
                    else float("nan")
                )
                actor = (
                    float(np.asarray(last_info["actor"]["actor_loss"]))
                    if last_info
                    else float("nan")
                )
                tau = float(np.asarray(last_info["curr_tau"])) if "curr_tau" in last_info else float("nan")
                writer.writerow(
                    {
                        "step": step,
                        "episodes": episodes,
                        "updates": updates,
                        "train_return": completed[0] if completed else "",
                        "dense_return": completed[1] if completed else "",
                        "episode_success": completed[2] if completed else "",
                        "eval_success_rate": eval_success if should_eval else "",
                        "eval_dense_return": eval_dense if should_eval else "",
                        "eval_train_return": eval_train if should_eval else "",
                        "critic_loss": critic,
                        "actor_loss": actor,
                        "tau": tau,
                    }
                )
                metrics_file.flush()
                if should_eval:
                    print(
                        f"[step {step:8d}] eval_success={eval_success:.3f} "
                        f"eval_dense_return={eval_dense:.2f} updates={updates} tau={tau:.4f}"
                    )

            if args.save_period > 0 and step % args.save_period == 0:
                print(f"[checkpoint] {_save_agent(agent, checkpoint_dir, step)}")

    env.close()
    final_path = _save_agent(agent, checkpoint_dir, step)
    print(f"[done] steps={step} episodes={episodes} updates={updates} checkpoint={final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
