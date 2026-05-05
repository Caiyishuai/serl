#!/usr/bin/env python3

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
import sys
import types
os.environ["WANDB_API_KEY"] = "5f07bbe343d183f389c30a3a6245463dca80ae0e"
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

from typing import Any, Dict, Optional
import pickle as pkl
import gymnasium as gymnasium
import mani_skill.envs
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import (
    evaluate,
    evaluate_with_trajectories,
    flatten,
    add_to,
)
from collections import defaultdict
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
from scipy.spatial.transform import Rotation

FLAGS = flags.FLAGS

# --- 环境参数：与 collect_mp_demos.py 默认值保持一致 ---
flags.DEFINE_string("env", "PushCube-v1", "Name of environment.")
flags.DEFINE_string("obs_mode", "rgb+state", "Observation mode for ManiSkill environment.")
flags.DEFINE_string("control_mode", "pd_ee_delta_pose", "Control mode for ManiSkill environment.")
flags.DEFINE_string("robot_uids", "panda_wristcam", "Robot UIDs for ManiSkill environment.")
flags.DEFINE_string("reward_mode", "normalized_dense",
                    "Reward mode (must match collect_mp_demos). "
                    "Options: normalized_dense, dense, sparse, none.")
flags.DEFINE_integer("max_episode_steps", 200,
                     "Max episode steps (must match collect_mp_demos, default 200).")

flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 1000, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("demo_batch_size", 0,
                     "Demo samples per batch when demo_path is set. "
                     "Online samples = batch_size - demo_batch_size. "
                     "Set to 0 to disable demo mixing. Default 64 (25% demo).")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 10_000_000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 1000, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 100, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

# --- UTD 精确控制：learner 侧阻塞等待，直到 UTD <= target ---
flags.DEFINE_float("target_utd", 0, #0.5
                   "Target UTD (grad_updates / new_transitions). 0 = disabled. "
                   "When >0, learner blocks until enough transitions arrive to maintain this UTD.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("server_port", 5488, "Learner trainer server port (change when running multiple experiments).")
flags.DEFINE_integer("broadcast_port", 5489, "Learner broadcast port (change with server_port).")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean("ignore_terminations", False,
                     "Ignore terminated signal from env (match official SERL demo pattern). "
                     "When True: masks=1.0 always, only reset at truncated (max_episode_steps).")

flags.DEFINE_boolean("debug", False, "Debug mode.")

flags.DEFINE_boolean("adaptive_tau_enabled", False, "Enable adaptive tau for target network update.")

# 新添加的参数
flags.DEFINE_float("critic_loss_threshold", 0.05, "Threshold for critic loss to determine tau adjustment (lower values = smaller dead zone).")
flags.DEFINE_float("tau_min", 0.001, "Minimum tau value.")
flags.DEFINE_float("tau_max", 0.05, "Maximum tau value.")
flags.DEFINE_float("tau_adjust_factor", 1.1, "Factor to multiply/divide tau by (closer to 1.0 = more gradual).")
flags.DEFINE_float("tau_adjust_tolerance", 0.2, "Tolerance range around threshold (dead zone = threshold ± tolerance).")

flags.DEFINE_boolean("potential_reward_shaping", True,
                     "Undo env potential-based reward shaping. "
                     "When True, reward = φ(s') - φ(s) instead of raw φ(s'). "
                     "Recommended for (normalized_)dense reward modes.")

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_bool("save_eval_video", False, "Whether to save evaluation trajectories as MP4 videos.")
flags.DEFINE_string("eval_video_dir", "eval_videos", "Directory to save evaluation MP4 videos.")

# --- Residual policy (AddPolicyActionWrapper) ---
flags.DEFINE_boolean("use_residual", False,
                     "Wrap env with AddPolicyActionWrapper so that DrQ learns a residual "
                     "delta on top of a remote base policy.")
flags.DEFINE_string("policy_uri", "ws://127.0.0.1:9001",
                    "WebSocket URI of the remote base policy (used when use_residual=True).")
flags.DEFINE_float("alpha_scale", 0.1,
                   "Scale factor for the residual delta_action: final = base + delta * alpha_scale.")
flags.DEFINE_string("policy_state_key", "observation/state",
                    "The obs key (in policy's namespace) that holds the state. "
                    "Used to transform 35-dim ManiSkill state → 8-dim eef state for the base policy.")
flags.DEFINE_boolean("eef_from_env", False,
                     "When True, extract 8-dim eef state directly from env.unwrapped.agent API "
                     "(robust to different state layouts). When False, parse from flat state by index.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def _get_success_from_info(info: dict) -> float:
    """从 env info 中取出 success (0/1)，保证返回值在 [0, 1]。"""
    flat = flatten(info)
    for key in ("success", "eval_success", "final.success", "final.eval_success"):
        if key in flat:
            v = flat[key]
            if hasattr(v, "ndim") and v.ndim > 0:
                v = np.asarray(v).flat[0]
            if hasattr(v, "item"):
                return float(np.clip(v.item(), 0.0, 1.0))
            return float(np.clip(float(v), 0.0, 1.0))
    return 0.0


def evaluate_with_episode_success(policy_fn, env, num_episodes: int) -> Dict[str, float]:
    """与 evaluate() 相同，但用「每条轨迹结束时的 success」做平均，得到 [0,1] 的 episode 成功率。
    这样 eval/success 不会出现 >1，且语义是「成功轨迹占比」。"""
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
    # 用 episode 成功率覆盖可能的 per-step 均值，保证 eval/success 恒在 [0,1]
    stats["success"] = float(np.mean(episode_successes))
    return stats


# ============================================================================
# ManiSkill obs → SERL flat dict  (与 collect_mp_demos._extract_obs 等价)
# ============================================================================
import torch
from gymnasium.spaces import Box as GymBox
from gymnasium.spaces import Dict as GymDict


class PotentialBasedRewardWrapper(gymnasium.Wrapper):
    """Undo potential-based reward shaping from ManiSkill dense rewards.

    ManiSkill's (normalized_)dense reward r(s) is a state-dependent potential φ(s).
    This wrapper converts it to difference-based reward:
        r_new(t) = φ(s_{t+1}) - φ(s_t)
    which is equivalent to PBRS with γ=1 and preserves the optimal policy under
    the true (sparse) reward while providing a richer learning signal.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_potential = 0.0

    def _compute_potential_at_current_state(self):
        """Compute φ(s) at the current env state without stepping."""
        base = self.env.unwrapped
        try:
            import torch as _torch
            eval_info = base.evaluate()
            obs = base.get_obs()
            zero_action = _torch.zeros(
                base.action_space.shape, dtype=_torch.float32
            )
            r = base.get_reward(obs=obs, action=zero_action, info=eval_info)
            if _torch.is_tensor(r):
                return float(r.cpu().item() if r.numel() == 1 else r.cpu().numpy().flat[0])
            return float(r)
        except Exception as e:
            print(f"[PotentialBasedRewardWrapper] Cannot compute potential: {e}, using 0.0")
            return 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._compute_potential_at_current_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if hasattr(reward, "cpu"):
            current_potential = float(
                reward.cpu().item() if reward.numel() == 1 else reward.cpu().numpy().flat[0]
            )
        else:
            current_potential = float(reward)

        shaped_reward = current_potential - self._prev_potential
        info["raw_potential"] = current_potential
        info["prev_potential"] = self._prev_potential
        self._prev_potential = current_potential
        return obs, shaped_reward, terminated, truncated, info


class ManiSkillMultiCameraWrapper(gymnasium.Wrapper):
    """将 ManiSkill 原始 obs 转换为 SERL 扁平格式。

    输出 dict:  state → (N,) float32;  每个相机 → (H, W, 3) uint8
    与 collect_mp_demos._extract_obs 完全一致。
    """

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
                if "rgb" not in cam_data:
                    continue
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
        if hasattr(action, "__array__"):
            action = np.asarray(action)
        obs, reward, terminated, truncated, info = self.env.step(action)

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
                if "rgb" not in cam_data:
                    continue
                rgb_data = cam_data["rgb"]
                if torch.is_tensor(rgb_data):
                    rgb_data = rgb_data.cpu().numpy()
                if rgb_data.ndim == 4 and rgb_data.shape[0] == 1:
                    rgb_data = rgb_data[0]
                obs_dict[cam_name] = rgb_data.astype(np.uint8)
        return obs_dict


# ============================================================================
# Residual policy helpers
# ============================================================================
import re


def get_task_prompt_from_env(env) -> str:
    """从 ManiSkill env 的 class docstring 中提取 **Task Description:** 部分。"""
    doc = getattr(env.unwrapped.__class__, "__doc__", None) or ""
    m = re.search(
        r"\*\*Task Description:\*\*\s*\n(.+?)(?:\n\s*\n|\n\s*\*\*|\Z)",
        doc,
        re.DOTALL,
    )
    if m:
        return " ".join(m.group(1).split()).strip()
    return f"Complete the manipulation task: {type(env.unwrapped).__name__}"


def extract_eef_from_flat_state(state: np.ndarray) -> np.ndarray:
    """从 ManiSkill 35-dim flat state 中提取 8-dim eef state（按 index 解析）。

    PushCube-v1 (panda, pd_ee_delta_pose, rgb+state) flat state layout:
      [0:9]   = agent/qpos  (7 arm joints + 2 gripper fingers)
      [9:18]  = agent/qvel  (9)
      [18:25] = extra/tcp_pose raw_pose = [px,py,pz, qw,qx,qy,qz]
      [25:28] = extra/goal_pos
      [28:35] = extra/obj_pose

    返回: (8,) float32 = eef_pos(3) + eef_rot_axis_angle(3) + gripper(2)
    """
    state = np.asarray(state).flatten()
    eef_pos = state[18:21].astype(np.float32)
    q_wxyz = state[21:25]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    rot_axis_angle = Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)
    gripper = state[7:9].astype(np.float32)
    return np.concatenate([eef_pos, rot_axis_angle, gripper]).astype(np.float32)


def extract_eef_from_env(env) -> np.ndarray:
    """直接从 ManiSkill env 的 agent API 获取 8-dim eef state，不依赖 flat state 的 index 布局。

    通过 env.unwrapped.agent 访问:
      - agent.tcp.pose.raw_pose  → [px, py, pz, qw, qx, qy, qz]  (7,)
      - agent.robot.get_qpos()   → 含 gripper finger joints (最后两个)

    返回: (8,) float32 = eef_pos(3) + eef_rot_axis_angle(3) + gripper(2)
    """
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    agent = base_env.agent

    # tcp pose: raw_pose = [px, py, pz, qw, qx, qy, qz]
    tcp_raw = agent.tcp.pose.raw_pose
    if torch.is_tensor(tcp_raw):
        tcp_raw = tcp_raw.cpu().numpy()
    tcp_raw = np.asarray(tcp_raw).flatten()
    eef_pos = tcp_raw[:3].astype(np.float32)
    q_wxyz = tcp_raw[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    rot_axis_angle = Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)

    # gripper: qpos 的最后两个 joint（panda_finger_joint1, panda_finger_joint2）
    qpos = agent.robot.get_qpos()
    if torch.is_tensor(qpos):
        qpos = qpos.cpu().numpy()
    qpos = np.asarray(qpos).flatten()
    gripper = qpos[-2:].astype(np.float32)

    return np.concatenate([eef_pos, rot_axis_angle, gripper]).astype(np.float32)


# ============================================================================
# 环境创建辅助
# ============================================================================
def _make_maniskill_env(render_mode=None):
    """统一创建 ManiSkill 环境，确保参数与 collect_mp_demos 一致。"""
    kwargs = dict(
        obs_mode=FLAGS.obs_mode,
        control_mode=FLAGS.control_mode,
        robot_uids=FLAGS.robot_uids,
        reward_mode=FLAGS.reward_mode,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gymnasium.make(FLAGS.env, **kwargs)


def _wrap_env(env, use_residual=False):
    """ManiSkill → (optional PBRS) → SERL flat obs → (optional Residual) → ChunkingWrapper(obs_horizon=1)"""
    if FLAGS.potential_reward_shaping:
        env = PotentialBasedRewardWrapper(env)
        print_green("PotentialBasedRewardWrapper enabled: reward = φ(s') - φ(s)")
    else:
        print_green("PotentialBasedRewardWrapper disabled: reward = φ(s')")

    task_prompt = get_task_prompt_from_env(env)
    env = ManiSkillMultiCameraWrapper(env)

    if use_residual:
        import sys
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from pi_link.remote_policy import RemotePolicy
        from pi_link.env_wrappers import AddPolicyActionWrapper

        def _select_action(result: dict):
            if "actions" in result:
                return result["actions"]
            if "action" in result:
                return result["action"]
            return result

        policy = RemotePolicy(uri_or_host=FLAGS.policy_uri, select_action=_select_action)
        print_green(f"RemotePolicy connected: {FLAGS.policy_uri}")
        print_green(f"  metadata: {policy.metadata}")
        print_green(f"  task_prompt: {task_prompt!r}")

        policy_state_key = FLAGS.policy_state_key
        use_eef_from_env = FLAGS.eef_from_env
        # 保存一份对底层 env 的引用，用于 extract_eef_from_env
        _raw_env_ref = env

        obs_mapping = {
            "state": "observation/state",
            "base_camera": "observation/image",
            "hand_camera": "observation/wrist_image",
        }

        def _base_policy(obs):
            obs = dict(obs)
            obs["prompt"] = task_prompt
            if policy_state_key in obs:
                if use_eef_from_env:
                    obs[policy_state_key] = extract_eef_from_env(_raw_env_ref)
                else:
                    obs[policy_state_key] = extract_eef_from_flat_state(obs[policy_state_key])
            return policy.step(obs)

        env = AddPolicyActionWrapper(
            env=env,
            base_policy=_base_policy,
            clip_action=False,
            allow_none_delta=True,
            prefetch_base_action=True,
            obs_mapping=obs_mapping,
            filter_unmapped_obs=False,
            remove_time_dim_for_policy=False,
            unbatch_obs_for_policy=False,
            broadcast_policy_action=False,
            alpha_scale=FLAGS.alpha_scale,
        )
        print_green(f"AddPolicyActionWrapper enabled (alpha_scale={FLAGS.alpha_scale}, "
                    f"eef_from_env={use_eef_from_env})")

    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    return env


# ============================================================================
# Demo 加载
# ============================================================================
def _adapt_demo_for_chunking(transition, env_obs_space=None):
    """给 demo transition 加上 ChunkingWrapper(obs_horizon=1) 的前导维度。

    兼容两种格式:
      格式A (无 batch 维): state=(N,),    image=(H,W,3)      → 加一维
      格式B (有 batch 维): state=(1,N),   image=(1,H,W,3)    → 已匹配，不加维
    ChunkingWrapper 期望: state=(1,N),   image=(1,H,W,3)
    """
    adapted = {}
    for k, v in transition.items():
        if k in ("observations", "next_observations"):
            new_obs = {}
            for ok, ov in v.items():
                arr = np.asarray(ov)
                if env_obs_space is not None:
                    space = env_obs_space.spaces.get(ok) if hasattr(env_obs_space, 'spaces') else None
                    if space is not None and arr.shape == space.shape:
                        new_obs[ok] = arr  # 已有正确 shape，无需扩维
                    else:
                        new_obs[ok] = np.expand_dims(arr, axis=0)
                else:
                    new_obs[ok] = np.expand_dims(arr, axis=0)
            adapted[k] = new_obs
        else:
            adapted[k] = v
    return adapted


def _validate_demo_vs_env(demo_sample, env_obs_space, env_act_space):
    """检查 demo transition 的 shape 是否与环境 obs/action space 匹配。"""
    obs = demo_sample["observations"]
    act = demo_sample["actions"]

    errors = []
    for key in env_obs_space:
        expected_shape = env_obs_space[key].shape
        if key not in obs:
            errors.append(f"  obs 缺少 key '{key}' (env 期望 shape {expected_shape})")
            continue
        actual_shape = np.asarray(obs[key]).shape
        if actual_shape != expected_shape:
            errors.append(f"  obs['{key}']: demo shape {actual_shape} != env 期望 {expected_shape}")

    act_arr = np.asarray(act)
    if act_arr.shape != env_act_space.shape:
        errors.append(f"  actions: demo shape {act_arr.shape} != env 期望 {env_act_space.shape}")

    return errors


# ============================================================================
# 视频工具
# ============================================================================
def _image_keys_from_obs(obs):
    return sorted([k for k in obs.keys() if k != "state"])


def _trajectory_frames_to_mp4(trajectory, image_keys, out_path, fps=20):
    obs_list = []
    if trajectory["observation"]:
        obs_list.append(trajectory["observation"][0])
    obs_list.extend(trajectory["next_observation"])
    if not obs_list:
        return
    if not image_keys:
        image_keys = _image_keys_from_obs(obs_list[0])
    frames = []
    for obs in obs_list:
        imgs = []
        for k in image_keys:
            if k not in obs:
                continue
            arr = np.asarray(obs[k])
            if arr.ndim == 4 and arr.shape[0] >= 1:
                arr = arr[-1]
            imgs.append(arr)
        if not imgs:
            continue
        combined = np.concatenate(imgs, axis=1)
        if combined.dtype != np.uint8:
            combined = np.clip(combined, 0, 255).astype(np.uint8)
        frames.append(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[Video] Saved {out_path} ({len(frames)} frames, {w}x{h})")


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """Actor loop."""
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.server_port, broadcast_port=FLAGS.broadcast_port),
        data_store,
        wait_for_server=True,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = _wrap_env(_make_maniskill_env())
    eval_env = RecordEpisodeStatistics(eval_env)

    obs, _ = env.reset()
    done = False

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

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            if hasattr(done, "cpu"):
                done = done.cpu().item() if done.numel() == 1 else done.cpu().numpy()
            if hasattr(truncated, "cpu"):
                truncated = truncated.cpu().item() if truncated.numel() == 1 else truncated.cpu().numpy()
            if hasattr(reward, "cpu"):
                reward = reward.cpu().item() if reward.numel() == 1 else reward.cpu().numpy()

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward

            if FLAGS.ignore_terminations:
                mask_val = 1.0 - float(truncated)
                done_val = bool(truncated)
                should_reset = truncated
            else:
                mask_val = 1.0 - float(done)
                done_val = done or truncated
                should_reset = done or truncated

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=mask_val,
                dones=done_val,
            )
            data_store.insert(transition)

            obs = next_obs
            if should_reset:
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                if FLAGS.save_eval_video:
                    evaluate_info, trajectories = evaluate_with_trajectories(
                        policy_fn=partial(agent.sample_actions, argmax=True),
                        env=eval_env,
                        num_episodes=FLAGS.eval_n_trajs,
                    )
                    # 用每条轨迹最后一步的 success 算 episode 成功率，保证 eval/success 在 [0,1]
                    episode_successes = [
                        _get_success_from_info(traj["info"][-1]) if traj["info"] else 0.0
                        for traj in trajectories
                    ]
                    evaluate_info["success"] = float(np.mean(episode_successes))
                    eval_image_keys = [k for k in eval_env.observation_space.keys() if k != "state"]
                    os.makedirs(FLAGS.eval_video_dir, exist_ok=True)
                    for ep_idx, traj in enumerate(trajectories):
                        out_path = os.path.join(
                            FLAGS.eval_video_dir,
                            f"step_{step}_ep_{ep_idx}.mp4",
                        )
                        _trajectory_frames_to_mp4(traj, eval_image_keys, out_path)
                else:
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
    """Learner loop."""
    wandb_logger = make_wandb_logger(
        project="maniskill_serl_pbr",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    update_steps = 0

    # UTD：用 replay_buffer.total_inserts 实时统计（server 线程插入时自增，无延迟）
    buffer_inserts_at_training_start = [None]  # list for closure mutability

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}

    server = TrainerServer(
        make_trainer_config(port_number=FLAGS.server_port, broadcast_port=FLAGS.broadcast_port),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    if demo_buffer is None or FLAGS.demo_batch_size <= 0:
        online_batch_size = FLAGS.batch_size
        demo_batch_size = 0
        demo_iterator = None
    else:
        demo_batch_size = min(FLAGS.demo_batch_size, FLAGS.batch_size - 1)
        online_batch_size = FLAGS.batch_size - demo_batch_size
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": demo_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )
        print_green(f"Demo mixing: {demo_batch_size} demo + "
                    f"{online_batch_size} online = {FLAGS.batch_size} total "
                    f"({100*demo_batch_size/FLAGS.batch_size:.0f}% demo)")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": online_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    timer = Timer()

    # --- UTD tracking：用 replay_buffer.total_inserts 实时统计 ---
    total_grad_updates = 0
    grad_updates_at_last_log = 0
    inserts_at_last_log = 0

    # 记录训练开始时 buffer 已有的 insert 数作为基线
    buffer_inserts_at_training_start[0] = replay_buffer.total_inserts

    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # UTD 精确控制：训练前阻塞等待，直到有足够 transition 使 UTD <= target
        if FLAGS.target_utd > 0:
            required_transitions = int(
                (total_grad_updates + FLAGS.critic_actor_ratio) / FLAGS.target_utd
            )
            while (replay_buffer.total_inserts - buffer_inserts_at_training_start[0]) < required_transitions:
                time.sleep(0.01)

        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(batch)

        with timer.context("train"):
            batch = next(replay_iterator)
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            # ── DEBUG PATCH: 打印 buffer 数据情况（step 0 和每 500 步）────────
            if update_steps in (0, 500, 1000) or (update_steps > 0 and update_steps % 2000 == 0):
                import numpy as _np
                def _show_batch(name, b):
                    sep = "─" * 60
                    print(f"\n{sep}\n[BufferDebug step={update_steps}] {name}\n{sep}")
                    obs = b.get("observations", {})
                    for k, v in obs.items():
                        arr = _np.asarray(v)
                        if arr.dtype == _np.uint8 or (hasattr(arr, 'dtype') and str(arr.dtype) == 'uint8'):
                            extra = f"uint8 [{arr.min()}, {arr.max()}]"
                        else:
                            extra = (f"[{float(arr.min()):.4f}, {float(arr.max()):.4f}]"
                                     f"  mean={float(arr.mean()):.4f}")
                        print(f"  obs[{k!r:15s}] shape={arr.shape}  dtype={arr.dtype}  {extra}")
                    for field in ("actions", "rewards", "masks", "dones"):
                        if field in b:
                            arr = _np.asarray(b[field])
                            print(f"  {field:10s} shape={arr.shape}  dtype={arr.dtype}"
                                  f"  [{float(arr.min()):.4f}, {float(arr.max()):.4f}]"
                                  f"  mean={float(arr.mean()):.4f}")
                    print(sep)

                # online batch（concat 前的）
                _online = next(replay_iterator)
                _show_batch("online_batch", _online)
                if demo_iterator is not None:
                    _demo = next(demo_iterator)
                    _show_batch("demo_batch", _demo)
                    _show_batch("concat_batch (online+demo)", batch)
            # ── END DEBUG PATCH ──────────────────────────────────────────────

            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        total_grad_updates += FLAGS.critic_actor_ratio

        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

            current_inserts = replay_buffer.total_inserts
            new_transitions = current_inserts - buffer_inserts_at_training_start[0]
            delta_grad = total_grad_updates - grad_updates_at_last_log
            delta_transitions = current_inserts - inserts_at_last_log

            utd_cumulative = (
                total_grad_updates / new_transitions
                if new_transitions > 0 else 0.0
            )
            utd_instantaneous = (
                delta_grad / delta_transitions
                if delta_transitions > 0 else 0.0
            )

            utd_log = {
                "utd/cumulative": utd_cumulative,
                "utd/instantaneous": utd_instantaneous,
                "utd/total_grad_updates": total_grad_updates,
                "utd/new_transitions_since_training": new_transitions,
                "utd/buffer_total_inserts": current_inserts,
                "utd/training_start_baseline": buffer_inserts_at_training_start[0],
                "utd/replay_buffer_size": len(replay_buffer),
                "utd/critic_actor_ratio": FLAGS.critic_actor_ratio,
            }
            if FLAGS.target_utd > 0:
                utd_log["utd/target_utd"] = FLAGS.target_utd
            wandb_logger.log(utd_log, step=update_steps)

            grad_updates_at_last_log = total_grad_updates
            inserts_at_last_log = current_inserts

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)
        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0

    rng = jax.random.PRNGKey(FLAGS.seed)

    # --- 打印环境配置，方便与 collect_mp_demos 对照 ---
    print_green(f"env={FLAGS.env}  obs_mode={FLAGS.obs_mode}  "
                f"control_mode={FLAGS.control_mode}  robot_uids={FLAGS.robot_uids}")
    print_green(f"reward_mode={FLAGS.reward_mode}  max_episode_steps={FLAGS.max_episode_steps}")
    print_green(f"adaptive_tau_enabled={FLAGS.adaptive_tau_enabled}")

    if FLAGS.render:
        env = _make_maniskill_env(render_mode="human")
    else:
        env = _make_maniskill_env()

    # actor 且启用 use_residual 时才包 AddPolicyActionWrapper；
    # learner 不需要连接 remote policy。
    env = _wrap_env(env, use_residual=(FLAGS.actor and FLAGS.use_residual))

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    # 打印 obs/action space 用于 debug
    print("shape of observation space and action space")
    print(env.observation_space)
    print(env.action_space)
    if FLAGS.use_residual:
        print_green(f"use_residual={FLAGS.use_residual}  policy_uri={FLAGS.policy_uri}  "
                    f"alpha_scale={FLAGS.alpha_scale}")

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
        adaptive_tau_enabled=FLAGS.adaptive_tau_enabled,
    )

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

                print_green(f"loading {len(trajs)} demo transitions from {FLAGS.demo_path}")

                # 取第一条做 shape 验证（加 ChunkingWrapper 维度后）
                if trajs:
                    sample_adapted = _adapt_demo_for_chunking(trajs[0], env.observation_space)
                    val_errors = _validate_demo_vs_env(
                        sample_adapted, env.observation_space, env.action_space
                    )
                    if val_errors:
                        print("\033[91m[ERROR] Demo 与环境 shape 不匹配:\033[0m")
                        for e in val_errors:
                            print(f"  {e}")
                        raise ValueError(
                            "Demo data shape mismatch. "
                            "请确认 collect_mp_demos 采集时的 env_id / obs_mode / "
                            "control_mode / robot_uids 与训练参数一致。"
                        )

                for traj in trajs:
                    demo_buffer.insert(_adapt_demo_for_chunking(traj, env.observation_space))

            print_green(f"demo buffer size: {len(demo_buffer)}")
        else:
            demo_buffer = None

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)

        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
