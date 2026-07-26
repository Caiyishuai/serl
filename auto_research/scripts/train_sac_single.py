"""
单进程 SAC / RLPD 训练脚本 (mac CPU 可跑) — SERL franka_sim PandaPickCube-v0。

对比 SERL 官方分布式 (learner/actor 分进程 + tmux + 网络) 的复杂度, 这里把 actor 采样
与 learner 更新放在同一进程的同一循环里, 便于在 mac 上快速跑通验证。

复用 serl_launcher 的:
  - make_sac_agent (SACAgent.create_states, 纯 MLP)  —— 支持论文 Rsync 的 adaptive_tau
  - ReplayBufferDataStore (在线 buffer + demo buffer)
  - update_high_utd (高 UTD 更新)  + RLPD 50/50 采样 (concat_batches)

模式:
  --demo_path 给定 → RLPD (离线 demo buffer + 在线 buffer 各采半 batch)
  --demo_path 不给 → 纯 online SAC
  --adaptive_tau  → 开启论文 Rsync 自适应 target 同步

用法 (必须在 serl 根目录之外, 脚本已显式加 sys.path):
  venv_serl/bin/python auto_research/scripts/train_sac_single.py \
      --env PandaPickCube-v0 --max_steps 3000 --time_limit_min 8 \
      --demo_path auto_research/data/demos_pickcube_state_20.pkl --adaptive_tau
"""
import argparse
import os
import sys
import time
import pickle as pkl
import numpy as np

SERL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(SERL_ROOT, "franka_sim"))

import warnings
warnings.filterwarnings("ignore")

# ---- tensorflow stub ----
# serl_launcher.common.typing 只用 tf.Tensor 做类型别名, train_utils 只在视频保存里用
# tf.io.gfile。为避免装 tensorflow(与 jax 0.4.35 / numpy 2.x 严重冲突), 注入轻量 stub。
import types as _types
if "tensorflow" not in sys.modules:
    import importlib.machinery as _mach
    _tf = _types.ModuleType("tensorflow")
    # flax.io 用 importlib.util.find_spec('tensorflow') 探测, 需要合法 __spec__
    _tf.__spec__ = _mach.ModuleSpec("tensorflow", loader=None)
    _tf.__version__ = "0.0.0-stub"
    class _Tensor:  # 仅用于类型别名 Union[..., tf.Tensor]
        pass
    _tf.Tensor = _Tensor
    _io = _types.ModuleType("tensorflow.io")
    _gfile = _types.ModuleType("tensorflow.io.gfile")
    _gfile.GFile = open  # 视频保存路径未走到; 退化为内置 open
    # flax.io 需要 gfile 的这些函数做 checkpoint IO
    import os as _os, shutil as _shutil, glob as _glob
    _gfile.exists = _os.path.exists
    _gfile.makedirs = lambda p: _os.makedirs(p, exist_ok=True)
    _gfile.glob = _glob.glob
    _gfile.remove = _os.remove
    _gfile.rmtree = _shutil.rmtree
    _gfile.rename = lambda a, b, overwrite=False: _os.replace(a, b)
    _gfile.copy = lambda a, b, overwrite=False: _shutil.copy(a, b)
    _gfile.isdir = _os.path.isdir
    _gfile.listdir = _os.listdir
    _io.gfile = _gfile
    _tf.io = _io
    # flax.io: from tensorflow import errors as tf_errors; tf_errors.NotFoundError
    _errors = _types.ModuleType("tensorflow.errors")
    _errors.NotFoundError = FileNotFoundError
    _tf.errors = _errors
    sys.modules["tensorflow"] = _tf
    sys.modules["tensorflow.io"] = _io
    sys.modules["tensorflow.io.gfile"] = _gfile
    sys.modules["tensorflow.errors"] = _errors

# ---- agentlace stub ----
# serl_launcher.data.data_store 只用 agentlace 的抽象基类 DataStoreBase (分布式 client/server
# 传输接口)。单进程训练用不到网络传输, 用一个接受 capacity 的空基类即可。
if "agentlace" not in sys.modules:
    _al = _types.ModuleType("agentlace")
    _al_data = _types.ModuleType("agentlace.data")
    _al_ds = _types.ModuleType("agentlace.data.data_store")

    class DataStoreBase:
        def __init__(self, capacity):
            self.capacity = capacity

        def latest_data_id(self):
            return 0

        def get_latest_data(self, from_id):
            raise NotImplementedError

    _al_ds.DataStoreBase = DataStoreBase
    _al_data.data_store = _al_ds
    _al.data = _al_data
    sys.modules["agentlace"] = _al
    sys.modules["agentlace.data"] = _al_data
    sys.modules["agentlace.data.data_store"] = _al_ds

import jax
import gymnasium as gym
import franka_sim  # noqa: F401  触发 env 注册

import flax.linen as nn
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.data.data_store import ReplayBufferDataStore
from serl_launcher.utils.train_utils import concat_batches


def make_sac_agent(seed, sample_obs, sample_action, adaptive_tau_enabled=False):
    """本地版 make_sac_agent, 直接调 SACAgent.create_states, 避开 launcher.py 顶部
    对 tensorflow_datasets 的 import (RLDS 路径, 本任务用不到)。超参照抄官方 launcher。"""
    return SACAgent.create_states(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.99,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
        adaptive_tau_enabled=adaptive_tau_enabled,
        critic_loss_threshold=0.3,
        tau_min=0.005,
        tau_max=0.2,
        tau_adjust_factor=1.2,
        tau_adjust_tolerance=0.4,
    )


class SparseRewardWrapper(gym.RewardWrapper):
    """把 PandaPickCube 的 dense reward 稀疏化。
    环境原文件里 reward_type=='01' 的分支被注释掉了(现有代码 bug),
    这里在外层用 wrapper 实现稀疏奖励: dense reward 超过阈值(接近抬起成功)才给 1, 否则 0。
    thresh 默认 0.6: dense=0.3*r_close+0.7*r_lift, r_lift 需 ~0.5+ 才达标, 对应方块被明显抬起。"""
    def __init__(self, env, thresh=0.6):
        super().__init__(env)
        self.thresh = thresh

    def reward(self, r):
        return 1.0 if float(r) >= self.thresh else 0.0


def build_env(env_name, sparse_reward=False, sparse_thresh=0.6):
    env = gym.make(env_name)
    if sparse_reward:
        env = SparseRewardWrapper(env, thresh=sparse_thresh)
    # PandaPickCube dict obs -> flat vector (官方 async_sac_state_sim 同款处理)
    env = gym.wrappers.FlattenObservation(env)
    return env


def load_demos_into_buffer(demo_path, buffer, obs_space):
    """把 gen_demos 保存的 pkl (list of transition dict, obs 为原始 dict) 展平后插入 buffer。"""
    with open(demo_path, "rb") as f:
        transitions = pkl.load(f)
    from gymnasium.spaces import flatten as gym_flatten
    # transition 里 observations 是原始 dict, 需要按 dict-space 展平成向量
    # 原始 env 的 obs space (未 flatten) 用来做 flatten 依据
    raw_env = gym.make("PandaPickCube-v0")
    raw_obs_space = raw_env.observation_space
    raw_env.close()
    n = 0
    for t in transitions:
        o = gym_flatten(raw_obs_space, t["observations"])
        no = gym_flatten(raw_obs_space, t["next_observations"])
        buffer.insert(dict(
            observations=np.asarray(o, dtype=np.float32),
            next_observations=np.asarray(no, dtype=np.float32),
            actions=np.asarray(t["actions"], dtype=np.float32),
            rewards=np.float32(t["rewards"]),
            masks=np.float32(t["masks"]),
            dones=bool(t["dones"]),
        ))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PandaPickCube-v0")
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--time_limit_min", type=float, default=8.0, help="墙钟上限(分), 到点即停")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--utd_ratio", type=int, default=4)
    ap.add_argument("--random_steps", type=int, default=300)
    ap.add_argument("--training_starts", type=int, default=300)
    ap.add_argument("--demo_path", default=None, help="离线 demo pkl; 给定则走 RLPD")
    ap.add_argument("--adaptive_tau", action="store_true", help="开启论文 Rsync 自适应 τ")
    ap.add_argument("--buffer_capacity", type=int, default=200000)
    ap.add_argument("--log_period", type=int, default=200)
    ap.add_argument("--sparse_reward", action="store_true", help="用 wrapper 把 dense reward 稀疏化(0/1)")
    ap.add_argument("--sparse_thresh", type=float, default=0.6)
    args = ap.parse_args()

    print(f"[cfg] jax devices = {jax.devices()}")
    print(f"[cfg] env={args.env} max_steps={args.max_steps} utd={args.utd_ratio} "
          f"rlpd={'YES' if args.demo_path else 'NO'} adaptive_tau={args.adaptive_tau} "
          f"reward={'SPARSE' if args.sparse_reward else 'DENSE'}")

    rng = jax.random.PRNGKey(args.seed)
    env = build_env(args.env, sparse_reward=args.sparse_reward, sparse_thresh=args.sparse_thresh)
    obs, _ = env.reset(seed=args.seed)
    sample_obs = obs
    sample_action = env.action_space.sample()

    agent = make_sac_agent(
        seed=args.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        adaptive_tau_enabled=args.adaptive_tau,
    )

    online_buffer = ReplayBufferDataStore(
        env.observation_space, env.action_space, capacity=args.buffer_capacity)

    demo_buffer = None
    if args.demo_path:
        demo_buffer = ReplayBufferDataStore(
            env.observation_space, env.action_space, capacity=args.buffer_capacity)
        n = load_demos_into_buffer(args.demo_path, demo_buffer, env.observation_space)
        print(f"[demo] loaded {n} demo transitions into demo_buffer (len={len(demo_buffer)})")

    online_iter = online_buffer.get_iterator(
        sample_args={"batch_size": args.batch_size // (2 if demo_buffer else 1)})
    demo_iter = None
    if demo_buffer:
        demo_iter = demo_buffer.get_iterator(sample_args={"batch_size": args.batch_size // 2})

    t0 = time.time()
    ep_ret, ep_len = 0.0, 0
    n_updates = 0
    returns_hist = []
    obs, _ = env.reset(seed=args.seed)

    for step in range(args.max_steps):
        # 墙钟保护
        if (time.time() - t0) / 60.0 > args.time_limit_min:
            print(f"[stop] hit time_limit {args.time_limit_min} min at step {step}")
            break

        # --- actor: 采样动作 ---
        if step < args.random_steps:
            action = env.action_space.sample()
        else:
            rng, key = jax.random.split(rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs), seed=key, argmax=False)
            action = np.asarray(jax.device_get(action))

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        online_buffer.insert(dict(
            observations=np.asarray(obs, dtype=np.float32),
            next_observations=np.asarray(next_obs, dtype=np.float32),
            actions=np.asarray(action, dtype=np.float32),
            rewards=np.float32(reward),
            masks=np.float32(1.0 - float(term)),  # 截断不算 done mask
            dones=bool(done),
        ))
        obs = next_obs
        ep_ret += float(reward); ep_len += 1
        if done:
            returns_hist.append(ep_ret)
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        # --- learner: 高 UTD 更新 ---
        if step >= args.training_starts and len(online_buffer) >= (args.batch_size // (2 if demo_buffer else 1)):
            batch = next(online_iter)
            if demo_iter is not None and len(demo_buffer) > 0:
                demo_batch = next(demo_iter)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=args.utd_ratio)
            n_updates += 1

            if step % args.log_period == 0:
                el = time.time() - t0
                crit = float(np.asarray(update_info["critic"]["critic_loss"]))
                actor = float(np.asarray(update_info["actor"]["actor_loss"]))
                recent = np.mean(returns_hist[-10:]) if returns_hist else float("nan")
                extra = ""
                if "curr_tau" in update_info:
                    extra = f" tau={float(np.asarray(update_info['curr_tau'])):.4f}"
                print(f"[step {step:5d}] t={el:5.1f}s updates={n_updates} "
                      f"critic_loss={crit:.4f} actor_loss={actor:.4f} "
                      f"recent_return={recent:.3f}{extra} "
                      f"online={len(online_buffer)}")

    el = time.time() - t0
    print(f"[done] steps trained, wall={el:.1f}s, updates={n_updates}, "
          f"episodes={len(returns_hist)}, "
          f"mean_return(all)={np.mean(returns_hist) if returns_hist else float('nan'):.3f}, "
          f"last10={np.mean(returns_hist[-10:]) if returns_hist else float('nan'):.3f}")
    env.close()


if __name__ == "__main__":
    main()
