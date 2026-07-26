"""在 SERL 环境 (venv_serl) 里, 用 ManiSkill 采集的**成功轨迹离线数据**做纯离线 RLPD 训练。

背景 (task B):
  ManiSkill 的 env 依赖 sapien/mani_skill, 只装在 maniskill 的 venv 里; SERL 的 venv_serl
  没有这些, 无法在 SERL 侧跑在线交互。因此这里做**纯离线** (offline RL): 只用 demo buffer,
  不与环境交互, 完全用 ManiSkill 的专家 transition 训练 SERL 的 SACAgent (含论文 adaptive_tau)。

  这验证了跨 venv 的完整闭环:
    maniskill venv 采数据 (官方 motionplanning demo 回放, success=100%)
      → convert_demo_h5.py 转 SERL 原生 stacked 格式
      → serl venv 用 serl_launcher 的 SACAgent 真正训练。

数据: convert_demo_h5.py 产出的 stacked pkl
      dict{observations,next_observations,actions,rewards,masks,dones}, 每个是 np.ndarray。

用法 (在 serl 根目录之外运行):
  venv_serl/bin/python auto_research/scripts/train_serl_offline_maniskill.py \
      --data /path/ms_pushcube_expert_serl.pkl --max_updates 3000 --adaptive_tau
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

# ---- tensorflow stub (与 train_sac_single.py 同, 避免装 tf 摧毁 jax 0.4.35) ----
import types as _types
if "tensorflow" not in sys.modules:
    import importlib.machinery as _mach
    _tf = _types.ModuleType("tensorflow")
    _tf.__spec__ = _mach.ModuleSpec("tensorflow", loader=None)
    _tf.__version__ = "0.0.0-stub"
    class _Tensor:
        pass
    _tf.Tensor = _Tensor
    _io = _types.ModuleType("tensorflow.io")
    _gfile = _types.ModuleType("tensorflow.io.gfile")
    _gfile.GFile = open
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
    _errors = _types.ModuleType("tensorflow.errors")
    _errors.NotFoundError = FileNotFoundError
    _tf.errors = _errors
    sys.modules["tensorflow"] = _tf
    sys.modules["tensorflow.io"] = _io
    sys.modules["tensorflow.io.gfile"] = _gfile
    sys.modules["tensorflow.errors"] = _errors

# ---- agentlace stub ----
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
import numpy as np
from gymnasium import spaces

import flax.linen as nn
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.data.replay_buffer import ReplayBuffer


def make_sac_agent(seed, sample_obs, sample_action, adaptive_tau_enabled=False):
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


def load_stacked(path):
    with open(path, "rb") as f:
        d = pkl.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="convert_demo_h5.py 产出的 SERL stacked pkl")
    ap.add_argument("--max_updates", type=int, default=3000)
    ap.add_argument("--time_limit_min", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--utd_ratio", type=int, default=4)
    ap.add_argument("--adaptive_tau", action="store_true")
    ap.add_argument("--sparse_reward", action="store_true",
                    help="把 dense reward 二值化: reward>=frac*max -> 1, 否则 0")
    ap.add_argument("--sparse_frac", type=float, default=0.7,
                    help="sparse 阈值 = sparse_frac * (该数据集 reward 最大值)")
    ap.add_argument("--log_period", type=int, default=200)
    ap.add_argument("--metrics_csv", type=str, default="",
                    help="若指定, 把 step,critic_loss,actor_loss,tau 每 record_period 步写入该 CSV, 用于画训练曲线")
    ap.add_argument("--record_period", type=int, default=10,
                    help="每多少个 update 记录一次 metrics 到 CSV")
    args = ap.parse_args()

    print(f"[cfg] jax devices = {jax.devices()}")
    d = load_stacked(args.data)
    N = d["observations"].shape[0]
    obs_dim = d["observations"].shape[1]
    act_dim = d["actions"].shape[1]
    print(f"[data] {args.data}")
    print(f"[data] N={N} obs_dim={obs_dim} act_dim={act_dim} "
          f"reward mean={d['rewards'].mean():.3f} max={d['rewards'].max():.3f}")

    # ---- 稀疏奖励: 相对阈值二值化 (各任务 dense reward 量级不同, 用 frac*max 统一) ----
    reward_mode = "dense"
    rewards = d["rewards"].astype(np.float32).copy()
    if args.sparse_reward:
        thr = args.sparse_frac * float(rewards.max())
        pos = float((rewards >= thr).mean())
        rewards = (rewards >= thr).astype(np.float32)
        reward_mode = f"sparse(thr={thr:.3f}={args.sparse_frac:.0%}*max, pos={pos:.1%})"
    d["rewards"] = rewards

    print(f"[cfg] mode=OFFLINE (demo-only) reward={reward_mode} "
          f"adaptive_tau={args.adaptive_tau} utd={args.utd_ratio} batch={args.batch_size}")

    # 用与数据匹配的 Box 空间构造 buffer / agent
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    buffer = ReplayBuffer(obs_space, act_space, capacity=N + 10)
    for i in range(N):
        buffer.insert(dict(
            observations=d["observations"][i],
            next_observations=d["next_observations"][i],
            actions=d["actions"][i],
            rewards=np.float32(d["rewards"][i]),
            masks=np.float32(d["masks"][i]),
            dones=bool(d["dones"][i]),
        ))
    print(f"[buffer] filled {len(buffer)} offline transitions")

    agent = make_sac_agent(
        seed=args.seed,
        sample_obs=obs_space.sample(),
        sample_action=act_space.sample(),
        adaptive_tau_enabled=args.adaptive_tau,
    )

    it = buffer.get_iterator(sample_args={"batch_size": args.batch_size})
    t0 = time.time()
    n_updates = 0
    csv_f = None
    if args.metrics_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_csv)), exist_ok=True)
        csv_f = open(args.metrics_csv, "w")
        csv_f.write("step,critic_loss,actor_loss,tau\n")
    crit = actor = float("nan")
    for step in range(args.max_updates):
        if (time.time() - t0) / 60.0 > args.time_limit_min:
            print(f"[stop] hit time_limit at update {step}")
            break
        batch = next(it)
        agent, info = agent.update_high_utd(batch, utd_ratio=args.utd_ratio)
        n_updates += 1
        crit = float(np.asarray(info["critic"]["critic_loss"]))
        actor = float(np.asarray(info["actor"]["actor_loss"]))
        tau = float(np.asarray(info["curr_tau"])) if "curr_tau" in info else float("nan")
        if csv_f is not None and step % args.record_period == 0:
            csv_f.write(f"{step},{crit:.6f},{actor:.6f},{tau:.6f}\n")
        if step % args.log_period == 0:
            extra = f" tau={tau:.4f}" if tau == tau else ""
            print(f"[upd {step:5d}] t={time.time()-t0:5.1f}s "
                  f"critic_loss={crit:.4f} actor_loss={actor:.4f}{extra}")
    if csv_f is not None:
        csv_f.close()

    # 简单评估: 用学到的 Q 估计 demo batch 上的 Q 值 & 策略动作与专家动作的 MSE
    eval_batch = next(it)
    pi_act = np.asarray(agent.sample_actions(
        observations=jax.device_put(eval_batch["observations"]), argmax=True))
    bc_mse = float(np.mean((pi_act - np.asarray(eval_batch["actions"])) ** 2))
    el = time.time() - t0
    print(f"[done] wall={el:.1f}s updates={n_updates} "
          f"final_critic_loss={crit:.4f} bc_action_mse={bc_mse:.4f}")


if __name__ == "__main__":
    main()
