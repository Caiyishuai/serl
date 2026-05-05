"""
一体化数据采集脚本：PPO采集 + 直接生成 SERL 可用的 pkl 文件

替代原来的三步流程:
  1. 3_collect_data_...py     → 生成 .h5 文件
  2. 4_convert_h5_to_pkl.py   → .h5 转 .pkl (重建 transition 结构, clip actions)
  3. fix_maniskill_demo_data.sh → 修复 image/state shape

本脚本在采集过程中直接构建 transition，不需要中间 .h5，也不需要 fix_demo_complete.py。

输出 pkl 格式（与 SERL 兼容）:
  List[dict], 每条 transition 包含:
    observations:      {'state': (1, state_dim), 'base_camera': (1,H,W,3), 'hand_camera': (1,H,W,3)}
    actions:           (n_act,)  float32，是否 clip 由 clip_actions 控制（默认不 clip）
    next_observations: 同 observations
    rewards:           float32 scalar
    masks:             float (0.0 if done, else 1.0)
    dones:             bool

为什么原来需要三步？
  - RecordEpisode 保存的 h5 是轨迹格式（不是 transition 列表），且图像/state 维度被
    squeezed（去掉了 num_envs=1 的 batch 维）。4_convert_h5_to_pkl 重新添加 batch 维、
    重组为 transition 列表并 clip actions。fix_demo_complete 再验证并修正 shape。
  - 本脚本直接在采集时捕获 obs（已有 batch 维），逐步构建 transition，省去中间转换。
"""

import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import shutil
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper


# ─────────────────────────────────────────────────────────────────────────────
# 势函数奖励包装器（与原脚本相同）
# ─────────────────────────────────────────────────────────────────────────────
class PotentialBasedRewardWrapper(gym.Wrapper):
    """normalized_dense reward → difference reward: r_new(t) = φ(s_{t+1}) - φ(s_t)"""

    def __init__(self, env):
        super().__init__(env)
        self._prev_potential = 0.0

    def _compute_potential_at_current_state(self):
        base = self.env.unwrapped
        try:
            import torch as _torch
            eval_info = base.evaluate()
            obs = base.get_obs()
            zero_action = _torch.zeros(base.action_space.shape, dtype=_torch.float32)
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


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent（与原脚本相同）
# ─────────────────────────────────────────────────────────────────────────────
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)), nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)), nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)), nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)), nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)), nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)), nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        from torch.distributions.normal import Normal
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def extract_state_tensor(obs_full, device):
    """提取 state 并转为 (1, state_dim) tensor（供模型推理）."""
    state = obs_full['state']
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()
    if isinstance(state, np.ndarray):
        return torch.from_numpy(state).float().to(device)
    return torch.tensor(state, dtype=torch.float32, device=device)


def obs_to_pkl(obs_full):
    """
    将 vectorized env 的观测转为 SERL pkl 格式。

    ManiSkill num_envs=1 时 obs 自带 batch 维:
      state:  (1, state_dim),  images: (1, H, W, 3)

    _adapt_demo_for_chunking (async_drq_sim_maniskill_ty.py) 期望输入为
    collect_mp_demos 格式（无 batch 维），然后做 expand_dims(axis=0):
      state:  (state_dim,)   → expand → (1, state_dim)
      images: (H, W, 3)      → expand → (1, H, W, 3)

    buffer insert 时对 pixel_key 做 [i] 索引（strip 第 0 维）:
      (1, H, W, 3)[0] → (H, W, 3) 存入  → sample 得 (batch, 2, H, W, 3) ✓

    如果保留 batch 维 (1, ...) 再经过 expand → (1, 1, ...):
      (1, 1, H, W, 3)[0] → (1, H, W, 3) 存入 → sample 得 (batch, 2, 1, H, W, 3) ✗
      与 online (batch, 2, H, W, 3) 不匹配，concat_batches 会出错。
    """
    obs_pkl = {}

    # state: (1, state_dim) → squeeze batch → (state_dim,)
    state = obs_full['state']
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()
    obs_pkl['state'] = state[0].astype(np.float32)  # (state_dim,)

    # 相机图像: (1, H, W, 3) → squeeze batch → (H, W, 3)
    sensor_data = obs_full.get('sensor_data', {})
    for cam_name in ['base_camera', 'hand_camera']:
        if cam_name in sensor_data and 'rgb' in sensor_data[cam_name]:
            rgb = sensor_data[cam_name]['rgb']
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            obs_pkl[cam_name] = rgb[0]  # (H, W, 3), uint8

    return obs_pkl


def scalar_reward(rew):
    """将各种形式的 reward 转为 np.float32 标量."""
    if isinstance(rew, torch.Tensor):
        rew = rew.cpu().numpy()
    if isinstance(rew, np.ndarray):
        return np.float32(rew.flat[0])
    return np.float32(rew)


def is_done(terminated, truncated):
    """从 (1,) 或标量 terminated/truncated 提取 bool done."""
    t = terminated.any() if isinstance(terminated, np.ndarray) else bool(terminated)
    tr = truncated.any() if isinstance(truncated, np.ndarray) else bool(truncated)
    return bool(t or tr)


def save_pkl(transitions, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(transitions, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  已保存: {path}  ({len(transitions)} transitions, {size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 配置参数（按需修改）
# ─────────────────────────────────────────────────────────────────────────────
# checkpoint_path = "runs/PlaceSphere-v1__1_ppo_fast__1__1770879246/ckpt_901.pt"
# env_id          = "PlaceSphere-v1"

checkpoint_path = "/home/caiyishuai/workspace/maniskill-ws/rl/runs/pushcube_ckpt_301.pt"
env_id          = "PushCube-v1"

reward_mode     = "normalized_dense_pbr"   # "sparse" | "dense" | "normalized_dense" | "normalized_dense_pbr"
robot_uids      = "panda_wristcam"

max_steps         = 100   # 单回合最大步数
max_success_data  = 20   # 目标成功回合数
max_episodes      = 300   # 最大尝试回合数

# pkl 输出路径（自动加上成功数量后缀）
# 可自行修改，例如:
#   serl_data_dir = "serl_data"
#   serl_data_dir = "/home/caiyishuai/workspace/maniskill-ws/serl_data"
#   serl_data_dir = "examples/async_drq_sim/maniskill_data"
serl_data_dir = "examples/async_drq_sim/maniskill_data"
pkl_prefix = f"mani_skill_{env_id.replace('-','_').lower()}_{reward_mode}"

# 是否 clip actions 到 [-1, 1]
# PPO 没有 tanh 输出，actions 可能超出 [-1,1]
# 原 4_convert_h5_to_pkl.py 默认 no_clip（与此保持一致）
# 如果下游 DrQ/SAC 使用 TanhNormal 且对范围敏感，可设为 True
clip_actions = True
action_clip_range = (-1.0, 1.0)

# 是否同时保存 h5 + 视频（与原脚本行为一致，可设为 False 以加速）
save_video_and_h5 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 创建环境
# ─────────────────────────────────────────────────────────────────────────────
env_kwargs = dict(
    obs_mode="rgb+state",
    control_mode="pd_ee_delta_pose",
    render_mode="all" if save_video_and_h5 else "sensors",
    sim_backend="physx_cuda",
    robot_uids=robot_uids,
)

use_pbr_wrapper = False
if reward_mode == "sparse":
    env_kwargs["reward_mode"] = "sparse"
elif reward_mode == "normalized_dense_pbr":
    env_kwargs["reward_mode"] = "normalized_dense"
    use_pbr_wrapper = True
elif reward_mode in ("dense", "normalized_dense"):
    env_kwargs["reward_mode"] = reward_mode

print(f"正在创建环境: {env_id}...")
env = gym.make(
    env_id,
    num_envs=1,
    reconfiguration_freq=1,
    human_render_camera_configs=dict(shader_pack="default"),
    **env_kwargs,
)
print("环境创建成功!")

if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)

if use_pbr_wrapper:
    env = PotentialBasedRewardWrapper(env)
    print("已包装 PotentialBasedRewardWrapper")

# 可选：h5 + 视频录制（保持与原脚本相同的备份功能）
if save_video_and_h5:
    output_dir = f"{os.path.dirname(checkpoint_path)}/collected_data_{reward_mode}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        save_trajectory=True,
        save_video=True,
        trajectory_name="trajectory",
        max_steps_per_video=max_steps,
        video_fps=30,
        record_reward=True,
        info_on_video=True,
    )
    print(f"h5/视频将同步保存到: {output_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# 加载 PPO 模型
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n加载模型: {checkpoint_path}")
n_act = math.prod(env.action_space.shape)
if isinstance(env.observation_space, gym.spaces.Dict):
    n_obs = math.prod(env.observation_space['state'].shape)
    print(f"  State 维度: {n_obs}")
    print(f"  图像分辨率: {env.observation_space['sensor_data']['base_camera']['rgb'].shape}")
else:
    n_obs = math.prod(env.observation_space.shape)
    print(f"  观测维度: {n_obs}")
print(f"  动作维度: {n_act}")

agent = Agent(n_obs, n_act, device=device)
agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
agent.eval()
print("模型加载完成!\n")

# ─────────────────────────────────────────────────────────────────────────────
# 数据采集主循环
# ─────────────────────────────────────────────────────────────────────────────
all_transitions = []   # 最终保存的 transitions（仅成功回合）
episode_buffer  = []   # 当前回合的 transitions 缓冲

success_data_count   = 0
total_episodes_tried = 0

print("=" * 80)
print(f"开始采集  目标: {max_success_data} 条成功数据  最大尝试: {max_episodes} 回合")
print("=" * 80 + "\n")

obs_full, _ = env.reset(seed=None)
state_tensor = extract_state_tensor(obs_full, device)

print("观测信息:")
if isinstance(obs_full, dict):
    print(f"  keys: {list(obs_full.keys())}")
    state_np = obs_full['state']
    if isinstance(state_np, torch.Tensor): state_np = state_np.cpu().numpy()
    print(f"  state shape: {state_np.shape}")
    print(f"  base_camera rgb shape: {obs_full['sensor_data']['base_camera']['rgb'].shape}")
print()

try:
    while success_data_count < max_success_data and total_episodes_tried < max_episodes:
        # ── 采集当前 obs（用于 transition 的 observations 字段）──────────────
        current_obs_pkl = obs_to_pkl(obs_full)

        # ── 模型推理（确定性策略）────────────────────────────────────────────
        with torch.no_grad():
            action = agent.actor_mean(state_tensor)  # (1, n_act)

        # ── 环境步进 ─────────────────────────────────────────────────────────
        obs_full, rew, terminated, truncated, info = env.step(action.cpu().numpy())

        # ── 构建 transition ──────────────────────────────────────────────────
        # actions: 压缩 batch 维 → (n_act,)，按配置决定是否 clip
        action_np = action.cpu().numpy()[0].astype(np.float32)
        if clip_actions:
            action_pkl = np.clip(action_np, action_clip_range[0], action_clip_range[1])
        else:
            action_pkl = action_np
        done       = is_done(terminated, truncated)

        episode_buffer.append({
            'observations':      current_obs_pkl,
            'actions':           action_pkl,
            'next_observations': obs_to_pkl(obs_full),
            'rewards':           scalar_reward(rew),
            'masks':             np.float32(0.0 if done else 1.0),
            'dones':             done,
        })

        # ── 更新模型输入 ──────────────────────────────────────────────────────
        state_tensor = extract_state_tensor(obs_full, device)

        # ── 回合结束处理 ──────────────────────────────────────────────────────
        if done:
            total_episodes_tried += 1

            success = info.get("success", False)
            if isinstance(success, (np.ndarray, torch.Tensor)):
                success = bool(success.flat[0] if hasattr(success, 'flat') else success.item())

            episode_return = info.get("episode", {}).get("return", 0.0)
            if isinstance(episode_return, (np.ndarray, torch.Tensor)):
                episode_return = float(episode_return.flat[0] if hasattr(episode_return, 'flat') else episode_return.item())

            if success:
                # 只保留成功回合的 transitions
                all_transitions.extend(episode_buffer)
                success_data_count += 1
                print(f"✓ 回合 {total_episodes_tried:3d} - 成功 | Return: {episode_return:7.2f} | "
                      f"成功数: {success_data_count}/{max_success_data} | "
                      f"累计 transitions: {len(all_transitions)}")
            else:
                print(f"✗ 回合 {total_episodes_tried:3d} - 失败 | Return: {episode_return:7.2f}")

            episode_buffer = []  # 清空缓冲（无论成败）

            # 重置环境
            if success_data_count < max_success_data and total_episodes_tried < max_episodes:
                obs_full, _ = env.reset()
                state_tensor = extract_state_tensor(obs_full, device)

except KeyboardInterrupt:
    print("\n\n用户中断，正在保存已采集的数据...")
except Exception as e:
    print(f"\n\n发生错误: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    env.close()

    print("\n" + "=" * 80)
    print("采集完成!")
    print(f"  总尝试回合数: {total_episodes_tried}")
    print(f"  成功回合数:   {success_data_count}")
    if total_episodes_tried > 0:
        print(f"  成功率:       {success_data_count / total_episodes_tried * 100:.2f}%")
    print(f"  总 transitions: {len(all_transitions)}")

    # ── 保存 pkl ──────────────────────────────────────────────────────────────
    if all_transitions:
        output_pkl = os.path.join(serl_data_dir, f"{pkl_prefix}_{success_data_count}.pkl")
        print(f"\n保存 pkl 文件...")
        save_pkl(all_transitions, output_pkl)

        # 打印示例 transition 结构（帮助验证格式）
        sample = all_transitions[0]
        print("\n示例 transition 结构:")
        print("  observations:")
        for k, v in sample['observations'].items():
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        print(f"  actions:  shape={sample['actions'].shape}, dtype={sample['actions'].dtype}, "
              f"range=[{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]"
              + (" (clipped)" if clip_actions else " (no clip)"))
        print(f"  rewards:  {sample['rewards']}")
        print(f"  masks:    {sample['masks']}")
        print(f"  dones:    {sample['dones']}")
    else:
        print("\n无成功数据，未生成 pkl 文件。")

    if save_video_and_h5 and 'output_dir' in dir():
        print(f"\nh5/视频保存位置: {output_dir}")

    print("=" * 80)
