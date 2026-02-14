import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用GPU 0，避免与其他训练任务冲突

from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import gymnasium as gym
import torch
import torch.nn as nn
from collections import defaultdict
import shutil
import math
import numpy as np


# 工具函数：从Dict观测中提取state
def extract_state_from_obs(obs):
    """
    从rgb+state的Dict观测中提取state部分
    obs: Dict包含 'sensor_param', 'sensor_data', 'state'
    返回: numpy array of state
    """
    if isinstance(obs, dict):
        # 提取state部分
        state = obs['state']
        # 确保是numpy数组
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return state
    else:
        # 如果不是Dict，直接返回
        return obs


# PPO Agent 定义
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        from torch.distributions.normal import Normal
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


# 配置参数
checkpoint_path = "runs/PlaceSphere-v1__1_ppo_fast__1__1770879246/ckpt_901.pt"
env_id = "PlaceSphere-v1" #"PlaceSphere-v1"  # PushCube-v1 PickCube-v1 PokeCube-v1 PlaceSphere-v1

robot_uids = "panda_wristcam"  # 可选: "panda", "panda_wristcam", "fetch" 等
max_steps = 100
max_success_data = 100  # 需要的成功数据条数
max_episodes = 100  # 最大尝试回合数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



env_kwargs = dict(
    obs_mode="rgb+state",  # 使用rgb+state模式来采集完整数据（含图像+state）
    control_mode="pd_ee_delta_pose",  # 匹配训练脚本的control_mode
    render_mode="all",  # 使用"all"模式录制传感器相机+人类渲染相机
    sim_backend="physx_cuda",
    robot_uids=robot_uids  # 设置机器人类型
)

print(f"正在创建环境: {env_id}...")
env = gym.make(
    env_id, 
    num_envs=1, 
    reconfiguration_freq=1,
    human_render_camera_configs=dict(shader_pack="default"),
    **env_kwargs
)
print("环境创建成功!")

# 如果动作空间是字典，需要展平
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)

# 打印可用的相机信息
print("\n环境信息:")
print(f"  机器人: {robot_uids}")
print(f"  观测模式: {env_kwargs['obs_mode']}")
print(f"  控制模式: {env_kwargs['control_mode']}")
print(f"  渲染模式: {env_kwargs['render_mode']}")
print()

output_dir = f"{os.path.dirname(checkpoint_path)}/collected_data"  # 视频保存目录

# 清理旧的视频文件（覆盖机制）
if os.path.exists(output_dir):
    print(f"正在清理旧的视频文件夹: {output_dir}")
    shutil.rmtree(output_dir)
    print("旧文件已清理!") 

print(f"正在初始化录制包装器...")
# render_mode="all" 会录制所有相机：
#   - 传感器相机: base_camera + hand_camera (for panda_wristcam)
#   - 人类渲染相机: render_camera (高质量渲染视角)
# save_trajectory=True 会保存轨迹数据，包含：
#   - 完整的rgb+state观测（包括相机图像和state向量）
#   - 动作序列
#   - 奖励、成功标志等信息
env = RecordEpisode(
    env, 
    output_dir=output_dir, 
    save_trajectory=True,  # 保存轨迹数据到.h5文件
    save_video=True,       # 保存视频到.mp4文件
    trajectory_name="trajectory", 
    max_steps_per_video=max_steps, 
    video_fps=30,
    record_reward=True,    # 在视频中显示奖励
    info_on_video=True     # 在视频中显示更多信息
)
print("录制包装器初始化完成!")
print(f"  轨迹数据格式: .h5 (HDF5)")
print(f"  视频格式: .mp4")
print(f"  保存路径: {output_dir}")

# 加载PPO模型
print(f"\n加载训练好的模型...")
n_act = math.prod(env.action_space.shape)

# 处理观测空间 - rgb+state模式下observation_space是Dict
if isinstance(env.observation_space, gym.spaces.Dict):
    # 模型使用state训练，所以只需要state维度
    n_obs = math.prod(env.observation_space['state'].shape)
    print(f"  观测模式: rgb+state (Dict)")
    print(f"  State维度: {n_obs}")
    print(f"  图像分辨率: {env.observation_space['sensor_data']['base_camera']['rgb'].shape}")
else:
    n_obs = math.prod(env.observation_space.shape)
    print(f"  观测维度: {n_obs}")

print(f"  动作维度: {n_act}")

agent = Agent(n_obs, n_act, device=device)
print(f"  加载checkpoint: {checkpoint_path}")
agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
agent.eval()
print("模型加载完成!\n")

success_data_count = 0
total_episodes_tried = 0
step_count = 0

print(f"="*80)
print(f"开始使用训练好的模型采集数据")
print(f"目标: {max_success_data} 条成功数据")
print(f"最大尝试回合数: {max_episodes}")
print(f"数据保存路径: {output_dir}")
print(f"="*80 + "\n")

obs_full, _ = env.reset(seed=None)
# 提取state部分给模型使用（模型是用state训练的）
state = extract_state_from_obs(obs_full)
# 转换为tensor
if isinstance(state, np.ndarray):
    state_tensor = torch.from_numpy(state).float().to(device)
else:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

print(f"观测信息:")
if isinstance(obs_full, dict):
    print(f"  完整观测包含: {list(obs_full.keys())}")
    print(f"  State shape: {state.shape}")
print()

try:
    while success_data_count < max_success_data and total_episodes_tried < max_episodes:
        # 使用模型生成动作（确定性策略，使用actor_mean）
        # 模型只需要state部分
        with torch.no_grad():
            action = agent.actor_mean(state_tensor)
        
        # 执行动作（完整的obs_full用于录制，包含rgb图像）
        obs_full, rew, terminated, truncated, info = env.step(action.cpu().numpy())
        step_count += 1
        
        # 从新的完整观测中提取state给模型
        state = extract_state_from_obs(obs_full)
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        
        # 检查是否有回合完成
        done = terminated or truncated
        if isinstance(done, np.ndarray):
            done = done.any()
            
        if done:
            total_episodes_tried += 1
            
            # 获取当前episode的成功状态
            success = info.get("success", False)
            if isinstance(success, np.ndarray):
                success = success.item()
                
            episode_return = info.get("episode", {}).get("return", 0.0)
            if isinstance(episode_return, (np.ndarray, torch.Tensor)):
                episode_return = episode_return.item()
            
            if success:
                success_data_count += 1
                print(f"✓ 回合 {total_episodes_tried:3d} - 成功 | Return: {episode_return:7.2f} | 成功数: {success_data_count}/{max_success_data}")
            else:
                print(f"✗ 回合 {total_episodes_tried:3d} - 失败 | Return: {episode_return:7.2f}")
            
            # 重置环境（如果还需要继续）
            if success_data_count < max_success_data and total_episodes_tried < max_episodes:
                obs_full, _ = env.reset()
                # 提取state给模型
                state = extract_state_from_obs(obs_full)
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).float().to(device)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                step_count = 0

except KeyboardInterrupt:
    print("\n\n用户中断，正在保存已采集的数据...")
except Exception as e:
    print(f"\n\n发生错误: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # 关闭环境
    env.close()
    
    print("\n" + "="*80)
    print("采集完成!")
    print(f"总尝试回合数: {total_episodes_tried}")
    print(f"成功回合数: {success_data_count}")
    if total_episodes_tried > 0:
        print(f"成功率: {success_data_count/total_episodes_tried*100:.2f}%")
    print(f"\n数据保存位置: {output_dir}")
    
    # 列出保存的文件
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        if files:
            print(f"\n已保存的文件 ({len(files)} 个):")
            video_files = [f for f in files if f.endswith('.mp4')]
            traj_files = [f for f in files if f.endswith('.h5')]
            
            if video_files:
                print(f"  视频文件 ({len(video_files)} 个):")
                for f in video_files[:5]:  # 只显示前5个
                    print(f"    - {f}")
                if len(video_files) > 5:
                    print(f"    ... 还有 {len(video_files)-5} 个视频文件")
            
            if traj_files:
                print(f"  轨迹文件 ({len(traj_files)} 个):")
                for f in traj_files[:5]:  # 只显示前5个
                    print(f"    - {f}")
                if len(traj_files) > 5:
                    print(f"    ... 还有 {len(traj_files)-5} 个轨迹文件")
    
    print("="*80)