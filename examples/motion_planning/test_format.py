"""
测试观测格式转换是否正确
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "franka_sim"))

from franka_sim.envs.panda_stack_gym_env import PandaStackGymEnv
import numpy as np

def format_observation(obs):
    """将环境的观测格式转换为训练所需格式"""
    # 扁平化 state
    state_list = []
    state_list.append(obs['state']['panda/tcp_pos'])  # 3
    state_list.append(obs['state']['panda/tcp_vel'])  # 3
    state_list.append(np.array([obs['state']['panda/gripper_pos']]))  # 1
    state_list.append(obs['state']['target_pillar_pos'])  # 3
    
    state = np.concatenate(state_list).astype(np.float32)  # (10,)
    
    # 组装输出
    formatted_obs = {
        'state': state[np.newaxis, :],  # (1, 10)
        'front': obs['images']['front'][np.newaxis, :],  # (1, H, W, 3)
        'wrist': obs['images']['wrist'][np.newaxis, :]   # (1, H, W, 3)
    }
    
    return formatted_obs

# 创建环境
env = PandaStackGymEnv(render_mode="rgb_array", image_obs=True)
obs, _ = env.reset()

print("原始观测格式:")
print(f"  观测键: {list(obs.keys())}")
if 'state' in obs:
    print(f"  state键: {list(obs['state'].keys())}")
    for key, value in obs['state'].items():
        if isinstance(value, np.ndarray):
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"    {key}: value={value}, type={type(value)}")
if 'images' in obs:
    print(f"  images键: {list(obs['images'].keys())}")
    for key, value in obs['images'].items():
        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")

# 转换格式
formatted_obs = format_observation(obs)

print("\n格式化后的观测:")
print(f"  观测键: {list(formatted_obs.keys())}")
for key, value in formatted_obs.items():
    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")

print("\n✓ 格式转换成功！")
print(f"state维度: {formatted_obs['state'].shape} (期望: (1, 10))")
print(f"front维度: {formatted_obs['front'].shape} (期望: (1, 128, 128, 3))")
print(f"wrist维度: {formatted_obs['wrist'].shape} (期望: (1, 128, 128, 3))")

env.close()

