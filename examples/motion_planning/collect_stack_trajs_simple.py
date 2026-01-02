"""
Stack task data collection script using motion planning
"""
import os
import pickle
import numpy as np
from pathlib import Path

# Import the PandaStackGymEnv from franka_sim
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "franka_sim"))
from franka_sim.envs.panda_stack_gym_env import PandaStackGymEnv

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# 全局 viewer 用于可视化
VIEWER = None



# 运动规划参数
lower_limit = -0.1
upper_limit = 0.1
max_dis = 0.5
min_dis = 0.05


def format_observation(obs):
    """
    将环境的观测格式转换为训练所需格式
    
    输入格式（from env):
    {
        'state': {
            'panda/tcp_pos': (3,),
            'panda/tcp_vel': (3,),
            'panda/gripper_pos': (),
            'target_pillar_pos': (3,)
        },
        'images': {
            'front': (H, W, 3),
            'wrist': (H, W, 3)
        }
    }
    
    输出格式（for training):
    {
        'state': (1, 10),  # [tcp_pos(3), tcp_vel(3), gripper_pos(1), target_pillar_pos(3)]
        'front': (1, H, W, 3),
        'wrist': (1, H, W, 3)
    }
    """
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


def step_collect_data(env, action, data_list, last_observations=None, task_stage=None):
    """执行一步并收集数据"""
    global VIEWER
    
    obs, rew, done, truncated, info = env.step(action)
    
    # 更新可视化
    if VIEWER is not None:
        VIEWER.sync()
    
    # 格式化观测
    formatted_obs = format_observation(obs)
    
    data_dict = {
        'observations': last_observations,
        'actions': action,
        'next_observations': formatted_obs,
        'rewards': rew,
        'masks': 1 - done,
        'dones': truncated or done,
    }
    
    if task_stage is not None:
        data_dict['task_stage'] = task_stage
        
    data_list.append(data_dict)
    return formatted_obs


def go_to_target(env, target_pos, data_list, task_stage=None, dead_zone=0.03):
    """移动到目标位置"""
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)
    max_steps = 200  # 防止无限循环
    step_count = 0
    
    while step_count < max_steps:
        # 从原始观测获取TCP位置
        tcp_pos = raw_obs["state"]["panda/tcp_pos"]
        delta_pos = target_pos - tcp_pos
        dis = np.linalg.norm(delta_pos)
        
        # 到达目标
        if dis < dead_zone:
            break
        
        # 计算移动方向
        direction = delta_pos / (dis + 1e-8)
        
        # 根据距离调整速度
        dis_clipped = np.clip(dis, min_dis, max_dis)
        dis_ratio = (dis_clipped - min_dis) / (max_dis - min_dis)
        speed = 0.1 + dis_ratio * 0.4  # 速度范围 [0.1, 0.5]
        
        # 接近目标时减速
        if dis < 0.1:
            speed *= 0.3
        
        # 限制单步最大移动量
        move = direction * speed
        move = np.clip(move, -0.1, 0.1)
        
        action = np.concatenate([move, [0]])
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        raw_obs = env._compute_observation()
        
        step_count += 1
    
    return obs


def close_gripper(env, data_list, task_stage=None, steps=15):
    """关闭夹爪"""
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)
    action = np.array([0, 0, 0, 1.0])
    
    for i in range(steps):
        last_gripper_pos = raw_obs["state"]["panda/gripper_pos"]
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        raw_obs = env._compute_observation()
        
        # 检查是否已经完全闭合
        if i > 5 and np.abs(raw_obs["state"]["panda/gripper_pos"] - last_gripper_pos) < 0.003:
            break
    
    return obs


def open_gripper(env, data_list, task_stage=None, steps=15):
    """打开夹爪"""
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)
    action = np.array([0, 0, 0, -1.0])
    
    for i in range(steps):
        last_gripper_pos = raw_obs["state"]["panda/gripper_pos"]
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        raw_obs = env._compute_observation()
        
        # 检查是否已经完全打开
        if i > 5 and np.abs(raw_obs["state"]["panda/gripper_pos"] - last_gripper_pos) < 0.003:
            break
    
    return obs


def check_success(env, threshold=0.04):
    """
    检查 stack 任务是否成功
    
    成功条件：block 应该在 pillar 顶部附近
    """
    # 直接从环境传感器读取位置
    block_pos = env._data.sensor("block_pos").data
    pillar_pos = env._data.sensor("target_pillar_pos").data
    
    # 常量
    PILLAR_HEIGHT = 0.08  # pillar 高度
    BLOCK_HEIGHT = 0.04   # block 高度
    
    # pillar 顶部 z 坐标
    pillar_top_z = pillar_pos[2] + PILLAR_HEIGHT / 2
    
    # 理想的 block 中心 z 坐标（在 pillar 顶部）
    target_block_z = pillar_top_z + BLOCK_HEIGHT / 2
    
    # 检查 x, y 方向是否对齐
    dist_xy = np.linalg.norm(block_pos[:2] - pillar_pos[:2])
    
    # 检查 z 方向是否在正确高度
    dist_z = np.abs(block_pos[2] - target_block_z)
    
    # 成功条件：水平距离小于阈值，且垂直距离也小于阈值
    success = (dist_xy < threshold) and (dist_z < threshold)
    
    return success


def collect_one_trajectory(env):
    """收集一条完整的 stack 轨迹"""
    data_list = []
    
    # 重置环境
    env.reset()
    obs = env._compute_observation()
    
    # 获取物体和目标位置 - 直接从环境传感器读取
    block_pos = env._data.sensor("block_pos").data.copy()
    pillar_pos = env._data.sensor("target_pillar_pos").data.copy()
    
    # 保存初始位置 - 包括 block 和 pillar 的位置
    # 形状为 (2, 3)：第一行是 block，第二行是 pillar
    initial_object_pos = np.vstack([block_pos, pillar_pos])
    
    # 常量定义
    PILLAR_HEIGHT = 0.08  # pillar 高度
    BLOCK_HEIGHT = 0.04   # block 高度 (半高 = 0.02)
    SAFE_HEIGHT = 0.15    # 安全高度
    
    # ========== 阶段 0: 移动到 block 正上方 ==========
    target = block_pos.copy()
    target[2] = block_pos[2] + 0.08  # 在 block 上方 8cm
    obs = go_to_target(env, target, data_list, task_stage=0)
    
    # ========== 阶段 1: 下降接近 block ==========
    target[2] = block_pos[2] - 0.015  # 稍微低于 block 中心
    obs = go_to_target(env, target, data_list, task_stage=1)
    
    # ========== 阶段 2: 关闭夹爪抓取 ==========
    obs = close_gripper(env, data_list, task_stage=2)
    
    # ========== 阶段 3: 抬起到安全高度 ==========
    raw_obs = env._compute_observation()
    tcp_pos = raw_obs["state"]["panda/tcp_pos"]
    target = tcp_pos.copy()
    target[2] = SAFE_HEIGHT
    obs = go_to_target(env, target, data_list, task_stage=3)
    
    # ========== 阶段 4: 水平移动到 pillar 上方 ==========
    target = pillar_pos.copy()
    target[2] = SAFE_HEIGHT
    obs = go_to_target(env, target, data_list, task_stage=4)
    
    # ========== 阶段 5: 下降到 pillar 顶部 ==========
    # pillar 顶部 z = pillar_pos[2] + PILLAR_HEIGHT/2 = 0.04 + 0.04 = 0.08
    # block 放置时中心应该在 0.08 + 0.02 = 0.10
    target_z = pillar_pos[2] + PILLAR_HEIGHT/2 + BLOCK_HEIGHT/2 + 0.005  # 稍微高一点
    target[2] = target_z
    obs = go_to_target(env, target, data_list, task_stage=5)
    
    # ========== 阶段 6: 打开夹爪放下 ==========
    obs = open_gripper(env, data_list, task_stage=6)
    
    # 等待几帧让物体稳定
    for _ in range(10):
        action = np.zeros(4)
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=6)
    
    # ========== 阶段 7: 抬起离开 ==========
    raw_obs = env._compute_observation()
    tcp_pos = raw_obs["state"]["panda/tcp_pos"]
    target = tcp_pos.copy()
    target[2] = SAFE_HEIGHT
    obs = go_to_target(env, target, data_list, task_stage=7)
    
    # 最终检查成功
    is_success = check_success(env, threshold=0.04)
    
    return data_list, is_success, initial_object_pos


def main():
    global VIEWER
    
    print(f"ROOT_PATH: {ROOT_PATH}")
    
    # 参数设置
    num_trajectories = 20  # 需要收集的成功轨迹数量
    render_visual = True  # 是否显示可视化窗口
    max_attempts = 100  # 最大尝试次数
    
    # 创建环境 - 使用 rgb_array 模式收集图像数据
    # 即使需要可视化，也使用 rgb_array 避免 EGL 冲突
    render_mode = "rgb_array"
    env = PandaStackGymEnv(render_mode=render_mode, image_obs=True)
    
    # 如果需要可视化，创建独立的 MuJoCo viewer
    if render_visual:
        try:
            import mujoco
            import mujoco.viewer
            # 创建被动 viewer（不会干扰环境）
            VIEWER = mujoco.viewer.launch_passive(env._model, env._data)
            print("✓ 可视化窗口已创建")
        except Exception as e:
            print(f"⚠ 无法创建可视化窗口: {e}")
            print("将继续收集数据但不显示可视化")
            VIEWER = None
    
    # 用于存储轨迹的列表
    trajectories = []
    success_count = 0
    attempt_count = 0
    
    print(f"\n开始收集 {num_trajectories} 条成功轨迹...")
    print(f"最大尝试次数: {max_attempts}")
    
    while success_count < num_trajectories and attempt_count < max_attempts:
        attempt_count += 1
        
        try:
            print(f"\n=== 尝试 {attempt_count}，已成功 {success_count}/{num_trajectories} ===")
            
            # 收集一条轨迹
            data_list, is_success, initial_object_pos = collect_one_trajectory(env)
            
            if is_success:
                # 转换为目标格式
                trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'next_observations': [],
                    'dones': [],
                    'infos': [],
                    'initial_object_pos': initial_object_pos  # shape (2, 3): [block_pos, pillar_pos]
                }
                
                for data in data_list:
                    trajectory['observations'].append(data['observations'])
                    trajectory['actions'].append(data['actions'])
                    trajectory['rewards'].append(data['rewards'])
                    trajectory['next_observations'].append(data['next_observations'])
                    trajectory['dones'].append(data['dones'])
                    trajectory['infos'].append({})
                
                trajectories.append(trajectory)
                print(f"✓ 轨迹成功！包含 {len(data_list)} 个转换")
                success_count += 1
            else:
                print(f"✗ 轨迹失败（未成功堆叠）")
            
        except Exception as e:
            print(f"✗ 轨迹 {attempt_count} 收集异常: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 关闭可视化窗口
    if VIEWER is not None:
        VIEWER.close()
    
    env.close()
    
    # 计算成功率和总转换数
    total_transitions = sum(len(traj['observations']) for traj in trajectories)
    success_rate = (success_count / attempt_count * 100) if attempt_count > 0 else 0
    print(f"\n" + "="*60)
    print(f"数据收集完成！")
    print(f"成功轨迹: {success_count}/{attempt_count} ({success_rate:.1f}%)")
    print(f"总转换数: {total_transitions}")
    print("="*60)
    
    if len(trajectories) == 0:
        print("没有收集到成功数据，退出")
        return
    
    # 保存为单个pkl文件，放在 async_drq_sim 目录下
    save_path = os.path.join(ROOT_PATH, "../async_drq_sim", f"stack_trajs_{success_count}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"\n保存数据到: {save_path}")
    print(f"文件包含 {len(trajectories)} 条轨迹，共 {total_transitions} 个转换")
    print("\n数据收集完成！")


if __name__ == "__main__":
    main()

