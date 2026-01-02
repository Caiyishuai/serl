"""
使用 motion planning 采集 stack 任务的成功轨迹
参考 go_to_target 和 close_gripper 的方式
"""
import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags

# Import franka_sim to ensure environments are registered
import franka_sim
import franka_sim.envs  # This triggers environment registration
from franka_sim.envs.panda_stack_gym_env import PandaStackGymEnv

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_trajs", 20, "Number of successful trajectories to collect.")
flags.DEFINE_string("output_dir", "./data/stack_trajs", "Output directory for trajectories.")
flags.DEFINE_float("success_threshold", 0.02, "Success threshold for block placement (m).")
flags.DEFINE_boolean("render", True, "Whether to render the environment.")
flags.DEFINE_float("action_scale", 0.1, "Action scale for position control.")

# 位置控制参数
LOWER_LIMIT = -0.1
UPPER_LIMIT = 0.1
MAX_DIS = 0.5
MIN_DIS = 0.05
POSITION_TOLERANCE = 0.01


def step_collect_data(env, action, data_list, last_observations=None, task_stage=None):
    """执行一步并收集数据"""
    obs, rew, done, truncated, info = env.step(action)
    data_dict = {
        'observations': last_observations,
        'actions': action,
        'next_observations': obs,
        'rewards': rew,
        'masks': 1 - done,
        'dones': truncated or done,
    }
    
    if task_stage is not None:
        data_dict['task_stage'] = task_stage
        
    data_list.append(data_dict)
    return obs


def go_to_target(env, target_pos, data_list, obs=None, task_stage=None, max_steps=500):
    """移动到目标位置"""
    if obs is None:
        obs = env._compute_observation()
    step_count = 0
    
    while step_count < max_steps:
        delta_pos = np.clip(target_pos - obs["state"]["panda/tcp_pos"], LOWER_LIMIT, UPPER_LIMIT)
        dis = np.linalg.norm(obs["state"]["panda/tcp_pos"] - target_pos)
        dis = np.clip(dis, MIN_DIS, MAX_DIS)
        
        # 根据距离调整速度
        dis_ratio = (dis - MIN_DIS) / (MAX_DIS - MIN_DIS)
        norm_delta_pos = delta_pos * (0.1 + dis_ratio * 2.5)
        
        action = np.concatenate([norm_delta_pos, [0]])  # 保持当前 gripper 状态
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        
        if FLAGS.render:
            env.render()
        
        # 检查是否到达目标
        if np.linalg.norm(obs["state"]["panda/tcp_pos"] - target_pos) < POSITION_TOLERANCE:
            # 等待速度也降下来
            tcp_vel = obs["state"].get("panda/tcp_vel", np.zeros(3))
            if np.linalg.norm(tcp_vel) < 0.01:
                break
        
        step_count += 1
    
    return obs


def close_gripper(env, data_list, obs=None, task_stage=None, max_steps=100):
    """闭合夹爪"""
    if obs is None:
        obs = env._compute_observation()
    action = np.concatenate([[0, 0, 0], [-1.0]])  # 闭合命令
    
    step_count = 0
    while step_count < max_steps:
        last_gripper_pos = obs["state"]["panda/gripper_pos"]
        if isinstance(last_gripper_pos, np.ndarray):
            last_gripper_pos = float(last_gripper_pos.item() if last_gripper_pos.ndim == 0 else last_gripper_pos[0])
        else:
            last_gripper_pos = float(last_gripper_pos)
        
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        
        if FLAGS.render:
            env.render()
        
        # 检查 gripper 是否停止变化
        current_gripper_pos = obs["state"]["panda/gripper_pos"]
        if isinstance(current_gripper_pos, np.ndarray):
            current_gripper_pos = float(current_gripper_pos.item() if current_gripper_pos.ndim == 0 else current_gripper_pos[0])
        else:
            current_gripper_pos = float(current_gripper_pos)
        
        if np.abs(current_gripper_pos - last_gripper_pos) < 0.005:
            # 如果已经闭合（接近0），停止
            if current_gripper_pos < 0.1:
                break
        
        step_count += 1
    
    return obs


def open_gripper(env, data_list, obs=None, task_stage=None, max_steps=100):
    """张开夹爪"""
    if obs is None:
        obs = env._compute_observation()
    action = np.concatenate([[0, 0, 0], [1.0]])  # 张开命令
    
    step_count = 0
    while step_count < max_steps:
        last_gripper_pos = obs["state"]["panda/gripper_pos"]
        if isinstance(last_gripper_pos, np.ndarray):
            last_gripper_pos = float(last_gripper_pos.item() if last_gripper_pos.ndim == 0 else last_gripper_pos[0])
        else:
            last_gripper_pos = float(last_gripper_pos)
        
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        
        if FLAGS.render:
            env.render()
        
        # 检查 gripper 是否停止变化
        current_gripper_pos = obs["state"]["panda/gripper_pos"]
        if isinstance(current_gripper_pos, np.ndarray):
            current_gripper_pos = float(current_gripper_pos.item() if current_gripper_pos.ndim == 0 else current_gripper_pos[0])
        else:
            current_gripper_pos = float(current_gripper_pos)
        
        if np.abs(current_gripper_pos - last_gripper_pos) < 0.005:
            # 如果已经张开（接近1），停止
            if current_gripper_pos > 0.9:
                break
        
        step_count += 1
    
    return obs


class PositionController:
    """简单的位置控制器：给定目标位置，移动到该位置并停止"""
    
    def __init__(self, action_scale=0.1, kp=2.0, kv=0.5, position_tolerance=0.01, velocity_tolerance=0.01):
        """
        Args:
            action_scale: 动作缩放
            kp: 位置增益
            kv: 速度增益（阻尼）
            position_tolerance: 位置容差，小于此距离认为到达
            velocity_tolerance: 速度容差，小于此速度认为停止
        """
        self.action_scale = action_scale
        self.kp = kp
        self.kv = kv
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.target_pos = None
        self.target_gripper = None
        self.last_tcp_pos = None
        self.last_tcp_vel = None
    
    def set_target(self, target_pos, target_gripper=None):
        """设置目标位置和夹爪状态"""
        self.target_pos = np.asarray(target_pos)
        self.target_gripper = target_gripper
    
    def compute_action(self, tcp_pos, tcp_vel=None):
        """
        计算动作，使夹爪移动到目标位置
        
        Args:
            tcp_pos: 当前 TCP 位置 [x, y, z]
            tcp_vel: 当前 TCP 速度 [vx, vy, vz]（可选）
        
        Returns:
            action: [x, y, z, grasp] 动作向量
        """
        if self.target_pos is None:
            return np.zeros(4)
        
        # 计算位置误差
        pos_error = self.target_pos - tcp_pos
        dist = np.linalg.norm(pos_error)
        
        # 如果已经到达目标位置（距离小于容差），停止移动
        if dist < self.position_tolerance:
            # 检查速度是否也足够小
            if tcp_vel is not None:
                vel_magnitude = np.linalg.norm(tcp_vel)
                if vel_magnitude < self.velocity_tolerance:
                    # 已经到达且停止，返回零动作
                    action_xyz = np.zeros(3)
                else:
                    # 位置到达但还有速度，添加阻尼
                    action_xyz = -self.kv * tcp_vel
            else:
                action_xyz = np.zeros(3)
        else:
            # 计算控制动作：PD 控制器
            # 位置项：朝向目标
            if dist > 1e-6:
                direction = pos_error / dist
            else:
                direction = np.zeros(3)
            
            # 位置控制项
            pos_action = self.kp * pos_error
            
            # 速度阻尼项（如果可用）
            if tcp_vel is not None:
                vel_action = -self.kv * tcp_vel
            else:
                vel_action = np.zeros(3)
            
            # 组合动作
            action_xyz = pos_action + vel_action
            
            # 限制动作幅度，避免过大
            action_magnitude = np.linalg.norm(action_xyz)
            max_action = 1.0  # 归一化动作的最大值
            if action_magnitude > max_action:
                action_xyz = action_xyz / action_magnitude * max_action
        
        # 夹爪控制
        if self.target_gripper is not None:
            action_gripper = self.target_gripper
        else:
            action_gripper = 0.0
        
        action = np.concatenate([action_xyz, [action_gripper]])
        return action
    
    def is_reached(self, tcp_pos, tcp_vel=None):
        """检查是否已到达目标位置"""
        if self.target_pos is None:
            return False
        
        pos_error = self.target_pos - tcp_pos
        dist = np.linalg.norm(pos_error)
        
        if dist < self.position_tolerance:
            if tcp_vel is not None:
                vel_magnitude = np.linalg.norm(tcp_vel)
                return vel_magnitude < self.velocity_tolerance
            return True
        return False


class MotionPlanner:
    """简单的 motion planner 用于 stack 任务"""
    
    def __init__(self, action_scale=0.1):
        self.action_scale = action_scale
        self.PILLAR_HEIGHT = 0.08
        self.BLOCK_HEIGHT = 0.04
        self.SAFE_LIFT_HEIGHT = 0.13  # pillar height + clearance
        self.block_initial_z = None  # 记录 block 的初始 z 位置
        self.last_tcp_pos = None  # 记录上一次 TCP 位置，用于计算速度
        self.grasping_started = False  # 标记是否已经开始抓取
        self.grasp_stable_count = 0  # 抓取稳定计数
        self.grasp_confirmed = False  # 标记抓取是否已确认（一旦确认就不再切换回未抓取状态）
        
        # 使用位置控制器
        self.controller = PositionController(action_scale=action_scale, kp=3.0, kv=0.8, 
                                            position_tolerance=0.01, velocity_tolerance=0.01)
        
    def _compute_smooth_action(self, tcp_pos, target_pos, tcp_vel=None, min_dist=0.01, max_speed=0.8, dead_zone=0.02):
        """
        计算平滑的动作，根据距离调整速度，避免震荡
        
        Args:
            tcp_pos: 当前 TCP 位置
            target_pos: 目标位置
            tcp_vel: 当前 TCP 速度（可选，用于阻尼）
            min_dist: 最小距离阈值，小于此距离时停止
            max_speed: 最大归一化速度（动作幅度）
            dead_zone: 死区大小，在此范围内完全停止
        
        Returns:
            action: 归一化的动作向量 [x, y, z]
        """
        direction = target_pos - tcp_pos
        dist = np.linalg.norm(direction)
        
        # 增大死区，在目标周围更大范围内完全停止，避免抖动
        if dist < dead_zone:
            return np.zeros(3)
        
        # 归一化方向
        if dist < 1e-6:
            return np.zeros(3)
        direction = direction / dist
        
        # 根据距离调整速度：使用更平滑的速度曲线
        # 距离越近，速度越小，避免过冲
        if dist > 0.15:
            # 远距离：使用较大速度
            speed = max_speed
        elif dist > 0.10:
            # 中远距离：线性减小
            speed = max_speed * 0.8 * (dist / 0.15)
        elif dist > 0.05:
            # 中距离：进一步减小
            speed = max_speed * 0.5 * (dist / 0.10)
        elif dist > 0.03:
            # 近距离：大幅减小
            speed = max_speed * 0.3 * (dist / 0.05)
        else:
            # 非常近：使用很小的速度
            speed = max_speed * 0.15 * (dist / 0.03)
        
        # 速度阻尼：检查是否在震荡
        if self.last_tcp_pos is not None:
            last_dist = np.linalg.norm(target_pos - self.last_tcp_pos)
            if dist >= last_dist * 1.01:  # 距离增加超过1%，说明在远离
                # 距离没有减小，可能过冲或震荡，大幅减小速度
                speed = speed * 0.1
            elif dist > last_dist * 0.99:  # 距离减小很慢
                # 接近速度很慢，进一步减小
                speed = speed * 0.4
        
        # 使用速度信息进行阻尼（如果可用）
        if tcp_vel is not None:
            vel_magnitude = np.linalg.norm(tcp_vel)
            # 如果速度方向与目标方向相反，说明在震荡
            if vel_magnitude > 0.005:
                vel_direction = tcp_vel / vel_magnitude
                dot_product = np.dot(direction, vel_direction)
                if dot_product < -0.3:  # 速度方向与目标方向相反
                    speed = speed * 0.2  # 大幅减小速度
                elif dot_product < 0.0:  # 速度方向与目标方向有一定夹角
                    speed = speed * 0.6  # 减小速度
        
        # 在死区附近大幅减小速度，避免抖动
        if dist < dead_zone * 2.0:
            speed = min(speed, 0.1)
        if dist < dead_zone * 1.5:
            speed = min(speed, 0.05)
        
        # 确保速度不会太小导致无法移动，也不会太大导致过冲
        speed = np.clip(speed, 0.02, max_speed)
        
        action = direction * speed
        return action
    
    def plan_action(self, obs, phase):
        """
        根据当前状态和阶段规划动作，使用位置控制器
        
        Phase 0: 移动到 block 上方
        Phase 1: 下降到 block 并抓取
        Phase 2: 提升 block
        Phase 3: 移动到 pillar 上方
        Phase 4: 下降到 pillar 顶部并放置
        Phase 5: 松开并提升
        """
        state = obs["state"]
        tcp_pos = state["panda/tcp_pos"]
        tcp_vel = state.get("panda/tcp_vel", None)  # 获取速度信息（如果可用）
        block_pos = state["block_pos"]
        pillar_pos = state["target_pillar_pos"]
        # Handle both scalar and array cases
        gripper_pos_val = state["panda/gripper_pos"]
        if isinstance(gripper_pos_val, np.ndarray):
            gripper_pos = float(gripper_pos_val.item() if gripper_pos_val.ndim == 0 else gripper_pos_val[0])
        else:
            gripper_pos = float(gripper_pos_val)
        
        # 计算目标位置
        pillar_top_z = pillar_pos[2] + self.PILLAR_HEIGHT / 2
        target_block_z = pillar_top_z + self.BLOCK_HEIGHT / 2
        
        action = np.zeros(4)  # [x, y, z, grasp]
        
        if phase == 0:  # 移动到 block 上方
            target_pos = block_pos.copy()
            target_pos[2] = 0.15  # 在 block 上方 15cm
            self.controller.set_target(target_pos, target_gripper=1.0 if gripper_pos < 0.8 else 0.0)
            action = self.controller.compute_action(tcp_pos, tcp_vel)
            
        elif phase == 1:  # 下降到 block 并抓取
            target_pos = block_pos.copy()
            target_pos[2] = block_pos[2] + 0.01  # 稍微高于 block 中心
            
            # 如果接近 block，开始闭合
            tcp_to_block_dist = np.linalg.norm(tcp_pos - block_pos)
            if tcp_to_block_dist < 0.08:
                if not self.grasping_started:
                    self.grasping_started = True
                    if FLAGS.render:
                        print(f"Starting grasp: dist={tcp_to_block_dist:.3f}, gripper={gripper_pos:.3f}")
                
                # 持续闭合
                if gripper_pos > 0.05:
                    target_gripper = -1.0  # 闭合
                else:
                    target_gripper = -0.3  # 接近闭合，缓慢闭合
            elif self.grasping_started:
                # 已经开始抓取，继续闭合
                if gripper_pos > 0.3:
                    target_gripper = -1.0
                elif gripper_pos > 0.15:
                    target_gripper = -0.8
                elif gripper_pos > 0.05:
                    target_gripper = -0.5
                else:
                    target_gripper = -0.2
            else:
                target_gripper = 0.0  # 保持张开
            
            self.controller.set_target(target_pos, target_gripper=target_gripper)
            action = self.controller.compute_action(tcp_pos, tcp_vel)
                
        elif phase == 2:  # 提升 block
            target_pos = tcp_pos.copy()
            target_pos[2] = self.SAFE_LIFT_HEIGHT
            # 如果 gripper 还没完全闭合，继续闭合
            target_gripper = -1.0 if gripper_pos > 0.2 else 0.0
            self.controller.set_target(target_pos, target_gripper=target_gripper)
            action = self.controller.compute_action(tcp_pos, tcp_vel)
            
        elif phase == 3:  # 移动到 pillar 上方
            target_pos = pillar_pos.copy()
            target_pos[2] = self.SAFE_LIFT_HEIGHT
            self.controller.set_target(target_pos, target_gripper=0.0)  # 保持闭合
            action = self.controller.compute_action(tcp_pos, tcp_vel)
            
        elif phase == 4:  # 下降到 pillar 顶部
            target_pos = pillar_pos.copy()
            target_pos[2] = target_block_z
            self.controller.set_target(target_pos, target_gripper=0.0)  # 保持闭合
            action = self.controller.compute_action(tcp_pos, tcp_vel)
            
        elif phase == 5:  # 松开并提升
            target_pos = tcp_pos.copy()
            target_pos[2] = self.SAFE_LIFT_HEIGHT
            target_gripper = 0.5 if gripper_pos < 0.8 else 0.0  # 缓慢张开
            self.controller.set_target(target_pos, target_gripper=target_gripper)
            action = self.controller.compute_action(tcp_pos, tcp_vel)
        
        # 更新上次位置
        self.last_tcp_pos = tcp_pos.copy()
        
        return action
    
    def get_phase(self, obs, block_initial_z=None):
        """根据当前状态判断应该处于哪个阶段"""
        state = obs["state"]
        tcp_pos = state["panda/tcp_pos"]
        block_pos = state["block_pos"]
        pillar_pos = state["target_pillar_pos"]
        # Handle both scalar and array cases
        gripper_pos_val = state["panda/gripper_pos"]
        if isinstance(gripper_pos_val, np.ndarray):
            gripper_pos = float(gripper_pos_val.item() if gripper_pos_val.ndim == 0 else gripper_pos_val[0])
        else:
            gripper_pos = float(gripper_pos_val)
        
        pillar_top_z = pillar_pos[2] + self.PILLAR_HEIGHT / 2
        block_bottom_z = block_pos[2] - self.BLOCK_HEIGHT / 2
        target_block_z = pillar_top_z + self.BLOCK_HEIGHT / 2
        
        # 检查是否已经成功放置
        dist_xy = np.linalg.norm(block_pos[:2] - pillar_pos[:2])
        dist_z = abs(block_bottom_z - pillar_top_z)
        if dist_xy < 0.03 and dist_z < 0.01 and gripper_pos > 0.5:
            return 6  # 完成
        
        # 检查是否已经抓取到 block
        # 通过检查 block 是否被提升来判断是否抓取（比初始位置高 2cm 以上）
        if block_initial_z is not None:
            block_lifted = block_pos[2] > block_initial_z + 0.02
        else:
            block_lifted = block_pos[2] > 0.05
        is_grasped = gripper_pos < 0.3  # 闭合状态
        tcp_near_block = np.linalg.norm(tcp_pos - block_pos) < 0.05  # TCP 接近 block
        
        # 判断是否成功抓取：gripper 闭合且 block 被提升，或者 TCP 接近 block 且 gripper 闭合且 block 稍微提升
        # 放宽条件：只要 gripper 闭合且 block 稍微提升就认为抓取了
        # 或者如果已经开始抓取且 gripper 已经闭合，也认为抓取了（即使 block 还没提升）
        # 一旦确认抓取，就不再切换回未抓取状态
        if self.grasp_confirmed:
            # 已经确认抓取，保持抓取状态（除非 gripper 完全张开，说明松开了）
            if gripper_pos > 0.7:
                # Gripper 完全张开，重置状态
                self.grasp_confirmed = False
                self.grasping_started = False
                is_grasping = False
            else:
                is_grasping = True
        else:
            # 判断是否抓取：gripper 闭合且 block 被提升，或者 gripper 闭合且接近 block
            is_grasping = (is_grasped and block_lifted) or \
                         (is_grasped and tcp_near_block and block_pos[2] > (block_initial_z + 0.01 if block_initial_z is not None else 0.03)) or \
                         (self.grasping_started and is_grasped and tcp_near_block)  # 已经开始抓取且已闭合
            
            # 如果确认抓取（gripper 闭合且 block 被提升），标记为已确认
            if is_grasped and block_lifted:
                self.grasp_confirmed = True
                if FLAGS.render:
                    print(f"Grasp confirmed: gripper={gripper_pos:.3f}, block_z={block_pos[2]:.3f}, initial_z={block_initial_z:.3f}")
        
        if not is_grasping:
            # 还没抓取
            tcp_to_block_xy = np.linalg.norm(tcp_pos[:2] - block_pos[:2])
            tcp_to_block_z = tcp_pos[2] - block_pos[2]
            
            # 如果已经开始抓取，必须保持在 phase 1，直到抓取成功或明确失败
            if self.grasping_started:
                # 正在闭合过程中，保持在 phase 1
                # 只有在距离 block 很远时才认为抓取失败
                if tcp_to_block_xy > 0.08 or tcp_to_block_z > 0.15:
                    # 距离太远，可能抓取失败，重置状态
                    self.grasping_started = False
                    self.grasp_confirmed = False
                    return 0  # 移动到 block 上方
                else:
                    # 还在抓取范围内，保持在 phase 1
                    return 1
            
            # 如果水平距离较远或高度太高，先移动到上方
            if tcp_to_block_xy > 0.03 or tcp_to_block_z > 0.1:
                return 0  # 移动到 block 上方
            else:
                return 1  # 下降到 block
        else:
            # 已经抓取
            # Phase 2: 提升 block
            # 如果 block 还没被提升到安全高度，继续提升
            # 一旦进入 phase 2，就保持在 phase 2，不再切换回 phase 1
            if block_pos[2] < self.SAFE_LIFT_HEIGHT - 0.02:  # 留一些余量
                return 2  # 提升 block（在 phase 2 中会继续闭合 gripper）
            # Phase 3: 移动到 pillar 上方
            tcp_to_pillar_xy = np.linalg.norm(tcp_pos[:2] - pillar_pos[:2])
            if tcp_to_pillar_xy > 0.03:
                return 3  # 移动到 pillar 上方
            # Phase 4: 下降到 pillar 顶部
            tcp_to_target_z = tcp_pos[2] - target_block_z
            if tcp_to_target_z > 0.02:  # 如果还在目标上方
                return 4  # 下降到 pillar 顶部
            # Phase 5: 松开
            else:
                return 5  # 松开
        
        return 0


def check_success(obs, threshold=0.02):
    """检查任务是否成功"""
    state = obs["state"]
    block_pos = state["block_pos"]
    pillar_pos = state["target_pillar_pos"]
    # Handle both scalar and array cases
    gripper_pos_val = state["panda/gripper_pos"]
    if isinstance(gripper_pos_val, np.ndarray):
        gripper_pos = float(gripper_pos_val.item() if gripper_pos_val.ndim == 0 else gripper_pos_val[0])
    else:
        gripper_pos = float(gripper_pos_val)
    
    PILLAR_HEIGHT = 0.08
    BLOCK_HEIGHT = 0.04
    
    # 计算 block 底部和 pillar 顶部的距离
    block_bottom_z = block_pos[2] - BLOCK_HEIGHT / 2
    pillar_top_z = pillar_pos[2] + PILLAR_HEIGHT / 2
    
    dist_xy = np.linalg.norm(block_pos[:2] - pillar_pos[:2])
    dist_z = abs(block_bottom_z - pillar_top_z)
    
    # 成功条件：水平距离和垂直距离都小于阈值，且 gripper 已张开
    success = dist_xy < threshold and dist_z < threshold and gripper_pos > 0.5
    
    return success


def main(_):
    # 创建环境
    env = PandaStackGymEnv(
        render_mode="human" if FLAGS.render else "rgb_array",
        image_obs=False,
        reward_type="dense",
        action_scale=np.asarray([FLAGS.action_scale, 1.0]),
    )
    
    # 创建输出目录
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    transition_data_list = []
    success_count = 0
    episode_count = 0
    
    pbar = tqdm(total=FLAGS.num_trajs, desc="Collecting trajectories")
    
    PILLAR_HEIGHT = 0.08
    BLOCK_HEIGHT = 0.04
    SAFE_LIFT_HEIGHT = 0.13
    
    while success_count < FLAGS.num_trajs:
        episode_count += 1
        obs, info = env.reset()
        
        data_list = []
        block_initial_z = obs["state"]["block_pos"][2]
        pillar_pos = obs["state"]["target_pillar_pos"]
        pillar_top_z = pillar_pos[2] + PILLAR_HEIGHT / 2
        target_block_z = pillar_top_z + BLOCK_HEIGHT / 2
        
        try:
            # Phase 0: 移动到 block 上方
            target = obs["state"]["block_pos"].copy()
            target[2] = 0.15  # 在 block 上方 15cm
            obs = go_to_target(env, target, data_list, task_stage=0)
            
            # Phase 1: 下降到 block 并抓取
            target = obs["state"]["block_pos"].copy()
            target[2] = target[2] + 0.01  # 稍微高于 block 中心
            obs = go_to_target(env, target, data_list, task_stage=1)
            
            # 闭合夹爪
            obs = close_gripper(env, data_list, task_stage=1)
            
            # Phase 2: 提升 block
            target = obs["state"]["panda/tcp_pos"].copy()
            target[2] = SAFE_LIFT_HEIGHT
            obs = go_to_target(env, target, data_list, task_stage=2)
            
            # Phase 3: 移动到 pillar 上方
            target = pillar_pos.copy()
            target[2] = SAFE_LIFT_HEIGHT
            obs = go_to_target(env, target, data_list, task_stage=3)
            
            # Phase 4: 下降到 pillar 顶部
            target = pillar_pos.copy()
            target[2] = target_block_z
            obs = go_to_target(env, target, data_list, task_stage=4)
            
            # Phase 5: 松开夹爪
            obs = open_gripper(env, data_list, task_stage=5)
            
            # 检查是否成功
            if check_success(obs, FLAGS.success_threshold):
                # 等待几帧确保稳定
                stable_count = 0
                for _ in range(10):
                    action = np.zeros(4)
                    obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=5)
                    if check_success(obs, FLAGS.success_threshold):
                        stable_count += 1
                
                if stable_count >= 8:
                    transition_data_list.extend(data_list)
                    success_count += 1
                    pbar.update(1)
                    pbar.set_description(
                        f"Success: {success_count}/{FLAGS.num_trajs}, "
                        f"Episodes: {episode_count}, "
                        f"Success rate: {success_count/episode_count*100:.1f}%"
                    )
        except Exception as e:
            if FLAGS.render:
                print(f"Episode {episode_count} failed: {e}")
            continue
    
    pbar.close()
    
    # 保存轨迹
    if transition_data_list:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(
            FLAGS.output_dir,
            f"stack_success_trajs_{len(transition_data_list)}_{timestamp}.pkl"
        )
        with open(output_path, "wb") as f:
            pkl.dump(transition_data_list, f)
        print(f"\n保存了 {len(transition_data_list)} 条成功轨迹到: {output_path}")
        print(f"总尝试次数: {episode_count}")
        print(f"成功率: {success_count/episode_count*100:.2f}%")
        
        # 按 task_stage 分组保存（可选）
        action_agent_num = 6  # 6 个阶段
        action_transition_data = [[] for _ in range(action_agent_num)]
        
        for transition_data in transition_data_list:
            if "task_stage" in transition_data:
                stage = transition_data["task_stage"]
                if 0 <= stage < action_agent_num:
                    action_transition_data[stage].append(transition_data)
        
        for i in range(action_agent_num):
            if action_transition_data[i]:
                stage_output_path = os.path.join(
                    FLAGS.output_dir,
                    f"stack_stage_{i}_{len(action_transition_data[i])}_{timestamp}.pkl"
                )
                with open(stage_output_path, "wb") as f:
                    pkl.dump(action_transition_data[i], f)
                print(f"保存了阶段 {i} 的 {len(action_transition_data[i])} 条数据到: {stage_output_path}")
    else:
        print("\n没有收集到成功的轨迹！")
    
    env.close()


if __name__ == "__main__":
    app.run(main)

