#!/usr/bin/env python3
"""
分析轨迹pkl文件的观察信息和统计数据
"""

import pickle as pkl
import numpy as np
from pathlib import Path
import sys

def analyze_pkl_file(pkl_path):
    """
    分析单个pkl文件
    
    Args:
        pkl_path: pkl文件路径
    """
    print(f"\n{'='*80}")
    print(f"分析文件: {pkl_path}")
    print(f"{'='*80}\n")
    
    # 加载数据
    with open(pkl_path, 'rb') as f:
        trajs = pkl.load(f)
    
    print(f"总轨迹数量: {len(trajs)}")
    
    # 检查第一个轨迹的结构
    if len(trajs) > 0:
        first_traj = trajs[0]
        print(f"\n轨迹数据结构 (keys): {list(first_traj.keys())}")
        
        # 分析observations结构 - 新格式是字典
        if 'observations' in first_traj:
            obs_data = first_traj['observations']
            print(f"\n--- Observation 结构 ---")
            
            # 新格式：observations是字典，包含state数组和images
            if isinstance(obs_data, dict):
                print(f"Observation keys: {list(obs_data.keys())}")
                
                # 分析state数组
                if 'state' in obs_data:
                    state_array = obs_data['state']
                    print(f"\nState (numpy array): shape={state_array.shape}, dtype={state_array.dtype}")
                    
                    if len(state_array.shape) == 2 and state_array.shape[1] == 22:
                        print(f"✓ State格式正确: (N, 22)")
                        print(f"\n第一个state的解析 (22维):")
                        s = state_array[0]
                        print(f"  [0]: gripper_pose = {s[0]:.4f}")
                        print(f"  [1:4]: tcp_force = {s[1:4]}")
                        print(f"  [4:10]: tcp_pose = {s[4:10]}")
                        print(f"  [10:13]: tcp_torque = {s[10:13]}")
                        print(f"  [13:19]: tcp_vel = {s[13:19]}")
                        print(f"  [19:22]: block_pos = {s[19:22]}")
                    else:
                        print(f"⚠ State shape: {state_array.shape}")
                
                # 分析images信息
                image_keys = [k for k in obs_data.keys() if k != 'state']
                if image_keys:
                    print(f"\nImages keys: {image_keys}")
                    for key in image_keys:
                        value = obs_data[key]
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            
            # 旧格式：observations是列表
            elif isinstance(obs_data, (list, np.ndarray)) and len(obs_data) > 0:
                first_obs = obs_data[0]
                print(f"⚠ 旧格式: observations是列表")
                print(f"Observation keys: {list(first_obs.keys()) if isinstance(first_obs, dict) else 'N/A'}")
                
                if isinstance(first_obs, dict) and 'state' in first_obs:
                    state = first_obs['state']
                    if isinstance(state, dict):
                        print(f"\nState keys: {list(state.keys())}")
                    elif isinstance(state, np.ndarray):
                        print(f"\nState array shape: {state.shape}")
    
    # 统计每个episode的长度
    episode_lengths = []
    for traj in trajs:
        if 'observations' in traj:
            obs_data = traj['observations']
            # 新格式：observations是字典，state是(N, 22)数组
            if isinstance(obs_data, dict) and 'state' in obs_data:
                episode_lengths.append(len(obs_data['state']))
            # 旧格式：observations是列表
            elif isinstance(obs_data, (list, np.ndarray)):
                episode_lengths.append(len(obs_data))
    
    print(f"\n--- Episode 长度统计 ---")
    print(f"Episode数量: {len(episode_lengths)}")
    print(f"平均长度: {np.mean(episode_lengths):.2f} steps")
    print(f"最短长度: {np.min(episode_lengths)} steps")
    print(f"最长长度: {np.max(episode_lengths)} steps")
    print(f"标准差: {np.std(episode_lengths):.2f} steps")
    print(f"中位数: {np.median(episode_lengths):.2f} steps")
    
    # 分析每个episode最后一个step的物体位置信息
    final_block_positions = []
    final_tcp_forces = []
    final_tcp_torques = []
    
    for traj in trajs:
        if 'observations' in traj:
            obs_data = traj['observations']
            
            # 新格式：observations是字典，state是(N, 22)数组
            if isinstance(obs_data, dict) and 'state' in obs_data:
                state_array = obs_data['state']
                if len(state_array) > 0:
                    # 获取最后一个state
                    last_state = state_array[-1]
                    
                    # State格式 (22维):
                    # [0]: gripper_pose, [1:4]: tcp_force, [4:10]: tcp_pose
                    # [10:13]: tcp_torque, [13:19]: tcp_vel, [19:22]: block_pos
                    if len(last_state) >= 22:
                        final_block_positions.append(last_state[19:22])  # block_pos
                        final_tcp_forces.append(last_state[1:4])         # tcp_force
                        final_tcp_torques.append(last_state[10:13])      # tcp_torque
                    elif len(last_state) == 19:
                        # 旧的19维格式（没有block_pos）
                        final_tcp_forces.append(last_state[1:4])
                        final_tcp_torques.append(last_state[10:13])
            
            # 旧格式：observations是列表
            elif isinstance(obs_data, (list, np.ndarray)) and len(obs_data) > 0:
                last_obs = obs_data[-1]
                
                if isinstance(last_obs, dict) and 'state' in last_obs:
                    state = last_obs['state']
                    
                    # 处理字典格式的state
                    if isinstance(state, dict):
                        if 'block_pos' in state:
                            final_block_positions.append(state['block_pos'])
                        if 'tcp_force' in state:
                            final_tcp_forces.append(state['tcp_force'])
                        if 'tcp_torque' in state:
                            final_tcp_torques.append(state['tcp_torque'])
    
    # 统计物体最终位置
    if final_block_positions and len(final_block_positions) > 0:
        final_block_positions = np.array(final_block_positions)
        
        # 确保数据是有效的
        if final_block_positions.size == 0:
            print(f"\n⚠ 没有找到有效的block_pos数据")
            return trajs, episode_lengths, np.array([])
        print(f"\n--- 最后一个Step的物体位置 (block_pos) 统计 ---")
        print(f"样本数量: {len(final_block_positions)}")
        print(f"平均值 (x, y, z): {np.mean(final_block_positions, axis=0)}")
        print(f"标准差 (x, y, z): {np.std(final_block_positions, axis=0)}")
        print(f"最小值 (x, y, z): {np.min(final_block_positions, axis=0)}")
        print(f"最大值 (x, y, z): {np.max(final_block_positions, axis=0)}")
        print(f"中位数 (x, y, z): {np.median(final_block_positions, axis=0)}")
        
        # 分析高度分布
        final_heights = final_block_positions[:, 2]
        print(f"\n物体最终高度 (z) 详细统计:")
        print(f"  平均高度: {np.mean(final_heights):.4f} m")
        print(f"  标准差: {np.std(final_heights):.4f} m")
        print(f"  最低高度: {np.min(final_heights):.4f} m")
        print(f"  最高高度: {np.max(final_heights):.4f} m")
        print(f"  高度范围: {np.max(final_heights) - np.min(final_heights):.4f} m")
    
    # 统计末端力
    if final_tcp_forces and len(final_tcp_forces) > 0:
        final_tcp_forces = np.array(final_tcp_forces)
        print(f"\n--- 最后一个Step的末端力 (tcp_force) 统计 ---")
        print(f"样本数量: {len(final_tcp_forces)}")
        print(f"平均值 (Fx, Fy, Fz): {np.mean(final_tcp_forces, axis=0)}")
        print(f"标准差 (Fx, Fy, Fz): {np.std(final_tcp_forces, axis=0)}")
        print(f"最小值 (Fx, Fy, Fz): {np.min(final_tcp_forces, axis=0)}")
        print(f"最大值 (Fx, Fy, Fz): {np.max(final_tcp_forces, axis=0)}")
        
        # 计算力的模
        force_magnitudes = np.linalg.norm(final_tcp_forces, axis=1)
        print(f"\n力的大小 (magnitude) 统计:")
        print(f"  平均力: {np.mean(force_magnitudes):.4f} N")
        print(f"  标准差: {np.std(force_magnitudes):.4f} N")
        print(f"  最小力: {np.min(force_magnitudes):.4f} N")
        print(f"  最大力: {np.max(force_magnitudes):.4f} N")
    
    # 统计末端力矩
    if final_tcp_torques and len(final_tcp_torques) > 0:
        final_tcp_torques = np.array(final_tcp_torques)
        print(f"\n--- 最后一个Step的末端力矩 (tcp_torque) 统计 ---")
        print(f"样本数量: {len(final_tcp_torques)}")
        print(f"平均值 (Tx, Ty, Tz): {np.mean(final_tcp_torques, axis=0)}")
        print(f"标准差 (Tx, Ty, Tz): {np.std(final_tcp_torques, axis=0)}")
        print(f"最小值 (Tx, Ty, Tz): {np.min(final_tcp_torques, axis=0)}")
        print(f"最大值 (Tx, Ty, Tz): {np.max(final_tcp_torques, axis=0)}")
        
        # 计算力矩的模
        torque_magnitudes = np.linalg.norm(final_tcp_torques, axis=1)
        print(f"\n力矩的大小 (magnitude) 统计:")
        print(f"  平均力矩: {np.mean(torque_magnitudes):.4f} Nm")
        print(f"  标准差: {np.std(torque_magnitudes):.4f} Nm")
        print(f"  最小力矩: {np.min(torque_magnitudes):.4f} Nm")
        print(f"  最大力矩: {np.max(torque_magnitudes):.4f} Nm")
    
    # 额外统计：分析整个轨迹的物体位置变化
    if len(final_block_positions) > 0:
        print(f"\n--- 额外分析: 轨迹中的物体位置变化 ---")
        
        all_block_positions = []
        initial_block_positions = []
        
        for traj in trajs:
            if 'observations' in traj:
                obs_data = traj['observations']
                
                # 新格式：observations是字典
                if isinstance(obs_data, dict) and 'state' in obs_data:
                    state_array = obs_data['state']
                    if len(state_array) > 0:
                        first_state = state_array[0]
                        # 从state数组提取block_pos [19:22]
                        if len(first_state) >= 22:
                            initial_block_positions.append(first_state[19:22])
                
                # 旧格式：observations是列表
                elif isinstance(obs_data, (list, np.ndarray)) and len(obs_data) > 0:
                    # 方式1：从initial_object_pos字段
                    if 'initial_object_pos' in traj and isinstance(traj['initial_object_pos'], np.ndarray):
                        initial_block_positions.append(traj['initial_object_pos'])
                    # 方式2：从第一个observation的state
                    else:
                        first_obs = obs_data[0]
                        if isinstance(first_obs, dict) and 'state' in first_obs:
                            state = first_obs['state']
                            if isinstance(state, dict) and 'block_pos' in state:
                                initial_block_positions.append(state['block_pos'])
        
        if initial_block_positions and len(initial_block_positions) > 0:
            initial_block_positions = np.array(initial_block_positions)
            print(f"\n初始物体位置统计:")
            print(f"  平均值 (x, y, z): {np.mean(initial_block_positions, axis=0)}")
            print(f"  标准差 (x, y, z): {np.std(initial_block_positions, axis=0)}")
            
            # 计算平均提升高度
            initial_heights = initial_block_positions[:, 2]
            avg_lift = np.mean(final_heights) - np.mean(initial_heights)
            print(f"\n平均提升高度: {avg_lift:.4f} m")
            print(f"提升高度范围: [{np.min(final_heights - initial_heights):.4f}, {np.max(final_heights - initial_heights):.4f}] m")
    else:
        # 如果没有找到final_block_positions，返回空数组
        if not final_block_positions or len(final_block_positions) == 0:
            final_block_positions = np.array([])
    
    return trajs, episode_lengths, final_block_positions


def compare_pkl_files(pkl_path1, pkl_path2):
    """
    比较两个pkl文件
    """
    print(f"\n{'#'*80}")
    print(f"# 比较两个PKL文件")
    print(f"{'#'*80}")
    
    trajs1, lengths1, blocks1 = analyze_pkl_file(pkl_path1)
    trajs2, lengths2, blocks2 = analyze_pkl_file(pkl_path2)
    
    # 比较统计
    print(f"\n{'='*80}")
    print(f"对比总结")
    print(f"{'='*80}\n")
    
    print(f"文件1: {Path(pkl_path1).name}")
    print(f"  轨迹数量: {len(trajs1)}")
    if len(lengths1) > 0:
        print(f"  平均episode长度: {np.mean(lengths1):.2f} ± {np.std(lengths1):.2f} steps")
    if isinstance(blocks1, np.ndarray) and blocks1.size > 0 and len(blocks1.shape) == 2:
        print(f"  最终平均高度: {np.mean(blocks1[:, 2]):.4f} ± {np.std(blocks1[:, 2]):.4f} m")
    
    print(f"\n文件2: {Path(pkl_path2).name}")
    print(f"  轨迹数量: {len(trajs2)}")
    if len(lengths2) > 0:
        print(f"  平均episode长度: {np.mean(lengths2):.2f} ± {np.std(lengths2):.2f} steps")
    if isinstance(blocks2, np.ndarray) and blocks2.size > 0 and len(blocks2.shape) == 2:
        print(f"  最终平均高度: {np.mean(blocks2[:, 2]):.4f} ± {np.std(blocks2[:, 2]):.4f} m")
    
    # 统计检验
    if (isinstance(blocks1, np.ndarray) and blocks1.size > 0 and len(blocks1.shape) == 2 and
        isinstance(blocks2, np.ndarray) and blocks2.size > 0 and len(blocks2.shape) == 2):
        try:
            from scipy import stats
            
            print(f"\n--- 统计检验 (最终物体高度) ---")
            t_stat, p_value = stats.ttest_ind(blocks1[:, 2], blocks2[:, 2])
            print(f"T检验 (双尾):")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_value:.6f}")
            if p_value < 0.05:
                print(f"  结论: 两个数据集的最终高度有显著差异 (p < 0.05)")
            else:
                print(f"  结论: 两个数据集的最终高度无显著差异 (p >= 0.05)")
        except Exception as e:
            print(f"\n⚠ 统计检验失败: {e}")


def main():
    """主函数"""
    # 定义pkl文件路径
    pkl_path1 = "/home/chenhaojun/workspace_cys/serl/examples/async_drq_sim/success_trajs_20_force_dense.pkl"
    pkl_path2 = "/home/chenhaojun/workspace_cys/serl/examples/async_drq_sim/fail_trajs_20_force_dense.pkl"
    
    # 检查文件是否存在
    if not Path(pkl_path1).exists():
        print(f"错误: 文件不存在 - {pkl_path1}")
        return
    
    if not Path(pkl_path2).exists():
        print(f"错误: 文件不存在 - {pkl_path2}")
        # 如果第二个文件不存在，只分析第一个
        analyze_pkl_file(pkl_path1)
        return
    
    # 比较两个文件
    compare_pkl_files(pkl_path1, pkl_path2)


if __name__ == "__main__":
    main()

