"""
将ManiSkill保存的.h5轨迹文件转换为pkl格式

⚠️ 重要说明：
- 相机名称必须保持ManiSkill原始名称：base_camera, hand_camera
- 不要映射为 front/wrist，否则会导致训练时找不到相机！

输入格式 (.h5):
- 按轨迹组织: traj_0, traj_1, ...
- 每个轨迹包含时间序列数据

输出格式 (.pkl):
- list of transitions (每个transition是一个dict)
- 每个transition包含: observations, actions, next_observations, rewards, masks, dones
- observations 包含: {'base_camera', 'hand_camera', 'state'}
"""

import os
import h5py
import pickle
import numpy as np
from typing import List, Dict
import argparse


def convert_h5_to_pkl(
    h5_file: str,
    output_pkl: str,
    state_dim: int = None,
    camera_mapping: Dict[str, str] = None,
    only_success: bool = False,
    clip_actions: bool = True,
    action_clip_range: tuple = (-1.0, 1.0),
    verbose: bool = True
):
    """
    将.h5文件转换为pkl格式
    
    Args:
        h5_file: 输入的.h5文件路径
        output_pkl: 输出的.pkl文件路径
        state_dim: 要使用的state维度（None表示使用全部）
        camera_mapping: 相机名称映射 {'H5中的名称': 'PKL中的名称'}
                       ⚠️ 默认保持原名称！不要改为front/wrist
        only_success: 是否只保存成功的轨迹
        clip_actions: 是否将actions clip到指定范围（默认True）
                     ⚠️ 重要！PPO采集的actions可能超出[-1,1]，需要clip以匹配DrQ/SAC
        action_clip_range: actions的clip范围（默认[-1, 1]）
        verbose: 是否打印详细信息
    """
    
    if camera_mapping is None:
        # 默认映射：保持ManiSkill原始相机名称
        # base_camera -> base_camera, hand_camera -> hand_camera
        camera_mapping = {
            'base_camera': 'base_camera',
            'hand_camera': 'hand_camera'
        }
    
    print(f"="*80)
    print(f"H5转PKL转换工具")
    print(f"="*80)
    print(f"输入文件: {h5_file}")
    print(f"输出文件: {output_pkl}")
    print(f"State维度: {state_dim if state_dim else '全部'}")
    print(f"相机映射: {camera_mapping}")
    print(f"只保存成功: {only_success}")
    print(f"Clip Actions: {clip_actions}")
    if clip_actions:
        print(f"Action范围: {action_clip_range}")
    print(f"="*80 + "\n")
    
    # 读取h5文件
    transitions = []
    
    # 统计 actions 的原始范围（用于后续分析）
    actions_stats = {
        'original_min': float('inf'),
        'original_max': float('-inf'),
        'clipped_count': 0,
        'total_count': 0
    }
    
    with h5py.File(h5_file, 'r') as f:
        # 获取所有轨迹
        traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
        
        print(f"找到 {len(traj_keys)} 条轨迹\n")
        
        total_transitions = 0
        success_count = 0
        
        for traj_key in traj_keys:
            traj = f[traj_key]
            
            # 检查是否成功（如果需要筛选）
            if only_success and 'success' in traj:
                # success是一个数组，检查最后一步是否成功
                if not traj['success'][-1]:
                    if verbose:
                        print(f"  跳过 {traj_key} (未成功)")
                    continue
            
            # 获取数据
            actions = traj['actions'][:]  # shape: (T, action_dim)
            rewards = traj['rewards'][:]  # shape: (T,)
            
            # 获取终止标志
            if 'terminated' in traj:
                terminated = traj['terminated'][:]
            else:
                terminated = np.zeros(len(actions), dtype=bool)
            
            if 'truncated' in traj:
                truncated = traj['truncated'][:]
            else:
                truncated = np.zeros(len(actions), dtype=bool)
            
            # 获取观测数据
            state = traj['obs/state'][:]  # shape: (T+1, state_dim)
            
            # 提取需要的state维度
            if state_dim is not None and state_dim < state.shape[1]:
                state = state[:, :state_dim]
            
            # 获取相机图像
            cameras = {}
            for source_cam, target_cam in camera_mapping.items():
                cam_path = f'obs/sensor_data/{source_cam}/rgb'
                if cam_path in traj:
                    cameras[target_cam] = traj[cam_path][:]  # shape: (T+1, H, W, 3)
            
            # 轨迹长度
            T = len(actions)
            
            # 转换为transitions
            for t in range(T):
                # 获取原始 action
                action = actions[t].astype(np.float32)
                
                # 统计原始范围
                actions_stats['original_min'] = min(actions_stats['original_min'], float(action.min()))
                actions_stats['original_max'] = max(actions_stats['original_max'], float(action.max()))
                actions_stats['total_count'] += 1
                
                # Clip actions（如果需要）
                if clip_actions:
                    action_original = action.copy()
                    action = np.clip(action, action_clip_range[0], action_clip_range[1]).astype(np.float32)
                    
                    # 检查是否被 clip
                    if not np.allclose(action_original, action):
                        actions_stats['clipped_count'] += 1
                
                # 构建observations (当前时刻)
                observations = {}
                for cam_name, cam_data in cameras.items():
                    # 添加batch维度 (1, H, W, 3)
                    observations[cam_name] = cam_data[t][None, ...]
                
                # 添加state，也加上batch维度 (1, state_dim)
                observations['state'] = state[t][None, ...].astype(np.float32)
                
                # 构建next_observations (下一时刻)
                next_observations = {}
                for cam_name, cam_data in cameras.items():
                    next_observations[cam_name] = cam_data[t+1][None, ...]
                next_observations['state'] = state[t+1][None, ...].astype(np.float32)
                
                # 计算done和mask
                done = bool(terminated[t] or truncated[t])
                mask = 0.0 if done else 1.0
                
                # 创建transition
                transition = {
                    'observations': observations,
                    'actions': action,  # 使用处理后的 action
                    'next_observations': next_observations,
                    'rewards': np.array(rewards[t], dtype=np.float32),
                    'masks': mask,
                    'dones': done
                }
                
                transitions.append(transition)
                total_transitions += 1
            
            if verbose:
                success_str = ""
                if 'success' in traj:
                    success_str = f" (成功: {traj['success'][-1]})"
                print(f"  处理 {traj_key}: {T} 步{success_str}")
            
            if 'success' in traj and traj['success'][-1]:
                success_count += 1
    
    # 保存为pkl
    print(f"\n保存数据到: {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(transitions, f)
    
    # 打印统计信息
    print(f"\n" + "="*80)
    print(f"转换完成!")
    print(f"总轨迹数: {len(traj_keys)}")
    if only_success:
        print(f"成功轨迹数: {success_count}")
    print(f"总transitions数: {total_transitions}")
    
    # 打印 actions 统计
    if clip_actions:
        print(f"\n⚠️ Actions Clipping 统计:")
        print(f"  原始范围: [{actions_stats['original_min']:.4f}, {actions_stats['original_max']:.4f}]")
        print(f"  Clip范围: {action_clip_range}")
        print(f"  被clip的transitions: {actions_stats['clipped_count']}/{actions_stats['total_count']} "
              f"({actions_stats['clipped_count']/actions_stats['total_count']*100:.2f}%)")
        
        if actions_stats['clipped_count'] > 0:
            print(f"\n  ✓ 已自动修正超出范围的actions")
            print(f"  ✓ 数据现在与DrQ/SAC的TanhNormal分布兼容")
        else:
            print(f"\n  ✓ 所有actions已在范围内，无需clip")
    else:
        print(f"\n⚠️ 警告：未启用action clipping")
        print(f"  原始actions范围: [{actions_stats['original_min']:.4f}, {actions_stats['original_max']:.4f}]")
        if actions_stats['original_min'] < -1.0 or actions_stats['original_max'] > 1.0:
            print(f"  ⚠️⚠️⚠️ Actions超出[-1,1]范围！")
            print(f"  这可能导致与DrQ/SAC训练时的actions分布不匹配")
            print(f"  建议：重新运行并启用 --clip_actions")
    
    # 打印示例数据结构
    if len(transitions) > 0:
        sample = transitions[0]
        print(f"\n示例transition结构:")
        print(f"  observations:")
        for key, value in sample['observations'].items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        print(f"  actions: shape={sample['actions'].shape}, dtype={sample['actions'].dtype}")
        print(f"  next_observations:")
        for key, value in sample['next_observations'].items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        print(f"  rewards: shape={sample['rewards'].shape}, dtype={sample['rewards'].dtype}")
        print(f"  masks: {sample['masks']}")
        print(f"  dones: {sample['dones']}")
    
    print(f"="*80)
    
    # ✅ 验证相机名称是否正确
    print(f"\n{'='*80}")
    print(f"✓ 验证生成的PKL文件")
    print(f"{'='*80}")
    
    if len(transitions) > 0:
        obs_keys = set(transitions[0]['observations'].keys())
        expected_keys = {'base_camera', 'hand_camera', 'state'}
        
        if obs_keys == expected_keys:
            print(f"✓ 相机名称正确: {obs_keys}")
            print(f"✓ 此文件可以直接用于ManiSkill训练！")
        else:
            missing = expected_keys - obs_keys
            extra = obs_keys - expected_keys
            print(f"⚠️  警告：相机名称可能不正确！")
            print(f"   当前keys: {obs_keys}")
            print(f"   期望keys: {expected_keys}")
            if missing:
                print(f"   缺少: {missing}")
            if extra:
                print(f"   多余: {extra}")
            print(f"\n⚠️  如果出现 'front', 'wrist' 等名称，训练时会出错！")
            print(f"   请使用 fix_camera_names.py 修复文件")
    
    print(f"="*80)
    
    return transitions


def main():
    # 🎯 默认配置（可以直接运行脚本）
    default_h5_file = "runs/PlaceSphere-v1__1_ppo_fast__1__1770879246/collected_data/trajectory.h5"
    default_output_pkl = "serl_data/mani_skill_place_sphere_100.pkl"
    
    parser = argparse.ArgumentParser(
        description='将ManiSkill的.h5文件转换为pkl格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️ 重要提示：
1. 相机名称必须保持ManiSkill原始名称（base_camera, hand_camera）
   不要使用 front/wrist 等自定义名称，否则训练时会出错！

2. Actions Clipping（默认启用）：
   - PPO采集的actions可能超出[-1,1]范围
   - DrQ/SAC使用TanhNormal，输出限制在[-1,1]
   - 自动clip可避免分布不匹配导致的训练不稳定
   - 如果不需要clip，使用 --no_clip_actions

示例用法：
  # 使用默认配置（直接运行，自动clip actions）
  python 4_convert_h5_to_pkl.py
  
  # 使用自定义文件
  python 4_convert_h5_to_pkl.py --h5_file trajectory.h5 --only_success
  
  # 禁用action clipping（不推荐）
  python 4_convert_h5_to_pkl.py --no_clip_actions
  
  # 自定义clip范围
  python 4_convert_h5_to_pkl.py --action_clip_min -0.99 --action_clip_max 0.99
        """
    )

    parser.add_argument('--h5_file', type=str, default=default_h5_file,
                        help=f'输入的.h5文件路径（默认：{default_h5_file}）')
    parser.add_argument('--output_pkl', type=str, default=default_output_pkl,
                        help=f'输出的.pkl文件路径（默认：{default_output_pkl}）')
    parser.add_argument('--state_dim', type=int, default=None,
                        help='使用的state维度（默认：使用全部维度）')
    parser.add_argument('--only_success', action='store_true',
                        help='只保存成功的轨迹')
    parser.add_argument('--keep_camera_names', action='store_true', default=True,
                        help='保持原始相机名称（默认：True，强烈推荐！）')
    parser.add_argument('--no_clip_actions', action='store_true',
                        help='禁用actions clipping（默认：启用clipping）')
    parser.add_argument('--action_clip_min', type=float, default=-1.0,
                        help='Actions最小值（默认：-1.0）')
    parser.add_argument('--action_clip_max', type=float, default=1.0,
                        help='Actions最大值（默认：1.0）')
    parser.add_argument('--no_verbose', action='store_true',
                        help='不打印详细信息')
    
    args = parser.parse_args()
    
    # 如果用户提供了 h5_file 但没有提供 output_pkl，
    # 且 h5_file 不是默认值，则自动生成 output_pkl
    if args.h5_file != default_h5_file and args.output_pkl == default_output_pkl:
        base_path = os.path.splitext(args.h5_file)[0]
        args.output_pkl = f"{base_path}.pkl"
        print(f"ℹ️  自动生成输出文件名: {args.output_pkl}\n")
    
    # ⚠️ 重要：保持原始相机名称，不进行映射！
    # 这样PKL文件中的相机名称会是 base_camera, hand_camera
    # 与ManiSkill环境期望的名称一致
    camera_mapping = {
        'base_camera': 'base_camera',
        'hand_camera': 'hand_camera'
    }
    
    print("⚠️  相机名称映射: base_camera -> base_camera, hand_camera -> hand_camera")
    print("   （保持原始名称以匹配ManiSkill环境）\n")
    
    # 执行转换
    convert_h5_to_pkl(
        h5_file=args.h5_file,
        output_pkl=args.output_pkl,
        state_dim=args.state_dim,
        camera_mapping=camera_mapping,
        only_success=args.only_success,
        clip_actions=not args.no_clip_actions,  # 默认启用clipping
        action_clip_range=(args.action_clip_min, args.action_clip_max),
        verbose=not args.no_verbose
    )


if __name__ == "__main__":
    main()
