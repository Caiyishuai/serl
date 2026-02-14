#!/usr/bin/env python3
"""
完整修复 demo 数据格式的脚本。

修复内容:
1. actions 归一化: 从原始控制空间归一化到 [-1, 1]
2. 图像 shape: 确保图像有正确的 stack 维度
3. state shape: 确保 state 有正确的 batch 维度

用法示例:
    # 自动从环境获取 action_space
    python fix_demo_complete.py demo.pkl --env PickCube-v1
    
    # 手动指定 action_space
    python fix_demo_complete.py demo.pkl --action-low -2.8 -1.7 --action-high 2.8 1.7
    
    # 同时指定 stack 数量
    python fix_demo_complete.py demo.pkl --env PickCube-v1 --num-stack 2
"""
import pickle
import numpy as np
from pathlib import Path
import sys
import argparse


def normalize_actions(actions, action_low, action_high):
    """将 actions 从 [action_low, action_high] 归一化到 [-1, 1]"""
    action_range = action_high - action_low
    normalized = 2.0 * (actions - action_low) / action_range - 1.0
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.astype(np.float32)


def fix_observations(obs_dict, num_stack=1):
    """
    修复 observations 的 shape。
    
    - 图像: 确保有 stack 维度 (num_stack, H, W, 3)
    - state: 确保有 batch 维度 (1, state_dim)
    """
    fixed_obs = {}
    
    for key, value in obs_dict.items():
        if key == "state":
            # state: 确保是 (1, state_dim) shape
            if value.ndim == 1:
                fixed_obs[key] = value[np.newaxis, :]
            elif value.ndim == 2 and value.shape[0] == 1:
                fixed_obs[key] = value
            else:
                print(f"警告: state 的 shape 异常: {value.shape}")
                fixed_obs[key] = value
        else:
            # 图像 keys: 需要 (num_stack, H, W, 3) shape
            if value.ndim == 4:
                # 已经是 (?, H, W, 3)
                if value.shape[0] == 1:
                    # (1, H, W, 3) -> (num_stack, H, W, 3)
                    img = value[0]
                    fixed_obs[key] = np.stack([img] * num_stack, axis=0)
                elif value.shape[0] == num_stack:
                    # 已经是正确的 shape
                    fixed_obs[key] = value
                else:
                    print(f"警告: 图像 {key} 的 shape 异常: {value.shape}")
                    fixed_obs[key] = value
            elif value.ndim == 3:
                # (H, W, 3) -> (num_stack, H, W, 3)
                fixed_obs[key] = np.stack([value] * num_stack, axis=0)
            else:
                print(f"警告: 图像 {key} 的 shape 异常: {value.shape}")
                fixed_obs[key] = value
    
    return fixed_obs


def fix_transition(transition, action_low, action_high, num_stack=1, 
                   normalize_actions_flag=True, auto_detect=False):
    """
    完整修复一条 transition。
    
    参数:
        transition: 原始 transition
        action_low: action space 下界
        action_high: action space 上界
        num_stack: 图像 stack 数量
        normalize_actions_flag: 是否归一化 actions
        auto_detect: 是否自动检测是否需要归一化
    """
    fixed = {}
    
    # 修复 observations
    fixed['observations'] = fix_observations(transition['observations'], num_stack)
    
    # 修复 next_observations
    fixed['next_observations'] = fix_observations(transition['next_observations'], num_stack)
    
    # 修复 actions
    actions = transition['actions']
    if normalize_actions_flag:
        needs_norm = True
        if auto_detect:
            # 如果已经在 [-1.1, 1.1] 范围内,认为不需要归一化
            needs_norm = np.any(actions < -1.1) or np.any(actions > 1.1)
        
        if needs_norm:
            fixed['actions'] = normalize_actions(actions, action_low, action_high)
        else:
            fixed['actions'] = actions.astype(np.float32)
    else:
        fixed['actions'] = actions.astype(np.float32)
    
    # 其他字段保持不变
    fixed['rewards'] = transition['rewards']
    fixed['masks'] = transition['masks']
    fixed['dones'] = transition['dones']
    
    return fixed


def print_transition_info(transition, name="Transition"):
    """打印 transition 的详细信息"""
    print(f"\n=== {name} ===")
    print(f"Keys: {list(transition.keys())}")
    
    print("\n--- observations ---")
    for key, value in transition['observations'].items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
              f"min={value.min():.3f}, max={value.max():.3f}")
    
    print("\n--- next_observations ---")
    for key, value in transition['next_observations'].items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
              f"min={value.min():.3f}, max={value.max():.3f}")
    
    print("\n--- actions ---")
    actions = transition['actions']
    print(f"  shape={actions.shape}, dtype={actions.dtype}")
    print(f"  min={actions.min():.3f}, max={actions.max():.3f}, "
          f"mean={actions.mean():.3f}, std={actions.std():.3f}")
    
    print("\n--- rewards ---")
    print(f"  type={type(transition['rewards'])}, value={transition['rewards']}")
    
    print("\n--- masks ---")
    print(f"  type={type(transition['masks'])}, value={transition['masks']}")
    
    print("\n--- dones ---")
    print(f"  type={type(transition['dones'])}, value={transition['dones']}")


def main():
    parser = argparse.ArgumentParser(description='完整修复 demo 数据格式')
    parser.add_argument('input_file', help='输入的 demo pkl 文件')
    parser.add_argument('--output', help='输出文件路径(默认为 input_file_fixed.pkl)')
    parser.add_argument('--action-low', type=float, nargs='+',
                       help='action space 的下界')
    parser.add_argument('--action-high', type=float, nargs='+',
                       help='action space 的上界')
    parser.add_argument('--env', default=None,
                       help='环境名称,用于自动获取 action_space')
    parser.add_argument('--num-stack', type=int, default=1,
                       help='图像 stack 数量(默认: 1)')
    parser.add_argument('--no-normalize-actions', action='store_true',
                       help='不归一化 actions')
    parser.add_argument('--auto-detect', action='store_true',
                       help='自动检测是否需要归一化')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed.pkl")
    
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {output_file}")
    print(f"图像 stack 数量: {args.num_stack}")
    print(f"归一化 actions: {not args.no_normalize_actions}")
    
    # 加载数据
    print(f"\n正在加载数据...")
    with open(args.input_file, 'rb') as f:
        transitions = pickle.load(f)
    
    print(f"加载了 {len(transitions)} 条 transitions")
    
    # 获取 action_space
    action_low = None
    action_high = None
    
    if not args.no_normalize_actions:
        if args.action_low and args.action_high:
            action_low = np.array(args.action_low, dtype=np.float32)
            action_high = np.array(args.action_high, dtype=np.float32)
            print(f"\n使用用户提供的 action_space:")
            print(f"  action_low: {action_low}")
            print(f"  action_high: {action_high}")
        elif args.env:
            print(f"\n从环境 {args.env} 中获取 action_space...")
            try:
                import gymnasium
                env = gymnasium.make(args.env)
                action_low = env.action_space.low.astype(np.float32)
                action_high = env.action_space.high.astype(np.float32)
                print(f"  action_low: {action_low}")
                print(f"  action_high: {action_high}")
                env.close()
            except Exception as e:
                print(f"错误: 无法从环境获取 action_space: {e}")
                print("\n请使用 --action-low 和 --action-high 手动指定")
                sys.exit(1)
        else:
            print("\n错误: 需要归一化 actions,但未提供 action_space")
            print("请使用 --env 指定环境,或使用 --action-low 和 --action-high 手动指定")
            print("或使用 --no-normalize-actions 跳过归一化")
            sys.exit(1)
    
    # 打印原始 transition 信息
    print_transition_info(transitions[0], "原始 Transition (第一条)")
    
    # 修复所有 transitions
    print(f"\n开始修复数据...")
    fixed_transitions = []
    
    for i, transition in enumerate(transitions):
        fixed = fix_transition(
            transition,
            action_low,
            action_high,
            num_stack=args.num_stack,
            normalize_actions_flag=not args.no_normalize_actions,
            auto_detect=args.auto_detect,
        )
        fixed_transitions.append(fixed)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(transitions)} 条 transitions")
    
    # 打印修复后的 transition 信息
    print_transition_info(fixed_transitions[0], "修复后 Transition (第一条)")
    
    # 统计所有 actions
    all_actions = np.array([t['actions'] for t in fixed_transitions])
    print(f"\n所有修复后的 actions 统计:")
    print(f"  shape: {all_actions.shape}")
    print(f"  min: {all_actions.min():.3f}")
    print(f"  max: {all_actions.max():.3f}")
    print(f"  mean: {all_actions.mean():.3f}")
    print(f"  std: {all_actions.std():.3f}")
    
    if not args.no_normalize_actions:
        out_of_range = np.sum((all_actions < -1.0) | (all_actions > 1.0))
        if out_of_range > 0:
            print(f"\n警告: 有 {out_of_range} 个 action 值超出 [-1, 1] (已 clip)")
    
    # 保存
    print(f"\n保存到 {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(fixed_transitions, f)
    
    file_size_mb = Path(output_file).stat().st_size / 1024 / 1024
    print(f"\n完成!")
    print(f"文件大小: {file_size_mb:.2f} MB")
    print(f"已保存 {len(fixed_transitions)} 条 transitions")


if __name__ == '__main__':
    main()
