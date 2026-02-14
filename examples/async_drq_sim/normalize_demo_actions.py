#!/usr/bin/env python3
"""
归一化 demo 数据的 actions 到 [-1, 1] 范围。

问题:
- Online 采集的数据: actions 已经归一化到 [-1, 1]
- Demo 数据: actions 是原始的控制空间数值(例如 -6.21 到 4.17)

解决方案:
根据环境的 action_space 进行归一化。
"""
import pickle
import numpy as np
from pathlib import Path
import sys
import argparse


def normalize_actions(actions, action_low, action_high):
    """
    将 actions 从 [action_low, action_high] 归一化到 [-1, 1]
    
    公式: normalized = 2 * (actions - action_low) / (action_high - action_low) - 1
    """
    action_range = action_high - action_low
    normalized = 2.0 * (actions - action_low) / action_range - 1.0
    
    # 确保在 [-1, 1] 范围内
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def denormalize_actions(normalized_actions, action_low, action_high):
    """
    将归一化的 actions 从 [-1, 1] 还原到 [action_low, action_high]
    
    公式: actions = (normalized + 1) / 2 * (action_high - action_low) + action_low
    """
    action_range = action_high - action_low
    actions = (normalized_actions + 1.0) / 2.0 * action_range + action_low
    
    return actions


def check_and_normalize_transition(transition, action_low, action_high, auto_detect=False):
    """
    检查并归一化单条 transition 的 actions。
    
    参数:
        transition: transition 字典
        action_low: action space 的下界
        action_high: action space 的上界
        auto_detect: 是否自动检测是否需要归一化(基于 actions 的范围)
    
    返回:
        normalized_transition: 归一化后的 transition
        was_normalized: 是否进行了归一化
    """
    actions = transition['actions']
    
    # 自动检测是否需要归一化
    needs_normalization = False
    if auto_detect:
        # 如果 actions 超出 [-1.1, 1.1] 范围,认为需要归一化
        # (留一点余量,因为可能有些 actions 略微超出 [-1, 1])
        if np.any(actions < -1.1) or np.any(actions > 1.1):
            needs_normalization = True
    else:
        needs_normalization = True
    
    if needs_normalization:
        normalized_transition = transition.copy()
        normalized_transition['actions'] = normalize_actions(actions, action_low, action_high)
        return normalized_transition, True
    else:
        return transition, False


def main():
    parser = argparse.ArgumentParser(description='归一化 demo 数据的 actions')
    parser.add_argument('input_file', help='输入的 demo pkl 文件')
    parser.add_argument('--output', help='输出文件路径(默认为 input_file_normalized.pkl)')
    parser.add_argument('--action-low', type=float, nargs='+', 
                       help='action space 的下界(多个值用空格分隔)')
    parser.add_argument('--action-high', type=float, nargs='+',
                       help='action space 的上界(多个值用空格分隔)')
    parser.add_argument('--auto-detect', action='store_true',
                       help='自动检测是否需要归一化(基于 actions 的范围)')
    parser.add_argument('--env', default='PickCube-v1',
                       help='环境名称,用于自动获取 action_space (默认: PickCube-v1)')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_normalized.pkl")
    
    # 加载数据
    print(f"加载数据: {args.input_file}")
    with open(args.input_file, 'rb') as f:
        transitions = pickle.load(f)
    
    print(f"加载了 {len(transitions)} 条 transitions")
    
    # 获取 action_space 的边界
    if args.action_low and args.action_high:
        action_low = np.array(args.action_low)
        action_high = np.array(args.action_high)
        print(f"使用用户提供的 action_space:")
        print(f"  action_low: {action_low}")
        print(f"  action_high: {action_high}")
    else:
        # 从环境中获取
        print(f"从环境 {args.env} 中获取 action_space...")
        try:
            import gymnasium
            env = gymnasium.make(args.env)
            action_low = env.action_space.low
            action_high = env.action_space.high
            print(f"  action_low: {action_low}")
            print(f"  action_high: {action_high}")
            env.close()
        except Exception as e:
            print(f"无法从环境获取 action_space: {e}")
            print("\n请使用 --action-low 和 --action-high 手动指定 action_space")
            print("示例: --action-low -2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175 -2.8973")
            print("      --action-high 2.8973 1.7628 2.8973 -0.0698 2.8973 3.7525 2.8973")
            sys.exit(1)
    
    # 检查第一条 transition 的 actions 范围
    first_actions = transitions[0]['actions']
    print(f"\n第一条 transition 的 actions:")
    print(f"  shape: {first_actions.shape}")
    print(f"  min: {first_actions.min()}")
    print(f"  max: {first_actions.max()}")
    print(f"  mean: {first_actions.mean()}")
    print(f"  std: {first_actions.std()}")
    
    # 统计所有 actions 的范围
    all_actions = np.array([t['actions'] for t in transitions])
    print(f"\n所有 actions 的统计:")
    print(f"  shape: {all_actions.shape}")
    print(f"  min: {all_actions.min()}")
    print(f"  max: {all_actions.max()}")
    print(f"  mean: {all_actions.mean()}")
    print(f"  std: {all_actions.std()}")
    
    # 归一化
    print(f"\n开始归一化...")
    normalized_transitions = []
    num_normalized = 0
    
    for i, transition in enumerate(transitions):
        normalized_transition, was_normalized = check_and_normalize_transition(
            transition, action_low, action_high, auto_detect=args.auto_detect
        )
        normalized_transitions.append(normalized_transition)
        if was_normalized:
            num_normalized += 1
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(transitions)} 条 transitions")
    
    print(f"\n归一化完成:")
    print(f"  归一化的数量: {num_normalized}/{len(transitions)}")
    
    # 检查归一化后的 actions 范围
    normalized_actions = np.array([t['actions'] for t in normalized_transitions])
    print(f"\n归一化后的 actions 统计:")
    print(f"  shape: {normalized_actions.shape}")
    print(f"  min: {normalized_actions.min()}")
    print(f"  max: {normalized_actions.max()}")
    print(f"  mean: {normalized_actions.mean()}")
    print(f"  std: {normalized_actions.std()}")
    
    # 检查是否有超出 [-1, 1] 的值
    out_of_range = np.sum((normalized_actions < -1.0) | (normalized_actions > 1.0))
    if out_of_range > 0:
        print(f"\n警告: 有 {out_of_range} 个 action 值超出 [-1, 1] 范围!")
        print("这些值已被 clip 到 [-1, 1]")
    
    # 保存
    print(f"\n保存到 {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(normalized_transitions, f)
    
    print(f"\n完成!")
    print(f"文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    
    # 可选: 测试反归一化
    print(f"\n测试反归一化 (前 3 个 action):")
    original = transitions[0]['actions'][:3]
    normalized = normalized_transitions[0]['actions'][:3]
    denormalized = denormalize_actions(normalized, action_low[:3], action_high[:3])
    print(f"  原始:     {original}")
    print(f"  归一化:   {normalized}")
    print(f"  反归一化: {denormalized}")
    print(f"  误差:     {np.abs(original - denormalized).max()}")


if __name__ == '__main__':
    main()
