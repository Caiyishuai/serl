"""
修复 demo 数据的 actions，将第 6 维度（gripper）归一化到 [-1, 1]
"""
import pickle
import numpy as np
import sys


def analyze_actions(transitions):
    """分析 actions 的分布"""
    all_actions = np.array([t['actions'] for t in transitions])
    
    print("=" * 80)
    print("Actions 分析")
    print("=" * 80)
    for i in range(all_actions.shape[1]):
        dim_actions = all_actions[:, i]
        print(f"\n维度 {i}:")
        print(f"  min={dim_actions.min():.4f}")
        print(f"  max={dim_actions.max():.4f}")
        print(f"  mean={dim_actions.mean():.4f}")
        print(f"  std={dim_actions.std():.4f}")
        
        # 检查是否超出 [-1, 1] 范围
        if dim_actions.min() < -1.5 or dim_actions.max() > 1.5:
            print(f"  ⚠️ 超出正常范围 [-1, 1]!")


def fix_gripper_actions(transitions, gripper_dim=6, target_range=(-1, 1)):
    """
    修复 gripper 维度的 actions
    
    参数:
        transitions: 原始 transitions
        gripper_dim: gripper 动作的维度索引（默认 6）
        target_range: 目标范围（默认 [-1, 1]）
    """
    # 收集所有 gripper actions
    all_actions = np.array([t['actions'] for t in transitions])
    gripper_actions = all_actions[:, gripper_dim]
    
    # 计算当前范围
    current_min = gripper_actions.min()
    current_max = gripper_actions.max()
    
    print("\n" + "=" * 80)
    print("Gripper Actions 修复")
    print("=" * 80)
    print(f"当前范围: [{current_min:.4f}, {current_max:.4f}]")
    print(f"目标范围: [{target_range[0]:.4f}, {target_range[1]:.4f}]")
    
    # 归一化到 target_range
    # 公式: new = (old - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    fixed_transitions = []
    for transition in transitions:
        fixed_transition = transition.copy()
        actions = transition['actions'].copy()
        
        # 归一化 gripper 维度
        old_val = actions[gripper_dim]
        normalized_val = (old_val - current_min) / (current_max - current_min + 1e-8)
        new_val = normalized_val * (target_range[1] - target_range[0]) + target_range[0]
        
        actions[gripper_dim] = new_val
        fixed_transition['actions'] = actions
        fixed_transitions.append(fixed_transition)
    
    # 验证修复后的范围
    fixed_actions = np.array([t['actions'] for t in fixed_transitions])
    fixed_gripper = fixed_actions[:, gripper_dim]
    
    print(f"修复后范围: [{fixed_gripper.min():.4f}, {fixed_gripper.max():.4f}]")
    print(f"修复后均值: {fixed_gripper.mean():.4f}")
    print(f"修复后标准差: {fixed_gripper.std():.4f}")
    
    return fixed_transitions


def main():
    if len(sys.argv) < 2:
        print("用法: python fix_demo_actions.py <demo_file.pkl> [--output output.pkl] [--gripper_dim N]")
        print("示例: python fix_demo_actions.py mani_skill_place_sphere_100.pkl --gripper_dim 6")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    gripper_dim = 6
    
    # 解析参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output":
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--gripper_dim":
            gripper_dim = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    if output_file is None:
        from pathlib import Path
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed_actions.pkl")
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"Gripper 维度: {gripper_dim}")
    
    # 加载数据
    print("\n加载数据...")
    with open(input_file, "rb") as f:
        transitions = pickle.load(f)
    
    print(f"加载了 {len(transitions)} 条 transitions")
    
    # 分析原始数据
    print("\n【原始数据分析】")
    analyze_actions(transitions)
    
    # 修复 gripper actions
    fixed_transitions = fix_gripper_actions(transitions, gripper_dim=gripper_dim)
    
    # 分析修复后的数据
    print("\n【修复后数据分析】")
    analyze_actions(fixed_transitions)
    
    # 保存
    print(f"\n保存到 {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(fixed_transitions, f)
    
    print(f"\n✅ 完成！已保存 {len(fixed_transitions)} 条 transitions")
    
    # 提供使用建议
    print("\n" + "=" * 80)
    print("使用建议")
    print("=" * 80)
    print(f"1. 请在训练脚本中使用修复后的文件:")
    print(f"   --demo_path {output_file}")
    print(f"\n2. 重新运行训练，检查收敛速度是否改善")
    print(f"\n3. 对比实验:")
    print(f"   - 不用 demo: baseline")
    print(f"   - 用原始 demo: 收敛慢")
    print(f"   - 用修复后 demo: 应该改善")


if __name__ == "__main__":
    main()
