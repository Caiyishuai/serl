"""
修复 Demo 数据的 Actions，将其 clip 到 [-1, 1] 范围
使其与 DrQ/SAC 的 TanhNormal distribution 匹配

问题：
- PPO 采集的 demo actions 没有被 clip，范围可能超出 [-1, 1]
- DrQ agent 使用 TanhNormal，输出自动限制在 [-1, 1]
- 分布不匹配导致训练不稳定

解决：
- 将 demo actions clip 到 [-1, 1]
- 保持其他数据不变
"""

import pickle
import numpy as np
import argparse
from pathlib import Path


def analyze_actions(transitions):
    """分析 actions 的统计信息"""
    all_actions = np.array([t['actions'] for t in transitions])
    
    print("\n" + "="*80)
    print("Actions 统计分析")
    print("="*80)
    print(f"总 transitions 数: {len(transitions)}")
    print(f"Actions shape: {all_actions.shape}")
    print(f"\n整体统计:")
    print(f"  Min: {all_actions.min():.4f}")
    print(f"  Max: {all_actions.max():.4f}")
    print(f"  Mean: {all_actions.mean():.4f}")
    print(f"  Std: {all_actions.std():.4f}")
    
    print(f"\n每个维度的统计:")
    for i in range(all_actions.shape[1]):
        dim_actions = all_actions[:, i]
        out_of_range = np.sum((dim_actions < -1) | (dim_actions > 1))
        pct = out_of_range / len(dim_actions) * 100
        
        print(f"  维度 {i}: min={dim_actions.min():7.4f}, "
              f"max={dim_actions.max():7.4f}, "
              f"mean={dim_actions.mean():7.4f}, "
              f"std={dim_actions.std():6.4f}, "
              f"超出范围: {out_of_range:4d} ({pct:5.2f}%)")
        
        if out_of_range > 0:
            print(f"           ⚠️ 此维度有 {out_of_range} 个样本超出 [-1, 1]")
    
    # 统计有多少 transitions 包含超出范围的 actions
    out_of_range_transitions = np.sum(
        np.any((all_actions < -1) | (all_actions > 1), axis=1)
    )
    print(f"\n包含超出范围 actions 的 transitions: {out_of_range_transitions}/{len(transitions)} "
          f"({out_of_range_transitions/len(transitions)*100:.2f}%)")
    

def fix_demo_actions(input_pkl, output_pkl, clip_range=(-1.0, 1.0), verbose=True):
    """
    修复 demo 数据的 actions，将其 clip 到指定范围
    
    Args:
        input_pkl: 输入的 pkl 文件
        output_pkl: 输出的 pkl 文件
        clip_range: clip 范围 (min, max)
        verbose: 是否打印详细信息
    """
    print(f"="*80)
    print(f"修复 Demo Actions")
    print(f"="*80)
    print(f"输入文件: {input_pkl}")
    print(f"输出文件: {output_pkl}")
    print(f"Clip 范围: {clip_range}")
    print(f"="*80)
    
    # 加载数据
    print(f"\n加载数据...")
    with open(input_pkl, 'rb') as f:
        transitions = pickle.load(f)
    
    print(f"加载了 {len(transitions)} 条 transitions")
    
    # 分析原始 actions
    print("\n【原始 Actions】")
    analyze_actions(transitions)
    
    # 修复 actions
    print(f"\n修复 actions...")
    fixed_transitions = []
    clip_count = 0
    
    for transition in transitions:
        fixed_transition = transition.copy()
        
        # Clip actions
        original_actions = transition['actions']
        clipped_actions = np.clip(
            original_actions,
            clip_range[0],
            clip_range[1]
        ).astype(np.float32)
        
        # 检查是否有改变
        if not np.allclose(original_actions, clipped_actions):
            clip_count += 1
        
        fixed_transition['actions'] = clipped_actions
        fixed_transitions.append(fixed_transition)
    
    print(f"修改了 {clip_count} 条 transitions 的 actions")
    
    # 分析修复后的 actions
    print("\n【修复后 Actions】")
    analyze_actions(fixed_transitions)
    
    # 保存
    print(f"\n保存到 {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(fixed_transitions, f)
    
    # 文件大小
    input_size = Path(input_pkl).stat().st_size / 1024 / 1024
    output_size = Path(output_pkl).stat().st_size / 1024 / 1024
    
    print(f"\n" + "="*80)
    print(f"完成！")
    print(f"输入文件大小: {input_size:.2f} MB")
    print(f"输出文件大小: {output_size:.2f} MB")
    print(f"="*80)
    
    # 验证
    print(f"\n验证数据格式...")
    sample = fixed_transitions[0]
    print(f"  observations keys: {list(sample['observations'].keys())}")
    print(f"  actions: shape={sample['actions'].shape}, dtype={sample['actions'].dtype}")
    print(f"  actions 范围: [{sample['actions'].min():.4f}, {sample['actions'].max():.4f}]")
    print(f"  ✓ 格式正确")


def main():
    parser = argparse.ArgumentParser(
        description='修复 Demo 数据的 Actions，clip 到 [-1, 1]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 修复 actions
  python 5_fix_demo_actions.py \\
      --input serl_data/mani_skill_place_sphere_100.pkl \\
      --output serl_data/mani_skill_place_sphere_100_fixed.pkl
  
  # 使用自定义 clip 范围
  python 5_fix_demo_actions.py \\
      --input demo.pkl \\
      --clip_min -0.99 \\
      --clip_max 0.99
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='输入的 pkl 文件')
    parser.add_argument('--output', type=str, default=None,
                        help='输出的 pkl 文件（默认：输入文件名_fixed.pkl）')
    parser.add_argument('--clip_min', type=float, default=-1.0,
                        help='Actions 最小值（默认：-1.0）')
    parser.add_argument('--clip_max', type=float, default=1.0,
                        help='Actions 最大值（默认：1.0）')
    
    args = parser.parse_args()
    
    # 自动生成输出文件名
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_fixed.pkl")
    
    # 执行修复
    fix_demo_actions(
        input_pkl=args.input,
        output_pkl=args.output,
        clip_range=(args.clip_min, args.clip_max)
    )


if __name__ == "__main__":
    main()
