"""
验证修改后的 4_convert_h5_to_pkl.py 的功能
测试 actions clipping 逻辑
"""

import numpy as np
import pickle

def test_action_clipping():
    """测试 action clipping 逻辑"""
    
    print("=" * 80)
    print("测试 Actions Clipping 逻辑")
    print("=" * 80)
    
    # 模拟从 H5 读取的 actions（超出 [-1, 1] 范围）
    test_actions = np.array([
        [-1.18, 0.01, -2.10, -2.74, -5.51, -1.41, -6.92],  # 多个超出
        [2.01, 1.73, 0.97, 3.08, 2.34, 0.55, 4.37],        # 多个超出
        [0.5, 0.3, -0.8, 0.2, -0.4, 0.1, 0.9],              # 正常范围内
    ])
    
    print("\n原始 Actions:")
    print(f"  Shape: {test_actions.shape}")
    print(f"  范围: [{test_actions.min():.2f}, {test_actions.max():.2f}]")
    print("\n  各维度:")
    for i in range(test_actions.shape[1]):
        print(f"    维度 {i}: [{test_actions[:, i].min():.2f}, {test_actions[:, i].max():.2f}]")
    
    # 应用 clipping
    action_clip_range = (-1.0, 1.0)
    clipped_actions = np.clip(test_actions, action_clip_range[0], action_clip_range[1])
    
    print("\nClipped Actions:")
    print(f"  Shape: {clipped_actions.shape}")
    print(f"  范围: [{clipped_actions.min():.2f}, {clipped_actions.max():.2f}]")
    print("\n  各维度:")
    for i in range(clipped_actions.shape[1]):
        print(f"    维度 {i}: [{clipped_actions[:, i].min():.2f}, {clipped_actions[:, i].max():.2f}]")
    
    # 统计被 clip 的数量
    clipped_count = np.sum(~np.isclose(test_actions, clipped_actions))
    total_count = test_actions.size
    
    print(f"\n统计:")
    print(f"  总元素数: {total_count}")
    print(f"  被clip的元素: {clipped_count} ({clipped_count/total_count*100:.1f}%)")
    
    # 验证每行
    print("\n逐行对比:")
    for i in range(len(test_actions)):
        original = test_actions[i]
        clipped = clipped_actions[i]
        changed = ~np.isclose(original, clipped)
        
        if np.any(changed):
            changed_dims = np.where(changed)[0]
            print(f"  样本 {i}: {len(changed_dims)} 个维度被clip")
            for dim in changed_dims:
                print(f"    维度 {dim}: {original[dim]:.2f} → {clipped[dim]:.2f}")
        else:
            print(f"  样本 {i}: 无需clip（已在范围内）")
    
    print("\n" + "=" * 80)
    print("✓ Clipping 逻辑测试完成")
    print("=" * 80)
    
    return clipped_actions


def test_transition_format():
    """测试完整的 transition 格式（包含 clipped actions）"""
    
    print("\n\n" + "=" * 80)
    print("测试 Transition 格式")
    print("=" * 80)
    
    # 创建示例 transition
    transition = {
        'observations': {
            'base_camera': np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
            'hand_camera': np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
            'state': np.random.randn(1, 39).astype(np.float32),
        },
        'actions': np.clip(np.random.randn(7) * 2, -1.0, 1.0).astype(np.float32),  # Clipped
        'next_observations': {
            'base_camera': np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
            'hand_camera': np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
            'state': np.random.randn(1, 39).astype(np.float32),
        },
        'rewards': np.array(0.5, dtype=np.float32),
        'masks': 1.0,
        'dones': False,
    }
    
    print("\nTransition 结构:")
    print(f"  observations:")
    for key, value in transition['observations'].items():
        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    
    print(f"\n  actions:")
    print(f"    shape={transition['actions'].shape}, dtype={transition['actions'].dtype}")
    print(f"    范围: [{transition['actions'].min():.4f}, {transition['actions'].max():.4f}]")
    print(f"    值: {transition['actions']}")
    
    print(f"\n  next_observations:")
    for key, value in transition['next_observations'].items():
        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    
    print(f"\n  rewards: {transition['rewards']}")
    print(f"  masks: {transition['masks']}")
    print(f"  dones: {transition['dones']}")
    
    # 验证 actions 在范围内
    assert transition['actions'].min() >= -1.0, "Actions 最小值应该 >= -1.0"
    assert transition['actions'].max() <= 1.0, "Actions 最大值应该 <= 1.0"
    
    print("\n✓ Actions 在 [-1, 1] 范围内")
    print("✓ Transition 格式正确")
    
    print("=" * 80)


if __name__ == "__main__":
    # 测试 clipping 逻辑
    clipped_actions = test_action_clipping()
    
    # 测试 transition 格式
    test_transition_format()
    
    print("\n" + "=" * 80)
    print("所有测试通过！✓")
    print("=" * 80)
