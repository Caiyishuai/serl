"""
修复 demo 数据格式，使其与 MemoryEfficientReplayBuffer 兼容。

问题：
1. 采集的 demo 数据图像 shape 是 (1, H, W, 3) - 这个格式是正确的（ChunkingWrapper 的输出）
2. 键名不匹配：demo 使用 "front"/"wrist"，但 ManiSkill 使用 "base_camera"/"hand_camera"
3. 需要确保所有字段的格式都正确

解决方案：
1. 重命名图像键
2. 验证所有字段的格式
"""
import pickle
import numpy as np
from pathlib import Path
import sys


def check_transition_format(transition):
    """检查 transition 的格式"""
    print("\n=== Transition 格式检查 ===")
    print(f"Keys: {transition.keys()}")
    
    print("\n--- observations ---")
    for key, value in transition["observations"].items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    print("\n--- next_observations ---")
    for key, value in transition["next_observations"].items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    print("\n--- actions ---")
    print(f"  shape={transition['actions'].shape}, dtype={transition['actions'].dtype}")
    
    print("\n--- rewards ---")
    print(f"  type={type(transition['rewards'])}, value={transition['rewards']}")
    
    print("\n--- masks ---")
    print(f"  type={type(transition['masks'])}, value={transition['masks']}")
    
    print("\n--- dones ---")
    print(f"  type={type(transition['dones'])}, value={transition['dones']}")


def fix_transition_format(transition, key_mapping=None):
    """
    修复 transition 格式以匹配 MemoryEfficientReplayBuffer 的期望。
    
    参数:
        transition: 原始 transition 字典
        key_mapping: 键名映射字典，例如 {"front": "base_camera", "wrist": "hand_camera"}
    
    返回:
        修复后的 transition 字典
    """
    if key_mapping is None:
        key_mapping = {}
    
    fixed_transition = {
        "observations": {},
        "next_observations": {},
        "actions": transition["actions"],
        "rewards": transition["rewards"],
        "masks": transition["masks"],
        "dones": transition["dones"],
    }
    
    # 修复 observations
    for key, value in transition["observations"].items():
        # 应用键名映射
        new_key = key_mapping.get(key, key)
        
        if key == "state":
            # state 应该保持 (1, state_dim) 的格式
            # 这是 ChunkingWrapper 的输出格式
            if value.ndim == 1:
                fixed_transition["observations"][new_key] = value[np.newaxis, :]
            else:
                fixed_transition["observations"][new_key] = value
        else:
            # 图像 keys: ChunkingWrapper 已经添加了 stack 维度
            # 原始格式: (1, H, W, 3) - 这个格式是正确的
            # 保持原样即可
            fixed_transition["observations"][new_key] = value
    
    # 修复 next_observations
    for key, value in transition["next_observations"].items():
        # 应用键名映射
        new_key = key_mapping.get(key, key)
        
        if key == "state":
            if value.ndim == 1:
                fixed_transition["next_observations"][new_key] = value[np.newaxis, :]
            else:
                fixed_transition["next_observations"][new_key] = value
        else:
            # 图像 keys: 保持原样
            fixed_transition["next_observations"][new_key] = value
    
    return fixed_transition


def main():
    if len(sys.argv) < 2:
        print("用法: python fix_demo_format.py <demo_file.pkl> [--key_mapping old1:new1,old2:new2] [--output output.pkl]")
        print("示例: python fix_demo_format.py franka_lift_cube_image_20_trajs.pkl --key_mapping front:base_camera,wrist:hand_camera")
        sys.exit(1)
    
    input_file = sys.argv[1]
    key_mapping = {}
    output_file = None
    
    # 解析命令行参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--key_mapping":
            # 解析键名映射，格式: old1:new1,old2:new2
            mapping_str = sys.argv[i + 1]
            for pair in mapping_str.split(','):
                old_key, new_key = pair.split(':')
                key_mapping[old_key.strip()] = new_key.strip()
            i += 2
        elif sys.argv[i] == "--output":
            output_file = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    if output_file is None:
        # 自动生成输出文件名
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed.pkl")
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"键名映射: {key_mapping}")
    
    # 加载数据
    print("\n加载数据...")
    with open(input_file, "rb") as f:
        transitions = pickle.load(f)
    
    print(f"加载了 {len(transitions)} 条 transitions")
    
    # 检查第一条 transition 的格式
    print("\n=== 原始格式 ===")
    check_transition_format(transitions[0])
    
    # 修复所有 transitions
    print("\n修复数据格式...")
    fixed_transitions = []
    for i, transition in enumerate(transitions):
        fixed_transition = fix_transition_format(transition, key_mapping=key_mapping)
        fixed_transitions.append(fixed_transition)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(transitions)} 条 transitions")
    
    # 检查修复后的格式
    print("\n=== 修复后的格式 ===")
    check_transition_format(fixed_transitions[0])
    
    # 保存修复后的数据
    print(f"\n保存到 {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(fixed_transitions, f)
    
    print(f"\n完成！已保存 {len(fixed_transitions)} 条 transitions 到 {output_file}")
    print(f"文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
