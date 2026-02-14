"""
检查 demo 数据格式是否与环境兼容
"""
import pickle
import sys
import numpy as np


def check_demo_format(demo_file):
    """检查 demo 数据的格式"""
    print(f"检查文件: {demo_file}")
    print("=" * 60)
    
    # 加载数据
    try:
        with open(demo_file, "rb") as f:
            transitions = pickle.load(f)
        print(f"✅ 成功加载 {len(transitions)} 条 transitions\n")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    if len(transitions) == 0:
        print("❌ 数据为空")
        return False
    
    # 检查第一条 transition
    transition = transitions[0]
    
    print("字段检查:")
    required_fields = ["observations", "next_observations", "actions", "rewards", "masks", "dones"]
    for field in required_fields:
        if field in transition:
            print(f"  ✅ {field}")
        else:
            print(f"  ❌ 缺少 {field}")
            return False
    
    print("\n--- observations 结构 ---")
    obs = transition["observations"]
    print(f"键: {list(obs.keys())}")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, value={value}")
    
    print("\n--- next_observations 结构 ---")
    next_obs = transition["next_observations"]
    print(f"键: {list(next_obs.keys())}")
    for key, value in next_obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}, value={value}")
    
    print("\n--- actions ---")
    actions = transition["actions"]
    if isinstance(actions, np.ndarray):
        print(f"  shape={actions.shape}, dtype={actions.dtype}")
        print(f"  范围: [{actions.min():.3f}, {actions.max():.3f}]")
    else:
        print(f"  type={type(actions)}")
    
    print("\n--- rewards ---")
    print(f"  type={type(transition['rewards'])}, value={transition['rewards']}")
    
    print("\n--- masks ---")
    print(f"  type={type(transition['masks'])}, value={transition['masks']}")
    
    print("\n--- dones ---")
    print(f"  type={type(transition['dones'])}, value={transition['dones']}")
    
    # 格式建议
    print("\n" + "=" * 60)
    print("格式分析:")
    
    issues = []
    suggestions = []
    
    # 检查图像键名
    image_keys = [k for k in obs.keys() if k != "state"]
    if "front" in image_keys or "wrist" in image_keys:
        issues.append("⚠️  图像键名使用了 'front'/'wrist'")
        suggestions.append("   需要映射到 'base_camera'/'hand_camera'")
        suggestions.append("   使用命令: python fix_demo_format.py <file> --key_mapping front:base_camera,wrist:hand_camera")
    
    if "base_camera" in image_keys or "hand_camera" in image_keys:
        print("✅ 图像键名正确 (使用 'base_camera'/'hand_camera')")
    
    # 检查图像格式
    for key in image_keys:
        img = obs[key]
        if isinstance(img, np.ndarray):
            if img.ndim == 4:
                if img.shape[0] == 1:
                    print(f"✅ {key} 格式正确: {img.shape} (ChunkingWrapper 格式)")
                else:
                    issues.append(f"⚠️  {key} 第一维不是 1: {img.shape}")
            else:
                issues.append(f"❌ {key} 维度错误: {img.shape}，应该是 (1, H, W, 3)")
    
    # 检查 state 格式
    if "state" in obs:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            if state.ndim == 2 and state.shape[0] == 1:
                print(f"✅ state 格式正确: {state.shape}")
            elif state.ndim == 1:
                issues.append(f"⚠️  state 维度错误: {state.shape}，应该是 (1, state_dim)")
                suggestions.append("   脚本会自动修复这个问题")
            else:
                issues.append(f"❌ state 格式错误: {state.shape}")
    
    # 打印问题和建议
    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(issue)
    
    if suggestions:
        print("\n建议:")
        for suggestion in suggestions:
            print(suggestion)
    
    if not issues and not suggestions:
        print("\n✅ 数据格式完全正确，可以直接使用！")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python check_demo_format.py <demo_file.pkl>")
        print("示例: python check_demo_format.py franka_lift_cube_image_20_trajs.pkl")
        sys.exit(1)
    
    demo_file = sys.argv[1]
    check_demo_format(demo_file)


if __name__ == "__main__":
    main()
