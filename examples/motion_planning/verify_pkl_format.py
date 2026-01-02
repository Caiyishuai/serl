"""
验证保存的pkl文件格式是否正确
"""
import pickle
import numpy as np
import sys

def verify_pkl_format(filepath):
    """验证pkl文件格式"""
    print(f"检查文件: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ 文件类型: {type(data)}")
    print(f"✓ 轨迹数量: {len(data)}")
    
    if len(data) == 0:
        print("✗ 文件为空")
        return False
    
    # 检查第一条轨迹的格式
    traj = data[0]
    print(f"\n检查第一条轨迹:")
    print(f"  类型: {type(traj)}")
    print(f"  键: {list(traj.keys())}")
    
    expected_keys = ['observations', 'actions', 'rewards', 'next_observations', 'dones', 'infos', 'initial_object_pos']
    for key in expected_keys:
        if key not in traj:
            print(f"✗ 缺少键: {key}")
            return False
        if key == 'initial_object_pos':
            val = traj[key]
            if isinstance(val, np.ndarray):
                print(f"  ✓ {key}: shape={val.shape}, dtype={val.dtype}")
                if val.shape == (2, 3):
                    print(f"      Block位置: {val[0]}")
                    print(f"      Pillar位置: {val[1]}")
            else:
                print(f"  ✓ {key}: type={type(val)}")
        else:
            print(f"  ✓ {key}: {len(traj[key]) if isinstance(traj[key], list) else type(traj[key])}")
    
    # 检查数据维度
    if len(traj['observations']) > 0:
        print(f"\n第一个观测:")
        obs = traj['observations'][0]
        print(f"  类型: {type(obs)}")
        if isinstance(obs, dict):
            print(f"  键: {list(obs.keys())}")
            for key, value in obs.items():
                if isinstance(value, dict):
                    print(f"    {key}: dict with keys {list(value.keys())}")
                elif isinstance(value, np.ndarray):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    
    if len(traj['actions']) > 0:
        action = traj['actions'][0]
        print(f"\n第一个动作:")
        print(f"  类型: {type(action)}")
        if isinstance(action, np.ndarray):
            print(f"  形状: {action.shape}")
            print(f"  值: {action}")
    
    print(f"\n✓ 格式验证通过！")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "../async_drq_sim/stack_trajs_20.pkl"
    
    verify_pkl_format(filepath)

