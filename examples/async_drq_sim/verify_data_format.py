#!/usr/bin/env python3
"""
验证新生成的pkl文件格式是否与参考文件一致
"""

import pickle as pkl
import numpy as np
from pathlib import Path

def check_format(pkl_path, name="文件"):
    """检查pkl文件格式"""
    print(f"\n{'='*80}")
    print(f"检查 {name}: {Path(pkl_path).name}")
    print(f"{'='*80}")
    
    with open(pkl_path, 'rb') as f:
        trajs = pkl.load(f)
    
    print(f"轨迹数量: {len(trajs)}")
    
    if len(trajs) == 0:
        print("警告: 没有轨迹数据")
        return
    
    traj = trajs[0]
    print(f"\n第一个轨迹的keys: {list(traj.keys())}")
    
    # 检查observations格式
    print(f"\nobservations:")
    if isinstance(traj['observations'], dict):
        print(f"  ✓ observations是字典")
        print(f"  keys: {list(traj['observations'].keys())}")
        
        if 'state' in traj['observations']:
            state = traj['observations']['state']
            print(f"\n  state:")
            print(f"    type: {type(state)}")
            print(f"    shape: {state.shape}")
            print(f"    dtype: {state.dtype}")
            
            if len(state.shape) == 2 and state.shape[1] == 22:
                print(f"    ✓ state格式正确: (N, 22)")
                print(f"\n    第一个state的解析:")
                s = state[0]
                print(f"      [0]: gripper_pose = {s[0]}")
                print(f"      [1:4]: tcp_force = {s[1:4]}")
                print(f"      [4:10]: tcp_pose = {s[4:10]}")
                print(f"      [10:13]: tcp_torque = {s[10:13]}")
                print(f"      [13:19]: tcp_vel = {s[13:19]}")
                print(f"      [19:22]: block_pos = {s[19:22]}")
            elif len(state.shape) == 2 and state.shape[1] == 19:
                print(f"    ⚠ state格式: (N, 19) - 旧格式，缺少block_pos")
            else:
                print(f"    ✗ state格式不正确，期望(N, 22)，实际{state.shape}")
        
        # 检查images
        image_keys = [k for k in traj['observations'].keys() if k != 'state']
        if image_keys:
            print(f"\n  images keys: {image_keys}")
            for img_key in image_keys:
                img = traj['observations'][img_key]
                print(f"    {img_key}: shape={img.shape}, dtype={img.dtype}")
    else:
        print(f"  ✗ observations不是字典")
    
    # 检查next_observations格式
    if 'next_observations' in traj:
        print(f"\nnext_observations:")
        if isinstance(traj['next_observations'], dict):
            print(f"  ✓ next_observations是字典")
            print(f"  keys: {list(traj['next_observations'].keys())}")
        else:
            print(f"  ✗ next_observations不是字典")
    
    # 检查infos
    if 'infos' in traj:
        print(f"\ninfos:")
        infos = traj['infos']
        print(f"  type: {type(infos)}")
        if isinstance(infos, (list, np.ndarray)) and len(infos) > 0:
            info = infos[0]
            print(f"  第一个info type: {type(info)}")
            if isinstance(info, dict):
                print(f"  第一个info keys: {list(info.keys())}")
                if 'original_state_obs' in info:
                    print(f"  ✓ 包含 original_state_obs")
                    orig_state = info['original_state_obs']
                    print(f"    original_state_obs keys: {list(orig_state.keys())}")
                else:
                    print(f"  - 不包含 original_state_obs (可能是可选的)")
    
    # 检查其他字段
    for key in ['actions', 'rewards', 'dones']:
        if key in traj:
            val = traj[key]
            print(f"\n{key}:")
            print(f"  type: {type(val)}")
            print(f"  shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
            print(f"  dtype: {val.dtype if hasattr(val, 'dtype') else 'N/A'}")


def main():
    """主函数"""
    reference_path = "/home/chenhaojun/workspace_cys/serl/examples/async_drq_sim/plug_insert_20_demos_2026-01-05_06-37-50.pkl"
    new_path = "/home/chenhaojun/workspace_cys/serl/examples/async_drq_sim/success_trajs_20_force_dense.pkl"
    
    # 检查参考文件
    if Path(reference_path).exists():
        check_format(reference_path, "参考文件")
    
    # 检查新文件
    if Path(new_path).exists():
        check_format(new_path, "新文件")
    else:
        print(f"\n警告: 新文件不存在: {new_path}")
        print("请先运行 1_eval_load_model_and_generate_data_force.py 生成数据")


if __name__ == "__main__":
    main()

