#!/usr/bin/env python3
"""
将 Mujoco Franka pkl 数据转换为 LeRobot 数据集格式

Usage:
    python data/mujoco_franka/convert_pkl_to_lerobot.py \
        --pkl_path data/mujoco_franka/success_trajs_100_dense.pkl \
        --repo_id franka_lift_cube_success_trajs_100_lerobot \
        --root data/mujoco_franka
"""

import argparse
import pickle
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Error: 'lerobot' library not found. Please install it.")
    exit(1)


def load_pkl_data(pkl_path):
    """加载 pkl 文件"""
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trajectories.")
    return data


def analyze_data_structure(data):
    """分析数据结构"""
    if not data:
        raise ValueError("Empty dataset")
    
    first_traj = data[0]
    first_obs = first_traj['observations'][0]
    first_action = first_traj['actions'][0]
    
    print(f"\nDataset structure:")
    print(f"  Number of trajectories: {len(data)}")
    print(f"  First trajectory length: {len(first_traj['observations'])}")
    print(f"  Observation keys: {list(first_obs.keys())}")
    print(f"  State shape: {first_obs['state'].shape}")
    print(f"  Front image shape: {first_obs['front'].shape}")
    print(f"  Wrist image shape: {first_obs['wrist'].shape}")
    print(f"  Action shape: {first_action.shape}")
    
    return first_obs, first_action


def convert_to_lerobot(pkl_path, repo_id, root=None, fps=60, robot_type="panda", task_name="Pick up the cube"):
    """转换 pkl 数据为 LeRobot 格式"""
    
    # 加载数据
    data = load_pkl_data(pkl_path)
    first_obs, first_action = analyze_data_structure(data)
    
    # 定义 features (不包含 task 字段)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(first_obs['state'].squeeze().shape),
            "names": ["joint_positions"]
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(first_action.shape),
            "names": ["joint_commands"]
        },
        "observation.images.front": {
            "dtype": "image",  # 改为 image 而不是 video
            "shape": tuple(first_obs['front'].squeeze().shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",  # 改为 image 而不是 video
            "shape": tuple(first_obs['wrist'].squeeze().shape),
            "names": ["height", "width", "channels"],
        }
    }
    
    # 设置输出路径
    if root is None:
        root = Path.cwd() / "data" / repo_id
    else:
        root = Path(root) / repo_id
    
    # 如果目录已存在，先删除
    if root.exists():
        print(f"Removing existing dataset at {root}")
        shutil.rmtree(root)
    
    print(f"\nCreating LeRobot dataset at: {root}")
    print(f"Features:")
    for key, val in features.items():
        print(f"  {key}: shape={val['shape']}, dtype={val['dtype']}")
    
    # 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=False,  # 改为 False，使用图像格式
        root=root
    )
    
    # 转换每个轨迹
    print(f"\nConverting {len(data)} trajectories...")
    for episode_idx, traj in enumerate(tqdm(data)):
        observations = traj['observations']
        actions = traj['actions']
        dones = traj['dones']
        
        num_frames = len(observations)
        
        for i in range(num_frames):
            obs = observations[i]
            
            # 处理图像: (1, H, W, C) -> (H, W, C)
            front_img = obs['front'].squeeze()
            wrist_img = obs['wrist'].squeeze()
            
            # 处理状态: (1, 7) -> (7,)
            state = obs['state'].squeeze().astype(np.float32)
            
            # 处理动作
            action = actions[i].astype(np.float32)
            
            # 创建帧字典
            # 重要: 'task' 字段会被 LeRobotDataset 自动处理，不在 features 中定义
            frame = {
                "observation.images.front": front_img,
                "observation.images.wrist": wrist_img,
                "observation.state": state,
                "action": action,
                "task": task_name,  # 每帧都需要包含 task
            }
            
            dataset.add_frame(frame)
        
        # 保存当前 episode
        dataset.save_episode()
    
    # 完成数据集创建 - 新版 lerobot 不需要显式调用 finalize
    print("\nDataset conversion complete!")
    
    print(f"\n✓ Dataset successfully saved to: {root}")
    print(f"  Total episodes: {len(data)}")
    print(f"  Total frames: {sum(len(traj['observations']) for traj in data)}")
    
    # 验证生成的文件
    print("\nGenerated files:")
    meta_dir = root / "meta"
    if meta_dir.exists():
        for file in sorted(meta_dir.iterdir()):
            if file.is_file():
                print(f"  ✓ {file.relative_to(root)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mujoco Franka pickle data to LeRobot dataset format"
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="data/mujoco_franka/success_trajs_100_dense.pkl",
        help="Path to input pickle file"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="franka_lift_cube_success_trajs_100_lerobot",
        help="Dataset repository ID (will be used as directory name)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/mujoco_franka",
        help="Root directory for the dataset (dataset will be created at root/repo_id)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="panda",
        help="Robot type"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Pick up the cube",
        help="Task description"
    )
    
    args = parser.parse_args()
    
    convert_to_lerobot(
        pkl_path=args.pkl_path,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task
    )


if __name__ == "__main__":
    main()

# cd /share_data/caiyishuai/openpi && chmod +x data/mujoco_franka/convert_pkl_to_lerobot.py && python3 data/mujoco_franka/convert_pkl_to_lerobot.py --pkl_path data/mujoco_franka/success_trajs_100_dense.pkl --repo_id franka_lift_cube_success_trajs_100_lerobot --root data/mujoco_franka --fps 60 --task "Pick up the cube"