import os
# 强制使用离线模式，避免连接 Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import numpy as np

# 数据集路径
DATA_ROOT = "tools/data"
REPO_ID = "franka_lift_cube_success_trajs_100_lerobot"

def main():
    print(f"Loading dataset from root={DATA_ROOT}, repo_id={REPO_ID}...")
    try:
        # 尝试使用 local_files_only 参数（如果支持）
        try:
            dataset = LeRobotDataset(root=DATA_ROOT, repo_id=REPO_ID, local_files_only=True)
        except TypeError:
             # 如果不支持 local_files_only，则仅依赖环境变量
            print("local_files_only argument not supported, relying on env vars...")
            dataset = LeRobotDataset(root=DATA_ROOT, repo_id=REPO_ID)
            
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        # 如果直接加载失败，尝试指定 repo_id (虽然这是本地路径)
        # 或者可能是版本问题，但通常 root 指向包含 meta 的目录即可
        return

    print(f"Dataset loaded. Number of episodes: {dataset.num_episodes}")
    print(f"Total frames: {len(dataset)}")

    # 获取第一个 episode 的索引范围
    episode_idx = 0
    from_idx = dataset.episode_data_index["from"][episode_idx]
    to_idx = dataset.episode_data_index["to"][episode_idx]
    
    print(f"Episode {episode_idx} range: {from_idx} to {to_idx}")

    # 读取第一帧和中间一帧
    frames_to_inspect = [from_idx, (from_idx + to_idx) // 2]
    
    output_dir = "inspected_images"
    os.makedirs(output_dir, exist_ok=True)

    for idx in frames_to_inspect:
        item = dataset[idx]
        print(f"Frame {idx} keys: {item.keys()}")
        
        # 提取图像
        # 根据 pkl_to_lerobot.py，keys 是 observation.images.front 和 observation.images.wrist
        for key in ["observation.images.front", "observation.images.wrist"]:
            if key in item:
                img_tensor = item[key]
                print(f"Frame {idx} - {key} shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
                
                # LeRobot 通常返回 pytorch tensor (C, H, W) 且范围可能是 [0, 1] 或 [0, 255]
                # pkl_to_lerobot.py 中看到是直接存的，通常是 uint8 [0, 255] (H, W, C) 或者被转换过
                # 如果是 tensor, 通常是 (C, H, W)
                
                img_np = img_tensor.numpy()
                
                # 如果是 (C, H, W)，转换为 (H, W, C)
                if img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # 归一化检查
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                
                save_path = f"{output_dir}/episode_{episode_idx}_frame_{idx}_{key.replace('.', '_')}.png"
                Image.fromarray(img_np).save(save_path)
                print(f"Saved image to {save_path}")

if __name__ == "__main__":
    main()

