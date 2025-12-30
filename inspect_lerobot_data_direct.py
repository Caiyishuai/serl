import pandas as pd
import torchvision
import torch
import os
import numpy as np
from PIL import Image

# 路径配置
DATA_ROOT = "data/serl/franka_sim_lift_100_40cm_lerobot"
META_PATH = f"{DATA_ROOT}/meta/episodes/chunk-000/file-000.parquet"
OUTPUT_DIR = "inspected_images"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 读取 Episode 元数据
    print(f"Reading metadata from {META_PATH}...")
    episodes_df = pd.read_parquet(META_PATH)
    
    # 获取第一个 episode (index 0)
    episode = episodes_df.iloc[5]
    print(f"Inspecting Episode {episode['episode_index']}")
    print(f"Length: {episode['length']} frames")
    
    # 两个视角的配置
    cameras = ["front", "wrist"]
    
    for cam in cameras:
        cam_key = f"observation.images.{cam}"
        
        # 获取视频文件路径信息
        chunk_idx = episode[f"videos/{cam_key}/chunk_index"]
        file_idx = episode[f"videos/{cam_key}/file_index"]
        
        # 视频路径
        video_path = f"{DATA_ROOT}/videos/{cam_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
        print(f"Reading video: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            continue
            
        # 3. 使用 torchvision 读取视频
        # torchvision.io.read_video 返回 (video, audio, info)
        # video tensor shape: (T, H, W, C)
        try:
            # 只读取所需的帧以节省内存/时间？
            # read_video 可以指定 start_pts 和 end_pts，但单位是 pts
            # 简单起见，如果视频不大，全部读取。或者尝试 stream="video"
            
            # 由于不知道 pts，我们先尝试全部读取（假设 chunk 比较小）
            # 或者使用 VideoReader
            from torchvision.io import VideoReader
            
            # 使用 VideoReader 更灵活
            reader = VideoReader(video_path, "video")
            
            # 提取该 episode 的所有帧
            length = episode['length']
            frames_to_extract = range(length)

            
            # 假设视频是从 0 开始对应 episode
            start_frame = 0 
            
            # VideoReader 迭代器方式，或者 seek
            # 为了准确 seek，我们需要知道 fps。 lerobot info.json 说 fps=60
            fps = 60.0
            
            for frame_offset in frames_to_extract:
                frame_idx = start_frame + frame_offset
                timestamp = frame_idx / fps
                
                reader.seek(timestamp)
                frame = next(reader) # 返回字典 {'data': tensor, 'pts': ...}
                
                img_tensor = frame['data'] # (C, H, W)
                
                # 转换为 PIL Image
                # torchvision 视频通常是 uint8 [0, 255]
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                save_path = f"{OUTPUT_DIR}/ep{episode['episode_index']}_{cam}_frame{frame_offset}.png"
                Image.fromarray(img_np).save(save_path)
                print(f"Saved {save_path}")
                
        except Exception as e:
            print(f"Error reading video with torchvision: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

