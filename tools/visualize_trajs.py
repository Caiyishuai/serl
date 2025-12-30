import pickle
import numpy as np
from PIL import Image
import os

file_path = 'examples/franka_lift_cube_image_20_trajs.pkl'
output_gif = 'replay_trajectory.gif'

def main():
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} transitions.")
    
    if len(data) == 0:
        print("No data found.")
        return

    # Check image size from the first element
    first_obs = data[0]['observations']
    front_img = first_obs['front']
    wrist_img = first_obs['wrist']
    
    # Remove batch dimension if present (1, H, W, C) -> (H, W, C)
    if front_img.ndim == 4:
        front_img = front_img[0]
    if wrist_img.ndim == 4:
        wrist_img = wrist_img[0]
        
    h, w, c = front_img.shape
    print(f"Image size (Front): {w}x{h}")
    print(f"Image size (Wrist): {wrist_img.shape[1]}x{wrist_img.shape[0]}")
    
    frames = []
    # Visualize first trajectory (assuming ~100 steps per traj, or just take first 200 frames)
    # The user said "20 trajs" and there are 2000 items, so 100 steps/traj is a good guess.
    # Let's visualize the first 100 frames.
    
    num_frames = min(len(data), 100)
    print(f"Generating GIF with first {num_frames} frames...")
    
    for i in range(num_frames):
        obs = data[i]['observations']
        img1 = obs['front']
        img2 = obs['wrist']
        
        if img1.ndim == 4: img1 = img1[0]
        if img2.ndim == 4: img2 = img2[0]
        
        # Ensure uint8
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        
        # Convert to PIL images
        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)
        
        # Concatenate horizontally
        total_width = pil_img1.width + pil_img2.width
        max_height = max(pil_img1.height, pil_img2.height)
        
        new_im = Image.new('RGB', (total_width, max_height))
        new_im.paste(pil_img1, (0, 0))
        new_im.paste(pil_img2, (pil_img1.width, 0))
        
        frames.append(new_im)
        
    # Save GIF
    frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
    print(f"Saved visualization to {output_gif}")

if __name__ == "__main__":
    main()

