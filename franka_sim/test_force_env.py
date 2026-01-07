#!/usr/bin/env python3
"""Test script for PandaPickCubeWithForce environment."""

import numpy as np
import gymnasium as gym
import franka_sim

def test_env_without_images():
    """Test environment without image observations."""
    print("=" * 60)
    print("Testing PandaPickCubeWithForce-v0 (state only)")
    print("=" * 60)
    
    env = gym.make("PandaPickCubeWithForce-v0")
    obs, info = env.reset()
    
    print("\nObservation space keys:", obs.keys())
    print("State keys:", obs["state"].keys())
    
    # Print shapes and sample values
    for key, value in obs["state"].items():
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Value: {value}")
    
    # Take a few steps
    print("\n" + "=" * 60)
    print("Taking 5 random steps...")
    print("=" * 60)
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Block pos: {obs['state']['block_pos']}")
        print(f"  Wrist force: {obs['state']['panda/wrist_force']}")
        print(f"  Joint torques: {obs['state']['panda/joint_torque']}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✓ State-only environment test passed!")


def test_env_with_images():
    """Test environment with image observations."""
    print("\n" + "=" * 60)
    print("Testing PandaPickCubeVisionWithForce-v0 (with images)")
    print("=" * 60)
    
    env = gym.make("PandaPickCubeVisionWithForce-v0")
    obs, info = env.reset()
    
    print("\nObservation space keys:", obs.keys())
    print("State keys:", obs["state"].keys())
    print("Image keys:", obs["images"].keys())
    
    # Print shapes
    print("\nState observations:")
    for key, value in obs["state"].items():
        print(f"  {key}: shape={value.shape}")
    
    print("\nImage observations:")
    for key, value in obs["images"].items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Verify block_pos is present even with image_obs=True
    assert "block_pos" in obs["state"], "block_pos should be in state!"
    print("\n✓ block_pos is present in state (as required)")
    
    # Take a step
    print("\n" + "=" * 60)
    print("Taking 1 step to verify all sensors work...")
    print("=" * 60)
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nReward: {reward:.4f}")
    print(f"Block pos: {obs['state']['block_pos']}")
    print(f"Wrist force: {obs['state']['panda/wrist_force']}")
    print(f"Joint torques shape: {obs['state']['panda/joint_torque'].shape}")
    print(f"Front image shape: {obs['images']['front'].shape}")
    print(f"Wrist image shape: {obs['images']['wrist'].shape}")
    
    env.close()
    print("\n✓ Vision environment test passed!")


if __name__ == "__main__":
    try:
        test_env_without_images()
        test_env_with_images()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNew environments registered successfully:")
        print("  - PandaPickCubeWithForce-v0")
        print("  - PandaPickCubeVisionWithForce-v0")
        print("\nThese environments include:")
        print("  ✓ Joint torques (7D)")
        print("  ✓ Wrist force (3D)")
        print("  ✓ Block position (always in state)")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

