
import sys
import os
import gymnasium as gym

# Try to import franka_sim and see where it comes from
try:
    import franka_sim
    print(f"franka_sim imported from: {franka_sim.__file__}")
except ImportError as e:
    print(f"Failed to import franka_sim: {e}")

# Check registry
env_id = "PandaPickCubeVision-v0"
if env_id in gym.envs.registry:
    print(f"Environment {env_id} found in registry.")
else:
    print(f"Environment {env_id} NOT found in registry.")
    print("Available envs:", [k for k in gym.envs.registry.keys() if "Panda" in k])

try:
    env = gym.make(env_id)
    print("Env created successfully")
except Exception as e:
    print(f"Failed to make env: {e}")

