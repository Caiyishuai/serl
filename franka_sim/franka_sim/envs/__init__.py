from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
    "PandaPickCubeGymEnv",
]

from gym.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs.panda_pick_gym_env:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs.panda_pick_gym_env:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
