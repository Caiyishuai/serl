from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.envs.panda_stack_gym_env import PandaStackGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
    "PandaPickCubeGymEnv",
    "PandaStackGymEnv",
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
register(
    id="PandaStack-v0",
    entry_point="franka_sim.envs.panda_stack_gym_env:PandaStackGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaStackVision-v0",
    entry_point="franka_sim.envs.panda_stack_gym_env:PandaStackGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
