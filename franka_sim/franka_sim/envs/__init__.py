from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.envs.panda_stack_gym_env import PandaStackGymEnv
from franka_sim.envs.panda_insert_gym_env import PandaInsertGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
    "PandaPickCubeGymEnv",
    "PandaStackGymEnv",
    "PandaInsertGymEnv",
]

from gymnasium.envs.registration import register

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
register(
    id="PandaInsert-v0",
    entry_point="franka_sim.envs.panda_insert_gym_env:PandaInsertGymEnv",
    max_episode_steps=500,
)
register(
    id="PandaInsertVision-v0",
    entry_point="franka_sim.envs.panda_insert_gym_env:PandaInsertGymEnv",
    max_episode_steps=500,
    kwargs={"image_obs": True},
)
