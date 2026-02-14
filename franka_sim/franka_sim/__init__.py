from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.envs.panda_pick_gym_env_with_force import PandaPickCubeGymEnvWithForce
from franka_sim.envs.panda_stack_gym_env import PandaStackGymEnv
from franka_sim.envs.panda_pick_gym_env_real_space import PandaPickCubeRealSpaceVisionGymEnv
__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
    "PandaPickCubeGymEnv",
    "PandaPickCubeGymEnvWithForce",
    "PandaStackGymEnv",
    "PandaPickCubeRealSpaceVisionGymEnv",
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
    id="PandaPickCubeWithForce-v0",
    entry_point="franka_sim.envs.panda_pick_gym_env_with_force:PandaPickCubeGymEnvWithForce",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVisionWithForce-v0",
    entry_point="franka_sim.envs.panda_pick_gym_env_with_force:PandaPickCubeGymEnvWithForce",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPickCubeRealSpaceVision-v0",
    entry_point="franka_sim.envs.panda_pick_gym_env_real_space:PandaPickCubeRealSpaceVisionGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)