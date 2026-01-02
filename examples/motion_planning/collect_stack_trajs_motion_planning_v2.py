from pathlib import Path
from typing import Any, Literal, Tuple, Dict
import copy
import pickle
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "../franka_sim/franka_sim/envs/xmls" / "arena_stack.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])


class PandaStackGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "dense",
    ):
        self._action_scale = action_scale
        self.reward_type = reward_type

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        state_space = {
            "panda/tcp_pos": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
            "panda/tcp_vel": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
            "panda/gripper_pos": spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            ),
            "block_pos": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
            "target_pillar_pos": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
        }

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "panda/tcp_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                            "target_pillar_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(state_space),
                }
            )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
        )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)

        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        x, y, z, grasp = action

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded()

        return obs, rew, terminated, False, {}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        pillar_pos = self._data.sensor("target_pillar_pos").data.astype(np.float32)
        obs["state"]["target_pillar_pos"] = pillar_pos

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        pillar_pos = self._data.sensor("target_pillar_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data

        # Constants
        PILLAR_HEIGHT = 0.08
        BLOCK_HEIGHT = 0.04
        SAFE_LIFT_HEIGHT = PILLAR_HEIGHT + 0.05  # 0.13m

        # Phase 1: Lift
        dist_tcp_block = np.linalg.norm(block_pos - tcp_pos)
        r_reach = (1 - np.tanh(10.0 * dist_tcp_block))

        z_score = (block_pos[2] - 0.02) / (SAFE_LIFT_HEIGHT - 0.02)
        r_lift = np.clip(z_score, 0.0, 1.0)

        # Phase 2: Place
        block_bottom_z = block_pos[2] - BLOCK_HEIGHT / 2
        pillar_top_z = pillar_pos[2] + PILLAR_HEIGHT / 2
        target_pos = pillar_pos.copy()
        target_pos[2] = pillar_top_z
        
        dist_xy = np.linalg.norm(block_pos[:2] - target_pos[:2])
        dist_z = block_bottom_z - pillar_top_z
        dist_block_target = np.sqrt(dist_xy**2 + dist_z**2)
        r_place = (1 - np.tanh(5.0 * dist_block_target))

        # Combine phases
        if block_pos[2] < SAFE_LIFT_HEIGHT:
            rew = 0.2 * r_reach + 0.8 * r_lift
        else:
            rew = 1.0 + r_place

        return rew


# 运动规划参数
lower_limit = -0.1
upper_limit = 0.1
max_dis = 0.5
min_dis = 0.05


def step_collect_data(env, action, data_list, last_observations=None, task_stage=None):
    """执行一步并收集数据"""
    obs, rew, done, truncated, info = env.step(action)
    data_dict = {
        'observations': last_observations,
        'actions': action,
        'next_observations': obs,
        'rewards': rew,
        'masks': 1 - done,
        'dones': truncated or done,
    }
    
    if task_stage is not None:
        data_dict['task_stage'] = task_stage
        
    data_list.append(data_dict)
    return obs


def go_to_target(env, target_pos, data_list, task_stage=None):
    """移动到目标位置"""
    obs = env._compute_observation()
    while True:
        delta_pos = np.clip(target_pos - obs["state"]["panda/tcp_pos"], lower_limit, upper_limit)
        dis = np.linalg.norm(obs["state"]["panda/tcp_pos"] - target_pos)
        dis = np.clip(dis, min_dis, max_dis)
        
        dis_ratio = (dis - min_dis) / (max_dis - min_dis)
        norm_delta_pos = delta_pos * (0.1 + dis_ratio * 2.5)

        action = np.concatenate([norm_delta_pos, [0]])
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        # 移除 env.render() 调用，因为在 human 模式下已经在 _compute_observation 中渲染

        if np.linalg.norm(obs["state"]["panda/tcp_pos"] - target_pos) < 0.05:
            break
    
    return obs


def close_gripper(env, data_list, task_stage=None):
    """关闭夹爪"""
    obs = env._compute_observation()
    action = np.array([0, 0, 0, 1])
    for _ in range(10):  # 执行固定步数确保夹爪完全闭合
        last_gripper_pos = obs["state"]["panda/gripper_pos"]
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        # 移除 env.render() 调用
        if np.abs(obs["state"]["panda/gripper_pos"] - last_gripper_pos) < 0.005:
            break
    return obs


def open_gripper(env, data_list, task_stage=None):
    """打开夹爪"""
    obs = env._compute_observation()
    action = np.array([0, 0, 0, -1])
    for _ in range(10):  # 执行固定步数确保夹爪完全打开
        last_gripper_pos = obs["state"]["panda/gripper_pos"]
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        # 移除 env.render() 调用
        if np.abs(obs["state"]["panda/gripper_pos"] - last_gripper_pos) < 0.005:
            break
    return obs


if __name__ == "__main__":
    
    print(f"ROOT_PATH: {ROOT_PATH}")
    
    max_transition_data_num = 20  # 收集的轨迹数量
    transition_data_list = []
    
    # 使用 human 模式进行可视化
    # 渲染会在 _compute_observation 中自动完成，不需要手动调用 render()
    env = PandaStackGymEnv(render_mode="human")
    
    for i in range(max_transition_data_num):
        print(f"\n=== 收集轨迹 {i+1}/{max_transition_data_num} ===")
        env.reset()
        
        data_list = []
        obs = env._compute_observation()
        
        # 获取物体和目标位置
        block_pos = obs["state"]["block_pos"].copy()
        pillar_pos = obs["state"]["target_pillar_pos"].copy()
        
        print(f"Block position: {block_pos}")
        print(f"Pillar position: {pillar_pos}")
        
        # ========== 阶段 0: 移动到 block 上方 ==========
        target = block_pos.copy()
        target[2] += 0.05  # 在 block 上方 5cm
        print(f"阶段 0: 移动到 block 上方 {target}")
        go_to_target(env, target, data_list, task_stage=0)
        
        # ========== 阶段 1: 下降到 block ==========
        target[2] = block_pos[2] - 0.02  # 稍微低于 block 中心
        print(f"阶段 1: 下降到 block {target}")
        go_to_target(env, target, data_list, task_stage=1)
        
        # ========== 阶段 2: 关闭夹爪抓取 ==========
        print(f"阶段 2: 关闭夹爪")
        close_gripper(env, data_list, task_stage=2)
        
        # ========== 阶段 3: 抬起 block 到安全高度 ==========
        SAFE_LIFT_HEIGHT = 0.15  # 15cm 安全高度
        target = obs["state"]["panda/tcp_pos"].copy()
        target[2] = SAFE_LIFT_HEIGHT
        print(f"阶段 3: 抬起到安全高度 {target}")
        go_to_target(env, target, data_list, task_stage=3)
        
        # ========== 阶段 4: 移动到 pillar 上方 ==========
        target = pillar_pos.copy()
        target[2] = SAFE_LIFT_HEIGHT  # 保持安全高度
        print(f"阶段 4: 移动到 pillar 上方 {target}")
        go_to_target(env, target, data_list, task_stage=4)
        
        # ========== 阶段 5: 下降到 pillar 顶部 ==========
        PILLAR_HEIGHT = 0.08
        target[2] = pillar_pos[2] + PILLAR_HEIGHT / 2 + 0.02  # pillar 顶部 + block 半高
        print(f"阶段 5: 下降到 pillar 顶部 {target}")
        go_to_target(env, target, data_list, task_stage=5)
        
        # ========== 阶段 6: 打开夹爪放下 block ==========
        print(f"阶段 6: 打开夹爪")
        open_gripper(env, data_list, task_stage=6)
        
        # ========== 阶段 7: 抬起离开 ==========
        target = obs["state"]["panda/tcp_pos"].copy()
        target[2] += 0.1  # 上升 10cm
        print(f"阶段 7: 抬起离开 {target}")
        go_to_target(env, target, data_list, task_stage=7)
        
        print(f"轨迹 {i+1} 收集完成，包含 {len(data_list)} 个转换")
        transition_data_list.extend(data_list)
    
    print(f"\n总共收集了 {len(transition_data_list)} 个转换")
    
    # 保存完整数据
    folder_name = f"{ROOT_PATH}/../../../../data/stack_trajs/panda_stack_{max_transition_data_num}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    with open(f"{folder_name}/demo_data.pkl", "wb") as f:
        pickle.dump(transition_data_list, f)
    print(f"保存完整数据到 {folder_name}/demo_data.pkl")
    
    # 按阶段保存数据
    action_agent_num = 8  # 8个阶段
    action_transition_data = [[] for _ in range(action_agent_num)]
    
    for transition_data in transition_data_list:
        stage = transition_data["task_stage"]
        action_transition_data[stage].append(transition_data)
    
    for i in range(action_agent_num):
        with open(f"{folder_name}/act_{i}.pkl", "wb") as f:
            pickle.dump(action_transition_data[i], f)
        print(f"保存阶段 {i} 数据到 {folder_name}/act_{i}.pkl，包含 {len(action_transition_data[i])} 个转换")
    
    env.close()
    print("\n数据收集完成！")

