"""
Panda Insertion Task (plug-into-socket).

At reset the robot already holds the red connector (红插头).  The connector body
is a kinematic child of the gripper "base" body in panda_insert.xml (no joint),
so it rigidly follows the gripper via MuJoCo's kinematics – no Python intervention.

The green socket (绿插座) is placed at a random XY position directly on the floor.
The agent must align the connector tip with the socket opening and push down.

Action space: 3-D continuous (Δx, Δy, Δz) – gripper open/close NOT included.
"""

from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena_insert.xml"

# Arm home pose (gripper points downward)
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))

# Hard Cartesian safety bounds for the TCP mocap target.
_CARTESIAN_BOUNDS = np.asarray([[0.20, -0.30, 0.00], [0.65, 0.30, 0.55]])

# Socket XY sampling region on the floor.
_SOCKET_SAMPLING_BOUNDS = np.asarray([[0.30, -0.12], [0.55, 0.12]])

# Floor-level z for socket placement (just above the floor plane).
_FLOOR_Z = 0.002

# How deep (m) the connector tip must go below the socket opening to count as success.
_INSERT_DEPTH = 0.020   # 2 cm

# Per-episode XY half-range around the socket (±4 cm).
_EPISODE_XY_HALF = 0.04

# Height above socket opening for the reset target position.
_RESET_HEIGHT = 0.10    # 10 cm above socket opening


class PandaInsertGymEnv(MujocoGymEnv):
    """Franka panda insertion (plug into socket) environment."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.01, 1.0]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 15.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "dense",   # "dense" or "binary"
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
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)   # front, wrist cameras
        self.image_obs = image_obs

        # ── Cache frequently-used model indices ──────────────────────────────
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id   = self._model.site("pinch").id

        # Gripper "base" body id – used to read orientation for mocap_quat at reset.
        self._base_body_id    = self._model.body("base").id

        # Per-episode Cartesian bounds (set in reset based on socket position)
        self._episode_bounds = _CARTESIAN_BOUNDS.copy()

        # ── Observation space ────────────────────────────────────────────────
        state_dict = {
            "panda/tcp_pos":    spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "panda/tcp_vel":    spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "panda/gripper_pos": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "socket_pos":       spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }

        if self.image_obs:
            self.observation_space = gym.spaces.Dict({
                "state": gym.spaces.Dict({
                    "panda/tcp_pos":    spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "panda/tcp_vel":    spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "panda/gripper_pos": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                    "socket_pos":       spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                }),
                "images": gym.spaces.Dict({
                    "front": gym.spaces.Box(
                        low=0, high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist": gym.spaces.Box(
                        low=0, high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    ),
                }),
            })
        else:
            self.observation_space = gym.spaces.Dict({
                "state": gym.spaces.Dict(state_dict),
            })

        # ── Action space: 3-D position delta only (no gripper) ──────────────
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0]),
            high=np.asarray([1.0,  1.0,  1.0]),
            dtype=np.float32,
        )

        # ── Renderers ─────────────────────────────────────────────────────────
        if self.image_obs:
            self._img_renderer = mujoco.Renderer(
                self._model,
                height=render_spec.height,
                width=render_spec.width,
            )
        else:
            self._img_renderer = None

        # Human display: mujoco passive viewer (GLFW, independent EGL context).
        self._passive_viewer = None
        if self.render_mode == "human":
            import mujoco.viewer as _mj_viewer
            self._passive_viewer = _mj_viewer.launch_passive(
                self._model, self._data
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset: arm to home, gripper closed, socket at random XY on floor,
        mocap target set to 10 cm above socket opening."""
        mujoco.mj_resetData(self._model, self._data)

        # Arm to home pose
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        self._data.ctrl[self._gripper_ctrl_id] = 255.0
        mujoco.mj_forward(self._model, self._data)

        # Randomise socket XY directly on the floor
        socket_xy = np.random.uniform(*_SOCKET_SAMPLING_BOUNDS)
        socket_z  = _FLOOR_Z
        socket_jnt_id   = self._model.joint("socket_joint").id
        socket_qpos_adr = self._model.jnt_qposadr[socket_jnt_id]
        self._data.qpos[socket_qpos_adr    : socket_qpos_adr + 3] = [*socket_xy, socket_z]
        self._data.qpos[socket_qpos_adr + 3: socket_qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self._model, self._data)

        # Socket opening height = socket_z + 0.050 (site at z=0.050 in socket frame)
        socket_opening_z = socket_z + 0.050

        # Per-episode Cartesian bounds: ±4 cm around socket XY, Z from floor to 15 cm above socket
        lo = np.array([socket_xy[0] - _EPISODE_XY_HALF,
                       socket_xy[1] - _EPISODE_XY_HALF,
                       socket_z])
        hi = np.array([socket_xy[0] + _EPISODE_XY_HALF,
                       socket_xy[1] + _EPISODE_XY_HALF,
                       socket_opening_z + _RESET_HEIGHT + 0.05])
        # Clamp to global safety bounds
        self._episode_bounds = np.stack([
            np.maximum(lo, _CARTESIAN_BOUNDS[0]),
            np.minimum(hi, _CARTESIAN_BOUNDS[1]),
        ])

        # Set mocap target to 10 cm above socket opening, then run warmup steps
        # so the opspace controller can move the arm to the starting position.
        target_pos = np.array([socket_xy[0], socket_xy[1], socket_opening_z + _RESET_HEIGHT])
        target_pos = np.clip(target_pos, *self._episode_bounds)
        self._data.mocap_pos[0] = target_pos

        # Get the "down" mocap orientation from the gripper base body
        base_quat = self._data.xquat[self._base_body_id].copy()
        self._data.mocap_quat[0] = base_quat

        # Warmup: run 200 control steps to let the arm converge to the target
        for _ in range(200):
            self._data.ctrl[self._gripper_ctrl_id] = 255.0
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

        mujoco.mj_forward(self._model, self._data)

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step: apply 3-D position delta; gripper stays closed."""
        x, y, z = action

        # Move mocap target within per-episode bounds
        pos  = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *self._episode_bounds)
        self._data.mocap_pos[0] = npos

        # Gripper always closed
        self._data.ctrl[self._gripper_ctrl_id] = 255.0

        # Run physics substeps
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

        # Update kinematics
        mujoco.mj_forward(self._model, self._data)

        obs  = self._compute_observation()
        rew  = self._compute_reward()
        done = self.time_limit_exceeded() or self._check_success()

        return obs, rew, done, False, {}

    def render(self):
        """Return [front_frame, wrist_frame] for image observations."""
        frames = []
        for cam_id in self.camera_id:
            if self._img_renderer is not None:
                self._img_renderer.update_scene(self._data, camera=cam_id)
                frames.append(self._img_renderer.render().copy())
            else:
                self._img_renderer = mujoco.Renderer(self._model)
                self._img_renderer.update_scene(self._data, camera=cam_id)
                frames.append(self._img_renderer.render().copy())
        return frames

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_observation(self) -> dict:
        obs = {"state": {}}

        obs["state"]["panda/tcp_pos"] = (
            self._data.sensor("2f85/pinch_pos").data.astype(np.float32)
        )
        obs["state"]["panda/tcp_vel"] = (
            self._data.sensor("2f85/pinch_vel").data.astype(np.float32)
        )
        obs["state"]["panda/gripper_pos"] = np.array(
            [self._data.ctrl[self._gripper_ctrl_id] / 255.0], dtype=np.float32
        )
        obs["state"]["socket_pos"] = (
            self._data.sensor("socket_pos").data.astype(np.float32)
        )

        if self.image_obs:
            front, wrist = self.render()
            obs["images"] = {"front": front, "wrist": wrist}

        # Sync passive viewer (human window) if active
        if self._passive_viewer is not None and self._passive_viewer.is_running():
            self._passive_viewer.sync()

        return obs

    def _compute_reward(self) -> float:
        tip_pos    = self._data.sensor("connector_tip_pos").data    # (3,)
        socket_pos = self._data.sensor("socket_opening_pos").data   # (3,)

        # Horizontal (XY) alignment distance
        xy_dist = np.linalg.norm(tip_pos[:2] - socket_pos[:2])

        # Vertical distance: positive when tip is ABOVE socket opening
        z_above = max(0.0, tip_pos[2] - socket_pos[2])

        # Phase 1: approach & align – exponential reward on combined distance
        approach_dist = np.sqrt(xy_dist ** 2 + z_above ** 2)
        r_approach = float(np.exp(-20.0 * approach_dist))

        # Phase 2: insertion depth (positive = tip below socket opening)
        insert_depth = max(0.0, socket_pos[2] - tip_pos[2])
        r_insert = np.clip(insert_depth / _INSERT_DEPTH, 0.0, 1.0)

        # Weight insertion by alignment to prevent hacking from far away
        rew = 0.4 * r_approach + 0.6 * r_insert * r_approach

        return float(rew)

    def _check_success(self) -> bool:
        tip_pos    = self._data.sensor("connector_tip_pos").data
        socket_pos = self._data.sensor("socket_opening_pos").data
        xy_dist    = np.linalg.norm(tip_pos[:2] - socket_pos[:2])
        return (xy_dist < 0.015) and (socket_pos[2] - tip_pos[2] >= _INSERT_DEPTH)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = PandaInsertGymEnv(render_mode="human")
    obs, _ = env.reset()
    print("Reset OK")
    print("TCP pos:", obs["state"]["panda/tcp_pos"])
    print("Socket pos:", obs["state"]["socket_pos"])
    for i in range(200):
        action = env.action_space.sample()
        obs, rew, done, _, _ = env.step(action)
        if i % 20 == 0:
            print(f"step {i}  reward={rew:.4f}  done={done}")
        if done:
            obs, _ = env.reset()
    env.close()

