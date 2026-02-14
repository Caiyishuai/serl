"""
Pick (lift cube) task data collection script using motion planning.
Uses PandaPickCubeRealSpaceVisionGymEnv (small workspace, 8cm lift success).
"""
import os
import pickle
import numpy as np
from pathlib import Path


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

from franka_sim.envs.panda_pick_gym_env_real_space import PandaPickCubeRealSpaceVisionGymEnv

VIEWER = None

# Motion planning: env action is [x,y,z,grasp] with pos scale 0.015 m per unit.
STEP_POS_SCALE = 0.015
MAX_DIS = 0.05
MIN_DIS = 0.005
DEAD_ZONE = 0.003  # 3 mm
SUCCESS_LIFT_HEIGHT = 0.08  # 8 cm, match env


def format_observation(obs):
    """
    Convert env observation to training format.

    Env (image_obs=True): state has panda/tcp_pos(3), panda/tcp_vel(3), panda/gripper_pos(1).
    No block_pos in state when image_obs=True.

    Output for training:
        state: (1, 7)  [tcp_pos(3), tcp_vel(3), gripper_pos(1)]
        front: (1, H, W, 3)
        wrist: (1, H, W, 3)
    """
    state_list = []
    state_list.append(obs["state"]["panda/tcp_pos"])
    state_list.append(obs["state"]["panda/tcp_vel"])
    state_list.append(np.array([obs["state"]["panda/gripper_pos"]], dtype=np.float32))
    state = np.concatenate(state_list).astype(np.float32)  # (7,)

    formatted_obs = {
        "state": state[np.newaxis, :],
        "front": obs["images"]["front"][np.newaxis, :],
        "wrist": obs["images"]["wrist"][np.newaxis, :],
    }
    return formatted_obs


def step_collect_data(env, action, data_list, last_observations=None, task_stage=None):
    """Execute one step and append transition to data_list."""
    global VIEWER

    obs, rew, done, truncated, info = env.step(action)

    if VIEWER is not None:
        VIEWER.sync()

    formatted_obs = format_observation(obs)
    data_dict = {
        "observations": last_observations,
        "actions": action.astype(np.float32),
        "next_observations": formatted_obs,
        "rewards": np.float32(rew),
        "masks": float(1 - done),
        "dones": bool(truncated or done),
    }
    data_list.append(data_dict)
    return formatted_obs


def go_to_target(env, target_pos, data_list, task_stage=None, dead_zone=DEAD_ZONE):
    """Move TCP to target_pos. Action space: [x,y,z,grasp] with pos scale STEP_POS_SCALE."""
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)
    max_steps = 300
    step_count = 0

    while step_count < max_steps:
        tcp_pos = raw_obs["state"]["panda/tcp_pos"]
        delta_pos = target_pos - tcp_pos
        dis = np.linalg.norm(delta_pos)

        if dis < dead_zone:
            break

        direction = delta_pos / (dis + 1e-8)
        dis_clipped = np.clip(dis, MIN_DIS, MAX_DIS)
        dis_ratio = (dis_clipped - MIN_DIS) / (MAX_DIS - MIN_DIS + 1e-8)
        speed = 0.005 + dis_ratio * 0.01  # max ~1.5cm per step
        if dis < 0.02:
            speed *= 0.5

        move_m = direction * min(speed, STEP_POS_SCALE)
        action_xyz = np.clip(move_m / STEP_POS_SCALE, -1.0, 1.0)
        action = np.concatenate([action_xyz.astype(np.float32), [np.float32(0)]])

        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        raw_obs = env._compute_observation()
        step_count += 1

    return obs


def close_gripper(env, data_list, task_stage=None, steps=20):
    """Close gripper."""
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for i in range(steps):
        last_gripper = raw_obs["state"]["panda/gripper_pos"]
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        raw_obs = env._compute_observation()
        if i > 5 and np.abs(raw_obs["state"]["panda/gripper_pos"] - last_gripper) < 0.003:
            break
    return obs


def open_gripper(env, data_list, task_stage=None, steps=20):
    """Open gripper."""
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)
    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    for i in range(steps):
        last_gripper = raw_obs["state"]["panda/gripper_pos"]
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=task_stage)
        raw_obs = env._compute_observation()
        if i > 5 and np.abs(raw_obs["state"]["panda/gripper_pos"] - last_gripper) < 0.003:
            break
    return obs


def check_success(env):
    """Success: block lifted by SUCCESS_LIFT_HEIGHT (8 cm)."""
    block_pos = env._data.sensor("block_pos").data
    z_init = getattr(env, "_z_init", block_pos[2] - SUCCESS_LIFT_HEIGHT)
    lift = block_pos[2] - z_init
    return lift >= SUCCESS_LIFT_HEIGHT - 0.005


def collect_one_trajectory(env):
    """
    Collect one pick trajectory: start above block (10 cm), go down, grasp, lift 8 cm.
    Returns: list of transitions, success bool.
    """
    data_list = []

    obs_dict, _ = env.reset()
    raw_obs = env._compute_observation()
    obs = format_observation(raw_obs)

    block_pos = env._data.sensor("block_pos").data.copy()
    z_init = block_pos[2]
    z_success = z_init + SUCCESS_LIFT_HEIGHT

    # Stage 0: Already above block at 10 cm. Move down to just above block (e.g. +2 cm), then into grasp height.
    target = block_pos.copy()
    target[2] = block_pos[2] + 0.02
    obs = go_to_target(env, target, data_list, task_stage=0, dead_zone=0.004)

    # Stage 1: Down to grasp (slightly below block center so fingers wrap).
    target[2] = block_pos[2] - 0.01
    obs = go_to_target(env, target, data_list, task_stage=1, dead_zone=0.004)

    # Stage 2: Close gripper.
    obs = close_gripper(env, data_list, task_stage=2)

    # Stage 3: Lift to success height (8 cm above initial).
    raw_obs = env._compute_observation()
    tcp_pos = raw_obs["state"]["panda/tcp_pos"]
    target = tcp_pos.copy()
    target[2] = z_success + 0.01
    obs = go_to_target(env, target, data_list, task_stage=3, dead_zone=0.005)

    for _ in range(5):
        action = np.zeros(4, dtype=np.float32)
        obs = step_collect_data(env, action, data_list, last_observations=obs, task_stage=3)

    is_success = check_success(env)
    return data_list, is_success


def main():
    global VIEWER

    print(f"ROOT_PATH: {ROOT_PATH}")

    num_trajectories = 20
    render_visual = True
    max_attempts = 100

    render_mode = "rgb_array"
    env = PandaPickCubeRealSpaceVisionGymEnv(render_mode=render_mode, image_obs=True)

    if render_visual:
        try:
            import mujoco
            import mujoco.viewer
            VIEWER = mujoco.viewer.launch_passive(env._model, env._data)
            print("Visualization window opened.")
        except Exception as e:
            print(f"Could not open viewer: {e}")
            VIEWER = None

    all_transitions = []
    success_count = 0
    attempt_count = 0

    print(f"\nCollecting {num_trajectories} successful pick trajectories (max attempts: {max_attempts})...")

    while success_count < num_trajectories and attempt_count < max_attempts:
        attempt_count += 1
        try:
            print(f"\n=== Attempt {attempt_count}, success {success_count}/{num_trajectories} ===")
            data_list, is_success = collect_one_trajectory(env)

            if is_success:
                all_transitions.extend(data_list)
                print(f"Trajectory success, {len(data_list)} transitions")
                success_count += 1
            else:
                print("Trajectory failed (block not lifted 8 cm).")

        except Exception as e:
            print(f"Attempt {attempt_count} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if VIEWER is not None:
        VIEWER.close()
    env.close()

    total_transitions = len(all_transitions)
    success_rate = (success_count / attempt_count * 100) if attempt_count > 0 else 0
    print(f"\n" + "=" * 60)
    print("Pick data collection finished.")
    print(f"Successful trajectories: {success_count}/{attempt_count} ({success_rate:.1f}%)")
    print(f"Total transitions: {total_transitions}")
    print("=" * 60)

    if total_transitions == 0:
        print("No successful data collected.")
        return

    save_path = os.path.join(ROOT_PATH, f"franka_lift_cube_image_{success_count}_trajs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_transitions, f)
    print(f"\nSaved to: {save_path}")
    print(f"File contains {success_count} trajectories, {total_transitions} transitions.")


if __name__ == "__main__":
    main()
