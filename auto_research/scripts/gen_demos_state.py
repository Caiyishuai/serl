"""
生成 PandaPickCube-v0 的离线 demo 数据 (state-only, 无图像, 无 GUI)。

基于 serl/tools/generate_demos.py 改进:
- image_obs=False (纯 state, 快)
- 去掉 cv2.imshow (mac 无窗口环境会卡死)
- 命令行可配 --num_trajs / --output
- 修正: 用 env 的稀疏成功奖励, 保存标准 SERL transition dict

用法(必须在 serl 根目录之外运行, 或显式加 sys.path):
  venv_serl/bin/python auto_research/scripts/gen_demos_state.py --num_trajs 20 --output auto_research/data/demos_pickcube_state_20.pkl
"""
import argparse
import os
import sys
import pickle as pkl
import numpy as np
from tqdm import tqdm

# 显式把 franka_sim 包目录加进 path, 避免 namespace package 陷阱
SERL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(SERL_ROOT, "franka_sim"))

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv  # noqa: E402


def rollout_scripted(env, max_steps=200):
    """脚本化运动规划抓方块, 返回 (transitions, success)。"""
    obs, _ = env.reset()
    data = env.unwrapped.data
    block_pos = data.sensor("block_pos").data.copy()

    hover_pos = block_pos.copy(); hover_pos[2] += 0.02
    grasp_pos = block_pos.copy()
    lift_pos = block_pos.copy(); lift_pos[2] += 0.6

    stages = [
        {"target": hover_pos, "gripper": 1.0, "wait": 10},
        {"target": grasp_pos, "gripper": 1.0, "wait": 10},
        {"target": grasp_pos, "gripper": 0.0, "wait": 30},
        {"target": lift_pos,  "gripper": 0.0, "wait": 20},
    ]

    episode = []
    stage_idx = 0
    wait_counter = 0
    stage_step = 0          # 当前阶段已用步数
    POS_TOL = 0.03          # 放宽位置容差 (opspace 控制器难达到 1cm)
    STAGE_MAX = 60          # 每阶段最多 60 步, 到点或超时都强制推进

    for _ in range(max_steps):
        if stage_idx >= len(stages):
            break
        tcp_pos = obs["state"]["panda/tcp_pos"]
        gripper_pos = obs["state"]["panda/gripper_pos"]
        gripper_pos = float(np.asarray(gripper_pos).reshape(-1)[0])

        stage = stages[stage_idx]
        target = stage["target"]
        gripper_target = stage["gripper"]
        wait_steps = stage["wait"]

        dist = np.linalg.norm(target - tcp_pos)
        stage_step += 1
        reached = dist < POS_TOL
        if reached or stage_step > STAGE_MAX:
            if wait_counter < wait_steps:
                wait_counter += 1
            else:
                stage_idx += 1
                wait_counter = 0
                stage_step = 0
                if stage_idx >= len(stages):
                    break
                continue

        action_pos = np.clip((target - tcp_pos) / 0.1, -1.0, 1.0)
        action_gripper = gripper_target - gripper_pos
        action = np.concatenate([action_pos, [action_gripper]]).astype(np.float32)

        next_obs, rew, done, truncated, info = env.step(action)
        episode.append({
            "observations": obs,
            "actions": action,
            "next_observations": next_obs,
            "rewards": float(rew),
            "masks": 1.0 - float(done),
            "dones": bool(done),
            "infos": info,
        })
        obs = next_obs
        if done or truncated:
            break

    final_block_z = env.unwrapped.data.sensor("block_pos").data[2]
    success = bool(final_block_z >= env.unwrapped._z_success)
    return episode, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_trajs", type=int, default=20)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--max_attempts", type=int, default=200)
    ap.add_argument("--keep_all", action="store_true",
                    help="保存所有轨迹(含未抓起成功的), 用作 RLPD 离线 buffer 数据")
    args = ap.parse_args()

    env = PandaPickCubeGymEnv(render_mode="rgb_array", image_obs=False, control_dt=0.02)

    transitions = []
    success_count = 0
    attempts = 0
    pbar = tqdm(total=args.num_trajs, desc="collecting demos")
    while success_count < args.num_trajs and attempts < args.max_attempts:
        attempts += 1
        episode, success = rollout_scripted(env)
        if success:
            transitions.extend(episode)
            success_count += 1
            pbar.update(1)
        elif args.keep_all:
            # 未成功也保留(dense reward 仍有信号), 用作离线 buffer
            transitions.extend(episode)
            pbar.update(1)
            success_count += 1  # 借用计数控制轨迹数
    pbar.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pkl.dump(transitions, f)
    print(f"[gen_demos] success={success_count}/{attempts} attempts, "
          f"total {len(transitions)} transitions -> {args.output}")


if __name__ == "__main__":
    main()
