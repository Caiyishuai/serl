#!/usr/bin/env python3
"""
make_residual_demos.py — Convert expert demos to residual demos using local ReFlow policy.

Residual demo:  a_residual = a_expert - a_base
where a_base is produced by the local ReFlowPolicy for each obs.

The output .pkl is compatible with async_drq_sim.py --demo_path.

Usage
-----
python examples/reflow_res_rl/make_residual_demos.py \
    --demo_path serl/examples/openpi/maniskill_demo_data/mani_skill_push_cube_dense_no_clip_100_fixed_complete.pkl \
    --output_path serl/examples/reflow_res_rl/demo_data/pushcube_residual.pkl \
    --reflow_ckpt runs/PushCube-v1__reflow__42__xxx/checkpoints/best_success_once.pt \
    --env PushCube-v1

If --skip_base_query, a_residual = a_expert (useful when ReFlow not available / testing).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SERL_ROOT = Path(__file__).resolve().parents[2]
if str(_SERL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERL_ROOT))

# ── Reuse helpers from openpi ─────────────────────────────────────────────────
from examples.openpi.async_drq_sim import extract_eef_from_env  # noqa: E402
from examples.reflow_res_rl.reflow_policy import ReFlowPolicy  # noqa: E402


def _strip_leading_one(obs: dict) -> dict:
    """Remove leading dim-1 added by ChunkingWrapper: (1,N)->(N,), (1,H,W,3)->(H,W,3)."""
    result = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        result[k] = arr
    return result


def make_residual_demos(args: argparse.Namespace) -> None:
    # ── Load expert demos ──────────────────────────────────────────────────────
    print(f"Loading expert demos: {args.demo_path}")
    with open(args.demo_path, "rb") as f:
        expert_trajs: list[dict] = pickle.load(f)
    print(f"  {len(expert_trajs)} transitions")

    sample_obs = expert_trajs[0]["observations"]
    for k, v in sample_obs.items():
        print(f"  obs['{k}']: shape={np.asarray(v).shape}")
    print(f"  actions: shape={np.asarray(expert_trajs[0]['actions']).shape}")

    # ── Init ReFlow policy (optional) ──────────────────────────────────────────
    policy: ReFlowPolicy | None = None
    if not args.skip_base_query:
        policy = ReFlowPolicy(
            ckpt_path=args.reflow_ckpt,
            obs_horizon=args.reflow_obs_horizon,
            pred_horizon=args.reflow_pred_horizon,
            act_horizon=args.reflow_act_horizon,
            hidden_dim=args.reflow_hidden_dim,
            n_layers=args.reflow_n_layers,
            time_emb_dim=args.reflow_time_emb_dim,
            max_denoising_steps=args.reflow_denoising_steps,
            img_size=args.reflow_img_size,
        )
        print("ReFlowPolicy loaded.")
    else:
        print("--skip_base_query: a_residual = a_expert (base action = 0)")

    # ── Build residual demos ───────────────────────────────────────────────────
    residual_trajs: list[dict] = []
    n_skipped = 0

    for i, traj in enumerate(expert_trajs):
        obs_raw  = traj["observations"]
        nobs_raw = traj["next_observations"]
        a_expert = np.asarray(traj["actions"], dtype=np.float32).reshape(-1)

        # Strip ChunkingWrapper leading dim if present
        obs_serl  = _strip_leading_one(obs_raw)
        nobs_serl = _strip_leading_one(nobs_raw)

        if policy is not None:
            try:
                # Extract 8-dim eef from obs["state"] using index-based parsing
                # (flat state index layout for PushCube PD-EEF):
                #   [18:21] = eef_pos,  [21:25] = eef_quat (wxyz),  [7:9] = gripper
                state_flat = np.asarray(obs_serl["state"]).flatten()
                from scipy.spatial.transform import Rotation
                eef_pos = state_flat[18:21].astype(np.float32)
                q_wxyz  = state_flat[21:25]
                q_xyzw  = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], np.float32)
                rot_aa  = Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)
                gripper = state_flat[7:9].astype(np.float32)
                eef_state = np.concatenate([eef_pos, rot_aa, gripper])  # (8,)

                # img keys: hand_camera → img_wrist, base_camera → img_third
                reflow_obs = {
                    "state":     eef_state,
                    "img_wrist": obs_serl["hand_camera"],
                    "img_third": obs_serl["base_camera"],
                }
                policy.reset()  # single-step query; clear buffer each time
                a_base = policy.step(reflow_obs)[0]  # take first step of chunk → (7,)
            except Exception as e:
                print(f"  [warn] step {i}: ReFlow query failed: {e}. Skipping.")
                n_skipped += 1
                continue

            a_residual = (a_expert - a_base).astype(np.float32)
        else:
            a_residual = a_expert.copy()

        residual_trajs.append(dict(
            observations=obs_serl,
            next_observations=nobs_serl,
            actions=a_residual,
            rewards=np.float32(traj.get("rewards", 0.0)),
            masks=np.float32(traj.get("masks", 1.0 - float(traj.get("dones", 0)))),
            dones=bool(traj.get("dones", False)),
        ))

        if (i + 1) % 100 == 0 or i == len(expert_trajs) - 1:
            print(f"  {i+1}/{len(expert_trajs)} transitions processed  skipped={n_skipped}")

    # ── Stats ──────────────────────────────────────────────────────────────────
    print(f"\n{len(residual_trajs)} residual transitions  ({n_skipped} skipped)")
    if residual_trajs:
        acts = np.stack([t["actions"] for t in residual_trajs])
        print(f"  residual action  mean={acts.mean(0).round(4)}")
        print(f"  residual action  std ={acts.std(0).round(4)}")
        print(f"  residual action  |max|={np.abs(acts).max(0).round(4)}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(residual_trajs, f)
    print(f"\nSaved → {out}")
    print("\nUsage in async_drq_sim.py:")
    print(f"  --demo_path {out} --demo_batch_size 32")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo_path", required=True, help="Input expert demo .pkl")
    p.add_argument("--output_path", required=True, help="Output residual demo .pkl")
    p.add_argument("--reflow_ckpt", default=None, help="ReFlow checkpoint .pt")
    p.add_argument("--skip_base_query", action="store_true",
                   help="Skip ReFlow query; residual = expert action")
    # ReFlow arch params (must match training)
    p.add_argument("--reflow_obs_horizon",      type=int, default=2)
    p.add_argument("--reflow_pred_horizon",     type=int, default=4)
    p.add_argument("--reflow_act_horizon",      type=int, default=4)
    p.add_argument("--reflow_hidden_dim",       type=int, default=512)
    p.add_argument("--reflow_n_layers",         type=int, default=4)
    p.add_argument("--reflow_time_emb_dim",     type=int, default=64)
    p.add_argument("--reflow_denoising_steps",  type=int, default=10)
    p.add_argument("--reflow_img_size",         type=int, default=128)
    # Env (for metadata only — not used for stepping)
    p.add_argument("--env", default="PushCube-v1")
    return p.parse_args()


if __name__ == "__main__":
    make_residual_demos(parse_args())
