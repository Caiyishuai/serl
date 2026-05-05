#!/usr/bin/env python3
"""
ManiFlow Policy Training on LeRobot dataset (same data as DP)

Architecture: TimmObsEncoder (ResNet18) + DiTX + Consistency Flow Matching
vs DP:        PlainConv(6ch)           + ConditionalUnet1D + DDPM

Key differences vs DP (3_train_dp_lerobot.py):
  - Visual encoder : ResNet18 per-camera (separate weights) instead of 6ch PlainConv
  - Backbone       : DiTX (Diffusion Transformer) instead of ConditionalUnet1D
  - Scheduler      : Consistency Flow Matching (10 steps) instead of DDPM (100 steps)
  - Loss           : 75% flow + 25% consistency  instead of pure MSE noise prediction
  - Normalizer     : LinearNormalizer on state/action (maps to [-1,1])

Usage:
    python maniflow_train/1_train_maniflow.py \
        --lerobot_path dp_data_collect/lerobot \
        --env_id PushCube-v1 \
        --action_key action.target_delta_pose \
        --state_key observation.state_eef \
        --total_iters 200000 \
        --sim_backend physx_cuda
"""

import os, sys, time, io, random, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq
import tyro
import wandb
from omegaconf import OmegaConf

# ── Add ManiFlow to python path ───────────────────────────────────────────────
_MANIFLOW_ROOT = Path(__file__).resolve().parent.parent / \
    "third_party/ManiFlow_Policy/ManiFlow"
if str(_MANIFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFLOW_ROOT))

try:
    from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
    from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
    from maniflow.model.common.normalizer import LinearNormalizer, get_image_range_normalizer
    from maniflow.model.diffusion.ema_model import EMAModel
except ImportError as e:
    print(f"[ERROR] Cannot import ManiFlow: {e}")
    print(f"  Make sure ManiFlow is at: {_MANIFLOW_ROOT}")
    sys.exit(1)

import mani_skill.envs  # noqa: F401
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper


# ═════════════════════════════════════════════════════════════════════════════
# Args
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Args:
    # ── Identity ───────────────────────────────────────────────────────────
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True
    track: bool = True
    wandb_project: str = "ManiSkill-ManiFlow"
    wandb_entity: Optional[str] = None

    # ── Data ───────────────────────────────────────────────────────────────
    lerobot_path: str = "dp_data_collect/lerobot"
    """Root of the lerobot dataset (contains <env_id>/ sub-directory)."""
    env_id: str = "PushCube-v1"
    action_key: str = "action.target_delta_pose"
    """Which action field to train on (same choices as DP script)."""
    state_key: str = "observation.state_eef"
    """observation.state_eef (8D) or observation.state_joint (9D)."""
    num_demos: Optional[int] = None

    # ── Training ───────────────────────────────────────────────────────────
    total_iters: int = 200_000
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-3
    lr_warmup_steps: int = 500
    num_workers: int = 0

    # ── ManiFlow / DiTX hyper-parameters ───────────────────────────────────
    obs_horizon: int = 2        # n_obs_steps
    act_horizon: int = 8        # n_action_steps (executed at each call)
    pred_horizon: int = 16      # horizon (total predicted sequence)

    # DiTX architecture
    n_layer: int = 4
    n_head: int = 4
    n_emb: int = 256
    diffusion_step_embed_dim: int = 128
    diffusion_target_t_embed_dim: int = 128

    # Vision encoder
    backbone: str = "resnet18"
    """timm model name for visual encoder. resnet18/resnet34/efficientnet_b0 etc."""
    share_rgb_model: bool = False
    """True: same backbone weights for wrist+third cameras."""

    # Flow / Consistency
    num_inference_steps: int = 10
    flow_batch_ratio: float = 0.75
    consistency_batch_ratio: float = 0.25
    denoise_timesteps: int = 10

    # ── Evaluation ─────────────────────────────────────────────────────────
    max_episode_steps: int = 100
    eval_freq: int = 5000
    log_freq: int = 500
    save_freq: int = 10000
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cuda"
    control_mode: str = "pd_ee_delta_pose"


# ═════════════════════════════════════════════════════════════════════════════
# Dataset  (reads same lerobot parquet produced by dp_data_collect/2_zarr_to_lerobot.py)
# ═════════════════════════════════════════════════════════════════════════════
def _decode_png(img_dict) -> np.ndarray:
    """{'bytes': b'...PNG...', 'path': None} → (H, W, 3) uint8"""
    return np.array(Image.open(io.BytesIO(img_dict["bytes"])), dtype=np.uint8)


class LeRobotManiFlowDataset(Dataset):
    """
    Load lerobot v0.4.x parquet dataset for ManiFlow training.

    Returns per-sample dict:
        obs:
            rgb_wrist  : (obs_h, 3, H, W)  float32 [0,1]
            rgb_third  : (obs_h, 3, H, W)  float32 [0,1]
            agent_pos  : (obs_h, state_dim) float32
        action     : (pred_h, act_dim)  float32
    """
    def __init__(self,
                 lerobot_path: str,
                 env_id: str,
                 action_key: str,
                 state_key: str,
                 obs_horizon: int,
                 pred_horizon: int,
                 device=None,
                 num_demos: Optional[int] = None):
        super().__init__()
        self.obs_horizon  = obs_horizon
        self.pred_horizon = pred_horizon
        self.device       = device

        data_dir = Path(lerobot_path) / env_id / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        assert parquet_files, f"No parquet files found in {data_dir}"

        # Map action key → parquet column name
        action_col_map = {
            "action.target_delta_pose":  "action.target_delta_pose",
            "action.target_pose":        "action.target_pose",
            "action.target_delta_joint": "action.target_delta_joint",
            "action.target_joint":       "action.target_joint",
            # fallback: plain "action" column
            "action":                    "action",
        }
        action_col = action_col_map.get(action_key, action_key)
        state_col  = state_key  # "observation.state_eef" or "observation.state_joint"
        is_delta   = "delta" in action_key

        print(f"Loading episodes …")
        self.trajectories: List[dict] = []
        for pf in tqdm(parquet_files, desc="episodes"):
            table = pq.read_table(pf)
            df    = table.to_pandas()

            # group by episode_index
            for ep_idx, ep_df in df.groupby("episode_index"):
                ep_df = ep_df.sort_values("frame_index")

                # ── action ──────────────────────────────────────────
                if action_col in ep_df.columns:
                    acts = np.stack(ep_df[action_col].tolist()).astype(np.float32)
                elif "action" in ep_df.columns:
                    acts = np.stack(ep_df["action"].tolist()).astype(np.float32)
                else:
                    raise KeyError(f"action column '{action_col}' not found. "
                                   f"Available: {list(ep_df.columns)}")
                if acts.ndim > 2:
                    acts = acts.reshape(len(acts), -1)

                # ── state ───────────────────────────────────────────
                if state_col in ep_df.columns:
                    states = np.stack(ep_df[state_col].tolist()).astype(np.float32)
                elif "observation.state" in ep_df.columns:
                    states = np.stack(ep_df["observation.state"].tolist()).astype(np.float32)
                else:
                    raise KeyError(f"state column '{state_col}' not found.")

                # ── images ──────────────────────────────────────────
                wrist_col = "observation.images.hand_camera"
                third_col = "observation.images.base_camera"
                if wrist_col not in ep_df.columns:
                    wrist_col = next((c for c in ep_df.columns
                                      if "wrist" in c or "hand" in c), None)
                if third_col not in ep_df.columns:
                    third_col = next((c for c in ep_df.columns
                                      if "base" in c or "third" in c), None)
                assert wrist_col and third_col, \
                    f"Cannot find camera columns. Available: {list(ep_df.columns)}"

                wrist_imgs = [_decode_png(v) for v in ep_df[wrist_col]]  # list of (H,W,3)
                third_imgs = [_decode_png(v) for v in ep_df[third_col]]

                H, W = wrist_imgs[0].shape[:2]
                wrist_arr = np.stack(wrist_imgs).astype(np.float32) / 255.0  # (T,H,W,3)
                third_arr = np.stack(third_imgs).astype(np.float32) / 255.0  # (T,H,W,3)
                # → (T, 3, H, W)
                wrist_arr = wrist_arr.transpose(0, 3, 1, 2)
                third_arr = third_arr.transpose(0, 3, 1, 2)

                self.trajectories.append({
                    "action":     torch.from_numpy(acts),
                    "state":      torch.from_numpy(states),
                    "rgb_wrist":  torch.from_numpy(wrist_arr),
                    "rgb_third":  torch.from_numpy(third_arr),
                })

                if num_demos is not None and len(self.trajectories) >= num_demos:
                    break
            if num_demos is not None and len(self.trajectories) >= num_demos:
                break

        self.act_dim   = self.trajectories[0]["action"].shape[1]
        self.state_dim = self.trajectories[0]["state"].shape[1]
        self.img_shape = (self.trajectories[0]["rgb_wrist"].shape[1],   # C=3
                          self.trajectories[0]["rgb_wrist"].shape[2],   # H
                          self.trajectories[0]["rgb_wrist"].shape[3])   # W
        self.is_delta  = is_delta

        if self.is_delta:
            self.pad_action_arm = torch.zeros(self.act_dim - 1)

        # ── pre-compute (traj, start, end) slices ────────────────────────
        pad_before = obs_horizon - 1
        pad_after  = pred_horizon - obs_horizon
        total_trans = 0
        self.slices: List[tuple] = []
        for i, traj in enumerate(self.trajectories):
            L = traj["action"].shape[0]
            total_trans += L
            for s in range(-pad_before, L - pred_horizon + pad_after):
                self.slices.append((i, s, s + pred_horizon))
        print(f"Total transitions: {total_trans} | Training slices: {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        ti, start, end = self.slices[idx]
        traj = self.trajectories[ti]
        L    = traj["action"].shape[0]

        # ── action seq ───────────────────────────────────────────────────
        act = traj["action"][max(0, start):end]
        if start < 0:
            act = torch.cat([act[0:1].expand(-start, -1), act], dim=0)
        if end > L:
            if self.is_delta:
                grip = act[-1, -1:]
                pad  = torch.cat([self.pad_action_arm, grip])
                act  = torch.cat([act, pad.unsqueeze(0).expand(end - L, -1)])
            else:
                act = torch.cat([act, act[-1:].expand(end - L, -1)])

        # ── obs seq (obs_horizon frames) ─────────────────────────────────
        def _slice(v):
            seq = v[max(0, start): start + self.obs_horizon]
            if start < 0:
                seq = torch.cat([seq[0:1].expand(-start, *[-1]*(seq.ndim-1)), seq])
            return seq

        state = _slice(traj["state"])       # (obs_h, state_dim)
        wrist = _slice(traj["rgb_wrist"])   # (obs_h, 3, H, W)
        third = _slice(traj["rgb_third"])   # (obs_h, 3, H, W)

        return {
            "obs": {
                "rgb_wrist": wrist,
                "rgb_third": third,
                "agent_pos": state,
            },
            "action": act,
        }


# ═════════════════════════════════════════════════════════════════════════════
# IterationBasedBatchSampler  (same as DP script)
# ═════════════════════════════════════════════════════════════════════════════
class IterationBasedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, num_iterations):
        self.batch_sampler   = batch_sampler
        self.num_iterations  = num_iterations

    def __iter__(self):
        it = 0
        while it < self.num_iterations:
            for batch in self.batch_sampler:
                if it >= self.num_iterations:
                    return
                yield batch
                it += 1

    def __len__(self):
        return self.num_iterations


def worker_init_fn(wid, base_seed=0):
    np.random.seed(base_seed + wid)
    random.seed(base_seed + wid)


# ═════════════════════════════════════════════════════════════════════════════
# Observation extraction from live GPU eval env
# ═════════════════════════════════════════════════════════════════════════════
def _quat2aa_batch(quat: torch.Tensor) -> torch.Tensor:
    qw  = quat[:, 0].clamp(-1.0, 1.0)
    den = (1.0 - qw * qw).sqrt().clamp(min=1e-7)
    return quat[:, 1:4] * (2.0 * qw.acos() / den).unsqueeze(-1)


def extract_state(envs, state_key: str, device: torch.device) -> torch.Tensor:
    base = envs.unwrapped
    if state_key == "observation.state_eef":
        tcp = base.agent.tcp.pose.raw_pose
        if tcp.ndim == 3:
            tcp = tcp.squeeze(1)
        tcp    = tcp.float()
        pos    = tcp[:, :3]
        rot_aa = _quat2aa_batch(tcp[:, 3:7])
        qpos   = base.agent.robot.get_qpos()
        if qpos.ndim == 3:
            qpos = qpos.squeeze(1)
        gripper = qpos[:, -2:].float()
        return torch.cat([pos, rot_aa, gripper], dim=-1).to(device)
    else:
        qpos = base.agent.robot.get_qpos()
        if qpos.ndim == 3:
            qpos = qpos.squeeze(1)
        return qpos.float().to(device)


def extract_rgb_dict(obs: dict, device: torch.device):
    """Returns {'rgb_wrist': (B,3,H,W), 'rgb_third': (B,3,H,W)} float32 [0,1]"""
    wrist = obs["sensor_data"]["hand_camera"]["rgb"]   # (B,H,W,3) uint8
    third = obs["sensor_data"]["base_camera"]["rgb"]   # (B,H,W,3) uint8
    if isinstance(wrist, np.ndarray):
        wrist = torch.from_numpy(wrist)
        third = torch.from_numpy(third)
    wrist = wrist.permute(0, 3, 1, 2).float().to(device) / 255.0  # (B,3,H,W)
    third = third.permute(0, 3, 1, 2).float().to(device) / 255.0
    return {"rgb_wrist": wrist, "rgb_third": third}


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_maniflow(n_episodes, policy, ema_policy, envs,
                      device, state_key, obs_horizon):
    ema_policy.eval()
    B    = envs.unwrapped.num_envs
    metrics = defaultdict(list)
    done_count = 0
    pbar = tqdm(total=n_episodes, desc="evaluating")

    def _build_obs(sbuf, rbuf_w, rbuf_t):
        return {
            "rgb_wrist": torch.stack(list(rbuf_w), dim=1),  # (B, T, 3, H, W)
            "rgb_third": torch.stack(list(rbuf_t), dim=1),
            "agent_pos": torch.stack(list(sbuf), dim=1),    # (B, T, state_dim)
        }

    while done_count < n_episodes:
        obs, _ = envs.reset()
        s0 = extract_state(envs, state_key, device)
        r0 = extract_rgb_dict(obs, device)

        state_buf = deque([s0] * obs_horizon, maxlen=obs_horizon)
        wrist_buf = deque([r0["rgb_wrist"]] * obs_horizon, maxlen=obs_horizon)
        third_buf = deque([r0["rgb_third"]] * obs_horizon, maxlen=obs_horizon)

        success_once   = torch.zeros(B, dtype=torch.bool, device=device)
        success_at_end = torch.zeros(B, dtype=torch.bool, device=device)
        trunc_all = False

        while not trunc_all:
            obs_dict = _build_obs(state_buf, wrist_buf, third_buf)
            result   = ema_policy.predict_action(obs_dict)
            action_seq = result["action"]  # (B, act_horizon, act_dim)

            for i in range(action_seq.shape[1]):
                obs, _, _, truncated, info = envs.step(action_seq[:, i])
                s = extract_state(envs, state_key, device)
                r = extract_rgb_dict(obs, device)
                state_buf.append(s)
                wrist_buf.append(r["rgb_wrist"])
                third_buf.append(r["rgb_third"])

                suc = info["success"].bool()
                success_once  |= suc
                success_at_end = suc
                if truncated.any():
                    trunc_all = True
                    break

        metrics["success_once"].append(success_once.float().cpu().numpy())
        metrics["success_at_end"].append(success_at_end.float().cpu().numpy())
        done_count += B
        pbar.update(B)

    pbar.close()
    ema_policy.train()
    return {k: np.concatenate(v) for k, v in metrics.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ═════════════════════════════════════════════════════════════════════════════
def save_ckpt(run_name, tag, policy, ema):
    ckpt_dir = Path(f"runs/{run_name}/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save EMA model weights (averaged_model IS ema_policy)
    torch.save({"policy": ema.averaged_model.state_dict()}, ckpt_dir / f"{tag}.pt")
    print(f"  Saved checkpoint: {ckpt_dir / tag}.pt")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = tyro.cli(Args)
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon

    if args.exp_name is None:
        args.exp_name = Path(__file__).stem
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Run    : {run_name}")
    print(f"Action : {args.action_key}")
    print(f"State  : {args.state_key}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = LeRobotManiFlowDataset(
        lerobot_path=args.lerobot_path,
        env_id=args.env_id,
        action_key=args.action_key,
        state_key=args.state_key,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        device=device,
        num_demos=args.num_demos,
    )
    act_dim   = dataset.act_dim
    state_dim = dataset.state_dim
    img_shape = dataset.img_shape   # (C=3, H, W)
    print(f"state_dim={state_dim}  act_dim={act_dim}  img={img_shape}")

    sampler        = RandomSampler(dataset, replacement=False)
    batch_samp     = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    iter_samp      = IterationBasedBatchSampler(batch_samp, args.total_iters)
    train_loader   = DataLoader(dataset, batch_sampler=iter_samp,
                                num_workers=args.num_workers,
                                worker_init_fn=lambda w: worker_init_fn(w, args.seed))

    # ── Normalizer: fit on training data ─────────────────────────────────
    print("Fitting normalizer …")
    all_states  = torch.cat([t["state"]  for t in dataset.trajectories], dim=0).numpy()
    all_actions = torch.cat([t["action"] for t in dataset.trajectories], dim=0).numpy()

    normalizer = LinearNormalizer()
    normalizer.fit({
        "agent_pos": all_states,
        "action":    all_actions,
    }, last_n_dims=1, mode="limits")
    # RGB: identity normalizer (obs_encoder handles /255 and imagenet norm internally)
    normalizer["rgb_wrist"] = get_image_range_normalizer()
    normalizer["rgb_third"] = get_image_range_normalizer()

    # ── shape_meta for TimmObsEncoder ─────────────────────────────────────
    # 'horizon' per obs key is required by TimmObsEncoder.output_shape()
    shape_meta = {
        "obs": {
            "rgb_wrist": {"shape": img_shape,    "type": "rgb",     "horizon": args.obs_horizon},
            "rgb_third": {"shape": img_shape,    "type": "rgb",     "horizon": args.obs_horizon},
            "agent_pos": {"shape": (state_dim,), "type": "low_dim", "horizon": args.obs_horizon},
        },
        "action": {"shape": (act_dim,)},
    }

    # ── Build TimmObsEncoder ───────────────────────────────────────────────
    print(f"Building obs_encoder: {args.backbone} …")
    obs_encoder = TimmObsEncoder(
        shape_meta=shape_meta,
        model_name=args.backbone,
        pretrained=False,
        frozen=False,
        global_pool="",
        feature_aggregation="avg",   # global avg pool → (B*T, feature_dim) 2D
        downsample_ratio=32,
        position_encording="sinusoidal",
        transforms=None,             # no augmentation during offline IL
        use_group_norm=True,         # BatchNorm → GroupNorm (more stable)
        share_rgb_model=args.share_rgb_model,
        imagenet_norm=True,
    )

    # ── compute obs_feature_dim from a dry run ────────────────────────────
    with torch.no_grad():
        _ex = {
            "rgb_wrist": torch.zeros(1, args.obs_horizon, *img_shape),
            "rgb_third": torch.zeros(1, args.obs_horizon, *img_shape),
            "agent_pos": torch.zeros(1, args.obs_horizon, state_dim),
        }
        obs_feature_dim = obs_encoder(_ex).shape[-1]
    visual_cond_len = 1        # single global token per obs step sequence
    print(f"obs_feature_dim={obs_feature_dim}  visual_cond_len={visual_cond_len}")

    # ── Build ManiFlowTransformerImagePolicy ──────────────────────────────
    # ManiFlow was designed with Hydra/OmegaConf: it expects DictConfig (not
    # plain dict) so that dict_apply doesn't recurse into per-key metadata.
    shape_meta_cfg = OmegaConf.create(shape_meta)
    print("Building ManiFlow policy (DiTX backbone) …")
    policy = ManiFlowTransformerImagePolicy(
        shape_meta=shape_meta_cfg,
        horizon=args.pred_horizon,
        n_action_steps=args.act_horizon,
        n_obs_steps=args.obs_horizon,
        num_inference_steps=args.num_inference_steps,
        obs_as_global_cond=True,
        diffusion_timestep_embed_dim=args.diffusion_step_embed_dim,
        diffusion_target_t_embed_dim=args.diffusion_target_t_embed_dim,
        visual_cond_len=visual_cond_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_emb=args.n_emb,
        qkv_bias=False,
        qk_norm=False,
        block_type="DiTX",
        obs_encoder=obs_encoder,
        language_conditioned=False,
        flow_batch_ratio=args.flow_batch_ratio,
        consistency_batch_ratio=args.consistency_batch_ratio,
        denoise_timesteps=args.denoise_timesteps,
        sample_t_mode_flow="beta",
        sample_t_mode_consistency="discrete",
        sample_dt_mode_consistency="uniform",
        sample_target_t_mode="relative",
    ).to(device)

    policy.set_normalizer(normalizer)

    # ── EMA ───────────────────────────────────────────────────────────────
    # EMAModel takes the shadow copy directly as averaged_model;
    # policy remains the trainable model; ema.step(policy) syncs weights.
    import copy as _copy
    ema_policy = _copy.deepcopy(policy)
    ema_policy.set_normalizer(normalizer)
    ema = EMAModel(model=ema_policy, update_after_step=0,
                   inv_gamma=1.0, power=0.75,
                   min_value=0.0, max_value=0.9999)
    # Restore training state on policy (EMAModel.__init__ calls eval/no_grad on its model)
    policy.train()
    policy.requires_grad_(True)

    # ── Optimizer & LR scheduler ──────────────────────────────────────────
    optimizer = policy.get_optimizer(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters,
                                  eta_min=args.lr * 0.01)

    # ── Eval environment ──────────────────────────────────────────────────
    print("\nCreating eval environment …")
    env_kwargs = dict(
        obs_mode="state+rgb",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        robot_uids="panda_wristcam",
        control_mode=args.control_mode,
        reward_mode="sparse",
        max_episode_steps=args.max_episode_steps,
    )
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # ── wandb ─────────────────────────────────────────────────────────────
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    # ── Training loop ─────────────────────────────────────────────────────
    best = {"success_once": -1.0, "success_at_end": -1.0}

    def evaluate_and_save_best(iteration):
        metrics = evaluate_maniflow(
            n_episodes=args.num_eval_episodes,
            policy=policy,
            ema_policy=ema_policy,
            envs=eval_envs,
            device=device,
            state_key=args.state_key,
            obs_horizon=args.obs_horizon,
        )
        log = {}
        for k, v in metrics.items():
            v_mean = float(v.mean())
            log[f"eval/{k}"] = v_mean
            if v_mean > best[k]:
                best[k] = v_mean
                save_ckpt(run_name, f"best_{k}", policy, ema)
                print(f"  New best {k}: {v_mean:.4f}")
        log["iteration"] = iteration
        if args.track:
            wandb.log(log, step=iteration)
        print(f"[eval iter={iteration}] " +
              " | ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "iteration"))

    print(f"\n{'='*70}")
    print(f"Training  total_iters={args.total_iters}  batch={args.batch_size}")
    print(f"  flow_ratio={args.flow_batch_ratio}  consistency_ratio={args.consistency_batch_ratio}")
    print(f"  inference_steps={args.num_inference_steps}  (vs DDPM 100)")
    print(f"{'='*70}\n")

    # warm-up lr
    def _get_warmup_lr(step):
        if step < args.lr_warmup_steps:
            return args.lr * step / max(1, args.lr_warmup_steps)
        return None

    policy.train()
    loss_buf = defaultdict(list)

    for iteration, batch in enumerate(
            tqdm(train_loader, total=args.total_iters, desc="Training"), start=1):

        # warmup
        wlr = _get_warmup_lr(iteration)
        if wlr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = wlr

        # move batch to device
        obs = {
            k: v.to(device) for k, v in batch["obs"].items()
        }
        actions = batch["action"].to(device)

        # ManiFlow forward: needs EMA model for consistency target
        loss, loss_dict = policy.compute_loss(
            {"obs": obs, "action": actions},
            ema_model=ema_policy,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if iteration >= args.lr_warmup_steps:
            scheduler.step()

        # update EMA: ema.averaged_model (= ema_policy) gets soft-updated from policy
        ema.step(policy)

        # log
        for k, v in loss_dict.items():
            loss_buf[k].append(float(v))

        if iteration % args.log_freq == 0:
            log = {f"train/{k}": np.mean(vs) for k, vs in loss_buf.items()}
            log["train/lr"]        = optimizer.param_groups[0]["lr"]
            log["iteration"]       = iteration
            tqdm.write(f"iter={iteration:6d}  loss={log['train/bc_loss']:.4f}"
                       f"  flow={log['train/loss_flow']:.4f}"
                       f"  ct={log['train/loss_ct']:.4f}"
                       f"  lr={log['train/lr']:.2e}")
            if args.track:
                wandb.log(log, step=iteration)
            loss_buf.clear()

        if iteration % args.eval_freq == 0:
            evaluate_and_save_best(iteration)
            policy.train()

        if args.save_freq and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), policy, ema)

    # final eval
    evaluate_and_save_best(args.total_iters)

    print(f"\nDone. Checkpoints saved to runs/{run_name}/checkpoints/")
    eval_envs.close()
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
