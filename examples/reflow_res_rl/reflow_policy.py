"""
reflow_policy.py — Local PyTorch ReFlow + ResNet18 base policy.

Acts as a drop-in for openpi's RemotePolicy: exposes a callable `.step(obs)`
that receives a SERL-style obs dict and returns an action chunk (act_horizon, 7).

The rolling obs-horizon buffer is maintained internally.

Usage:
    policy = ReFlowPolicy(
        ckpt_path="runs/.../checkpoints/best_success_once.pt",
        device="cuda",
    )
    policy.reset()                  # call on every env.reset()
    action_chunk = policy.step(obs) # obs: {state:(8,), img_wrist:(H,W,3), img_third:(H,W,3)}
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── Resolve paths to reflow + diffusion_policy ────────────────────────────────
_SERL_ROOT    = Path(__file__).resolve().parents[2]          # …/serl/
_MS_WS        = _SERL_ROOT.parent / "maniskill-ws"
_REFLOW_DIR   = _MS_WS / "bc_reflow_train"
_RF_DIR       = _MS_WS / "bc_reflow_train" / "rectified flow"
_DP_DIR       = _MS_WS / "bc_diffusion_policy" / "diffusion_policy"

for _p in [str(_REFLOW_DIR), str(_RF_DIR), str(_DP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reflow import ReFlowMLP                                          # noqa: E402
from model.flow.mlp_flow import FlowMLP                               # noqa: E402
from diffusion_policy.model.vision.multi_image_obs_encoder import (   # noqa: E402
    MultiImageObsEncoder,
)
from diffusion_policy.model.vision.model_getter import get_resnet     # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# VisualEncoder (same architecture as reflow_train/1_train_reflow.py)
# ─────────────────────────────────────────────────────────────────────────────
class VisualEncoder(nn.Module):
    """2×ResNet18 visual encoder matching the training script."""

    def __init__(self, img_size: int = 128):
        super().__init__()
        shape_meta = {
            "obs": {
                "rgb_wrist": {"shape": (3, img_size, img_size), "type": "rgb"},
                "rgb_third": {"shape": (3, img_size, img_size), "type": "rgb"},
            }
        }
        self.encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=get_resnet("resnet18"),
            crop_shape=None,       # eval: no random crop
            random_crop=False,
            use_group_norm=True,
            imagenet_norm=True,
        )
        self.feat_dim: int = self.encoder.output_shape()[0]  # 1024 (512×2)

    def forward(self, wrist: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W), (B,T,3,H,W) → (B,T,feat_dim)"""
        B, T = wrist.shape[:2]
        w = wrist.flatten(end_dim=1)
        t = third.flatten(end_dim=1)
        with torch.no_grad():
            feat = self.encoder({"rgb_wrist": w, "rgb_third": t})
        return feat.reshape(B, T, -1)


# ─────────────────────────────────────────────────────────────────────────────
# ReFlowPolicy
# ─────────────────────────────────────────────────────────────────────────────
class ReFlowPolicy:
    """
    Local ReFlow policy.  Wraps ReFlowMLP + VisualEncoder loaded from a
    checkpoint saved by reflow_train/1_train_reflow.py.

    Parameters
    ----------
    ckpt_path     : path to .pt checkpoint
    device        : torch device string
    obs_horizon   : observation window length (must match training)
    pred_horizon  : prediction horizon (must match training)
    act_horizon   : how many actions to return per chunk
    hidden_dim, n_layers, time_emb_dim, max_denoising_steps :
                    FlowMLP / ReFlowMLP hyper-params (must match training)
    img_size      : camera image size in pixels
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        obs_horizon: int = 2,
        pred_horizon: int = 4,
        act_horizon: int = 4,
        hidden_dim: int = 512,
        n_layers: int = 4,
        time_emb_dim: int = 64,
        max_denoising_steps: int = 10,
        img_size: int = 128,
    ):
        self.device       = torch.device(device)
        self.obs_horizon  = obs_horizon
        self.act_horizon  = act_horizon

        # ── Load checkpoint ───────────────────────────────────────────────────
        print(f"[ReFlowPolicy] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.state_min  = ckpt["state_min"].to(self.device)   # (state_dim,)
        self.state_max  = ckpt["state_max"].to(self.device)
        self.action_min = ckpt["action_min"].to(self.device)   # (act_dim,)
        self.action_max = ckpt["action_max"].to(self.device)

        state_dim = self.state_min.shape[0]
        act_dim   = self.action_min.shape[0]

        # ── Visual encoder ────────────────────────────────────────────────────
        if "visual_enc" in ckpt:
            self.visual_enc = VisualEncoder(img_size=img_size).to(self.device)
            self.visual_enc.load_state_dict(ckpt["visual_enc"])
            self.visual_enc.eval()
            aug_obs_dim = state_dim + self.visual_enc.feat_dim
            print(f"[ReFlowPolicy] VisualEncoder loaded  feat_dim={self.visual_enc.feat_dim}")
        else:
            self.visual_enc = None
            aug_obs_dim = state_dim
            print("[ReFlowPolicy] state-only mode (no visual_enc in checkpoint)")

        # ── ReFlowMLP ─────────────────────────────────────────────────────────
        network = FlowMLP(
            action_dim=act_dim,
            horizon_steps=pred_horizon,
            obs_dim=aug_obs_dim,
            cond_steps=obs_horizon,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_emb_dim=time_emb_dim,
        )
        self.model = ReFlowMLP(
            network=network,
            device=self.device,
            horizon_steps=pred_horizon,
            action_dim=act_dim,
            act_min=-1.0,
            act_max=1.0,
            obs_dim=aug_obs_dim,
            max_denoising_steps=max_denoising_steps,
            seed=0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[ReFlowPolicy] ReFlowMLP loaded  "
              f"state_dim={state_dim}  act_dim={act_dim}  obs_dim={aug_obs_dim}")

        # ── Rolling obs buffer ────────────────────────────────────────────────
        self._state_buf: deque = deque(maxlen=obs_horizon)
        self._wrist_buf: deque = deque(maxlen=obs_horizon)
        self._third_buf: deque = deque(maxlen=obs_horizon)

    # ── Normalization helpers ─────────────────────────────────────────────────
    def _norm_state(self, s: torch.Tensor) -> torch.Tensor:
        lo, hi = self.state_min, self.state_max
        return 2.0 * (s - lo) / (hi - lo + 1e-8) - 1.0

    def _denorm_action(self, a: torch.Tensor) -> torch.Tensor:
        lo, hi = self.action_min, self.action_max
        return (a + 1.0) / 2.0 * (hi - lo + 1e-8) + lo

    # ── Public API ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear rolling obs buffer — call on every env.reset()."""
        self._state_buf.clear()
        self._wrist_buf.clear()
        self._third_buf.clear()

    @torch.no_grad()
    def step(self, obs: dict) -> np.ndarray:
        """
        obs keys (after AddPolicyActionWrapper key remapping):
            state     : (8,) float32  — 8-dim eef state
            img_wrist : (H, W, 3) uint8  — wrist camera
            img_third : (H, W, 3) uint8  — third camera

        Returns (act_horizon, act_dim) float32 action chunk.
        The AddPolicyActionWrapper will buffer this and execute one step at a time.
        """
        dev = self.device

        # ── Parse + convert obs ───────────────────────────────────────────────
        state = torch.from_numpy(
            np.asarray(obs["state"], dtype=np.float32)
        ).to(dev)  # (state_dim,)

        # ── Update rolling buffer ─────────────────────────────────────────────
        self._state_buf.append(state)

        if self.visual_enc is not None:
            # Images: (H,W,3) uint8 → (3,H,W) float32 [0,1]
            def _img_to_chw(img: np.ndarray) -> torch.Tensor:
                arr = np.asarray(img)
                if arr.ndim == 4:      # (1,H,W,3) from ChunkingWrapper edge case
                    arr = arr[0]
                return torch.from_numpy(arr).permute(2, 0, 1).float().to(dev) / 255.0

            wrist = _img_to_chw(obs["img_wrist"])   # (3,H,W)
            third = _img_to_chw(obs["img_third"])
            self._wrist_buf.append(wrist)
            self._third_buf.append(third)

        # Left-pad if buffer not yet full (first obs_horizon steps)
        while len(self._state_buf) < self.obs_horizon:
            self._state_buf.appendleft(state)
            if self.visual_enc is not None:
                self._wrist_buf.appendleft(wrist)
                self._third_buf.appendleft(third)

        # ── Build batched tensors (B=1) ───────────────────────────────────────
        state_seq = torch.stack(list(self._state_buf)).unsqueeze(0)   # (1,T,D)

        # ── Visual features (or state-only) ───────────────────────────────────
        state_n = self._norm_state(state_seq)
        if self.visual_enc is not None:
            wrist_seq = torch.stack(list(self._wrist_buf)).unsqueeze(0)
            third_seq = torch.stack(list(self._third_buf)).unsqueeze(0)
            vis_feat  = self.visual_enc(wrist_seq, third_seq)          # (1,T,V)
            aug_state = torch.cat([state_n, vis_feat], dim=-1)
        else:
            aug_state = state_n

        # ── ReFlow inference ──────────────────────────────────────────────────
        sample  = self.model.forward({"state": aug_state})            # Sample
        acts_n  = sample.trajectories[:, : self.act_horizon]          # (1,H,A)
        actions = self._denorm_action(acts_n)[0]                      # (H,A)

        return actions.cpu().numpy().astype(np.float32)               # (act_horizon, act_dim)
