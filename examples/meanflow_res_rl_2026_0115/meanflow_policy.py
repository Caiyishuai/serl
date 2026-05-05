"""
meanflow_policy.py — Local PyTorch ManiFlow + TimmObsEncoder policy.

Acts as a drop-in for openpi's RemotePolicy: exposes a callable `.step(obs)`
that receives a SERL-style obs dict and returns an action chunk (act_horizon, 7).

The rolling obs-horizon buffer is maintained internally.

Usage:
    policy = ManiFlowPolicy(
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
from omegaconf import OmegaConf

# ── Resolve paths to ManiFlow ─────────────────────────────────────────────────
_SERL_ROOT    = Path(__file__).resolve().parents[2]          # …/serl/
_MANIFLOW_ROOT = _SERL_ROOT / "third_party/ManiFlow_Policy/ManiFlow"

if str(_MANIFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MANIFLOW_ROOT))

try:
    from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
    from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
    from maniflow.model.common.normalizer import LinearNormalizer, get_image_range_normalizer
except ImportError as e:
    print(f"[ERROR] Cannot import ManiFlow: {e}")
    print(f"  Make sure ManiFlow is at: {_MANIFLOW_ROOT}")
    sys.exit(1)


class ManiFlowPolicy:
    """
    Local ManiFlow policy. Wraps ManiFlowTransformerImagePolicy + TimmObsEncoder
    loaded from a checkpoint saved by maniflow_train/1_train_maniflow.py.

    Parameters
    ----------
    ckpt_path     : path to .pt checkpoint (contains policy state_dict)
    device        : torch device string
    obs_horizon   : observation window length (must match training)
    pred_horizon  : prediction horizon (must match training)
    act_horizon   : how many actions to return per chunk
    num_inference_steps : number of denoising steps (must match training)
    backbone      : timm model name for visual encoder (e.g., "resnet18")
    n_layer, n_head, n_emb, diffusion_step_embed_dim, diffusion_target_t_embed_dim :
                    DiTX hyper-params (must match training)
    img_size      : camera image size in pixels
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        num_inference_steps: int = 10,
        backbone: str = "resnet18",
        n_layer: int = 4,
        n_head: int = 4,
        n_emb: int = 256,
        diffusion_step_embed_dim: int = 128,
        diffusion_target_t_embed_dim: int = 128,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        denoise_timesteps: int = 10,
        img_size: int = 128,
    ):
        self.device       = torch.device(device)
        self.obs_horizon  = obs_horizon
        self.act_horizon  = act_horizon
        self.pred_horizon = pred_horizon
        self.state_dim    = 8   # default: eef state (pos:3 + rot_aa:3 + gripper:2)
        self.act_dim      = 7   # action dim
        self.img_size     = img_size

        print(f"[ManiFlowPolicy] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # ── Build shape_meta (required by TimmObsEncoder) ──────────────────────
        shape_meta = {
            "obs": {
                "rgb_wrist": {"shape": (3, img_size, img_size), "type": "rgb",     "horizon": obs_horizon},
                "rgb_third": {"shape": (3, img_size, img_size), "type": "rgb",     "horizon": obs_horizon},
                "agent_pos": {"shape": (self.state_dim,),       "type": "low_dim", "horizon": obs_horizon},
            },
            "action": {"shape": (self.act_dim,)},
        }

        # ── Build TimmObsEncoder ───────────────────────────────────────────────
        print(f"[ManiFlowPolicy] Building TimmObsEncoder: {backbone} …")
        obs_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name=backbone,
            pretrained=False,
            frozen=False,
            global_pool="",
            feature_aggregation="avg",
            downsample_ratio=32,
            position_encording="sinusoidal",
            transforms=None,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )

        # ── Build ManiFlowTransformerImagePolicy ──────────────────────────────
        shape_meta_cfg = OmegaConf.create(shape_meta)
        print("[ManiFlowPolicy] Building ManiFlow policy (DiTX backbone) …")
        self.policy = ManiFlowTransformerImagePolicy(
            shape_meta=shape_meta_cfg,
            horizon=pred_horizon,
            n_action_steps=act_horizon,
            n_obs_steps=obs_horizon,
            num_inference_steps=num_inference_steps,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=diffusion_step_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            visual_cond_len=1,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            obs_encoder=obs_encoder,
            language_conditioned=False,
            flow_batch_ratio=flow_batch_ratio,
            consistency_batch_ratio=consistency_batch_ratio,
            denoise_timesteps=denoise_timesteps,
            sample_t_mode_flow="beta",
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform",
            sample_target_t_mode="relative",
        ).to(self.device)

        # ── Load policy weights ───────────────────────────────────────────────
        if "policy" in ckpt:
            self.policy.load_state_dict(ckpt["policy"])
        else:
            # Fallback: assume entire checkpoint is policy state_dict
            self.policy.load_state_dict(ckpt)
        self.policy.eval()
        print("[ManiFlowPolicy] Policy loaded successfully")

        # ── Setup normalizer ─────────────────────────────────────────────────
        normalizer = LinearNormalizer()
        normalizer["agent_pos"] = LinearNormalizer()
        normalizer["action"] = LinearNormalizer()
        normalizer["rgb_wrist"] = get_image_range_normalizer()
        normalizer["rgb_third"] = get_image_range_normalizer()
        self.policy.set_normalizer(normalizer)

        # ── Rolling obs buffer ───────────────────────────────────────────────
        self._state_buf: deque = deque(maxlen=obs_horizon)
        self._wrist_buf: deque = deque(maxlen=obs_horizon)
        self._third_buf: deque = deque(maxlen=obs_horizon)

    # ── Public API ───────────────────────────────────────────────────────────
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

        # ── Parse + convert obs ──────────────────────────────────────────────
        state = torch.from_numpy(
            np.asarray(obs["state"], dtype=np.float32)
        ).to(dev)  # (state_dim,)

        # ── Convert images: (H,W,3) uint8 → (3,H,W) float32 [0,1] ────────────
        def _img_to_chw(img: np.ndarray) -> torch.Tensor:
            arr = np.asarray(img)
            if arr.ndim == 4:      # (1,H,W,3) from ChunkingWrapper edge case
                arr = arr[0]
            return torch.from_numpy(arr).permute(2, 0, 1).float().to(dev) / 255.0

        wrist = _img_to_chw(obs["img_wrist"])   # (3,H,W)
        third = _img_to_chw(obs["img_third"])

        # ── Update rolling buffers ───────────────────────────────────────────
        self._state_buf.append(state)
        self._wrist_buf.append(wrist)
        self._third_buf.append(third)

        # Left-pad if buffer not yet full (first obs_horizon steps)
        while len(self._state_buf) < self.obs_horizon:
            self._state_buf.appendleft(state)
            self._wrist_buf.appendleft(wrist)
            self._third_buf.appendleft(third)

        # ── Build batched tensors (B=1) ──────────────────────────────────────
        wrist_seq = torch.stack(list(self._wrist_buf)).unsqueeze(0)   # (1,T,3,H,W)
        third_seq = torch.stack(list(self._third_buf)).unsqueeze(0)   # (1,T,3,H,W)
        state_seq = torch.stack(list(self._state_buf)).unsqueeze(0)   # (1,T,state_dim)

        # ── ManiFlow inference ───────────────────────────────────────────────
        obs_dict = {
            "rgb_wrist": wrist_seq,
            "rgb_third": third_seq,
            "agent_pos": state_seq,
        }
        result = self.policy.predict_action(obs_dict)
        action_seq = result["action"]  # (1, act_horizon, act_dim)
        actions = action_seq[0]  # (act_horizon, act_dim)

        return actions.cpu().numpy().astype(np.float32)  # (act_horizon, act_dim)
