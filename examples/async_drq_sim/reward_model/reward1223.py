import os
import io
import time
import math
import json
import csv
import random
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
from collections import OrderedDict

import numpy as np
import pandas as pd
import imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
try:
    import pytorch_lightning as pl
    PL_AVAILABLE = True
except ImportError:
    PL_AVAILABLE = False
    # Define a dummy class to avoid NameError when subclassing
    class DummyLightningModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pass
    # Mock pl namespace
    class pl:
        LightningModule = DummyLightningModule
        class callbacks:
            ModelCheckpoint = object
        class Trainer:
            pass
        @staticmethod
        def seed_everything(seed):
            pass
from torchvision import transforms
from transformers import AutoModel, AutoConfig, get_cosine_schedule_with_warmup

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    from torchmetrics.functional.classification import binary_auroc
except ImportError:
    binary_auroc = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set precision for potential Tensor Core usage
torch.set_float32_matmul_precision('high')
BACKBONE_TYPE = "facebook/dinov3-vits16plus-pretrain-lvd1689m"

class GaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ----------------------------------------------------------------
# 1. Model Component: Multimodal Reward Head
# ----------------------------------------------------------------
class RewardHead(nn.Module):
    """
    A temporal reward head that processes a sequence of fused features.
    Uses a Transformer Encoder to aggregate temporal information and an MLP for the final score.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        # Transformer Encoder for temporal aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # MLP for reward prediction
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape [batch_size, seq_len, feature_dim]
        Returns:
            Reward score of shape [batch_size, 1]
        """
        # [B, T, D] -> [B, T, D]
        x = self.temporal_transformer(x)
        # Use the embedding of the last time step
        x_last = x[:, -1, :]
        # [B, D] -> [B, 1]
        return self.mlp(x_last)


# ----------------------------------------------------------------
# 2. Main Model: Ensemble Reward Model (Lightning Module)
# ----------------------------------------------------------------
class BadassRewardModel(pl.LightningModule):
    """
    Ensemble Reward Model combining visual (base/hand cameras) and proprioceptive state features.
    Uses a frozen or unfrozen Vision Transformer backbone (DINOv2/v3).
    """
    def __init__(
        self,
        ensemble_size: int = 3,
        window_size: int = 3,
        lr: float = 1e-4,
        freeze_backbone: bool = True,
        backbone_type: str = BACKBONE_TYPE,
        state_dim: int = 7,
        max_epochs: int = 20,
        compile_heads: bool = False,
        bounded_temperature: float = 2.0,
        aux_progress_weight: float = 1.0,
        range_push_weight: float = 0.0,
        range_push_threshold: float = 0.8,
        range_push_margin: float = 0.9,
        weight_decay: float = 1e-4,
        scheduler: str = 'cosine',
        dropout: float = 0.1,
        warmup_ratio: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        logger.info(f"Loading backbone: {backbone_type}...")
        self.backbone_config = AutoConfig.from_pretrained(backbone_type)
        self.backbone = AutoModel.from_pretrained(backbone_type)
        self.backbone_embed_dim = self.backbone_config.hidden_size

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Projection layers to map features to a common dimension
        self.base_proj = nn.Linear(self.backbone_embed_dim, 128)
        self.hand_proj = nn.Linear(self.backbone_embed_dim, 128)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.total_feature_dim = 128 + 128 + 64

        # Ensemble of reward heads
        self.heads = nn.ModuleList([
            RewardHead(feature_dim=self.total_feature_dim, dropout=dropout) for _ in range(ensemble_size)
        ])

        # Optional: Compile heads for speed (PyTorch 2.0+)
        self._try_compile_heads(compile_heads)

    def _try_compile_heads(self, compile_heads: bool):
        global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        if compile_heads and hasattr(torch, "compile") and global_rank == 0:
            try:
                self.base_proj = torch.compile(self.base_proj)
                self.hand_proj = torch.compile(self.hand_proj)
                self.state_proj = torch.compile(self.state_proj)
                self.heads = nn.ModuleList([torch.compile(h) for h in self.heads])
                logger.info("torch.compile enabled for projections and heads.")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

    def forward_features(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and fuse features from inputs."""
        base_imgs = batch_data['base_img']   # [B, W, 3, 224, 224]
        hand_imgs = batch_data['hand_img']   # [B, W, 3, 224, 224]
        states = batch_data['state']         # [B, W, state_dim]

        b, w = base_imgs.shape[:2]
        
        # Flatten time dimension for backbone processing
        flat_base = base_imgs.view(b * w, 3, 224, 224)
        flat_hand = hand_imgs.view(b * w, 3, 224, 224)

        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            base_out = self.backbone(flat_base)
            hand_out = self.backbone(flat_hand)
            # Use CLS token (index 0)
            base_feat = base_out.last_hidden_state[:, 0, :]
            hand_feat = hand_out.last_hidden_state[:, 0, :]

        # Project features
        base_emb = self.base_proj(base_feat)
        hand_emb = self.hand_proj(hand_feat)
        
        flat_state = states.view(b * w, -1)
        state_emb = self.state_proj(flat_state)

        # Concatenate: [B*W, D_base+D_hand+D_state]
        fused_feat = torch.cat([base_emb, hand_emb, state_emb], dim=-1)
        # Reshape back to [B, W, D_total]
        return fused_feat.view(b, w, -1)

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns:
            rewards: [ensemble_size, batch_size, 1]
        """
        features = self.forward_features(batch_data)
        # Stack outputs from all heads: [E, B, 1]
        rewards = torch.stack([head(features) for head in self.heads], dim=0)
        return rewards

    @torch.no_grad()
    def predict_reward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference helper: returns mean reward across ensemble."""
        was_training = self.training
        self.eval()
        try:
            raw_rewards = self(batch_data) # [E, B, 1]
            return raw_rewards.mean(dim=0)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_reward_bounded(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference helper: returns bounded reward (-1 to 1)."""
        raw = self.predict_reward(batch_data)
        return torch.tanh(raw / self.hparams.bounded_temperature)

    def training_step(self, batch, batch_idx):
        obs_early = batch['early']
        obs_late = batch['late']
        label = batch.get('label', None) # 1.0 if late > early else 0.0

        r_early_all = self(obs_early)  # [E, B, 1]
        r_late_all = self(obs_late)    # [E, B, 1]

        if label is None:
            # Default to ranking: late should be higher than early? 
            # If no label, assume late > early implicitly or handle data error.
            # Here we assume target=1.
            target = torch.ones_like(r_early_all[0])
        else:
            target = label.to(dtype=r_early_all.dtype).view(-1, 1)

        # Apply Label Smoothing
        if self.hparams.label_smoothing > 0.0:
            # target (0 or 1) -> (alpha or 1-alpha)
            # For BCE with logits, targets are probabilities.
            # smoothed = target * (1 - alpha) + 0.5 * alpha
            alpha = self.hparams.label_smoothing
            target = target * (1.0 - alpha) + 0.5 * alpha

        # 1. Bradley-Terry / Cross Entropy Loss for Ranking
        loss_ranking = 0.0
        for i in range(self.hparams.ensemble_size):
            diff = r_late_all[i] - r_early_all[i]
            loss_ranking += F.binary_cross_entropy_with_logits(diff, target)
        loss_ranking /= self.hparams.ensemble_size

        # 2. Auxiliary Progress Regression Loss
        loss_progress = None
        loss_mse = None
        loss_push = None
        if ("progress_target_early" in batch) and ("progress_target_late" in batch) and (self.hparams.aux_progress_weight > 0):
            loss_progress, loss_mse, loss_push = self._compute_progress_loss(
                r_early_all, r_late_all, 
                batch["progress_target_early"], batch["progress_target_late"]
            )
        
        # Total loss
        loss = loss_ranking
        if loss_progress is not None:
            loss += self.hparams.aux_progress_weight * loss_progress

        # Log individual loss components
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log('train/loss_ranking', loss_ranking, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        if loss_progress is not None:
            self.log('train/loss_progress', loss_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        if loss_mse is not None:
            self.log('train/loss_mse', loss_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        if loss_push is not None and loss_push.item() > 0:
            self.log('train/loss_range_push', loss_push, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        
        return loss

    def _compute_progress_loss(self, r_early_all, r_late_all, y_early, y_late):
        """Helper to compute auxiliary regression and range push losses.
        
        Returns:
            tuple: (loss_progress_total, loss_mse, loss_push)
                - loss_progress_total: total progress loss (MSE + range_push if enabled)
                - loss_mse: MSE loss component
                - loss_push: Range push loss component (0 if disabled)
        """
        y_early = y_early.to(dtype=r_early_all.dtype).view(-1, 1)
        y_late = y_late.to(dtype=r_early_all.dtype).view(-1, 1)

        raw_early = r_early_all.mean(0)
        raw_late = r_late_all.mean(0)
        bounded_early = torch.tanh(raw_early / self.hparams.bounded_temperature)
        bounded_late = torch.tanh(raw_late / self.hparams.bounded_temperature)

        # MSE Loss against progress target
        loss_mse = 0.5 * (F.mse_loss(bounded_early, y_early) + F.mse_loss(bounded_late, y_late))
        
        # Optional: Range Push Loss (force values to boundaries)
        loss_push = None
        if self.hparams.range_push_weight > 0:
            loss_push = self._compute_range_push_loss(bounded_early, bounded_late, y_early, y_late)
            loss_progress_total = loss_mse + self.hparams.range_push_weight * loss_push
        else:
            loss_push = torch.tensor(0.0, device=loss_mse.device, dtype=loss_mse.dtype)
            loss_progress_total = loss_mse

        # Logging Correlation
        with torch.no_grad():
            self._log_progress_correlation(y_early, y_late, bounded_early, bounded_late, prefix='train')
            
        return loss_progress_total, loss_mse, loss_push

    def _compute_range_push_loss(self, b_early, b_late, y_early, y_late):
        thr = float(self.hparams.range_push_threshold)
        margin = float(self.hparams.range_push_margin)
        
        y = torch.cat([y_early, y_late], dim=0)
        b = torch.cat([b_early, b_late], dim=0)

        mask_low = (y <= -thr)
        mask_high = (y >= thr)
        
        loss_low = F.relu(b[mask_low] - (-margin)).mean() if mask_low.any() else b.new_tensor(0.0)
        loss_high = F.relu((+margin) - b[mask_high]).mean() if mask_high.any() else b.new_tensor(0.0)
        return loss_low + loss_high

    def _log_progress_correlation(self, y_early, y_late, b_early, b_late, prefix):
        """
        Compute Pearson correlation between progress targets and bounded rewards.
        
        WARNING: This metric is unreliable with small validation sets (< 10 samples).
        With only 2-3 samples, correlation can easily be 1.0 even if the model is poor.
        Use val/loss or val/monotonicity for checkpoint selection instead.
        """
        y = torch.cat([y_early, y_late], dim=0).squeeze(-1)
        b = torch.cat([b_early, b_late], dim=0).squeeze(-1)
        # Avoid NaN if variance is 0
        if y.numel() < 2: 
            return
        
        # Need at least 3 samples for meaningful correlation
        if y.numel() < 3:
            logger.warning(f"[{prefix}/progress_corr] Only {y.numel()} samples, correlation may be unreliable")
            
        y0 = y - y.mean()
        b0 = b - b.mean()
        denom = (y0.square().mean().sqrt() * b0.square().mean().sqrt()).clamp_min(1e-8)
        corr = (y0 * b0).mean() / denom
        self.log(f'{prefix}/progress_corr', corr, on_step=(prefix=='train'), on_epoch=True, prog_bar=False, sync_dist=(prefix!='train'))

    def validation_step(self, batch, batch_idx):
        obs_early = batch['early']
        obs_late = batch['late']
        label = batch.get('label', None)
        
        r_early_all = self(obs_early)
        r_late_all = self(obs_late)
        r_early = r_early_all.mean(dim=0)
        r_late = r_late_all.mean(dim=0)

        if label is None:
            target = torch.ones_like(r_early)
        else:
            target = label.to(dtype=r_early.dtype).view(-1, 1)

        diff = (r_late - r_early)
        loss = F.binary_cross_entropy_with_logits(diff, target)

        pred_bin = (diff > 0).float()
        acc = (pred_bin == (target > 0.5).float()).float().mean()

        log_data = {
            'val/loss': loss,
        }
        
        # Optional Metrics
        if batch.get('label_time') is not None:
            target_t = batch['label_time'].to(dtype=r_early.dtype).view(-1, 1)
            log_data['val/acc_time'] = (pred_bin == (target_t > 0.5).float()).float().mean()
            
        if batch.get('label_proxy') is not None:
            target_p = batch['label_proxy'].to(dtype=r_early.dtype).view(-1, 1)
            log_data['val/acc_proxy'] = (pred_bin == (target_p > 0.5).float()).float().mean()

        self.log_dict(log_data, prog_bar=True, sync_dist=False)

        # Progress Correlation Logging for Validation
        if ("progress_target_early" in batch) and ("progress_target_late" in batch):
            y_early = batch["progress_target_early"].to(dtype=r_early.dtype).view(-1, 1)
            y_late = batch["progress_target_late"].to(dtype=r_early.dtype).view(-1, 1)
            bounded_early = torch.tanh(r_early / self.hparams.bounded_temperature)
            bounded_late = torch.tanh(r_late / self.hparams.bounded_temperature)
            
            # Re-use the logging helper
            self._log_progress_correlation(y_early, y_late, bounded_early, bounded_late, prefix='val')

        return acc

    def configure_optimizers(self):
        # Different learning rates for backbone vs heads
        params = [
            {'params': self.heads.parameters(), 'lr': self.hparams.lr},
            {'params': self.base_proj.parameters(), 'lr': self.hparams.lr},
            {'params': self.hand_proj.parameters(), 'lr': self.hparams.lr},
            {'params': self.state_proj.parameters(), 'lr': self.hparams.lr},
        ]
        if not self.hparams.freeze_backbone:
            params.append({'params': self.backbone.parameters(), 'lr': self.hparams.lr * 0.1})

        weight_decay = getattr(self.hparams, 'weight_decay', 1e-4)
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=weight_decay)
        
        scheduler_type = getattr(self.hparams, 'scheduler', 'cosine')
        
        if scheduler_type == 'plateau':
            # ReduceLROnPlateau: 当 val loss 不下降时降低学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # CosineAnnealingLR with optional Warmup
            # Calculate total steps for warmup
            total_steps = self.hparams.max_epochs * len(self.trainer.datamodule.train_dataloader()) \
                          if self.trainer and self.trainer.datamodule else self.hparams.max_epochs * 100
            
            warmup_steps = int(total_steps * getattr(self.hparams, 'warmup_ratio', 0.0))
            
            if warmup_steps > 0:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=warmup_steps, 
                    num_training_steps=total_steps
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step", # Warmup schedulers update every step
                        "frequency": 1,
                    },
                }
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.max_epochs,
                    eta_min=1e-6,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }


# ----------------------------------------------------------------
# 3. Model for TorchScript Export (No Lightning dependency)
# ----------------------------------------------------------------
class PureBadassRewardModel(nn.Module):
    """
    A pure nn.Module version of BadassRewardModel for JIT export.
    Mirrors the structure exactly to load state_dict directly.
    """
    def __init__(
        self,
        ensemble_size: int = 3,
        window_size: int = 5,
        freeze_backbone: bool = True,
        backbone_type: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        state_dim: int = 7,
        bounded_temperature: float = 2.0,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.window_size = window_size
        self.freeze_backbone = freeze_backbone
        self.backbone_type = backbone_type
        self.state_dim = state_dim
        self.bounded_temperature = bounded_temperature

        logger.info(f"[export] Loading backbone: {backbone_type}...")
        self.backbone_config = AutoConfig.from_pretrained(backbone_type)
        self.backbone = AutoModel.from_pretrained(backbone_type)
        self.backbone_embed_dim = self.backbone_config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.base_proj = nn.Linear(self.backbone_embed_dim, 128)
        self.hand_proj = nn.Linear(self.backbone_embed_dim, 128)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.total_feature_dim = 128 + 128 + 64

        self.heads = nn.ModuleList([
            RewardHead(feature_dim=self.total_feature_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)
            for _ in range(ensemble_size)
        ])

    def forward_features(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        base_imgs = batch_data["base_img"]
        hand_imgs = batch_data["hand_img"]
        states = batch_data["state"]

        b, w = base_imgs.shape[:2]
        flat_base = base_imgs.reshape(b * w, 3, 224, 224)
        flat_hand = hand_imgs.reshape(b * w, 3, 224, 224)

        with torch.set_grad_enabled(not self.freeze_backbone):
            base_out = self.backbone(flat_base)
            hand_out = self.backbone(flat_hand)
            base_feat = base_out.last_hidden_state[:, 0, :]
            hand_feat = hand_out.last_hidden_state[:, 0, :]

        base_emb = self.base_proj(base_feat)
        hand_emb = self.hand_proj(hand_feat)
        flat_state = states.reshape(b * w, -1)
        state_emb = self.state_proj(flat_state)

        fused = torch.cat([base_emb, hand_emb, state_emb], dim=-1)
        return fused.reshape(b, w, -1)

    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.forward_features(batch_data)
        rewards = torch.stack([h(feats) for h in self.heads], dim=0)
        return rewards

    @torch.no_grad()
    def predict_reward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        raw = self(batch_data)
        return raw.mean(dim=0)

    @torch.no_grad()
    def predict_reward_bounded(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw = self.predict_reward(batch_data)
        return torch.tanh(raw / self.bounded_temperature)


class VideoCaptureCache:
    """Simple LRU cache for video readers to avoid reopening files frequently."""
    def __init__(self, max_size=8):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get_reader(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        
        if len(self.cache) >= self.max_size:
            old_path, old_reader = self.cache.popitem(last=False)
            old_reader.close()
            
        reader = imageio.get_reader(path, 'ffmpeg')
        self.cache[path] = reader
        return reader
        
    def __del__(self):
        for reader in self.cache.values():
            try:
                reader.close()
            except:
                pass


# ----------------------------------------------------------------
# 4. Dataset: Lerobot Reward Dataset
# ----------------------------------------------------------------
class LerobotRewardDataset(Dataset):
    """
    Dataset for loading robot interaction data from Parquet files (and videos).
    Generates pairs of (early, late) windows for ranking.
    """
    def __init__(
        self,
        data_root: str,
        window_size: int = 5,
        min_gap: int = 15,
        max_gap: int = 60,
        split: str = 'train',
        task_names: Optional[List[str]] = None,
        image_size: int = 224,
        aug: bool = True,
        cache_index: bool = True,
        index_cache_dir: Optional[str] = None,
        chunk_cache_size: int = 2,
        pair_sampling: str = "uniform",
        label_mode: str = "proxy",
    ):
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.split = split
        self.is_train = (split == 'train')
        self.image_size = image_size
        self.aug = aug and self.is_train
        # Use a larger stride for training to reduce index size, smaller for val
        self.index_stride = 3 if self.is_train else 1
        self.cache_index = cache_index
        self.chunk_cache_size = chunk_cache_size
        self.pair_sampling = pair_sampling
        self.label_mode = label_mode

        self._chunk_cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self.video_cache = VideoCaptureCache(max_size=4)

        # Default data keys
        self.key_base = 'observation.images.base_camera'
        self.key_hand = 'observation.images.hand_camera'
        self.key_state = 'observation.state'
        
        self.load_from_video = self._detect_video_loading()
        self._setup_transforms()

        # Build or load dataset index
        self.episode_records, self.indices = self._load_or_build_index(task_names, split, index_cache_dir)
        logger.info(f"[{split}] Loaded {len(self.episode_records)} episodes, {len(self.indices)} samples.")

    def _detect_video_loading(self) -> bool:
        """Detect if we need to load images from video files."""
        info_path = self.data_root / "meta/info.json"
        load_video = False
        
        if info_path.exists():
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                feats = info.get("features", {})
                
                # Check for alternative key names
                if "observation.images.front" in feats:
                    self.key_base = "observation.images.front"
                if "observation.images.wrist" in feats:
                    self.key_hand = "observation.images.wrist"
                
                # Check dtype
                if self.key_base in feats and feats[self.key_base].get("dtype") == "video":
                    load_video = True
                    logger.info("Detected video features in info.json.")
            except Exception as e:
                logger.warning(f"Error parsing info.json: {e}")

        # Fallback check
        if not load_video and (self.data_root / "videos").exists():
            load_video = True
            logger.info("Detected 'videos' directory, enabling video loading fallback.")
            
        return load_video

    def _setup_transforms(self):
        """Configure image transformations."""
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.eval_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])
        
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.2,
            ),
            transforms.ToTensor(),
            GaussianNoise(mean=0.0, std=0.05),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
        ])

    def _rows_numpy(self, rows) -> np.ndarray:
        """Ensure rows are numpy array."""
        if isinstance(rows, np.ndarray):
            return rows
        if torch.is_tensor(rows):
            return rows.detach().cpu().numpy()
        return np.asarray(rows, dtype=np.int64)

    def _index_cache_path(self, split: str, index_cache_dir: Optional[str]) -> Path:
        base_dir = Path(index_cache_dir) if index_cache_dir else (self.data_root / "meta")
        base_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{split}_ws{self.window_size}_g{self.min_gap}-{self.max_gap}_stride{self.index_stride}"
        return base_dir / f"index_cache_{tag}.pt"

    def _load_or_build_index(self, task_names, split, index_cache_dir: Optional[str]):
        """Load index from cache or build it from scratch."""
        cache_path = self._index_cache_path(split, index_cache_dir)
        
        if self.cache_index and cache_path.exists():
            try:
                # Use torch.load for simple serialization
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                ep = payload["episode_records"]
                indices = payload["indices"]
                
                # Ensure compatibility
                for r in ep:
                    r["rows"] = self._rows_numpy(r["rows"])
                
                # Check for empty validation cache
                if split == 'val' and len(indices) == 0:
                    logger.warning(f"[{split}] Cached index is empty. Rebuilding...")
                    # Fall through to rebuild
                else:
                    return ep, indices
            except Exception as e:
                logger.warning(f"Index cache load failed: {e}. Rebuilding...")

        # If we reached here, we need to build
        global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        
        # Only rank 0 builds the index to avoid race conditions
        if global_rank == 0:
            episode_records, indices = self._build_index(task_names, split)
            
            if self.cache_index:
                try:
                    tmp_path = cache_path.with_suffix(".pt.tmp")
                    torch.save({"episode_records": episode_records, "indices": indices}, tmp_path)
                    os.replace(tmp_path, cache_path)
                    logger.info(f"[{split}] Index cached to {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save index cache: {e}")
            
            return episode_records, indices
        
        # Other ranks wait for rank 0
        t0 = time.time()
        timeout_s = 600
        while not cache_path.exists():
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Timed out waiting for index cache: {cache_path}")
            time.sleep(1.0)
            
        return self._load_or_build_index(task_names, split, index_cache_dir)

    def _build_index(self, task_names, split):
        """
        Builds the index by scanning all parquet files.
        Strategy: Scan ALL episodes first, then filter based on split.
        This ensures validation set is not empty if data exists.
        """
        all_episode_records = []
        all_episode_samples = []  # (ep_idx, ep_rec_idx, start_t, is_train_ep)
        
        # Determine task directories
        if (self.data_root / 'meta' / 'info.json').exists():
            task_dirs = [self.data_root]
        elif task_names is None:
            task_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        else:
            task_dirs = [self.data_root / t for t in task_names]

        # 1. Scan and collect ALL episodes first
        all_episode_indices = []  # Collect all unique episode indices across all chunks
        
        for task_dir in task_dirs:
            if not task_dir.exists(): continue
            
            data_dir = task_dir / 'data'
            chunk_files = sorted(data_dir.glob('**/*.parquet'))
            
            for chunk_file in chunk_files:
                try:
                    columns = [self.key_state, 'episode_index', 'frame_index'] if self.load_from_video \
                              else [self.key_base, self.key_hand, self.key_state, 'episode_index', 'frame_index']
                    df = pd.read_parquet(chunk_file, columns=columns)
                    unique_eps = df['episode_index'].unique()
                    all_episode_indices.extend(unique_eps.tolist())
                except Exception:
                    continue
        
        # Get unique episode indices and sort them
        all_episode_indices = sorted(set(all_episode_indices))
        total_episodes_found = len(all_episode_indices)
        
        # 2. First pass: Collect all VALID episodes (using training criteria to be more lenient)
        # This ensures we only count episodes that can actually be used
        valid_episode_indices = []
        val_valid_episodes = []  # Episodes that can be used for validation
        episode_lengths = {}  # Store episode lengths for later use
        
        for task_dir in task_dirs:
            if not task_dir.exists(): continue
            
            data_dir = task_dir / 'data'
            chunk_files = sorted(data_dir.glob('**/*.parquet'))
            
            for chunk_file in chunk_files:
                try:
                    columns = [self.key_state, 'episode_index', 'frame_index'] if self.load_from_video \
                              else [self.key_base, self.key_hand, self.key_state, 'episode_index', 'frame_index']
                    df = pd.read_parquet(chunk_file, columns=columns)
                except Exception:
                    continue

                unique_eps = df['episode_index'].unique()

                for ep_idx in unique_eps:
                    if ep_idx in episode_lengths:
                        continue  # Already processed
                    
                    ep_mask = (df["episode_index"] == ep_idx)
                    row_ids = df.loc[ep_mask].index.to_numpy(dtype=np.int64)
                    ep_len = int(len(row_ids))
                    episode_lengths[ep_idx] = ep_len
                    
                    # Check if episode can be used for training or validation
                    # For training: valid_len = ep_len - window_size + 1
                    # For validation: valid_len = ep_len - window_size * 2 - min_gap
                    train_valid_len = ep_len - self.window_size + 1
                    val_valid_len = ep_len - self.window_size * 2 - self.min_gap
                    
                    # Store both validities
                    if train_valid_len > 0:
                        valid_episode_indices.append(ep_idx)
                    if val_valid_len > 0:
                        val_valid_episodes.append(ep_idx)
        
        # Sort valid episodes
        valid_episode_indices = sorted(valid_episode_indices)
        total_valid_episodes = len(valid_episode_indices)
        valid_episode_indices_set = set(valid_episode_indices)  # For fast lookup
        
        # For validation set: only use episodes that can actually be used for validation
        val_valid_episodes = sorted(val_valid_episodes)
        if len(val_valid_episodes) >= 11:
            # Use last 11 episodes that can be used for validation
            val_episode_indices = set(val_valid_episodes[-11:])
        else:
            # If not enough episodes can be used for validation, use all available
            val_episode_indices = set(val_valid_episodes)
        
        # Training set = all valid episodes minus validation episodes
        train_episode_count = total_valid_episodes - len(val_episode_indices)
        
        logger.info(f"[split] Total episode indices found: {total_episodes_found}, Valid episodes: {total_valid_episodes}")
        logger.info(f"[split] Training episodes: {train_episode_count}, Validation episodes: {len(val_episode_indices)}")
        
        # 3. Second pass: Scan and collect episodes with split information
        skipped_episodes = 0  # Count episodes that are too short
        for task_dir in task_dirs:
            if not task_dir.exists(): continue
            
            try:
                data_dir = task_dir / 'data'
                chunk_files = sorted(data_dir.glob('**/*.parquet'))
                
                for chunk_file in chunk_files:
                    try:
                        # Use pandas to quickly read index columns
                        columns = [self.key_state, 'episode_index', 'frame_index'] if self.load_from_video \
                                  else [self.key_base, self.key_hand, self.key_state, 'episode_index', 'frame_index']
                        df = pd.read_parquet(chunk_file, columns=columns)
                    except Exception:
                        continue

                    unique_eps = df['episode_index'].unique()

                    for ep_idx in unique_eps:
                        # Only process episodes that are valid (in valid_episode_indices_set)
                        if ep_idx not in valid_episode_indices_set:
                            continue
                        
                        # Determine split based on episode index
                        is_train_ep = ep_idx not in val_episode_indices
                        
                        ep_mask = (df["episode_index"] == ep_idx)
                        row_ids = df.loc[ep_mask].index.to_numpy(dtype=np.int64)
                        ep_len = int(len(row_ids))

                        # Determine valid length based on sampling strategy
                        # Use looser criteria for non-uniform sampling (biased_gap) or validation
                        if is_train_ep and self.pair_sampling == "uniform":
                            valid_len = ep_len - self.window_size + 1
                        else:
                            # Use min_gap to maximize samples for validation / biased sampling
                            valid_len = ep_len - self.window_size * 2 - self.min_gap

                        if valid_len > 0:
                            ep_rec_idx = len(all_episode_records)
                            all_episode_records.append({
                                "chunk": str(chunk_file),
                                "rows": row_ids,
                            })
                            
                            # Add all possible start indices
                            for start_t in range(0, valid_len, self.index_stride):
                                all_episode_samples.append((ep_idx, ep_rec_idx, start_t, is_train_ep))
                        else:
                            skipped_episodes += 1
                                
            except Exception as e:
                logger.warning(f"Error loading {task_dir}: {e}")

        if skipped_episodes > 0:
            logger.info(f"[split] Skipped {skipped_episodes} episodes that were too short (ep_len < required minimum)")

        # 4. Filter for requested split
        episode_records = []
        indices = []
        
        # Map old record index to new record index
        ep_rec_mapping = {}  # old_idx -> new_idx
        
        for ep_idx, old_ep_rec_idx, start_t, is_train_ep in all_episode_samples:
            # Filter logic
            if split == 'train' and not is_train_ep: continue
            if split == 'val' and is_train_ep: continue
            
            # If included, ensure record is in new list
            if old_ep_rec_idx not in ep_rec_mapping:
                new_ep_rec_idx = len(episode_records)
                ep_rec_mapping[old_ep_rec_idx] = new_ep_rec_idx
                episode_records.append(all_episode_records[old_ep_rec_idx])
            
            indices.append((ep_rec_mapping[old_ep_rec_idx], start_t))

        if split == 'val' and len(episode_records) == 0:
            logger.warning(f"[{split}] Warning: Validation set is empty!")

        return episode_records, indices

    def _get_chunk_df(self, chunk_path: str) -> pd.DataFrame:
        """LRU cache for parquet chunks."""
        if chunk_path in self._chunk_cache:
            # Move to end (MRU)
            df = self._chunk_cache.pop(chunk_path)
            self._chunk_cache[chunk_path] = df
            return df

        columns = [self.key_state, "episode_index", "frame_index"] if self.load_from_video \
                  else [self.key_base, self.key_hand, self.key_state, "episode_index", "frame_index"]
            
        df = pd.read_parquet(chunk_path, columns=columns)
        self._chunk_cache[chunk_path] = df
        
        # Evict if full
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False) # Pop FIFO (LRU)
        return df

    def _get_video_path(self, chunk_path: str, key: str) -> str:
        """Resolve video path from parquet chunk path."""
        p = Path(chunk_path)
        # Assuming structure: .../data/chunk-XXX/file.parquet -> .../videos/key/chunk-XXX/file.mp4
        # Simple string replacement fallback
        s = str(chunk_path)
        return s.replace("/data/", f"/videos/{key}/").replace(".parquet", ".mp4")

    def _read_video_frame(self, chunk_path: str, key: str, frame_idx: int):
        vid_path = self._get_video_path(chunk_path, key)
        try:
            reader = self.video_cache.get_reader(vid_path)
            frame = reader.get_data(frame_idx)
            return frame
        except Exception:
            # Return black frame on error
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _decode_image(self, image_bytes):
        """Decode image from bytes or dict."""
        if isinstance(image_bytes, dict) and 'bytes' in image_bytes:
            image_bytes = image_bytes['bytes']

        if isinstance(image_bytes, bytes):
            try:
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if self.aug:
                    return self.train_transform(img)
                return self.eval_transform(img)
            except Exception:
                pass
        return torch.zeros(3, self.image_size, self.image_size)

    def _process_state(self, state_array):
        """Process and normalize state vector."""
        if isinstance(state_array, np.ndarray):
            tensor = torch.from_numpy(state_array.copy()).float()
        else:
            tensor = torch.tensor(state_array).float()
        
        # Specific logic for this dataset: extract relevant joints
        if tensor.shape[0] >= 26:
             part1 = tensor[0:9]
             part2 = tensor[19:26]
             return torch.cat([part1, part2])
        return tensor

    def _proxy_obj_to_goal_dist(self, state_array) -> torch.Tensor:
        """Calculate proxy reward (distance) from state."""
        if isinstance(state_array, np.ndarray):
            st = torch.from_numpy(state_array.copy()).float()
        else:
            st = torch.tensor(state_array).float()
        
        # Specific index for goal distance
        if st.shape[0] > 42:
            v = st[39:42]
            return torch.linalg.norm(v, ord=2)
        return torch.tensor(0.0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_rec_idx, early_start = self.indices[idx]
        rec = self.episode_records[ep_rec_idx]
        
        # Load data for this window
        chunk_path = rec["chunk"]
        df = self._get_chunk_df(chunk_path)
        row_ids = self._rows_numpy(rec["rows"])
        ep_len = int(len(row_ids))
        denom = max(1, ep_len - 1)

        # Helper to get window data
        def get_window_data(start_idx):
            base_imgs, hand_imgs, states = [], [], []
            last_raw_state = None
            for i in range(self.window_size):
                rid = int(row_ids[start_idx + i])
                row = df.loc[rid]
                
                # Load images
                if self.load_from_video:
                    frame_idx = int(row["frame_index"]) if "frame_index" in row else rid
                    base_np = self._read_video_frame(chunk_path, self.key_base, frame_idx)
                    hand_np = self._read_video_frame(chunk_path, self.key_hand, frame_idx)
                    
                    if self.aug:
                        base_imgs.append(self.train_transform(base_np))
                        hand_imgs.append(self.train_transform(hand_np))
                    else:
                        base_imgs.append(self.eval_transform(base_np))
                        hand_imgs.append(self.eval_transform(hand_np))
                else:
                    base_imgs.append(self._decode_image(row[self.key_base]))
                    hand_imgs.append(self._decode_image(row[self.key_hand]))
                
                states.append(self._process_state(row[self.key_state]))
                if i == self.window_size - 1:
                    last_raw_state = row[self.key_state]

            out = {
                'base_img': torch.stack(base_imgs),
                'hand_img': torch.stack(hand_imgs),
                'state': torch.stack(states)
            }
            if self.label_mode == "proxy" and last_raw_state is not None:
                out["proxy_dist"] = self._proxy_obj_to_goal_dist(last_raw_state)
            return out

        def get_progress(start_idx):
            end_t = start_idx + (self.window_size - 1)
            p = float(end_t) / float(denom)
            return 2.0 * p - 1.0  # Normalize to [-1, 1]

        # Determine sampling strategy for pair
        max_start = max(0, ep_len - self.window_size)
        
        # 1. Uniform Sampling (Random pair) - mostly for training
        if self.is_train and (self.pair_sampling == "uniform"):
            s0 = random.randint(0, max_start)
            # dynamic_min_gap = max(self.min_gap, int(ep_len * 0.15))
            dynamic_min_gap = max(self.min_gap, int(ep_len * 0.15))
            min_sep = self.window_size + dynamic_min_gap
            s1 = None
            # Try to find a valid pair
            for _ in range(32):
                cand = random.randint(0, max_start)
                if abs(cand - s0) >= min_sep:
                    s1 = cand
                    break
            # Fallback
            if s1 is None:
                s1 = max(0, min(max_start, s0 + min_sep))
                if s1 == s0: s1 = max(0, min(max_start, s0 - min_sep))

            w0 = get_window_data(s0)
            w1 = get_window_data(s1)
            
            # Determine label
            l_time = 1.0 if (s1 > s0) else 0.0
            if ("proxy_dist" in w0) and ("proxy_dist" in w1):
                l_proxy = 1.0 if (w1["proxy_dist"] < w0["proxy_dist"]) else 0.0
            else:
                l_proxy = l_time

            label = l_proxy if self.label_mode == "proxy" else l_time
            early, late = w0, w1
            prog_early, prog_late = get_progress(s0), get_progress(s1)

            # Random flip for augmentation
            if random.random() < 0.5:
                label = 1.0 - label
                l_time = 1.0 - l_time
                l_proxy = 1.0 - l_proxy
                early, late = late, early
                prog_early, prog_late = prog_late, prog_early

        # 2. Biased Gap Sampling (Deterministic pair based on current index) - for validation
        else:
            gap = random.randint(self.min_gap, self.max_gap) if self.is_train else self.min_gap
            
            s_early = early_start
            s_late = min(s_early + self.window_size + gap, max_start)
            # Ensure s_late is at least distinct if possible
            s_late = max(s_early, s_late)

            early = get_window_data(s_early)
            late = get_window_data(s_late)
            prog_early, prog_late = get_progress(s_early), get_progress(s_late)

            l_time = 1.0 # By definition late > early in time
            if ("proxy_dist" in early) and ("proxy_dist" in late):
                l_proxy = 1.0 if (late["proxy_dist"] < early["proxy_dist"]) else 0.0
            else:
                l_proxy = 1.0

            label = l_proxy if self.label_mode == "proxy" else l_time

            # Deterministic flip for validation based on index
            flip = (random.random() < 0.5) if self.is_train else ((idx + ep_rec_idx) % 2 == 1)
            
            if flip:
                label = 1.0 - label
                l_time = 1.0 - l_time
                l_proxy = 1.0 - l_proxy
                early, late = late, early
                prog_early, prog_late = prog_late, prog_early

        return {
            'early': early,
            'late': late,
            'label': torch.tensor(label, dtype=torch.float32),
            'label_time': torch.tensor(l_time, dtype=torch.float32),
            'label_proxy': torch.tensor(l_proxy, dtype=torch.float32),
            'progress_target_early': torch.tensor(prog_early, dtype=torch.float32),
            'progress_target_late': torch.tensor(prog_late, dtype=torch.float32),
        }

    def get_window_by_end(self, ep_rec_idx: int, end_t: int) -> Dict[str, torch.Tensor]:
        """
        Extract a window of observations ending at a specific time step.
        Used primarily by the validation callback.
        """
        rec = self.episode_records[ep_rec_idx]
        chunk_path = rec["chunk"]
        df = self._get_chunk_df(chunk_path)
        row_ids = self._rows_numpy(rec["rows"])
        ep_len = int(len(row_ids))

        # Clamp end_t to valid range
        if end_t < self.window_size - 1:
            end_t = self.window_size - 1
        if end_t > ep_len - 1:
            end_t = ep_len - 1

        start_idx = end_t - (self.window_size - 1)

        base_imgs, hand_imgs, states = [], [], []
        for i in range(self.window_size):
            rid = int(row_ids[start_idx + i])
            row = df.loc[rid]
            
            # Load images
            if self.load_from_video:
                frame_idx = int(row["frame_index"]) if "frame_index" in row else rid
                base_np = self._read_video_frame(chunk_path, self.key_base, frame_idx)
                hand_np = self._read_video_frame(chunk_path, self.key_hand, frame_idx)
                
                # Use eval transform (deterministic)
                base_imgs.append(self.eval_transform(base_np))
                hand_imgs.append(self.eval_transform(hand_np))
            else:
                base_imgs.append(self._decode_image(row[self.key_base]))
                hand_imgs.append(self._decode_image(row[self.key_hand]))
                
            states.append(self._process_state(row[self.key_state]))

        return {
            "base_img": torch.stack(base_imgs),
            "hand_img": torch.stack(hand_imgs),
            "state": torch.stack(states),
        }


# ----------------------------------------------------------------
# 5. Callbacks
# ----------------------------------------------------------------
class LossSummaryCallback(pl.Callback):
    """Callback to print and save detailed loss breakdown at the end of each training epoch."""
    
    def __init__(self, save_csv: bool = True, csv_path: Optional[str] = None):
        """
        Args:
            save_csv: Whether to save results to CSV file
            csv_path: Path to CSV file (if None, auto-generate from log dir)
        """
        super().__init__()
        self.save_csv = save_csv
        self.csv_path = csv_path
        self.results = []  # Store results for CSV export
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: BadassRewardModel) -> None:
        """Initialize CSV file at training start."""
        if not trainer.is_global_zero or not self.save_csv:
            return
        
        # Determine CSV path
        if self.csv_path is None:
            log_dir = Path(trainer.logger.log_dir) if hasattr(trainer.logger, 'log_dir') else Path("lightning_logs")
            self.csv_path = log_dir / "loss_summary.csv"
        
        # Create CSV file with header
        self.csv_columns = [
            'epoch', 'train/loss', 'train/loss_ranking', 'train/loss_progress', 
            'train/loss_mse', 'train/loss_range_push', 'val/loss', 
            'val/acc_time', 'val/acc_proxy', 'val/monotonicity'
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_columns)
        
        logger.info(f"Loss summary will be saved to: {self.csv_path}")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: BadassRewardModel) -> None:
        """Print and save detailed loss breakdown at the end of each training epoch."""
        if not trainer.is_global_zero:
            return
        
        # Get logged metrics
        metrics = trainer.callback_metrics
        
        # Collect loss components
        loss_items = []
        row_data = {
            'epoch': trainer.current_epoch,
            'train/loss': None,
            'train/loss_ranking': None,
            'train/loss_progress': None,
            'train/loss_mse': None,
            'train/loss_range_push': 0.0,
            'val/loss': None,
            'val/acc_time': None,
            'val/acc_proxy': None,
            'val/monotonicity': None,
        }
        
        if 'train/loss_epoch' in metrics:
            val = metrics['train/loss_epoch'].item()
            loss_items.append(f"Total: {val:.4f}")
            row_data['train/loss'] = val
        
        if 'train/loss_ranking' in metrics:
            val = metrics['train/loss_ranking'].item()
            loss_items.append(f"Ranking: {val:.4f}")
            row_data['train/loss_ranking'] = val
        
        if 'train/loss_progress' in metrics:
            val = metrics['train/loss_progress'].item()
            loss_items.append(f"Progress: {val:.4f}")
            row_data['train/loss_progress'] = val
        
        if 'train/loss_mse' in metrics:
            val = metrics['train/loss_mse'].item()
            loss_items.append(f"MSE: {val:.4f}")
            row_data['train/loss_mse'] = val
        
        if 'train/loss_range_push' in metrics:
            push_val = metrics['train/loss_range_push'].item()
            if push_val > 0:
                loss_items.append(f"RangePush: {push_val:.4f}")
            row_data['train/loss_range_push'] = push_val
        
        # Add validation metrics if available
        if 'val/loss' in metrics:
            row_data['val/loss'] = metrics['val/loss'].item()
        if 'val/acc_time' in metrics:
            row_data['val/acc_time'] = metrics['val/acc_time'].item()
        if 'val/acc_proxy' in metrics:
            row_data['val/acc_proxy'] = metrics['val/acc_proxy'].item()
        if 'val/monotonicity' in metrics:
            row_data['val/monotonicity'] = metrics['val/monotonicity'].item()
        
        # Print to console
        if loss_items:
            logger.info(f"[Epoch {trainer.current_epoch}] Loss breakdown: " + " | ".join(loss_items))
        
        # Save to CSV
        if self.save_csv:
            self.results.append(row_data)
            self._save_to_csv()
    
    def _save_to_csv(self):
        """Save accumulated results to CSV file."""
        if not self.csv_path or not self.results:
            return
        
        # Write all rows with fixed column order
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.results)


class ValTrajectoryRewardBinsCallback(pl.Callback):
    """
    Validation callback to compute and log average reward over trajectory bins (0-10%, ..., 90-100%).
    Helps visualize if the reward is monotonic with respect to progress.
    """
    def __init__(
        self,
        val_dataset: "LerobotRewardDataset",
        bins: int = 10,
        samples_per_bin: int = 1,
        batch_size: int = 32,
        tanh_temperature: float = 2.0,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.bins = bins
        self.samples_per_bin = samples_per_bin
        self.batch_size = batch_size
        self.tanh_temperature = tanh_temperature

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BadassRewardModel) -> None:
        if not trainer.is_global_zero:
            return

        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device

        sum_bins = torch.zeros(self.bins, dtype=torch.float64)
        sum_sq_bins = torch.zeros(self.bins, dtype=torch.float64)
        cnt_bins = torch.zeros(self.bins, dtype=torch.float64)

        # Sample episodes to evaluate
        indices = list(range(len(self.val_dataset.episode_records)))
        if len(indices) > 20:
            indices = random.sample(indices, 20)
            
        for ep_rec_idx in indices:
            rec = self.val_dataset.episode_records[ep_rec_idx]
            ep_len = int(len(rec["rows"]))
            if ep_len < self.val_dataset.window_size:
                continue

            # Generate sample points for each bin
            batch_windows, batch_bin_ids = [], []
            
            for bi in range(self.bins):
                lo, hi = bi / self.bins, (bi + 1) / self.bins
                
                # Map bin percentage to time indices
                lo_t = int(math.floor(lo * (ep_len - 1)))
                hi_t = int(math.floor(hi * (ep_len - 1)))
                # Constrain to valid window ends
                lo_t = max(lo_t, self.val_dataset.window_size - 1)
                hi_t = min(max(hi_t, self.val_dataset.window_size - 1), ep_len - 1)
                
                if hi_t < lo_t: continue

                # Sample time steps
                if self.samples_per_bin <= 1:
                    ts = [(lo_t + hi_t) // 2]
                else:
                    ts = torch.linspace(lo_t, hi_t, steps=self.samples_per_bin).round().long().tolist()
                    ts = list(dict.fromkeys(ts))

                for t in ts:
                    batch_windows.append(self.val_dataset.get_window_by_end(ep_rec_idx, int(t)))
                    batch_bin_ids.append(bi)

            # Batched inference
            if not batch_windows: continue
            
            # Process in mini-batches
            for i in range(0, len(batch_windows), self.batch_size):
                batch = batch_windows[i:i+self.batch_size]
                ids = batch_bin_ids[i:i+self.batch_size]
                
                base = torch.stack([w["base_img"] for w in batch], dim=0).to(device)
                hand = torch.stack([w["hand_img"] for w in batch], dim=0).to(device)
                state = torch.stack([w["state"] for w in batch], dim=0).to(device)

                raw = pl_module.predict_reward({"base_img": base, "hand_img": hand, "state": state}).squeeze(-1)
                rewards = torch.tanh(raw / self.tanh_temperature).detach().cpu().double()

                for r, bi in zip(rewards.tolist(), ids):
                    sum_bins[bi] += r
                    sum_sq_bins[bi] += (r * r)
                    cnt_bins[bi] += 1

        # Aggregate statistics
        mean_bins = sum_bins / torch.clamp(cnt_bins, min=1.0)
        var_bins = (sum_sq_bins / torch.clamp(cnt_bins, min=1.0)) - (mean_bins ** 2)
        std_bins = torch.sqrt(torch.clamp(var_bins, min=0.0))

        # Log results
        parts = []
        for i in range(self.bins):
            parts.append(f"{int(100*i/self.bins)}-{int(100*(i+1)/self.bins)}%: {mean_bins[i].item():+.4f}±{std_bins[i].item():.4f}")
            pl_module.log(f"val/reward_bin_{i}", float(mean_bins[i].item()), prog_bar=False, sync_dist=False)
            
        # Monotonicity Loss (Ranking Quality)
        diffs = mean_bins[1:] - mean_bins[:-1]
        monotonicity = (diffs > 0).float().mean().item()  # Convert to Python float to avoid device sync issues
        pl_module.log('val/monotonicity', monotonicity, prog_bar=True, sync_dist=False)
            
        logger.info(f"[val] Reward bins: " + " | ".join(parts))
        pl_module.train(was_training)


# ----------------------------------------------------------------
# 6. Utils & Main Entry
# ----------------------------------------------------------------
def is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

@torch.no_grad()
def evaluate_reward_model(model_path, val_loader, device="auto", return_predictions=False, model_type="auto"):
    """
    Standalone evaluation function for exported models (JIT or PyTorch).
    
    Args:
        model_path: Path to the model file
        val_loader: Validation DataLoader
        device: Device to run inference on ("auto", "cuda", or "cpu").
        return_predictions: If True, returns (metrics, predictions, ground_truths)
        model_type: "auto" (detect), "jit", or "pytorch"
    
    Returns:
        metrics dict, or (metrics, predictions, ground_truths) if return_predictions=True
    """
    model_path = Path(model_path)
    
    # Auto-detect model type
    if model_type == "auto":
        try:
            # Try loading as JIT first
            test_model = torch.jit.load(str(model_path), map_location="cpu")
            model_type = "jit"
            logger.info("[evaluate_reward_model] Detected JIT model")
        except:
            # If JIT fails, assume PyTorch
            model_type = "pytorch"
            logger.info("[evaluate_reward_model] Detected PyTorch model")
    
    # Load model based on type
    if model_type == "jit":
        # JIT model loading
        if device == "auto":
            map_location = "cpu"
        elif str(device).startswith("cuda") and torch.cuda.is_available():
            map_location = device
        else:
            map_location = "cpu"
        
        model = torch.jit.load(str(model_path), map_location=map_location)
        model.eval()
        
        # Detect model's expected device by running a dummy forward pass
        model_device = None
        if device == "auto":
            # Try GPU first if available
            if torch.cuda.is_available():
                try:
                    dummy_base = torch.randn(1, 5, 3, 224, 224, device="cuda")
                    dummy_hand = torch.randn(1, 5, 3, 224, 224, device="cuda")
                    dummy_state = torch.randn(1, 5, 7, device="cuda")
                    _ = model(dummy_base, dummy_hand, dummy_state)
                    model_device = torch.device("cuda")
                    logger.info("[evaluate_reward_model] JIT model runs on GPU (as traced)")
                except (RuntimeError, Exception) as e:
                    if "device" in str(e).lower() or "cuda" in str(e).lower():
                        logger.info(f"[evaluate_reward_model] GPU test failed, trying CPU: {e}")
                        model_device = None
                    else:
                        raise
            
            # Try CPU if GPU failed or not available
            if model_device is None:
                try:
                    dummy_base = torch.randn(1, 5, 3, 224, 224, device="cpu")
                    dummy_hand = torch.randn(1, 5, 3, 224, 224, device="cpu")
                    dummy_state = torch.randn(1, 5, 7, device="cpu")
                    _ = model(dummy_base, dummy_hand, dummy_state)
                    model_device = torch.device("cpu")
                    logger.info("[evaluate_reward_model] JIT model runs on CPU (as traced)")
                except RuntimeError as e:
                    logger.error(f"[evaluate_reward_model] Both GPU and CPU failed: {e}")
                    raise RuntimeError(f"Cannot determine model device: {e}") from e
        else:
            # Use the requested device
            if str(device).startswith("cuda") and torch.cuda.is_available():
                model_device = torch.device(device)
            elif str(device).startswith("cuda"):
                logger.warning(f"CUDA device '{device}' requested but not available. Falling back to CPU.")
                model_device = torch.device("cpu")
            else:
                model_device = torch.device(device)
            logger.info(f"[evaluate_reward_model] Using requested device: {model_device}")
        
        eval_device = model_device
        
        # Wrapper for JIT model inference
        def predict_fn(base, hand, state):
            return model(base, hand, state)
            
    else:
        # PyTorch model loading
        model = load_pytorch_model(str(model_path), device=device)
        eval_device = next(model.parameters()).device
        
        # Wrapper for PyTorch model inference
        def predict_fn(base, hand, state):
            return model.predict_reward_bounded({
                "base_img": base,
                "hand_img": hand,
                "state": state
            })

    preds_list, gts_list = [], []

    for batch in val_loader:
        if isinstance(batch, dict):
            # Check for pair-wise keys in batch
            if "early" in batch and isinstance(batch["early"], dict):
                # Use 'early' part for evaluation check
                sub_batch = batch["early"]
                base = sub_batch.get("base_img")
                hand = sub_batch.get("hand_img")
                state = sub_batch.get("state")
                gt = batch.get("label")
            else:
                base = batch.get("base_img") or batch.get("base")
                hand = batch.get("hand_img") or batch.get("hand")
                state = batch.get("state")
                gt = batch.get("label")
        elif isinstance(batch, (tuple, list)) and len(batch) >= 3:
            base, hand, state = batch[0], batch[1], batch[2]
            gt = batch[3] if len(batch) > 3 else None
        else:
            continue

        if base is None or hand is None:
             continue

        # Move inputs to the same device as model expects
        base = base.to(eval_device, non_blocking=False)
        hand = hand.to(eval_device, non_blocking=False)
        if state is not None:
            state = state.to(eval_device, non_blocking=False)

        pred = predict_fn(base, hand, state)
        preds_list.append(pred.detach().float().reshape(-1).cpu())
        
        if gt is not None:
            gts_list.append(gt.detach().float().reshape(-1).cpu())

    if len(preds_list) == 0:
        return {"n": 0, "pred_mean": 0.0, "pred_std": 0.0}

    preds = torch.cat(preds_list, dim=0)
    gts = torch.cat(gts_list, dim=0) if gts_list else None

    metrics = {
        "n": int(preds.numel()),
        "pred_mean": float(preds.mean().item()),
        "pred_std": float(preds.std(unbiased=False).item()),
    }
    return (metrics, preds.numpy(), gts.numpy()) if return_predictions else metrics


def export_to_pytorch(model: pl.LightningModule, path: Path, use_half: bool = False):
    """
    Export the model as a pure PyTorch model (no JIT, no Lightning dependency).
    Saves state_dict and config, can be loaded on any device.
    
    Args:
        model: PyTorch Lightning model
        path: Path to save the exported model (.pt file)
        use_half: If True, convert model to half precision (FP16) before export.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Exporting PyTorch model to {path} (FP16={use_half})...")
        
        hparams = model.hparams
        
        # Create pure model and load weights
        dropout = getattr(hparams, 'dropout', 0.1)
        export_model = PureBadassRewardModel(
            ensemble_size=int(hparams.ensemble_size),
            window_size=int(hparams.window_size),
            freeze_backbone=bool(hparams.freeze_backbone),
            backbone_type=str(hparams.backbone_type),
            state_dim=int(hparams.state_dim),
            bounded_temperature=float(hparams.bounded_temperature),
            head_dropout=float(dropout),
        )
        
        # Load weights from the trained model
        state_dict = model.state_dict()
        export_model.load_state_dict(state_dict, strict=False)
        
        if use_half:
            export_model = export_model.half()
            
        export_model.eval()
        
        # Save model state_dict and config
        save_data = {
            'state_dict': export_model.state_dict(),
            'config': {
                'ensemble_size': int(hparams.ensemble_size),
                'window_size': int(hparams.window_size),
                'freeze_backbone': bool(hparams.freeze_backbone),
                'backbone_type': str(hparams.backbone_type),
                'state_dim': int(hparams.state_dim),
                'bounded_temperature': float(hparams.bounded_temperature),
                'use_half': use_half
            }
        }
        
        torch.save(save_data, path)
        logger.info(f"✅ PyTorch export successful: {path} (can run on any device)")
        return True
    except Exception as e:
        logger.error(f"❌ PyTorch export failed: {e}")
        return False


def load_pytorch_model(model_path: str, device: str = "auto"):
    """
    Load a pure PyTorch model exported by export_to_pytorch.
    
    Args:
        model_path: Path to the saved model (.pt file)
        device: Device to load on ("auto", "cuda", or "cpu")
    
    Returns:
        Loaded PureBadassRewardModel instance
    """
    # Determine device
    if device == "auto":
        load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif str(device).startswith("cuda"):
        if torch.cuda.is_available():
            load_device = torch.device(device)
        else:
            logger.warning(f"CUDA device '{device}' requested but not available. Falling back to CPU.")
            load_device = torch.device("cpu")
    else:
        load_device = torch.device(device)
    
    logger.info(f"Loading PyTorch model from {model_path} on {load_device}...")
    
    # Load saved data
    save_data = torch.load(model_path, map_location=load_device)
    config = save_data['config']
    state_dict = save_data['state_dict']
    
    # Create model
    model = PureBadassRewardModel(
        ensemble_size=config['ensemble_size'],
        window_size=config['window_size'],
        freeze_backbone=config['freeze_backbone'],
        backbone_type=config['backbone_type'],
        state_dim=config['state_dim'],
        bounded_temperature=config['bounded_temperature'],
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(load_device)
    
    logger.info(f"✅ Model loaded successfully on {load_device}")
    return model


def export_to_jit(model: pl.LightningModule, path: Path, device: str = "auto"):
    """
    Export the model to TorchScript.
    
    Args:
        model: PyTorch Lightning model
        path: Path to save the exported model
        device: Device to trace on ("auto", "cuda", or "cpu"). 
                "auto" tries GPU first, falls back to CPU if GPU fails.
    """
    try:
        logger.info(f"Exporting model to {path}...")
        
        # Reconstruct pure model to strip Lightning artifacts
        hparams = model.hparams
        dropout = getattr(hparams, 'dropout', 0.1)
        export_model = PureBadassRewardModel(
            ensemble_size=int(hparams.ensemble_size),
            window_size=int(hparams.window_size),
            freeze_backbone=bool(hparams.freeze_backbone),
            backbone_type=str(hparams.backbone_type),
            state_dim=int(hparams.state_dim),
            bounded_temperature=float(hparams.bounded_temperature),
            head_dropout=float(dropout),
        )
        
        # Load weights from the trained model
        state_dict = model.state_dict()
        export_model.load_state_dict(state_dict, strict=False)
        export_model.eval()
        
        # Create wrapper for inference signature
        class InferenceWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, base_img, hand_img, state):
                return self.m.predict_reward_bounded({"base_img": base_img, "hand_img": hand_img, "state": state})

        wrapper = InferenceWrapper(export_model)
        
        # Determine trace device
        if device == "auto":
            # Try GPU first if available
            if torch.cuda.is_available():
                trace_device = torch.device("cuda")
                logger.info("[export] Attempting to trace on GPU...")
            else:
                trace_device = torch.device("cpu")
                logger.info("[export] GPU not available, tracing on CPU...")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            trace_device = torch.device("cuda")
            logger.info("[export] Tracing on GPU (requested)...")
        else:
            trace_device = torch.device("cpu")
            logger.info("[export] Tracing on CPU (requested)...")
        
        # Move model to trace device
        wrapper = wrapper.to(trace_device)
        
        # Create dummy inputs on the trace device
        dummy_base = torch.randn(1, hparams.window_size, 3, 224, 224, device=trace_device)
        dummy_hand = torch.randn(1, hparams.window_size, 3, 224, 224, device=trace_device)
        dummy_state = torch.randn(1, hparams.window_size, hparams.state_dim, device=trace_device)
        
        # Try tracing on the selected device
        trace_success = False
        try:
            # Verify the model works before tracing
            with torch.no_grad():
                _ = wrapper(dummy_base, dummy_hand, dummy_state)
            logger.info(f"[export] Model forward pass verified on {trace_device} before tracing")
            
            # Trace the model - this records the device state
            # check_trace=False because DINOv3's internal operations may vary slightly
            scripted_model = torch.jit.trace(wrapper, (dummy_base, dummy_hand, dummy_state), strict=False, check_trace=False)
            
            # Verify the traced model works
            with torch.no_grad():
                _ = scripted_model(dummy_base, dummy_hand, dummy_state)
            logger.info(f"[export] Traced model verified on {trace_device}")
            trace_success = True
            
        except Exception as e:
            if device == "auto" and trace_device.type == "cuda":
                # Fallback to CPU if GPU tracing failed
                logger.warning(f"[export] GPU tracing failed: {e}")
                logger.info("[export] Falling back to CPU tracing...")
                trace_device = torch.device("cpu")
                wrapper = wrapper.cpu()
                dummy_base = dummy_base.cpu()
                dummy_hand = dummy_hand.cpu()
                dummy_state = dummy_state.cpu()
                
                # Retry on CPU
                with torch.no_grad():
                    _ = wrapper(dummy_base, dummy_hand, dummy_state)
                logger.info("[export] Model forward pass verified on CPU before tracing")
                
                scripted_model = torch.jit.trace(wrapper, (dummy_base, dummy_hand, dummy_state), strict=False, check_trace=False)
                
                with torch.no_grad():
                    _ = scripted_model(dummy_base, dummy_hand, dummy_state)
                logger.info("[export] Traced model verified on CPU")
                trace_success = True
            else:
                # If not auto or already on CPU, raise the error
                logger.error(f"[export] Tracing failed on {trace_device}: {e}")
                raise RuntimeError(f"Cannot trace model on {trace_device}: {e}") from e
        
        if not trace_success:
            raise RuntimeError("Tracing failed on all devices")
        
        # Save the traced model
        torch.jit.save(scripted_model, path)
        device_str = "GPU" if trace_device.type == "cuda" else "CPU"
        logger.info(f"✅ Export successful: {path} (model traced and runs on {device_str})")
        return True
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Data & Training
    parser.add_argument('--data_root', type=str, default='data/lerobot_processed/PickCube-v1')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['bf16-mixed', '16-mixed', '32-true'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau'], 
                        help='Learning rate scheduler: cosine (CosineAnnealingLR) or plateau (ReduceLROnPlateau)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability for RewardHead (default: 0.1)')
    
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio for cosine scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--fp16_export', action='store_true', help='Export model in FP16')
    
    # Model config
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--unfreeze_backbone', action='store_true', default=False)
    parser.add_argument('--compile_heads', action='store_true', default=False)
    parser.add_argument('--no_compile_heads', action='store_true', default=False)
    
    # Dataset config
    parser.add_argument('--no_index_cache', action='store_true', default=False)
    parser.add_argument('--train_label_mode', type=str, default='proxy', choices=['proxy', 'time'])
    parser.add_argument('--val_label_mode', type=str, default='proxy', choices=['proxy', 'time'])
    parser.add_argument('--train_pair_sampling', type=str, default='uniform', choices=['uniform', 'biased_gap'])
    
    # Eval & Loss config
    parser.add_argument('--traj_eval_batch_size', type=int, default=32)
    parser.add_argument('--traj_eval_samples_per_bin', type=int, default=1)
    parser.add_argument('--traj_eval_tanh_temperature', type=float, default=2.0)
    parser.add_argument('--bounded_temperature', type=float, default=2.0)
    parser.add_argument('--aux_progress_weight', type=float, default=1.0)
    parser.add_argument('--range_push_weight', type=float, default=0.0)
    parser.add_argument('--range_push_threshold', type=float, default=0.8)
    parser.add_argument('--range_push_margin', type=float, default=0.9)
    
    args = parser.parse_args()
    pl.seed_everything(42)

    # Initialize Datasets
    train_dataset = LerobotRewardDataset(
        args.data_root, split='train', aug=True,
        cache_index=not args.no_index_cache,
        label_mode=args.train_label_mode,
        pair_sampling=args.train_pair_sampling,
    )
    val_dataset = LerobotRewardDataset(
        args.data_root, split='val', aug=False,
        cache_index=not args.no_index_cache,
        label_mode=args.val_label_mode,
        pair_sampling="biased_gap",
    )

    # Validate Data
    if len(val_dataset) == 0:
        logger.error("❌ Validation dataset is empty! Training cannot proceed properly.")
        raise RuntimeError("Empty validation set. Check data path or split logic.")

    # DataLoaders
    loader_kwargs = dict(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Initialize Model
    model = BadassRewardModel(
        ensemble_size=3,
        window_size=3,
        freeze_backbone=(not args.unfreeze_backbone),
        max_epochs=args.max_epochs,
        compile_heads=(args.compile_heads and (not args.no_compile_heads)),
        bounded_temperature=args.bounded_temperature,
        aux_progress_weight=args.aux_progress_weight,
        range_push_weight=args.range_push_weight,
        range_push_threshold=args.range_push_threshold,
        range_push_margin=args.range_push_margin,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        dropout=args.dropout,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
    )

    # Callbacks
    # Use val/loss instead of val/progress_corr because:
    # 1. val/progress_corr is unstable with small validation sets (only 3 samples)
    # 2. val/loss is more reliable and reflects overall model performance
    # 3. val/monotonicity is also a good alternative but requires more samples
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        mode='min',  # Lower loss is better
        filename='reward-{epoch:02d}-{val/loss:.4f}',
        save_top_k=3
    )
    
    traj_bins_callback = ValTrajectoryRewardBinsCallback(
        val_dataset=val_dataset,
        bins=10,
        samples_per_bin=args.traj_eval_samples_per_bin,
        batch_size=args.traj_eval_batch_size,
        tanh_temperature=args.traj_eval_tanh_temperature,
    )
    
    loss_summary_callback = LossSummaryCallback()

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_true" if args.gpus > 1 else "auto",
        precision=args.precision,
        log_every_n_steps=1,
        check_val_every_n_epoch=1, # Validate every epoch
        callbacks=[checkpoint_callback, traj_bins_callback, loss_summary_callback],
        gradient_clip_val=1.0,
        use_distributed_sampler=True,
        val_check_interval=1.0 
    )

    if is_rank0():
        logger.info(f"Starting training on {args.gpus} GPUs...")

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Export (Rank 0 only)
    if is_rank0():
        best_path = checkpoint_callback.best_model_path
        if best_path:
            save_dir = Path(best_path).parent.parent
            pytorch_path = save_dir / "reward_model.pt"
            export_to_pytorch(model, pytorch_path, use_half=args.fp16_export)
            
            
            logger.info("Evaluating PyTorch model...")
            metrics = evaluate_reward_model(pytorch_path, val_loader, device="auto", return_predictions=False, model_type="pytorch")
            print(f"[Final Test] Metrics: {metrics}")
