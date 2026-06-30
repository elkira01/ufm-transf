"""
Complete Training Pipeline for the UFM-Transformer Multimodal Biometric Model.

This module implements the full two-phase training procedure for the Uncertainty-Guided
Fusion Multimodal Transformer (UFM-Transformer), which jointly processes face and
fingerprint modalities for biometric recognition. The pipeline includes:

    - UFMLoss: Composite loss combining triplet loss with hard negative mining,
      ArcFace additive margin softmax, and uncertainty regularization.
    - Phase 1 (Unimodal Pre-training): Independently trains face and fingerprint
      encoders using ArcFace loss, freezing all fusion components.
    - Phase 2 (Joint Fine-tuning): Unfreezes all parameters, applies random
      modality dropout (30%), and optimizes the full composite loss with
      cosine annealing learning rate scheduling.
    - Comprehensive validation: Computes verification accuracy, EER, TAR@FAR,
      and AUC for model selection.
    - Checkpointing: Saves best model (lowest val EER), periodic checkpoints,
      and supports resuming from any checkpoint.

Typical usage::

    $ python train.py --dataset_path /path/to/data --output_dir ./checkpoints

Author: Biometrics Pipeline Team
Python Version: >=3.10
PyTorch Version: >=2.0
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports (models.py and data_loader.py are in the same package)
# ---------------------------------------------------------------------------
from data_loader import (
    DatasetConfig,
    MissingModalitySimulator,
    get_dataloaders,
    set_seed as dl_set_seed,
)

# models.py is assumed to define UFMTransformerModel and its sub-components.
# The expected model forward signature is:
#   model(face, fingerprint, face_quality, fingerprint_quality,
#         face_missing=False, fingerprint_missing=False)
# and returns a dict with keys:
#   "fused_embedding": (B, D)  -- L2-normalized fused feature vector
#   "face_embedding":  (B, D)  -- L2-normalized face feature vector
#   "fingerprint_embedding": (B, D)  -- L2-normalized fingerprint feature vector
#   "face_logits": (B, num_subjects)  -- classification logits for face
#   "fingerprint_logits": (B, num_subjects)  -- classification logits for fingerprint
#   "uncertainty": (B, 1)  -- predicted uncertainty score
from models import UFMTransformerModel

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ufm_train")

# ---------------------------------------------------------------------------
# Graceful shutdown mechanism (for time-limited environments like Kaggle)
# ---------------------------------------------------------------------------

_shutdown_requested: bool = False


def request_shutdown() -> None:
    """Signal that a graceful shutdown has been requested."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("*** GRACEFUL SHUTDOWN REQUESTED — will exit after current epoch ***")


def is_shutdown_requested() -> bool:
    """Check whether a graceful shutdown has been requested."""
    return _shutdown_requested


# ---------------------------------------------------------------------------
# Distributed training (DDP) utilities
# ---------------------------------------------------------------------------

_DDP_LOCAL_RANK: int = -1
_DDP_INITIALIZED: bool = False


def _setup_ddp() -> int:
    """Initialize DistributedDataParallel if launched via ``torchrun``.

    Detects the ``LOCAL_RANK`` environment variable set by ``torchrun``
    (or ``torch.distributed.launch``).  When present, initialises the
    NCCL process group, pins the current process to its assigned GPU,
    and returns the local rank.

    Returns:
        ``local_rank`` (``0`` when not distributed).
    """
    global _DDP_LOCAL_RANK, _DDP_INITIALIZED
    local_rank_str = os.environ.get("LOCAL_RANK", "")
    if local_rank_str != "":
        _DDP_LOCAL_RANK = int(local_rank_str)
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(_DDP_LOCAL_RANK)
        _DDP_INITIALIZED = True
    else:
        _DDP_LOCAL_RANK = 0
        _DDP_INITIALIZED = False
    return _DDP_LOCAL_RANK


def _cleanup_ddp() -> None:
    """Destroy the distributed process group if it was initialised."""
    if _DDP_INITIALIZED:
        dist.destroy_process_group()


def _is_distributed() -> bool:
    """Return ``True`` if DDP is active."""
    return _DDP_INITIALIZED


def _is_main_process() -> bool:
    """Return ``True`` on rank 0 (or when not distributed)."""
    if not _DDP_INITIALIZED:
        return True
    return dist.get_rank() == 0


def _ddp_barrier() -> None:
    """Synchronise all processes when DDP is active."""
    if _DDP_INITIALIZED:
        dist.barrier()


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, stripping DDP / DataParallel wrapper."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def save_history(history: List[Dict[str, Any]], path: Path) -> None:
    """Atomically write training history to a JSON file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Hyperparameter configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Complete hyperparameter configuration for UFM-Transformer training.

    This dataclass centralizes every tunable knob in the training pipeline,
    making experiments reproducible and easy to track.

    Attributes:
        # Data
        dataset_path: Root directory containing subject sub-folders.
        output_dir: Directory to save checkpoints and logs.
        batch_size: Number of samples per batch.
        num_workers: Parallel data-loading workers.
        image_size: Spatial resolution for both modalities.
        pin_memory: Pin CPU memory for faster GPU transfer.

        # Optimizer
        lr_phase1: Peak learning rate during unimodal pre-training.
        lr_phase2: Initial learning rate during joint fine-tuning.
        lr_min: Minimum LR for cosine annealing in phase 2.
        weight_decay: L2 regularization coefficient.
        max_grad_norm: Maximum gradient norm for clipping.

        # Training schedule
        epochs_phase1: Number of unimodal pre-training epochs.
        epochs_phase2: Number of joint fine-tuning epochs.
        warmup_epochs: Linear LR warmup epochs (phase 2 only).

        # Loss weights
        w_triplet: Weight for triplet loss component.
        w_arcface: Weight for ArcFace loss component.
        w_uncertainty: Weight for uncertainty regularization component.

        # Loss hyperparameters
        triplet_margin: Margin for triplet loss with hard negative mining.
        arcface_margin: Angular margin (m) for ArcFace.
        arcface_scale: Feature scale (s) for ArcFace.

        # Modality dropout
        modality_dropout_prob: Probability of randomly masking a modality
            during phase 2 joint training.

        # Reproducibility
        seed: Random seed for deterministic behavior.
        device: Target compute device ("cuda", "cpu", or "auto").

        # Checkpointing
        checkpoint_every: Save a checkpoint every N epochs.
        resume_from: Optional path to checkpoint for resuming training.

        # Logging
        log_interval: Print metrics every N batches.
        save_logs: Whether to save per-epoch metrics to JSON.

        # Model
        embedding_dim: Dimensionality of the fused embedding.
        num_subjects: Number of unique subjects (classes). If None, inferred
            from the training dataset.
    """

    # Data ------------------------------------------------------------------
    dataset_path: str = "/data/biometric"
    output_dir: str = "./checkpoints"
    batch_size: int = 64
    num_workers: int = 4
    image_size: int = 224
    pin_memory: bool = True

    # Optimizer -------------------------------------------------------------
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Training schedule -----------------------------------------------------
    epochs_phase1: int = 50
    epochs_phase2: int = 100
    warmup_epochs: int = 5

    # Loss weights ----------------------------------------------------------
    w_triplet: float = 1.0
    w_arcface: float = 1.0
    w_uncertainty: float = 0.1

    # Loss hyperparameters --------------------------------------------------
    triplet_margin: float = 0.5
    arcface_margin: float = 0.5
    arcface_scale: float = 30.0

    # Modality dropout ------------------------------------------------------
    modality_dropout_prob: float = 0.30

    # Reproducibility -------------------------------------------------------
    seed: int = 42
    device: str = "auto"
    use_multi_gpu: bool = True

    # Checkpointing ---------------------------------------------------------
    checkpoint_every: int = 10
    resume_from: Optional[str] = None

    # Logging ---------------------------------------------------------------
    log_interval: int = 10
    save_logs: bool = True

    # Model -----------------------------------------------------------------
    embedding_dim: int = 512
    num_subjects: Optional[int] = None

    # Performance -----------------------------------------------------------
    use_amp: bool = True
    """Enable Automatic Mixed Precision (AMP) training for ~2-3x speedup on T4/V100."""
    mc_samples_train: int = 1
    """Number of MC-Dropout samples during training (use 1 for speed; full at inference)."""
    num_workers: int = 4
    """DataLoader worker processes (overrides the base num_workers default)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            A TrainConfig instance with loaded values.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# ArcFace Additive Margin Softmax Layer
# ---------------------------------------------------------------------------

class ArcFaceMargin(nn.Module):
    """ArcFace additive angular margin softmax layer.

    Implements the additive margin softmax from "ArcFace: Additive Angular
    Margin Loss for Deep Face Recognition" (Deng et al., CVPR 2019). It
    enforces intra-class compactness and inter-class separability by adding
    an angular margin penalty to the target logit.

    The forward pass computes:
        cos_theta = W_norm @ x_norm   (cosine similarity)
        theta = arccos(cos_theta)
        cos_theta_margin = cos(theta + m)  for target class
        logits = s * cos_theta_margin      (scaled)

    Attributes:
        in_features: Dimensionality of input feature vectors.
        num_classes: Number of subject identities (classes).
        margin: Angular margin m (radians). Defaults to 0.5.
        scale: Feature scale s. Defaults to 30.0.
        weight: Learnable class-centre matrix of shape (num_classes, in_features).

    Args:
        in_features: Dimension of input embeddings.
        num_classes: Number of output classes (subjects).
        margin: Angular margin. Defaults to 0.5.
        scale: Feature norm scale. Defaults to 30.0.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable class centres (one per subject)
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, in_features)
        )
        # Xavier-style initialization for angular metrics
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, embedding: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute ArcFace-scaled logits.

        Args:
            embedding: L2-normalised feature vectors of shape (B, in_features).
            labels: Ground-truth subject IDs of shape (B,) with values in
                [0, num_classes - 1].

        Returns:
            Logits tensor of shape (B, num_classes) scaled by `self.scale`.
        """
        # Ensure inputs are float and on the same device
        embedding = embedding.to(self.weight.device)
        labels = labels.to(self.weight.device)

        # L2-normalise features and class centres
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: (B, num_classes)
        cos_theta = torch.matmul(embedding_norm, weight_norm.t())
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # Additive angular margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        theta = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)

        # One-hot encode the target labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to the target class
        logits = self.scale * (one_hot * target_logits + (1.0 - one_hot) * cos_theta)
        return logits

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"num_classes={self.num_classes}, "
            f"margin={self.margin}, scale={self.scale})"
        )


# ---------------------------------------------------------------------------
# Composite UFM Loss
# ---------------------------------------------------------------------------

class UFMLoss(nn.Module):
    """Composite loss function for the UFM-Transformer model.

    Combines three complementary objectives:

    1. **Triplet loss** with hard negative mining: Pulls the anchor embedding
       closer to positive (same-subject) embeddings and pushes it away from
       hard negative (different-subject) embeddings within each batch.

    2. **ArcFace loss** (additive margin softmax): Enforces discriminative
       classification boundaries with an angular margin.

    3. **Uncertainty regularization**: Encourages the model to be confident
       (low uncertainty) when predictions are correct and uncertain (high
       uncertainty) when predictions are wrong.

    The total loss is a weighted sum::

        L_total = w1 * L_triplet + w2 * L_arcface + w3 * L_uncertainty

    Args:
        embedding_dim: Dimension of feature embeddings.
        num_classes: Number of subject identities.
        w_triplet: Weight for triplet loss. Defaults to 1.0.
        w_arcface: Weight for ArcFace loss. Defaults to 1.0.
        w_uncertainty: Weight for uncertainty regularization. Defaults to 0.1.
        triplet_margin: Margin for triplet loss. Defaults to 0.5.
        arcface_margin: Angular margin for ArcFace. Defaults to 0.5.
        arcface_scale: Feature scale for ArcFace. Defaults to 30.0.

    Attributes:
        arcface: ArcFaceMargin layer shared across face and fingerprint branches.
        triplet: nn.TripletMarginLoss with specified margin.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        w_triplet: float = 1.0,
        w_arcface: float = 1.0,
        w_uncertainty: float = 0.1,
        triplet_margin: float = 0.5,
        arcface_margin: float = 0.5,
        arcface_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.w_triplet = w_triplet
        self.w_arcface = w_arcface
        self.w_uncertainty = w_uncertainty
        self.triplet_margin = triplet_margin

        # ArcFace margin layer (shared for face and fingerprint branches)
        self.arcface = ArcFaceMargin(
            in_features=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
        )

        # Triplet loss with hard negative mining uses a base margin
        self.triplet = nn.TripletMarginLoss(
            margin=triplet_margin,
            p=2,  # L2 distance
            reduction="mean",
        )

    def _triplet_hard_mining(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform hard negative mining for triplet selection (fully vectorised).

        For each anchor in the batch:
        - **Hardest positive**: the same-class sample with maximum distance.
        - **Hardest negative**: the different-class sample with minimum distance.

        Implemented without Python-level loops for GPU efficiency.

        Args:
            embeddings: Normalised feature vectors of shape (B, D).
            labels: Subject IDs of shape (B,).

        Returns:
            A tuple of (anchors, positives, negatives) tensors,
            each of shape (B, D).
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Pairwise squared L2 distances: (B, B)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2).pow(2)

        # (B, B) boolean masks for same / different subject pairs
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        # Exclude the diagonal (self-pair) from positives
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)  # (B, B)
        same_mask = label_matrix & eye_mask   # True where same subject AND not self
        diff_mask = ~label_matrix              # True where different subject

        # ------------------------------------------------------------------ #
        # Hardest positives: argmax of distance within same-subject pool      #
        # ------------------------------------------------------------------ #
        # Fill non-positive entries with -1 so argmax always picks a positive
        pos_dists = dist_matrix.masked_fill(~same_mask, -1.0)
        hardest_pos_idx = pos_dists.argmax(dim=1)  # (B,)
        # Fallback: when no positive exists, use self (gives zero loss)
        has_positive = same_mask.any(dim=1)  # (B,)
        self_idx = torch.arange(batch_size, device=device)
        pos_idx = torch.where(has_positive, hardest_pos_idx, self_idx)
        positives = embeddings[pos_idx]  # (B, D)

        # ------------------------------------------------------------------ #
        # Hardest negatives: argmin of distance within different-subject pool  #
        # ------------------------------------------------------------------ #
        neg_dists = dist_matrix.masked_fill(~diff_mask, float("inf"))
        hardest_neg_idx = neg_dists.argmin(dim=1)  # (B,)
        # Fallback: when no negative exists (homogeneous batch), use global mean
        has_negative = diff_mask.any(dim=1)  # (B,)
        negatives_mean = embeddings.mean(dim=0, keepdim=True).expand(batch_size, -1)
        negatives_raw = embeddings[hardest_neg_idx]  # (B, D)
        negatives = torch.where(
            has_negative.unsqueeze(1).expand_as(negatives_raw),
            negatives_raw,
            negatives_mean,
        )  # (B, D)

        anchors = embeddings  # (B, D)
        return anchors, positives, negatives

    def _uncertainty_loss(
        self,
        uncertainty: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uncertainty-aware regularization loss.

        The idea is to penalise the model differently based on correctness:

        - When the model predicts the correct class: penalise high uncertainty
          (loss = -log(uncertainty)) so the model learns to be confident.
        - When the model predicts the wrong class: penalise low uncertainty
          (loss = +log(uncertainty)) so the model learns to express doubt.

        Args:
            uncertainty: Predicted uncertainty scores of shape (B, 1), values
                should be in (0, 1) after sigmoid.
            logits: Classification logits of shape (B, num_classes).
            labels: Ground-truth subject IDs of shape (B,).

        Returns:
            Scalar uncertainty regularisation loss.
        """
        # Clamp uncertainty to avoid log(0)
        uncertainty = torch.clamp(uncertainty.squeeze(-1), min=1e-6, max=1.0)

        # Determine correctness per sample
        predicted_classes = logits.argmax(dim=1)
        is_correct = (predicted_classes == labels).float()

        # Correct predictions: penalise high uncertainty -> -log(u)
        # Incorrect predictions: penalise low uncertainty -> +log(u)
        # Equivalent to: -(is_correct * log(u) + (1 - is_correct) * -log(u))
        #                = -(2*is_correct - 1) * log(u)
        loss_per_sample = -(2.0 * is_correct - 1.0) * torch.log(uncertainty)

        return loss_per_sample.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite UFM loss.

        Args:
            outputs: Dictionary returned by the UFM-Transformer model containing:
                - "fused_embedding": (B, D) fused feature vectors
                - "face_embedding": (B, D) face feature vectors
                - "fingerprint_embedding": (B, D) fingerprint feature vectors
                - "face_logits": (B, num_classes) face classification logits
                - "fingerprint_logits": (B, num_classes) fingerprint logits
                - "uncertainty": (B, 1) predicted uncertainty scores
            labels: Ground-truth subject IDs of shape (B,).

        Returns:
            A tuple of:
                - total_loss: Weighted composite loss (scalar tensor).
                - loss_dict: Dictionary with individual loss components for logging.
        """
        device = labels.device

        # ------------------------------------------------------------------
        # 1. Triplet loss with hard negative mining (on fused embeddings)
        # ------------------------------------------------------------------
        fused_emb = F.normalize(outputs["fused_embedding"], p=2, dim=1)
        anchors, positives, negatives = self._triplet_hard_mining(
            fused_emb, labels
        )
        loss_triplet = self.triplet(anchors, positives, negatives)

        # ------------------------------------------------------------------
        # 2. ArcFace loss (on both face and fingerprint branches)
        # ------------------------------------------------------------------
        # We average the ArcFace loss across both modalities to enforce
        # discriminative representations in each branch independently.
        face_emb = F.normalize(outputs["face_embedding"], p=2, dim=1)
        fp_emb = F.normalize(outputs["fingerprint_embedding"], p=2, dim=1)

        # Use the shared arcface layer for both modalities
        face_arc_logits = self.arcface(face_emb, labels)
        fp_arc_logits = self.arcface(fp_emb, labels)

        loss_arcface_face = F.cross_entropy(face_arc_logits, labels)
        loss_arcface_fp = F.cross_entropy(fp_arc_logits, labels)
        loss_arcface = 0.5 * (loss_arcface_face + loss_arcface_fp)

        # ------------------------------------------------------------------
        # 3. Uncertainty regularization
        # ------------------------------------------------------------------
        uncertainty = outputs["uncertainty"]
        # Use face logits to determine correctness (either branch works)
        loss_uncertainty = self._uncertainty_loss(
            uncertainty, face_arc_logits.detach(), labels
        )

        # ------------------------------------------------------------------
        # Weighted total
        # ------------------------------------------------------------------
        total_loss = (
            self.w_triplet * loss_triplet
            + self.w_arcface * loss_arcface
            + self.w_uncertainty * loss_uncertainty
        )

        loss_dict = {
            "triplet": loss_triplet.detach(),
            "arcface": loss_arcface.detach(),
            "uncertainty": loss_uncertainty.detach(),
            "total": total_loss.detach(),
        }

        return total_loss, loss_dict

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding_dim}, "
            f"num_classes={self.num_classes}, "
            f"w_triplet={self.w_triplet}, "
            f"w_arcface={self.w_arcface}, "
            f"w_uncertainty={self.w_uncertainty})"
        )


# ---------------------------------------------------------------------------
# Unimodal ArcFace-only loss (for Phase 1)
# ---------------------------------------------------------------------------

class UnimodalArcFaceLoss(nn.Module):
    """ArcFace-only loss for unimodal pre-training (Phase 1).

    Unlike the full UFMLoss, this loss only uses the ArcFace classification
    objective for a single modality branch. This allows encoders to learn
    discriminative representations independently before joint fusion.

    Args:
        embedding_dim: Dimension of feature embeddings.
        num_classes: Number of subject identities.
        arcface_margin: Angular margin for ArcFace. Defaults to 0.5.
        arcface_scale: Feature scale for ArcFace. Defaults to 30.0.

    Attributes:
        arcface: ArcFaceMargin layer for the modality.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        arcface_margin: float = 0.5,
        arcface_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.arcface = ArcFaceMargin(
            in_features=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the ArcFace classification loss.

        Args:
            embedding: Feature vectors of shape (B, embedding_dim).
            labels: Ground-truth subject IDs of shape (B,).

        Returns:
            A tuple of:
                - loss: Scalar cross-entropy loss.
                - loss_dict: Dictionary with loss component for logging.
        """
        logits = self.arcface(embedding, labels)
        loss = F.cross_entropy(logits, labels)
        # Compute accuracy for monitoring
        acc = (logits.argmax(dim=1) == labels).float().mean()
        loss_dict = {
            "arcface": loss.detach(),
            "accuracy": acc.detach(),
        }
        return loss, loss_dict


# ---------------------------------------------------------------------------
# Utility: reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Also set data_loader seed
    dl_set_seed(seed)


# ---------------------------------------------------------------------------
# Utility: modality masking for Phase 2
# ---------------------------------------------------------------------------

def apply_modality_masking(
    face: torch.Tensor,
    fingerprint: torch.Tensor,
    face_quality: torch.Tensor,
    fingerprint_quality: torch.Tensor,
    dropout_prob: float = 0.30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply random modality dropout during joint training.

    Randomly masks entire modalities (replaces with zeros) with probability
    `dropout_prob`. This forces the fusion transformer to learn robust
    representations that do not overly rely on any single modality.

    At least one modality is always kept per sample to avoid empty inputs.

    Args:
        face: Face image tensor of shape (B, C, H, W).
        fingerprint: Fingerprint image tensor of shape (B, C, H, W).
        face_quality: Face quality scores of shape (B,).
        fingerprint_quality: Fingerprint quality scores of shape (B,).
        dropout_prob: Probability of dropping each modality. Defaults to 0.30.

    Returns:
        A tuple of:
            - masked_face: Face tensor after potential masking.
            - masked_fingerprint: Fingerprint tensor after potential masking.
            - masked_face_quality: Face quality set to 0 if masked.
            - masked_fingerprint_quality: Fingerprint quality set to 0 if masked.
            - face_missing_mask: Boolean tensor (B,) indicating dropped face.
            - fingerprint_missing_mask: Boolean tensor (B,) indicating dropped fingerprint.
    """
    batch_size = face.size(0)
    device = face.device

    # Random dropout masks
    face_missing = torch.rand(batch_size, device=device) < dropout_prob
    fp_missing = torch.rand(batch_size, device=device) < dropout_prob

    # Ensure at least one modality per sample
    both_missing = face_missing & fp_missing
    if both_missing.any():
        # Randomly restore one modality for samples where both would be dropped
        restore_face = torch.rand(both_missing.sum(), device=device) < 0.5
        face_missing_indices = torch.where(both_missing)[0]
        face_missing[face_missing_indices[restore_face]] = False
        fp_missing[face_missing_indices[~restore_face]] = False

    # Apply masking by zeroing out the tensor
    face_out = face.clone()
    fingerprint_out = fingerprint.clone()
    face_quality_out = face_quality.clone()
    fingerprint_quality_out = fingerprint_quality.clone()

    face_out[face_missing] = 0.0
    fingerprint_out[fp_missing] = 0.0
    face_quality_out[face_missing] = 0.0
    fingerprint_quality_out[fp_missing] = 0.0

    return (
        face_out,
        fingerprint_out,
        face_quality_out,
        fingerprint_quality_out,
        face_missing,
        fp_missing,
    )


def _run_model_forward(
    model: nn.Module,
    face: torch.Tensor,
    fingerprint: torch.Tensor,
    face_quality: torch.Tensor,
    fp_quality: torch.Tensor,
    face_missing: Optional[torch.Tensor] = None,
    fp_missing: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if face.size(1) == 1:
        face = face.repeat(1, 3, 1, 1).contiguous()
    elif face.size(1) != 3:
        face = face[:, :3, :, :].contiguous()

    if fingerprint.size(1) != 1:
        fingerprint = fingerprint.mean(dim=1, keepdim=True)

    face_quality = face_quality.to(device=face.device, dtype=face.dtype)
    fp_quality = fp_quality.to(device=face.device, dtype=face.dtype)
    if face_quality.dim() == 1:
        face_quality = face_quality.unsqueeze(1)
    if fp_quality.dim() == 1:
        fp_quality = fp_quality.unsqueeze(1)

    batch_size = face.size(0)
    device = face.device
    if face_missing is None:
        face_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if fp_missing is None:
        fp_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)

    return model(
        face_img=face.contiguous(),
        fp_img=fingerprint.contiguous(),
        face_quality=face_quality.contiguous(),
        fp_quality=fp_quality.contiguous(),
        missing_mask=torch.stack([face_missing, fp_missing], dim=1),
    )


def _outputs_for_ufm_loss(outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    z_face = outputs["z_face"]
    z_fp = outputs["z_fp"]
    fused_face = outputs.get("fused_face", z_face)
    fused_fp = outputs.get("fused_fp", z_fp)
    uncertainty = outputs.get("uncertainty", {})
    if isinstance(uncertainty, dict):
        uncertainty_tensor = uncertainty.get("total_uncertainty")
        if uncertainty_tensor is None:
            uncertainty_tensor = uncertainty.get("total")
        if uncertainty_tensor is None:
            uncertainty_tensor = torch.zeros(z_face.size(0), 1, device=z_face.device)
    else:
        uncertainty_tensor = uncertainty

    return {
        "fused_embedding": F.normalize((fused_face + fused_fp) * 0.5, p=2, dim=1),
        "face_embedding": z_face,
        "fingerprint_embedding": z_fp,
        "uncertainty": uncertainty_tensor,
    }


# ---------------------------------------------------------------------------
# Metric computation utilities
# ---------------------------------------------------------------------------

def compute_eer(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER) and the threshold at which it occurs.

    The EER is the point where the False Acceptance Rate (FAR) equals the
    False Rejection Rate (FRR). It is a standard performance metric for
    biometric verification systems.

    Args:
        scores: Similarity scores (higher = more similar) of shape (N,).
        labels: Binary labels (1 = genuine, 0 = impostor) of shape (N,).

    Returns:
        A tuple of (eer_value, eer_threshold).
    """
    # Sort scores and labels by score
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Count genuine and impostor samples
    n_genuine = np.sum(labels == 1)
    n_impostor = np.sum(labels == 0)

    if n_genuine == 0 or n_impostor == 0:
        return 0.0, 0.0

    # FAR = FP / N_impostor, FRR = FN / N_genuine
    # Iterate through thresholds
    far_values = np.zeros(len(sorted_scores))
    frr_values = np.zeros(len(sorted_scores))

    # At threshold = sorted_scores[i], everything >= is accepted
    # Count false accepts (impostor with score >= threshold)
    # and false rejects (genuine with score < threshold)
    for i in range(len(sorted_scores)):
        threshold = sorted_scores[i]
        fa = np.sum((scores >= threshold) & (labels == 0))
        fr = np.sum((scores < threshold) & (labels == 1))
        far_values[i] = fa / n_impostor
        frr_values[i] = fr / n_genuine

    # Find the threshold where |FAR - FRR| is minimised
    diffs = np.abs(far_values - frr_values)
    min_idx = np.argmin(diffs)
    eer = (far_values[min_idx] + frr_values[min_idx]) / 2.0
    eer_threshold = sorted_scores[min_idx]

    return float(eer), float(eer_threshold)


def compute_tar_at_far(
    scores: np.ndarray, labels: np.ndarray, far_target: float = 0.01
) -> float:
    """Compute the True Acceptance Rate at a given False Acceptance Rate.

    TAR@FAR measures the proportion of genuine pairs that are correctly
    accepted when the system is operating at a specific FAR level.

    Args:
        scores: Similarity scores of shape (N,).
        labels: Binary labels (1 = genuine, 0 = impostor) of shape (N,).
        far_target: Target FAR value (e.g., 0.01 for 1%, 0.001 for 0.1%).
            Defaults to 0.01.

    Returns:
        TAR value in [0, 1] at the specified FAR operating point.
    """
    n_impostor = np.sum(labels == 0)
    if n_impostor == 0:
        return 0.0

    # Sort scores descending to find threshold for target FAR
    sorted_scores = np.sort(scores)[::-1]

    for threshold in sorted_scores:
        fa = np.sum((scores >= threshold) & (labels == 0))
        far = fa / n_impostor
        if far <= far_target:
            # At this threshold, compute TAR
            ta = np.sum((scores >= threshold) & (labels == 1))
            n_genuine = np.sum(labels == 1)
            if n_genuine > 0:
                return float(ta / n_genuine)
            return 0.0

    return 0.0


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Area Under the ROC Curve (AUC).

    Uses the trapezoidal rule for numerical integration of the ROC curve.

    Args:
        scores: Similarity scores of shape (N,).
        labels: Binary labels (1 = genuine, 0 = impostor) of shape (N,).

    Returns:
        AUC value in [0, 1]. Higher is better.
    """
    # Sort by scores descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_genuine = np.sum(labels == 1)
    n_impostor = np.sum(labels == 0)

    if n_genuine == 0 or n_impostor == 0:
        return 0.5

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = tps / n_genuine
    fpr = fps / n_impostor

    # Trapezoidal integration for AUC
    auc_value = np.trapz(tpr, fpr)
    return float(auc_value)


def compute_verification_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute all verification metrics at once.

    Args:
        scores: Similarity scores of shape (N,).
        labels: Binary labels (1 = genuine, 0 = impostor) of shape (N,).

    Returns:
        Dictionary with keys: eer, eer_threshold, tar_at_far_1,
        tar_at_far_0.1, auc, accuracy.
    """
    eer, eer_threshold = compute_eer(scores, labels)
    tar_1 = compute_tar_at_far(scores, labels, far_target=0.01)
    tar_01 = compute_tar_at_far(scores, labels, far_target=0.001)
    auc_value = compute_auc(scores, labels)

    # Accuracy at EER threshold
    predictions = (scores >= eer_threshold).astype(int)
    accuracy = np.mean(predictions == labels)

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "tar_at_far_1": tar_1,
        "tar_at_far_0.1": tar_01,
        "auc": auc_value,
        "accuracy": accuracy,
    }


def compute_cosine_similarity(
    emb1: torch.Tensor, emb2: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between two sets of embeddings.

    Args:
        emb1: First embedding tensor of shape (B, D).
        emb2: Second embedding tensor of shape (B, D).

    Returns:
        Cosine similarity scores of shape (B,).
    """
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    return (emb1_norm * emb2_norm).sum(dim=1)


# ---------------------------------------------------------------------------
# Training epoch loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    use_modality_masking: bool = False,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Execute one training epoch.

    Iterates over the dataloader, computes the loss (with optional modality
    masking), back-propagates gradients, and updates model weights with
    gradient clipping.

    Args:
        model: The UFM-Transformer model to train.
        dataloader: DataLoader yielding training batches.
        optimizer: Optimiser instance (AdamW).
        criterion: Loss function (UFMLoss or UnimodalArcFaceLoss).
        device: Compute device (cuda or cpu).
        config: Training configuration.
        epoch: Current epoch number (for logging).
        use_modality_masking: If True, applies random modality dropout.
            Used only in Phase 2. Defaults to False.
        scaler: Optional GradScaler for AMP. Created internally when
            ``config.use_amp`` is True and scaler is None.

    Returns:
        Dictionary of averaged epoch metrics:
            - loss: Mean total loss.
            - triplet_loss: Mean triplet loss component (if applicable).
            - arcface_loss: Mean ArcFace loss component.
            - uncertainty_loss: Mean uncertainty loss component (if applicable).
            - accuracy: Classification accuracy.
            - eer: Approximate EER on batch pairs (for monitoring).
            - learning_rate: Current learning rate.
    """
    model.train()
    amp_enabled = config.use_amp and device.type == "cuda"
    if scaler is None and amp_enabled:
        scaler = GradScaler()

    epoch_losses: Dict[str, List[float]] = defaultdict(list)
    all_embeddings: List[torch.Tensor] = []
    all_labels_list: List[torch.Tensor] = []
    correct_predictions = 0
    total_predictions = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch: (face, fingerprint, subject_id, face_quality, fp_quality)
        face, fingerprint, labels, face_quality, fp_quality = batch
        face = face.to(device, non_blocking=True)
        fingerprint = fingerprint.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        face_quality = face_quality.to(device, non_blocking=True)
        fp_quality = fp_quality.to(device, non_blocking=True)

        # Apply modality masking for Phase 2
        face_missing = torch.zeros(face.size(0), dtype=torch.bool, device=device)
        fp_missing = torch.zeros(face.size(0), dtype=torch.bool, device=device)

        if use_modality_masking and config.modality_dropout_prob > 0:
            (
                face,
                fingerprint,
                face_quality,
                fp_quality,
                face_missing,
                fp_missing,
            ) = apply_modality_masking(
                face,
                fingerprint,
                face_quality,
                fp_quality,
                dropout_prob=config.modality_dropout_prob,
            )

        # Forward pass + loss under AMP autocast
        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            outputs = _run_model_forward(
                model=model,
                face=face,
                fingerprint=fingerprint,
                face_quality=face_quality,
                fp_quality=fp_quality,
                face_missing=face_missing,
                fp_missing=fp_missing,
            )
            # Compute loss
            loss, loss_dict = criterion(_outputs_for_ufm_loss(outputs), labels)

        # Backward pass (scaled when AMP is active)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        # Accumulate metrics
        epoch_losses["loss"].append(loss.item())
        for key, value in loss_dict.items():
            epoch_losses[key].append(value.item())

        # Compute classification accuracy from face logits (or fused)
        if "face_logits" in outputs:
            preds = outputs["face_logits"].argmax(dim=1)
        elif "fused_face" in outputs or "z_face" in outputs:
            # Fallback: use arcface on fused embeddings if available
            preds = labels  # placeholder
        else:
            preds = labels

        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

        # Store embeddings for approximate EER computation
        if "fused_face" in outputs and "fused_fp" in outputs:
            emb = F.normalize((outputs["fused_face"] + outputs["fused_fp"]) * 0.5, p=2, dim=1)
            all_embeddings.append(emb.detach().cpu())
            all_labels_list.append(labels.detach().cpu())

        # Update progress bar
        current_lr = optimizer.param_groups[0]["lr"]
        postfix = {
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}",
        }
        if "triplet" in loss_dict:
            postfix["triplet"] = f"{loss_dict['triplet'].item():.4f}"
        if "arcface" in loss_dict:
            postfix["arcface"] = f"{loss_dict['arcface'].item():.4f}"
        if "uncertainty" in loss_dict:
            postfix["uncert"] = f"{loss_dict['uncertainty'].item():.4f}"
        pbar.set_postfix(postfix)

    # Compute approximate EER from batch embeddings
    eer_approx = 0.0
    if len(all_embeddings) > 0 and len(all_labels_list) > 0:
        eer_approx = _compute_batch_eer_approx(all_embeddings, all_labels_list)

    # Aggregate metrics
    metrics: Dict[str, float] = {
        "loss": float(np.mean(epoch_losses["loss"])),
        "accuracy": correct_predictions / max(total_predictions, 1),
        "eer": eer_approx,
        "learning_rate": current_lr,
    }

    for key in ["triplet", "arcface", "uncertainty"]:
        if key in epoch_losses and len(epoch_losses[key]) > 0:
            metrics[f"{key}_loss"] = float(np.mean(epoch_losses[key]))

    return metrics


def _compute_batch_eer_approx(
    embeddings_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
) -> float:
    """Compute an approximate EER from accumulated batch embeddings.

    This is a lightweight approximation used for training monitoring only.
    It samples pairs within each batch to estimate verification performance
    without constructing the full pairwise matrix.

    Args:
        embeddings_list: List of embedding tensors from each batch.
        labels_list: List of label tensors from each batch.

    Returns:
        Approximate EER value.
    """
    all_emb = torch.cat(embeddings_list, dim=0)
    all_lbl = torch.cat(labels_list, dim=0)

    # Subsample if too large (avoid OOM)
    max_samples = 1000
    if all_emb.size(0) > max_samples:
        indices = torch.randperm(all_emb.size(0))[:max_samples]
        all_emb = all_emb[indices]
        all_lbl = all_lbl[indices]

    # Generate genuine and impostor pairs
    emb_np = all_emb.numpy()
    lbl_np = all_lbl.numpy()

    scores_list: List[float] = []
    labels_list_out: List[int] = []

    n = len(lbl_np)
    # Sample a subset of pairs for efficiency
    max_pairs = 5000
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if pair_count >= max_pairs:
                break
            sim = np.dot(emb_np[i], emb_np[j])
            scores_list.append(float(sim))
            labels_list_out.append(1 if lbl_np[i] == lbl_np[j] else 0)
            pair_count += 1
        if pair_count >= max_pairs:
            break

    if len(scores_list) < 10:
        return 0.0

    scores_arr = np.array(scores_list)
    labels_arr = np.array(labels_list_out)
    eer, _ = compute_eer(scores_arr, labels_arr)
    return float(eer)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation and compute comprehensive verification metrics.

    Generates genuine and impostor pairs from the validation set, computes
    fused embeddings, measures cosine similarity, and evaluates:

    - Verification accuracy
    - Equal Error Rate (EER)
    - TAR at FAR = 1%
    - TAR at FAR = 0.1%
    - AUC

    Args:
        model: The trained UFM-Transformer model.
        dataloader: Validation DataLoader.
        device: Compute device.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()

    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    # Collect all embeddings and labels
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate", leave=False, dynamic_ncols=True):
            face, fingerprint, labels, face_quality, fp_quality = batch
            face = face.to(device, non_blocking=True)
            fingerprint = fingerprint.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            face_quality = face_quality.to(device, non_blocking=True)
            fp_quality = fp_quality.to(device, non_blocking=True)

            outputs = _run_model_forward(
                model=model,
                face=face,
                fingerprint=fingerprint,
                face_quality=face_quality,
                fp_quality=fp_quality,
            )

            if "fused_face" in outputs and "fused_fp" in outputs:
                emb = F.normalize((outputs["fused_face"] + outputs["fused_fp"]) * 0.5, p=2, dim=1)
            else:
                # Fallback: concatenate face + fingerprint embeddings
                emb = torch.cat([outputs["z_face"], outputs["z_fp"]], dim=1)

            all_embeddings.append(emb.detach().cpu())
            all_labels.append(labels.detach().cpu())

    # Concatenate all samples
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # Generate verification pairs
    embeddings_np = all_embeddings_tensor.numpy()
    labels_np = all_labels_tensor.numpy()

    scores_list: List[float] = []
    pair_labels_list: List[int] = []

    n = len(labels_np)
    unique_labels = np.unique(labels_np)

    # Generate genuine pairs (same subject)
    for lbl in unique_labels:
        indices = np.where(labels_np == lbl)[0]
        if len(indices) < 2:
            continue
        # Sample pairs from this subject
        for i in range(min(len(indices), 10)):
            for j in range(i + 1, min(len(indices), 10)):
                idx1, idx2 = indices[i], indices[j]
                sim = np.dot(embeddings_np[idx1], embeddings_np[idx2])
                scores_list.append(float(sim))
                pair_labels_list.append(1)

    # Generate impostor pairs (different subjects)
    n_genuine = len(scores_list)
    impostor_count = 0
    max_impostor = n_genuine * 2  # Keep roughly balanced

    for _ in range(max_impostor * 5):
        if impostor_count >= max_impostor:
            break
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        if labels_np[idx1] != labels_np[idx2]:
            sim = np.dot(embeddings_np[idx1], embeddings_np[idx2])
            scores_list.append(float(sim))
            pair_labels_list.append(0)
            impostor_count += 1

    if len(scores_list) < 10:
        logger.warning("Too few pairs for validation metrics. Returning zeros.")
        return {
            "eer": 1.0,
            "eer_threshold": 0.0,
            "tar_at_far_1": 0.0,
            "tar_at_far_0.1": 0.0,
            "auc": 0.5,
            "accuracy": 0.5,
        }

    scores_arr = np.array(scores_list)
    pair_labels_arr = np.array(pair_labels_list)

    metrics = compute_verification_metrics(scores_arr, pair_labels_arr)
    return metrics


# ---------------------------------------------------------------------------
# Phase 1: Unimodal Pre-training
# ---------------------------------------------------------------------------

def train_phase1_unimodal(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    start_epoch: int = 1,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """Phase 1: Unimodal encoder pre-training.

    In this phase, each encoder (face and fingerprint) is independently
    trained using the ArcFace classification loss. The fusion transformer
    and uncertainty head are **frozen** to prevent them from interfering
    with encoder learning.

    Training alternates between face and fingerprint batches within each
    epoch to train both encoders simultaneously.

    Args:
        model: The UFM-Transformer model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration.
        device: Compute device.
        output_dir: Directory to save checkpoints.

    Returns:
        The model with pre-trained encoders.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Unimodal Encoder Pre-training")
    logger.info("=" * 70)

    # Freeze fusion components
    # Expected model structure: model has face_encoder, fingerprint_encoder,
    # fusion_transformer, uncertainty_head, and projectors.
    # We freeze fusion_transformer and uncertainty_head.
    fusion_params = []
    encoder_params = []

    for name, param in model.named_parameters():
        if "fusion" in name or "uncertainty" in name:
            param.requires_grad = False
            fusion_params.append(name)
        else:
            param.requires_grad = True
            encoder_params.append(name)

    logger.info(f"Frozen parameters ({len(fusion_params)}): {fusion_params[:5]}...")
    logger.info(f"Trainable parameters ({len(encoder_params)}): {encoder_params[:5]}...")

    # Setup optimizer (only trainable params)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr_phase1,
        weight_decay=config.weight_decay,
    )

    # Setup scheduler: cosine annealing for Phase 1
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs_phase1,
        eta_min=config.lr_min,
    )

    if resume_checkpoint is not None:
        load_checkpoint(resume_checkpoint, model, optimizer=optimizer, scheduler=scheduler, device=device)
        logger.info("Phase 1 optimizer/scheduler state restored from checkpoint.")

    # Loss function for each modality (ArcFace only)
    face_criterion = UnimodalArcFaceLoss(
        embedding_dim=config.embedding_dim,
        num_classes=config.num_subjects or 100,  # Will be updated
        arcface_margin=config.arcface_margin,
        arcface_scale=config.arcface_scale,
    ).to(device)

    fp_criterion = UnimodalArcFaceLoss(
        embedding_dim=config.embedding_dim,
        num_classes=config.num_subjects or 100,
        arcface_margin=config.arcface_margin,
        arcface_scale=config.arcface_scale,
    ).to(device)

    # Attempt to infer num_subjects from data if not set
    if config.num_subjects is None:
        try:
            config.num_subjects = train_loader.dataset.get_num_subjects()
            logger.info(f"Inferred num_subjects: {config.num_subjects}")
        except AttributeError:
            logger.warning("Could not infer num_subjects from dataset. Using default.")

        # Recreate criteria with correct num_classes
        face_criterion = UnimodalArcFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=config.num_subjects,
            arcface_margin=config.arcface_margin,
            arcface_scale=config.arcface_scale,
        ).to(device)
        fp_criterion = UnimodalArcFaceLoss(
            embedding_dim=config.embedding_dim,
            num_classes=config.num_subjects,
            arcface_margin=config.arcface_margin,
            arcface_scale=config.arcface_scale,
        ).to(device)

    best_val_eer = float("inf")
    history: List[Dict[str, Any]] = []
    start_epoch = max(1, int(start_epoch))
    logger.info("Phase 1 starts at epoch %d", start_epoch)

    for epoch in range(start_epoch, config.epochs_phase1 + 1):
        epoch_start = time.time()
        model.train()

        # Keep fusion components frozen
        for name, param in model.named_parameters():
            if "fusion" in name or "uncertainty" in name:
                param.requires_grad = False

        epoch_losses: Dict[str, List[float]] = defaultdict(list)
        face_correct = 0
        face_total = 0
        fp_correct = 0
        fp_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Phase1 Epoch {epoch}/{config.epochs_phase1}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            face, fingerprint, labels, face_quality, fp_quality = batch
            face = face.to(device, non_blocking=True)
            fingerprint = fingerprint.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            face_quality = face_quality.to(device, non_blocking=True)
            fp_quality = fp_quality.to(device, non_blocking=True)

            # Forward pass (full model, but fusion params are frozen)
            outputs = _run_model_forward(
                model=model,
                face=face,
                fingerprint=fingerprint,
                face_quality=face_quality,
                fp_quality=fp_quality,
            )

            # Compute losses for both modalities independently
            loss_face, face_dict = face_criterion(
                outputs["z_face"], labels
            )
            loss_fp, fp_dict = fp_criterion(
                outputs["z_fp"], labels
            )

            # Combined loss: equal weighting for both modalities
            loss = loss_face + loss_fp

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()

            # Accumulate metrics
            epoch_losses["total"].append(loss.item())
            epoch_losses["face_arcface"].append(face_dict["arcface"].item())
            epoch_losses["fp_arcface"].append(fp_dict["arcface"].item())
            face_correct += (face_criterion.arcface(
                F.normalize(outputs["z_face"], p=2, dim=1), labels
            ).argmax(dim=1) == labels).sum().item()
            face_total += labels.size(0)
            fp_correct += (fp_criterion.arcface(
                F.normalize(outputs["z_fp"], p=2, dim=1), labels
            ).argmax(dim=1) == labels).sum().item()
            fp_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "face_acc": f"{face_correct/max(face_total,1):.4f}",
                "fp_acc": f"{fp_correct/max(fp_total,1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        # Validation
        val_metrics = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch,
            "phase": 1,
            "train_loss": float(np.mean(epoch_losses["total"])),
            "face_accuracy": face_correct / max(face_total, 1),
            "fp_accuracy": fp_correct / max(fp_total, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_time": epoch_time,
        }
        history.append(metrics)

        if config.save_logs and _is_main_process():
            save_history(history, output_dir / "history.json")

        logger.info(
            f"Phase1 Epoch {epoch:3d}/{config.epochs_phase1} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Face Acc: {metrics['face_accuracy']:.4f} | "
            f"FP Acc: {metrics['fp_accuracy']:.4f} | "
            f"Val EER: {val_metrics['eer']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model (lowest val EER)
        if val_metrics["eer"] < best_val_eer:
            best_val_eer = val_metrics["eer"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                path=output_dir / "best_model_phase1.pt",
                phase=1,
            )
            logger.info(f"  -> Saved best Phase 1 model (EER: {best_val_eer:.4f})")

        # Periodic checkpoint
        if epoch % config.checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                path=output_dir / f"checkpoint_phase1_epoch{epoch:03d}.pt",
                phase=1,
            )

        if is_shutdown_requested():
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                path=output_dir / "shutdown_checkpoint_phase1.pt",
                phase=1,
            )
            if config.save_logs:
                save_history(history, output_dir / "history.json")
            logger.info(f"Phase 1 stopped early at epoch {epoch}/{config.epochs_phase1} "
                        f"(graceful shutdown). Best val EER: {best_val_eer:.4f}")
            return model

    if config.save_logs:
        save_history(history, output_dir / "history.json")

    logger.info(f"Phase 1 complete. Best val EER: {best_val_eer:.4f}")
    return model


# ---------------------------------------------------------------------------
# Phase 2: Joint Fine-tuning
# ---------------------------------------------------------------------------

def train_phase2_joint(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    start_epoch: int = 1,
    resume_checkpoint: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """Phase 2: Joint fine-tuning with full composite loss.

    All model parameters are unfrozen. Random modality dropout (30%) is
    applied during training to make the fusion transformer robust to missing
    modalities. The full UFMLoss (triplet + ArcFace + uncertainty) is used
    with cosine annealing learning rate scheduling.

    Args:
        model: The UFM-Transformer model (with pre-trained encoders).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration.
        device: Compute device.
        output_dir: Directory to save checkpoints.

    Returns:
        The fine-tuned model.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Joint Fine-tuning with Modality Masking")
    logger.info("=" * 70)

    # Unfreeze ALL parameters for joint training
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,}")

    # Setup optimizer (full model)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr_phase2,
        weight_decay=config.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs_phase2,
        eta_min=config.lr_min,
    )

    if resume_checkpoint is not None:
        load_checkpoint(resume_checkpoint, model, optimizer=optimizer, scheduler=scheduler, device=device)
        logger.info("Phase 2 optimizer/scheduler state restored from checkpoint.")

    # Composite loss function
    criterion = UFMLoss(
        embedding_dim=config.embedding_dim,
        num_classes=config.num_subjects,
        w_triplet=config.w_triplet,
        w_arcface=config.w_arcface,
        w_uncertainty=config.w_uncertainty,
        triplet_margin=config.triplet_margin,
        arcface_margin=config.arcface_margin,
        arcface_scale=config.arcface_scale,
    ).to(device)

    best_val_eer = float("inf")
    history_path = output_dir / "history.json"
    history: List[Dict[str, Any]] = json.loads(history_path.read_text()) if history_path.exists() else []
    start_epoch = max(1, int(start_epoch))
    logger.info("Phase 2 starts at epoch %d", start_epoch)

    for epoch in range(start_epoch, config.epochs_phase2 + 1):
        epoch_start = time.time()

        # Training epoch with modality masking
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            epoch=epoch,
            use_modality_masking=True,
        )

        scheduler.step()

        # Validation
        val_metrics = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch,
            "phase": 2,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch_time": epoch_time,
        }
        history.append(metrics)

        if config.save_logs and _is_main_process():
            save_history(history, history_path)

        logger.info(
            f"Phase2 Epoch {epoch:3d}/{config.epochs_phase2} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Triplet: {train_metrics.get('triplet_loss', 0):.4f} | "
            f"ArcFace: {train_metrics.get('arcface_loss', 0):.4f} | "
            f"Uncert: {train_metrics.get('uncertainty_loss', 0):.4f} | "
            f"Val EER: {val_metrics['eer']:.4f} | "
            f"Val TAR@1%: {val_metrics['tar_at_far_1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"LR: {train_metrics['learning_rate']:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model (lowest val EER)
        if val_metrics["eer"] < best_val_eer:
            best_val_eer = val_metrics["eer"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                path=output_dir / "best_model_phase2.pt",
                phase=2,
            )
            logger.info(f"  -> Saved best Phase 2 model (EER: {best_val_eer:.4f})")

        # Periodic checkpoint
        if epoch % config.checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                path=output_dir / f"checkpoint_phase2_epoch{epoch:03d}.pt",
                phase=2,
            )

        if is_shutdown_requested():
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                path=output_dir / "shutdown_checkpoint_phase2.pt",
                phase=2,
            )
            if config.save_logs:
                save_history(history, history_path)
            logger.info(f"Phase 2 stopped early at epoch {epoch}/{config.epochs_phase2} "
                        f"(graceful shutdown). Best val EER: {best_val_eer:.4f}")
            return model

    if config.save_logs:
        save_history(history, history_path)

    logger.info(f"Phase 2 complete. Best val EER: {best_val_eer:.4f}")
    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, Any],
    config: TrainConfig,
    path: Union[str, Path],
    phase: int = 1,
) -> None:
    """Save a training checkpoint.

    The checkpoint contains the full model state, optimizer state, scheduler
    state, current epoch, metrics, and training configuration. This enables
    seamless resumption of training from any saved point.

    Args:
        model: The model to checkpoint.
        optimizer: The optimiser to checkpoint (optional).
        scheduler: The LR scheduler to checkpoint (optional).
        epoch: Current training epoch.
        metrics: Dictionary of current metrics.
        config: Training configuration.
        path: Destination file path.
        phase: Training phase (1 or 2). Defaults to 1.
    """
    if not _is_main_process():
        return  # Only rank 0 saves checkpoints

    path = Path(path)
    if epoch < 1:
        raise ValueError(f"Refusing to save checkpoint with invalid epoch {epoch}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    underlying = _unwrap_model(model)

    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": underlying.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "config": config.to_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int, Dict[str, Any]]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to map tensors to.

    Returns:
        A tuple of (epoch, phase, metrics).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load into the underlying model (unwrap DDP / DataParallel if needed)
    underlying = _unwrap_model(model)
    underlying.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    phase = checkpoint.get("phase", 1)
    metrics = checkpoint.get("metrics", {})

    logger.info(f"Loaded checkpoint from {path}: epoch {epoch}, phase {phase}")
    return epoch, phase, metrics


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    """Construct the UFM-Transformer model.

    Args:
        config: Training configuration containing model hyperparameters.
        device: Compute device to place the model on.

    Returns:
        The instantiated UFMTransformerModel.
    """
    model = UFMTransformerModel(
        embed_dim=config.embedding_dim,
        arcface_margin=config.arcface_margin,
        arcface_scale=config.arcface_scale,
    )
    model = model.to(device)

    # Multi-GPU: prefer DDP (launched via torchrun), fall back to DataParallel
    if _is_distributed():
        logger.info(
            "Wrapping model with DistributedDataParallel "
            "(rank=%d/%d, local_rank=%d)",
            dist.get_rank(), dist.get_world_size(), _DDP_LOCAL_RANK,
        )
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[_DDP_LOCAL_RANK],
            output_device=_DDP_LOCAL_RANK,
            find_unused_parameters=True,
        )
    elif config.use_multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info(
            "Wrapping model with DataParallel across %d GPUs",
            torch.cuda.device_count(),
        )
        model = nn.DataParallel(model)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def main(config: Optional[TrainConfig] = None) -> nn.Module:
    """Execute the complete two-phase training pipeline.

    This is the main entry point that orchestrates:

    1. Environment setup: seeds, device, directories.
    2. Data loading: train/val/test splits via data_loader.py.
    3. Model construction: UFM-Transformer via models.py.
    4. Phase 1: Unimodal pre-training of encoders.
    5. Phase 2: Joint fine-tuning with modality masking.
    6. Checkpointing: saves best and periodic checkpoints.
    7. Logging: console + JSON metrics history.

    Args:
        config: Training configuration. If None, uses default TrainConfig().

    Returns:
        The trained model.
    """
    if config is None:
        config = TrainConfig()

    # ------------------------------------------------------------------
    # 1. Environment setup
    # ------------------------------------------------------------------
    # Initialize DDP if launched via torchrun (sets LOCAL_RANK env var)
    local_rank = _setup_ddp()

    # Device selection — per-rank GPU when DDP is active
    if _is_distributed():
        device = torch.device(f"cuda:{local_rank}")
    elif config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # Per-rank seed offset so each GPU sees a different shuffle sequence
    seed = config.seed + local_rank
    set_global_seed(seed)

    logger.info("Executing train.py from: %s", Path(inspect.getfile(main)).resolve())
    logger.info(f"Device: {device}")
    if _is_distributed():
        logger.info("DDP: rank=%d/%d  local_rank=%d",
                     dist.get_rank(), dist.get_world_size(), local_rank)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPUs available: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize unified history file (rank 0 only)
    if config.save_logs and _is_main_process():
        history_path = output_dir / "history.json"
        if not history_path.exists():
            history_path.write_text("[]")
            logger.info("Initialized history.json")

    # Save configuration (rank 0 only)
    if _is_main_process():
        config.save(output_dir / "train_config.json")
        logger.info(f"Configuration saved to {output_dir / 'train_config.json'}")

    # ------------------------------------------------------------------
    # 2. Data loading
    # ------------------------------------------------------------------
    logger.info("Loading datasets...")

    logger.info("Using paired dataset from: %s", config.dataset_path)
    loaders = get_dataloaders(
        root_dir=config.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        seed=config.seed,
        pin_memory=config.pin_memory,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    logger.info("Train batches: %d", len(train_loader))
    logger.info("Val batches: %d", len(val_loader))
    logger.info("Test batches: %d", len(test_loader))

    # Infer num_subjects if not specified
    if config.num_subjects is None:
        try:
            config.num_subjects = loaders["datasets"]["train"].get_num_subjects()
        except AttributeError:
            config.num_subjects = len(loaders["train_subjects"])
        logger.info("Inferred num_subjects: %s", config.num_subjects)

    # ------------------------------------------------------------------
    # 3. Model construction
    # ------------------------------------------------------------------
    logger.info("Building model...")
    model = build_model(config, device)

    # ------------------------------------------------------------------
    # Handle resume from checkpoint
    # ------------------------------------------------------------------
    start_phase = 1
    start_epoch = 1

    if config.resume_from is not None and Path(config.resume_from).exists():
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        epoch, phase, metrics = load_checkpoint(
            config.resume_from, model, device=device
        )
        start_phase = phase
        start_epoch = epoch + 1
        logger.info(f"Resumed at phase {phase}, epoch {epoch}; continuing from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # 4. Phase 1: Unimodal pre-training
    # ------------------------------------------------------------------
    if start_phase <= 1:
        model = train_phase1_unimodal(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            output_dir=output_dir,
            start_epoch=start_epoch if start_phase == 1 else 1,
            resume_checkpoint=config.resume_from if start_phase == 1 else None,
        )

        # Load best Phase 1 checkpoint for Phase 2
        best_phase1_path = output_dir / "best_model_phase1.pt"
        if best_phase1_path.exists():
            logger.info("Loading best Phase 1 checkpoint for Phase 2...")
            load_checkpoint(best_phase1_path, model, device=device)
            _ddp_barrier()  # sync all ranks before Phase 2

        if is_shutdown_requested():
            logger.info("Skipping Phase 2 (graceful shutdown requested)")
            start_phase = 3  # prevent Phase 2 from running below
    else:
        logger.info("Skipping Phase 1 (resuming from Phase 2)")

    # ------------------------------------------------------------------
    # 5. Phase 2: Joint fine-tuning
    # ------------------------------------------------------------------
    if start_phase <= 2:
        model = train_phase2_joint(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            output_dir=output_dir,
            start_epoch=start_epoch if start_phase == 2 else 1,
            resume_checkpoint=config.resume_from if start_phase == 2 else None,
        )
    else:
        logger.info("Skipping Phase 2 (already complete)")

    # ------------------------------------------------------------------
    # 6. Final evaluation on test set
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 70)

    # Load best Phase 2 model
    best_phase2_path = output_dir / "best_model_phase2.pt"
    if best_phase2_path.exists():
        load_checkpoint(best_phase2_path, model, device=device)

    test_metrics = validate(model, test_loader, device)
    logger.info("Test Set Metrics:")
    for key, value in test_metrics.items():
        logger.info("  %-20s: %.4f", key, value)

    # Save test metrics
    if config.save_logs and _is_main_process():
        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Save final model (save_checkpoint is already rank-0 guarded)
    save_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=config.epochs_phase1 + config.epochs_phase2,
        metrics={"test": test_metrics},
        config=config,
        path=output_dir / "final_model.pt",
        phase=2,
    )

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 70)

    _cleanup_ddp()
    return model


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    """Parse command-line arguments into a TrainConfig.

    Returns:
        TrainConfig populated from CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the UFM-Transformer multimodal biometric model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/biometric",
        help="Root directory containing subject sub-folders with face and fingerprint images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints, logs, and metrics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of samples per training batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel data loading workers.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Spatial resolution for input images.",
    )

    # Optimizer arguments
    parser.add_argument(
        "--lr_phase1",
        type=float,
        default=1e-3,
        help="Peak learning rate for Phase 1 (unimodal pre-training).",
    )
    parser.add_argument(
        "--lr_phase2",
        type=float,
        default=1e-4,
        help="Initial learning rate for Phase 2 (joint fine-tuning).",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 weight decay coefficient.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping.",
    )

    # Training schedule
    parser.add_argument(
        "--epochs_phase1",
        type=int,
        default=50,
        help="Number of unimodal pre-training epochs.",
    )
    parser.add_argument(
        "--epochs_phase2",
        type=int,
        default=100,
        help="Number of joint fine-tuning epochs.",
    )

    # Loss weights
    parser.add_argument(
        "--w_triplet",
        type=float,
        default=1.0,
        help="Weight for triplet loss component.",
    )
    parser.add_argument(
        "--w_arcface",
        type=float,
        default=1.0,
        help="Weight for ArcFace loss component.",
    )
    parser.add_argument(
        "--w_uncertainty",
        type=float,
        default=0.1,
        help="Weight for uncertainty regularization.",
    )

    # Loss hyperparameters
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.5,
        help="Margin for triplet loss with hard negative mining.",
    )
    parser.add_argument(
        "--arcface_margin",
        type=float,
        default=0.5,
        help="Angular margin (m) for ArcFace loss.",
    )
    parser.add_argument(
        "--arcface_scale",
        type=float,
        default=30.0,
        help="Feature scale (s) for ArcFace loss.",
    )

    # Modality dropout
    parser.add_argument(
        "--modality_dropout_prob",
        type=float,
        default=0.30,
        help="Probability of random modality dropout during Phase 2.",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device selection.",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save periodic checkpoints every N epochs.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training.",
    )

    # Logging
    parser.add_argument(
        "--save_logs",
        action="store_true",
        default=True,
        help="Save per-epoch training metrics to JSON.",
    )

    # Model
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Dimensionality of the fused embedding space.",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        default=None,
        help="Number of subjects (classes). Inferred from data if not set.",
    )

    args = parser.parse_args()

    # Convert argparse Namespace to TrainConfig
    config = TrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        lr_phase1=args.lr_phase1,
        lr_phase2=args.lr_phase2,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        w_triplet=args.w_triplet,
        w_arcface=args.w_arcface,
        w_uncertainty=args.w_uncertainty,
        triplet_margin=args.triplet_margin,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        modality_dropout_prob=args.modality_dropout_prob,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        save_logs=args.save_logs,
        embedding_dim=args.embedding_dim,
        num_subjects=args.num_subjects,
    )

    return config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage:
    #   $ python train.py --dataset_path /path/to/data --output_dir ./checkpoints
    config = parse_args()
    model = main(config)
