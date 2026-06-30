"""
Multimodal Face + Fingerprint Biometric Data Loading and Preprocessing Pipeline.

This module provides a complete, production-ready data pipeline for multimodal
biometric recognition using face and fingerprint modalities. It includes:

    - MultimodalDataset: Loads paired face/fingerprint images per subject
    - Modality-specific augmentations for face and fingerprint images
    - Missing modality simulation for robust multimodal training
    - Pair generation for verification tasks (genuine + impostor pairs)
    - Heuristic quality estimation based on Laplacian variance and contrast
    - Subject-disjoint train/val/test splitting

Typical usage:
    >>> from data_loader import get_dataloaders
    >>> train_loader, val_loader, test_loader = get_dataloaders(
    ...     root_dir="/path/to/dataset", batch_size=32
    ... )
    >>> for face, fingerprint, subject_id, q_face, q_fp in train_loader:
    ...     # Training loop
    ...     pass

Author: Biometrics Pipeline Team
Python Version: >=3.10
PyTorch Version: >=2.0
"""

from __future__ import annotations

import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, torch, and random.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Custom Transform Classes
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Add Gaussian noise to a tensor image.

    This transform operates on tensor inputs (C, H, W) in the [0, 1] range.
    It adds zero-mean Gaussian noise with a configurable standard deviation,
    then clamps the result back to [0, 1].

    Args:
        std_min: Minimum standard deviation of the noise. Defaults to 0.01.
        std_max: Maximum standard deviation of the noise. Defaults to 0.05.
        p: Probability of applying the transform. Defaults to 0.5.

    Example:
        >>> transform = GaussianNoise(std_min=0.01, std_max=0.05, p=0.5)
        >>> noisy_tensor = transform(image_tensor)
    """

    def __init__(self, std_min: float = 0.01, std_max: float = 0.05, p: float = 0.5) -> None:
        if not (0 <= std_min <= std_max):
            raise ValueError("std_min must be <= std_max and both >= 0")
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        self.std_min = std_min
        self.std_max = std_max
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to the input tensor.

        Args:
            tensor: Input image tensor of shape (C, H, W) with values in [0, 1].

        Returns:
            Noisy tensor of same shape, clamped to [0, 1].
        """
        if random.random() < self.p:
            std = random.uniform(self.std_min, self.std_max)
            noise = torch.randn_like(tensor) * std
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std_min={self.std_min}, std_max={self.std_max}, p={self.p})"


class RandomOcclusionFace:
    """Apply random block occlusion to face images.

    Simulates real-world occlusion scenarios (e.g., masks, sunglasses, hands)
    by overlaying a random rectangular block on the image.

    Args:
        p: Probability of applying occlusion. Defaults to 0.3.
        min_size: Minimum occlusion block size as fraction of image. Defaults to 0.1.
        max_size: Maximum occlusion block size as fraction of image. Defaults to 0.4.
        fill_value: Value to fill the occluded region with. Defaults to 0 (black).

    Example:
        >>> occlude = RandomOcclusionFace(p=0.3, min_size=0.1, max_size=0.4)
        >>> occluded = occlude(face_tensor)
    """

    def __init__(
        self,
        p: float = 0.3,
        min_size: float = 0.1,
        max_size: float = 0.4,
        fill_value: float = 0.0,
    ) -> None:
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        if not (0 < min_size <= max_size < 1):
            raise ValueError("min_size and max_size must be in (0, 1)")
        self.p = p
        self.min_size = min_size
        self.max_size = max_size
        self.fill_value = fill_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random block occlusion to a face tensor.

        Args:
            tensor: Input face tensor of shape (C, H, W).

        Returns:
            Tensor with random block occlusion applied.
        """
        if random.random() >= self.p:
            return tensor

        _, h, w = tensor.shape
        # Random occlusion block size
        occ_h = int(random.uniform(self.min_size, self.max_size) * h)
        occ_w = int(random.uniform(self.min_size, self.max_size) * w)
        # Random top-left position
        top = random.randint(0, max(0, h - occ_h))
        left = random.randint(0, max(0, w - occ_w))

        tensor_occluded = tensor.clone()
        tensor_occluded[:, top : top + occ_h, left : left + occ_w] = self.fill_value
        return tensor_occluded

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, min_size={self.min_size}, "
            f"max_size={self.max_size}, fill_value={self.fill_value})"
        )


class RandomOcclusionFingerprint:
    """Apply random minutiae dropout occlusion to fingerprint images.

    Simulates poor ridge clarity by randomly dropping small circular regions,
    mimicking missing minutiae points in fingerprint scans.

    Args:
        p: Probability of applying occlusion. Defaults to 0.3.
        num_regions: Number of circular dropout regions. Defaults to (3, 10).
        radius: Radius of each dropout region in pixels. Defaults to (2, 8).
        fill_value: Fill value for occluded regions. Defaults to 0.

    Example:
        >>> occlude = RandomOcclusionFingerprint(p=0.3, num_regions=(3, 10), radius=(2, 8))
        >>> occluded = occlude(fp_tensor)
    """

    def __init__(
        self,
        p: float = 0.3,
        num_regions: Tuple[int, int] = (3, 10),
        radius: Tuple[int, int] = (2, 8),
        fill_value: float = 0.0,
    ) -> None:
        if not (0 <= p <= 1):
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.num_regions = num_regions
        self.radius = radius
        self.fill_value = fill_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply minutiae dropout to a fingerprint tensor.

        Args:
            tensor: Input fingerprint tensor of shape (C, H, W).

        Returns:
            Tensor with random minutiae dropout applied.
        """
        if random.random() >= self.p:
            return tensor

        c, h, w = tensor.shape
        tensor_occluded = tensor.clone()
        num_regions = random.randint(*self.num_regions)

        for _ in range(num_regions):
            radius = random.randint(*self.radius)
            center_y = random.randint(radius, h - radius - 1)
            center_x = random.randint(radius, w - radius - 1)

            # Create a circular mask
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            dist = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
            mask = dist <= radius
            tensor_occluded[:, mask] = self.fill_value

        return tensor_occluded

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, num_regions={self.num_regions}, "
            f"radius={self.radius}, fill_value={self.fill_value})"
        )


class RandomQualityDegradation:
    """Simulate real-world quality degradation on images.

    This composite transform randomly applies a sequence of degradation
    operations (blur, noise, occlusion) with configurable probabilities.
    It is modality-agnostic and works with both face and fingerprint tensors.

    Args:
        blur_prob: Probability of applying Gaussian blur. Defaults to 0.3.
        noise_prob: Probability of adding Gaussian noise. Defaults to 0.3.
        occlusion_prob: Probability of applying occlusion. Defaults to 0.2.
        blur_kernel_size: Kernel size for Gaussian blur. Defaults to (3, 7).
        noise_std_range: Standard deviation range for noise. Defaults to (0.01, 0.05).
        occlusion_transform: Optional custom occlusion transform. If None,
            uses a simple block occlusion.

    Example:
        >>> degrade = RandomQualityDegradation(blur_prob=0.3, noise_prob=0.3)
        >>> degraded = degrade(image_tensor)
    """

    def __init__(
        self,
        blur_prob: float = 0.3,
        noise_prob: float = 0.3,
        occlusion_prob: float = 0.2,
        blur_kernel_size: Tuple[int, int] = (3, 7),
        noise_std_range: Tuple[float, float] = (0.01, 0.05),
        occlusion_transform: Optional[Callable] = None,
    ) -> None:
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.occlusion_prob = occlusion_prob
        self.blur_kernel_size = blur_kernel_size
        self.noise_std_range = noise_std_range
        self.occlusion_transform = occlusion_transform

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random quality degradation to an image tensor.

        Args:
            tensor: Input image tensor of shape (C, H, W) in [0, 1].

        Returns:
            Degraded tensor of same shape.
        """
        degraded = tensor

        # Random Gaussian blur (applied via PIL for proper kernel behavior)
        if random.random() < self.blur_prob:
            kernel = random.choice(
                [k for k in range(self.blur_kernel_size[0], self.blur_kernel_size[1] + 1, 2)]
            )
            sigma = random.uniform(0.5, 2.0)
            # Convert to PIL, apply blur, convert back
            degraded_pil = TF.to_pil_image(degraded)
            degraded_pil = degraded_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
            degraded = TF.to_tensor(degraded_pil)

        # Random Gaussian noise
        if random.random() < self.noise_prob:
            std = random.uniform(*self.noise_std_range)
            noise = torch.randn_like(degraded) * std
            degraded = torch.clamp(degraded + noise, 0.0, 1.0)

        # Random occlusion
        if random.random() < self.occlusion_prob:
            if self.occlusion_transform is not None:
                degraded = self.occlusion_transform(degraded)
            else:
                # Simple block occlusion fallback
                _, h, w = degraded.shape
                occ_h = int(random.uniform(0.05, 0.2) * h)
                occ_w = int(random.uniform(0.05, 0.2) * w)
                top = random.randint(0, max(0, h - occ_h))
                left = random.randint(0, max(0, w - occ_w))
                degraded[:, top : top + occ_h, left : left + occ_w] = 0.0

        return degraded

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(blur_prob={self.blur_prob}, "
            f"noise_prob={self.noise_prob}, occlusion_prob={self.occlusion_prob})"
        )


# ---------------------------------------------------------------------------
# Missing Modality Simulation
# ---------------------------------------------------------------------------

class MissingModalitySimulator:
    """Simulate missing modalities during training.

    Randomly drops one of the available modalities (face or fingerprint) with
    a configurable probability. When a modality is missing, a zero tensor is
    returned along with a binary flag indicating the missing state.

    This enables training robust multimodal fusion networks that can handle
    incomplete inputs at inference time.

    Args:
        drop_face_prob: Probability of dropping the face modality. Defaults to 0.15.
        drop_fingerprint_prob: Probability of dropping the fingerprint modality.
            Defaults to 0.15.
        placeholder_value: Value to fill missing modalities with. Defaults to 0.0.

    Raises:
        ValueError: If drop probabilities are invalid or their sum >= 1.

    Example:
        >>> simulator = MissingModalitySimulator(drop_face_prob=0.15, drop_fingerprint_prob=0.15)
        >>> face, fingerprint, flags = simulator(face_tensor, fingerprint_tensor)
        >>> # flags is a dict: {"face_missing": bool, "fingerprint_missing": bool}
    """

    def __init__(
        self,
        drop_face_prob: float = 0.15,
        drop_fingerprint_prob: float = 0.15,
        placeholder_value: float = 0.0,
    ) -> None:
        if not (0 <= drop_face_prob <= 1):
            raise ValueError("drop_face_prob must be in [0, 1]")
        if not (0 <= drop_fingerprint_prob <= 1):
            raise ValueError("drop_fingerprint_prob must be in [0, 1]")
        if drop_face_prob + drop_fingerprint_prob >= 1.0:
            warnings.warn(
                f"Sum of drop probabilities ({drop_face_prob + drop_fingerprint_prob}) "
                f"is >= 1.0. This may cause both modalities to be dropped simultaneously."
            )

        self.drop_face_prob = drop_face_prob
        self.drop_fingerprint_prob = drop_fingerprint_prob
        self.placeholder_value = placeholder_value

    def __call__(
        self, face: torch.Tensor, fingerprint: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, bool]]:
        """Apply random modality dropout.

        Args:
            face: Face image tensor of shape (C, H, W).
            fingerprint: Fingerprint image tensor of shape (C, H, W).

        Returns:
            A tuple containing:
                - face tensor (original or placeholder)
                - fingerprint tensor (original or placeholder)
                - flags dict with keys "face_missing" and "fingerprint_missing"
        """
        face_missing = random.random() < self.drop_face_prob
        fingerprint_missing = random.random() < self.drop_fingerprint_prob

        # Ensure at least one modality is present during training
        if face_missing and fingerprint_missing:
            # Randomly keep one modality to avoid completely empty samples
            if random.random() < 0.5:
                face_missing = False
            else:
                fingerprint_missing = False

        face_out = (
            torch.full_like(face, self.placeholder_value) if face_missing else face
        )
        fingerprint_out = (
            torch.full_like(fingerprint, self.placeholder_value)
            if fingerprint_missing
            else fingerprint
        )

        flags = {
            "face_missing": face_missing,
            "fingerprint_missing": fingerprint_missing,
        }

        return face_out, fingerprint_out, flags

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(drop_face_prob={self.drop_face_prob}, "
            f"drop_fingerprint_prob={self.drop_fingerprint_prob})"
        )


# ---------------------------------------------------------------------------
# Quality Estimation (Placeholder Heuristic)
# ---------------------------------------------------------------------------

class QualityEstimator:
    """Heuristic quality estimator for biometric images.

    Estimates image quality using a combination of Laplacian variance (blur
    detection) and local contrast measures. Higher scores indicate better quality.

    This is a placeholder implementation that will be replaced by a learned
    quality estimation network in future iterations.

    The quality score is computed as a weighted combination:
        score = w1 * normalized_laplacian + w2 * normalized_contrast

    Both components are normalized to [0, 1] using reference values derived
    from typical biometric image statistics.

    Args:
        laplacian_weight: Weight for Laplacian variance component. Defaults to 0.6.
        contrast_weight: Weight for local contrast component. Defaults to 0.4.
        reference_laplacian_max: Reference max Laplacian variance for normalization.
            Defaults to 500.0.
        reference_contrast_max: Reference max contrast for normalization.
            Defaults to 100.0.

    Example:
        >>> estimator = QualityEstimator()
        >>> quality_score = estimator.estimate(tensor)  # float in [0, 1]
    """

    def __init__(
        self,
        laplacian_weight: float = 0.6,
        contrast_weight: float = 0.4,
        reference_laplacian_max: float = 500.0,
        reference_contrast_max: float = 100.0,
    ) -> None:
        if abs(laplacian_weight + contrast_weight - 1.0) > 1e-6:
            raise ValueError("laplacian_weight + contrast_weight must equal 1.0")

        self.laplacian_weight = laplacian_weight
        self.contrast_weight = contrast_weight
        self.reference_laplacian_max = reference_laplacian_max
        self.reference_contrast_max = reference_contrast_max

    def estimate(self, tensor: torch.Tensor) -> float:
        """Estimate the quality of a single image tensor.

        Args:
            tensor: Image tensor of shape (C, H, W) with values in [0, 1].

        Returns:
            Quality score in the range [0, 1]. Higher is better.
        """
        # Convert to numpy grayscale for OpenCV-style operations
        if tensor.shape[0] == 3:
            # RGB to grayscale: weighted average
            gray_np = (
                0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            ).numpy()
        else:
            gray_np = tensor[0].numpy()

        # Convert to [0, 255] range for Laplacian computation
        gray_np = (gray_np * 255).astype(np.float32)

        # 1. Laplacian variance (blur detection)
        # Higher variance = sharper image = better quality
        laplacian = cv2_laplacian(gray_np)
        laplacian_var = laplacian.var()
        laplacian_score = min(laplacian_var / self.reference_laplacian_max, 1.0)

        # 2. Local contrast (standard deviation of local patches)
        # Higher local contrast = better ridge/structure visibility
        local_std = compute_local_contrast(gray_np, window_size=16)
        contrast_score = min(local_std / self.reference_contrast_max, 1.0)

        # Combined weighted score
        quality = (
            self.laplacian_weight * laplacian_score
            + self.contrast_weight * contrast_score
        )

        return float(np.clip(quality, 0.0, 1.0))

    def __call__(self, tensor: torch.Tensor) -> float:
        """Callable interface for use in transforms pipeline.

        Args:
            tensor: Image tensor of shape (C, H, W).

        Returns:
            Quality score in [0, 1].
        """
        return self.estimate(tensor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(laplacian_weight={self.laplacian_weight}, "
            f"contrast_weight={self.contrast_weight})"
        )


def cv2_laplacian(gray_img: np.ndarray) -> np.ndarray:
    """Compute Laplacian of a grayscale image using NumPy.

    Pure NumPy implementation of the Laplacian operator for edge/sharpness
    detection. Uses a 3x3 kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]].

    Args:
        gray_img: 2D numpy array of shape (H, W) with dtype float32.

    Returns:
        Laplacian response map of shape (H, W).
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Pad the image to handle borders
    padded = np.pad(gray_img, pad_width=1, mode="reflect")
    # Apply kernel via convolution
    result = (
        padded[0:-2, 1:-1] * kernel[0, 1]
        + padded[1:-1, 0:-2] * kernel[1, 0]
        + padded[1:-1, 1:-1] * kernel[1, 1]
        + padded[1:-1, 2:] * kernel[1, 2]
        + padded[2:, 1:-1] * kernel[2, 1]
    )
    return result


def compute_local_contrast(
    gray_img: np.ndarray, window_size: int = 16
) -> float:
    """Compute mean local contrast using sliding window standard deviation.

    Divides the image into non-overlapping windows and computes the mean
    standard deviation across all windows. This measures local intensity
    variation which correlates with ridge/texture visibility.

    Args:
        gray_img: 2D numpy array of shape (H, W) with dtype float32.
        window_size: Size of local windows. Defaults to 16.

    Returns:
        Mean local contrast value.
    """
    h, w = gray_img.shape
    if h < window_size or w < window_size:
        return float(np.std(gray_img))

    # Trim to multiple of window_size
    h_trim = h - (h % window_size)
    w_trim = w - (w % window_size)
    img_trim = gray_img[:h_trim, :w_trim]

    # Reshape into windows: (num_windows_h, num_windows_w, window_size, window_size)
    windows = img_trim.reshape(
        h_trim // window_size, window_size, w_trim // window_size, window_size
    )
    windows = windows.transpose(0, 2, 1, 3).reshape(-1, window_size, window_size)

    # Compute standard deviation for each window
    local_stds = np.std(windows, axis=(1, 2))
    return float(np.mean(local_stds))


# ---------------------------------------------------------------------------
# Transform Composition Builders
# ---------------------------------------------------------------------------

def get_face_transforms(
    image_size: int = 224,
    is_training: bool = True,
    apply_quality_degradation: bool = True,
) -> Callable:
    """Build the transform pipeline for face images.

    Training transforms include augmentation suitable for face recognition:
    geometric augmentations (flip, rotation), photometric augmentations
    (color jitter, blur), and structural augmentations (random erasing, occlusion).

    Validation transforms are deterministic: just resize and normalize.

    Args:
        image_size: Target square image size. Defaults to 224.
        is_training: Whether to use training augmentations. Defaults to True.
        apply_quality_degradation: Whether to apply quality degradation simulation.
            Defaults to True.

    Returns:
        Composed transform callable.
    """
    base_transforms: List[Callable] = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]

    if is_training:
        augmentation_transforms: List[Callable] = [
            # Geometric augmentations
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-10, 10)),
            # Photometric augmentations
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ]

        # Quality degradation simulation
        if apply_quality_degradation:
            augmentation_transforms.append(
                RandomQualityDegradation(
                    blur_prob=0.2,
                    noise_prob=0.2,
                    occlusion_prob=0.15,
                    occlusion_transform=RandomOcclusionFace(p=0.15),
                )
            )

        # Random occlusion (always applied during training)
        augmentation_transforms.append(RandomOcclusionFace(p=0.2))

        # Structural augmentations (applied after ToTensor)
        tensor_transforms: List[Callable] = [
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]

        return T.Compose(
            base_transforms[:1]  # Resize first
            + augmentation_transforms[:3]  # Geometric before ToTensor
            + base_transforms[1:]  # ToTensor
            + augmentation_transforms[3:]  # Post-tensor augmentations
            + tensor_transforms
        )
    else:
        return T.Compose(base_transforms)


def get_fingerprint_transforms(
    image_size: int = 224,
    is_training: bool = True,
    apply_quality_degradation: bool = True,
) -> Callable:
    """Build the transform pipeline for fingerprint images.

    Training transforms include augmentation specific to fingerprint images:
    rotation (finger placement variation), Gaussian noise (sensor noise),
    random erasing, and minutiae dropout occlusion.

    Validation transforms are deterministic: resize and normalize.

    Args:
        image_size: Target square image size. Defaults to 224.
        is_training: Whether to use training augmentations. Defaults to True.
        apply_quality_degradation: Whether to apply quality degradation simulation.
            Defaults to True.

    Returns:
        Composed transform callable.
    """
    base_transforms: List[Callable] = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]

    if is_training:
        augmentation_transforms: List[Callable] = [
            T.RandomRotation(degrees=(-15, 15)),
            GaussianNoise(std_min=0.01, std_max=0.05, p=0.5),
        ]

        if apply_quality_degradation:
            augmentation_transforms.append(
                RandomQualityDegradation(
                    blur_prob=0.25,
                    noise_prob=0.25,
                    occlusion_prob=0.2,
                    occlusion_transform=RandomOcclusionFingerprint(p=0.2),
                )
            )

        # Fingerprint-specific occlusion (minutiae dropout)
        augmentation_transforms.append(RandomOcclusionFingerprint(p=0.2))

        tensor_transforms: List[Callable] = [
            T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]

        return T.Compose(
            base_transforms[:1]  # Resize
            + augmentation_transforms[:1]  # Rotation before ToTensor
            + base_transforms[1:]  # ToTensor
            + augmentation_transforms[1:]  # Post-tensor augmentations
            + tensor_transforms
        )
    else:
        return T.Compose(base_transforms)


# ---------------------------------------------------------------------------
# Multimodal Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration dataclass for the multimodal biometric dataset.

    Attributes:
        root_dir: Root directory containing subject subdirectories.
        face_prefix: Filename prefix for face images. Defaults to "face".
        fingerprint_prefix: Filename prefix for fingerprint images.
            Defaults to "fingerprint".
        image_extensions: Supported image file extensions.
            Defaults to (".jpg", ".jpeg", ".png", ".bmp").
        image_size: Target image size for both modalities. Defaults to 224.
        train_split: Fraction of subjects for training. Defaults to 0.7.
        val_split: Fraction of subjects for validation. Defaults to 0.15.
        test_split: Fraction of subjects for testing. Defaults to 0.15.
        seed: Random seed for reproducible splitting. Defaults to 42.
        drop_face_prob: Probability of dropping face modality. Defaults to 0.15.
        drop_fingerprint_prob: Probability of dropping fingerprint modality.
            Defaults to 0.15.
        apply_quality_degradation: Whether to simulate quality degradation.
            Defaults to True.
        pairs_per_subject: Number of pairs to generate per subject. Defaults to 5.
    """

    root_dir: str
    face_prefix: str = "face"
    fingerprint_prefix: str = "fingerprint"
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    image_size: int = 224
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    drop_face_prob: float = 0.15
    drop_fingerprint_prob: float = 0.15
    apply_quality_degradation: bool = True
    pairs_per_subject: int = 5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")


class MultimodalBiometricDataset(Dataset):
    """Multimodal biometric dataset for face + fingerprint recognition.

    Loads paired face and fingerprint images organized by subject directories.
    Each subject folder contains face images (prefixed with 'face_') and
    fingerprint images (prefixed with 'fingerprint_').

    Expected directory structure::

        root_dir/
            subject_001/
                face_001.jpg, face_002.jpg, ...
                fingerprint_001.jpg, fingerprint_002.jpg, ...
            subject_002/
                ...

    Returns tuples of::
        (face_tensor, fingerprint_tensor, subject_id, face_quality, fingerprint_quality)

    Args:
        config: DatasetConfig instance with dataset parameters.
        split: Which split to use - "train", "val", or "test". Defaults to "train".
        modality_simulator: Optional MissingModalitySimulator for training.
        face_transform: Optional custom transform for face images.
        fingerprint_transform: Optional custom transform for fingerprint images.

    Attributes:
        subjects: List of subject directory names.
        subject_to_id: Mapping from subject name to integer ID.
        samples: List of (face_path, fingerprint_path, subject_id) tuples.

    Example:
        >>> config = DatasetConfig(root_dir="/data/biometric")
        >>> train_dataset = MultimodalBiometricDataset(config, split="train")
        >>> face, fp, sid, qf, qfp = train_dataset[0]
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        modality_simulator: Optional[MissingModalitySimulator] = None,
        face_transform: Optional[Callable] = None,
        fingerprint_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.config = config
        self.split = split
        self.modality_simulator = modality_simulator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build modality-specific transforms
        is_training = split == "train"
        self.face_transform = face_transform or get_face_transforms(
            image_size=config.image_size,
            is_training=is_training,
            apply_quality_degradation=config.apply_quality_degradation,
        )
        self.fingerprint_transform = fingerprint_transform or get_fingerprint_transforms(
            image_size=config.image_size,
            is_training=is_training,
            apply_quality_degradation=config.apply_quality_degradation,
        )

        # Quality estimator (placeholder heuristic)
        self.quality_estimator = QualityEstimator()

        # Parse dataset directory structure
        self.subjects: List[str] = []
        self.subject_to_id: Dict[str, int] = {}
        self.samples: List[Tuple[str, str, int]] = []

        self._parse_dataset()

    def _parse_dataset(self) -> None:
        """Parse the dataset directory and build the sample list.

        Walks through the root directory, identifies subject folders,
        and collects all valid face-fingerprint pairs.

        Raises:
            FileNotFoundError: If root_dir does not exist.
            ValueError: If no valid subjects or samples are found.
        """
        root = Path(self.config.root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root}")

        subject_dirs = sorted(
            [
                d.name
                for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

        if not subject_dirs:
            raise ValueError(f"No subject directories found in {root}")

        # Assign integer IDs to subjects
        self.subjects = subject_dirs
        self.subject_to_id = {name: idx for idx, name in enumerate(self.subjects)}

        # Collect all samples
        for subject_name in self.subjects:
            subject_dir = root / subject_name
            subject_id = self.subject_to_id[subject_name]

            # Find face and fingerprint images
            face_images = self._find_images(subject_dir, self.config.face_prefix)
            fingerprint_images = self._find_images(
                subject_dir, self.config.fingerprint_prefix
            )

            if not face_images or not fingerprint_images:
                continue  # Skip subjects missing one modality entirely

            # Create pairs by cycling through available images
            for i in range(max(len(face_images), len(fingerprint_images))):
                face_path = face_images[i % len(face_images)]
                fp_path = fingerprint_images[i % len(fingerprint_images)]
                self.samples.append((str(face_path), str(fp_path), subject_id))

        if not self.samples:
            raise ValueError(
                f"No valid face-fingerprint pairs found in {root}. "
                f"Ensure subject directories contain '{self.config.face_prefix}_*' "
                f"and '{self.config.fingerprint_prefix}_*' images."
            )

    def _find_images(self, directory: Path, prefix: str) -> List[Path]:
        """Find image files with given prefix in a directory.

        Args:
            directory: Path to the subject directory.
            prefix: Filename prefix to match (e.g., "face" or "fingerprint").

        Returns:
            Sorted list of matching image file paths.
        """
        images = []
        for ext in self.config.image_extensions:
            images.extend(directory.glob(f"{prefix}_*{ext}"))
            images.extend(directory.glob(f"{prefix}*{ext}"))  # Also match without underscore
        return sorted(images)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
        """Get a single sample from the dataset.

        Args:
            idx: Integer index of the sample.

        Returns:
            A tuple containing:
                - face_tensor: Face image tensor of shape (C, H, W)
                - fingerprint_tensor: Fingerprint image tensor of shape (C, H, W)
                - subject_id: Integer subject identifier
                - face_quality: Quality score for face in [0, 1]
                - fingerprint_quality: Quality score for fingerprint in [0, 1]

        Raises:
            IndexError: If idx is out of range.
        """
        if not (0 <= idx < len(self.samples)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        face_path, fp_path, subject_id = self.samples[idx]

        # Load and transform images
        face_img = Image.open(face_path).convert("RGB")
        fp_img = Image.open(fp_path).convert("RGB")

        face_tensor = self.face_transform(face_img)
        fingerprint_tensor = self.fingerprint_transform(fp_img)

        # Apply missing modality simulation if configured (training only)
        if self.modality_simulator is not None and self.split == "train":
            face_tensor, fingerprint_tensor, _ = self.modality_simulator(
                face_tensor, fingerprint_tensor
            )

        # Compute quality scores
        face_quality = self.quality_estimator.estimate(face_tensor)
        fingerprint_quality = self.quality_estimator.estimate(fingerprint_tensor)

        return face_tensor, fingerprint_tensor, subject_id, face_quality, fingerprint_quality

    def get_subject_samples(self, subject_id: int) -> List[int]:
        """Get all sample indices belonging to a specific subject.

        Args:
            subject_id: Integer subject identifier.

        Returns:
            List of sample indices for the subject.
        """
        return [i for i, (_, _, sid) in enumerate(self.samples) if sid == subject_id]

    def get_num_subjects(self) -> int:
        """Return the number of unique subjects in the dataset."""
        return len(self.subjects)


# ---------------------------------------------------------------------------
# Subject-Disjoint Splitting
# ---------------------------------------------------------------------------

def create_subject_disjoint_splits(
    config: DatasetConfig,
    modality_simulator: Optional[MissingModalitySimulator] = None,
) -> Tuple[MultimodalBiometricDataset, MultimodalBiometricDataset, MultimodalBiometricDataset]:
    """Create train/val/test splits with NO subject overlap.

    This is critical for biometric evaluation - subjects in the test set
    must not appear in training or validation. The split is performed at
    the subject level to ensure fair evaluation of generalization to
    unseen identities.

    Args:
        config: Dataset configuration.
        modality_simulator: Optional modality dropout simulator for training.

    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset).

    Raises:
        ValueError: If there are not enough subjects for the requested split.
    """
    set_seed(config.seed)

    root = Path(config.root_dir)
    subject_dirs = sorted(
        [
            d.name
            for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    )

    if len(subject_dirs) < 3:
        raise ValueError(
            f"Need at least 3 subjects for train/val/test split, found {len(subject_dirs)}"
        )

    # Shuffle and split subjects
    indices = list(range(len(subject_dirs)))
    random.shuffle(indices)

    n_train = int(len(subject_dirs) * config.train_split)
    n_val = int(len(subject_dirs) * config.val_split)
    # Test gets the remainder to avoid rounding issues
    n_test = len(subject_dirs) - n_train - n_val

    train_subjects = set(subject_dirs[i] for i in indices[:n_train])
    val_subjects = set(subject_dirs[i] for i in indices[n_train : n_train + n_val])
    test_subjects = set(subject_dirs[i] for i in indices[n_train + n_val :])

    # Create full dataset first
    full_dataset = MultimodalBiometricDataset(
        config=config,
        split="train",
        modality_simulator=modality_simulator,
    )

    # Filter samples by split
    train_samples = [
        i for i, (_, _, sid) in enumerate(full_dataset.samples)
        if full_dataset.subjects[sid] in train_subjects
    ]
    val_samples = [
        i for i, (_, _, sid) in enumerate(full_dataset.samples)
        if full_dataset.subjects[sid] in val_subjects
    ]
    test_samples = [
        i for i, (_, _, sid) in enumerate(full_dataset.samples)
        if full_dataset.subjects[sid] in test_subjects
    ]

    # Build split-specific datasets
    train_dataset = _SplitSubset(full_dataset, train_samples, "train", modality_simulator)
    val_dataset = _SplitSubset(full_dataset, val_samples, "val", None)
    test_dataset = _SplitSubset(full_dataset, test_samples, "test", None)

    return train_dataset, val_dataset, test_dataset


class _SplitSubset(Dataset):
    """Internal subset wrapper for creating subject-disjoint splits.

    This wrapper creates a view into the full dataset containing only
    samples from subjects in a specific split. It manages split-specific
    transforms and modality simulation.

    Args:
        base_dataset: The full MultimodalBiometricDataset.
        indices: Sample indices to include in this split.
        split: Split name ("train", "val", or "test").
        modality_simulator: Optional modality simulator (training only).
    """

    def __init__(
        self,
        base_dataset: MultimodalBiometricDataset,
        indices: List[int],
        split: str,
        modality_simulator: Optional[MissingModalitySimulator] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.split = split
        self.modality_simulator = modality_simulator

        # Update transforms for this split
        is_training = split == "train"
        self.face_transform = get_face_transforms(
            image_size=base_dataset.config.image_size,
            is_training=is_training,
            apply_quality_degradation=base_dataset.config.apply_quality_degradation,
        )
        self.fingerprint_transform = get_fingerprint_transforms(
            image_size=base_dataset.config.image_size,
            is_training=is_training,
            apply_quality_degradation=base_dataset.config.apply_quality_degradation,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
        """Get a sample from this split.

        Args:
            idx: Index within this split (not the base dataset).

        Returns:
            Sample tuple: (face, fingerprint, subject_id, face_quality, fp_quality)
        """
        base_idx = self.indices[idx]
        face_path, fp_path, subject_id = self.base_dataset.samples[base_idx]

        face_img = Image.open(face_path).convert("RGB")
        fp_img = Image.open(fp_path).convert("RGB")

        face_tensor = self.face_transform(face_img)
        fingerprint_tensor = self.fingerprint_transform(fp_img)

        if self.modality_simulator is not None:
            face_tensor, fingerprint_tensor, _ = self.modality_simulator(
                face_tensor, fingerprint_tensor
            )

        face_quality = self.base_dataset.quality_estimator.estimate(face_tensor)
        fingerprint_quality = self.base_dataset.quality_estimator.estimate(fingerprint_tensor)

        return face_tensor, fingerprint_tensor, subject_id, face_quality, fingerprint_quality

    @property
    def config(self) -> DatasetConfig:
        """Access the underlying dataset configuration."""
        return self.base_dataset.config

    @property
    def subjects(self) -> List[str]:
        """Get unique subjects in this split."""
        unique_ids = set()
        for i in self.indices:
            _, _, sid = self.base_dataset.samples[i]
            unique_ids.add(sid)
        return [self.base_dataset.subjects[sid] for sid in sorted(unique_ids)]

    def get_num_subjects(self) -> int:
        """Return the number of unique subjects in this split."""
        return len(self.subjects)


# ---------------------------------------------------------------------------
# Pair Generation for Verification
# ---------------------------------------------------------------------------

def generate_pairs(
    dataset: Union[MultimodalBiometricDataset, _SplitSubset],
    pairs_per_subject: int = 5,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """Generate balanced genuine and impostor pairs for verification.

    Creates pairs of sample indices for biometric verification tasks.
    Genuine pairs (label=1) are from the same subject, impostor pairs
    (label=0) are from different subjects. The number of genuine and
    impostor pairs is balanced.

    Args:
        dataset: The dataset to generate pairs from.
        pairs_per_subject: Number of genuine pairs to generate per subject.
            An equal number of impostor pairs will be generated. Defaults to 5.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        List of tuples (idx1, idx2, label) where:
            - idx1, idx2 are sample indices in the dataset
            - label is 1 for genuine pairs, 0 for impostor pairs

    Example:
        >>> pairs = generate_pairs(train_dataset, pairs_per_subject=10)
        >>> for idx1, idx2, label in pairs:
        ...     face1, fp1, sid1, _, _ = dataset[idx1]
        ...     face2, fp2, sid2, _, _ = dataset[idx2]
        ...     # Train verification model
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build subject-to-samples mapping
    subject_samples: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(dataset)):
        if hasattr(dataset, "base_dataset"):
            # _SplitSubset case - need to map through indices
            if i >= len(dataset.indices):
                continue
            base_idx = dataset.indices[i]
            _, _, subject_id = dataset.base_dataset.samples[base_idx]
            subject_samples[subject_id].append(i)
        else:
            _, _, subject_id = dataset.samples[i]
            subject_samples[subject_id].append(i)

    subjects = list(subject_samples.keys())
    if len(subjects) < 2:
        raise ValueError("Need at least 2 subjects to generate pairs")

    genuine_pairs: List[Tuple[int, int, int]] = []
    impostor_pairs: List[Tuple[int, int, int]] = []

    # Generate genuine pairs (same subject)
    for subject_id in subjects:
        samples = subject_samples[subject_id]
        if len(samples) < 2:
            continue

        # Generate genuine pairs by random sampling without replacement
        num_pairs = min(pairs_per_subject, len(samples) * (len(samples) - 1) // 2)
        pairs_created = 0
        attempts = 0
        max_attempts = num_pairs * 100

        used_pairs = set()
        while pairs_created < num_pairs and attempts < max_attempts:
            attempts += 1
            i, j = random.sample(samples, 2)
            pair_key = tuple(sorted([i, j]))
            if pair_key not in used_pairs:
                used_pairs.add(pair_key)
                genuine_pairs.append((i, j, 1))
                pairs_created += 1

    # Generate impostor pairs (different subjects) - balanced with genuine
    target_impostor = len(genuine_pairs)
    impostor_created = 0
    attempts = 0
    max_attempts = target_impostor * 100

    used_impostor_pairs = set()
    while impostor_created < target_impostor and attempts < max_attempts:
        attempts += 1
        subj_a, subj_b = random.sample(subjects, 2)
        if subj_a == subj_b:
            continue
        sample_a = random.choice(subject_samples[subj_a])
        sample_b = random.choice(subject_samples[subj_b])
        pair_key = tuple(sorted([sample_a, sample_b]))
        if pair_key not in used_impostor_pairs:
            used_impostor_pairs.add(pair_key)
            impostor_pairs.append((sample_a, sample_b, 0))
            impostor_created += 1

    # Combine and shuffle
    all_pairs = genuine_pairs + impostor_pairs
    random.shuffle(all_pairs)

    return all_pairs


class VerificationPairDataset(Dataset):
    """Dataset wrapper for verification pair training.

    Wraps an existing dataset and provides paired samples for
    binary verification (genuine vs. impostor) training.

    Args:
        base_dataset: The underlying dataset to sample from.
        pairs: List of (idx1, idx2, label) tuples from generate_pairs().

    Example:
        >>> pairs = generate_pairs(dataset, pairs_per_subject=10)
        >>> pair_dataset = VerificationPairDataset(dataset, pairs)
        >>> loader = DataLoader(pair_dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        base_dataset: Union[MultimodalBiometricDataset, _SplitSubset],
        pairs: List[Tuple[int, int, int]],
    ) -> None:
        self.base_dataset = base_dataset
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a verification pair.

        Args:
            idx: Index of the pair.

        Returns:
            A tuple containing:
                - face1: Face tensor of first sample
                - fingerprint1: Fingerprint tensor of first sample
                - face2: Face tensor of second sample
                - fingerprint2: Fingerprint tensor of second sample
                - label: 1 for genuine, 0 for impostor
        """
        idx1, idx2, label = self.pairs[idx]

        face1, fp1, sid1, qf1, qfp1 = self.base_dataset[idx1]
        face2, fp2, sid2, qf2, qfp2 = self.base_dataset[idx2]

        return face1, fp1, face2, fp2, label


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    seed: int = 42,
    drop_face_prob: float = 0.15,
    drop_fingerprint_prob: float = 0.15,
    apply_quality_degradation: bool = True,
    pairs_per_subject: int = 5,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    """Create train/val/test DataLoaders with subject-disjoint splits.

    Factory function that creates the complete data pipeline including:
    subject-disjoint splitting, modality-specific transforms, missing modality
    simulation, and DataLoader creation with appropriate configuration.

    Args:
        root_dir: Root directory of the dataset.
        batch_size: Batch size for all DataLoaders. Defaults to 32.
        num_workers: Number of worker processes for data loading. Defaults to 4.
        image_size: Target image size. Defaults to 224.
        seed: Random seed for reproducibility. Defaults to 42.
        drop_face_prob: Probability of dropping face modality. Defaults to 0.15.
        drop_fingerprint_prob: Probability of dropping fingerprint modality.
            Defaults to 0.15.
        apply_quality_degradation: Whether to simulate quality degradation.
            Defaults to True.
        pairs_per_subject: Number of pairs per subject for verification.
            Defaults to 5.
        pin_memory: Whether to pin memory for GPU transfer. Defaults to True.

    Returns:
        Dictionary with keys:
            - "train": Training DataLoader
            - "val": Validation DataLoader
            - "test": Test DataLoader
            - "datasets": Dict of train/val/test datasets
            - "train_subjects": List of training subject names
            - "val_subjects": List of validation subject names
            - "test_subjects": List of test subject names

    Example:
        >>> loaders = get_dataloaders("/data/biometric", batch_size=64)
        >>> for face, fp, sid, qf, qfp in loaders["train"]:
        ...     # Training loop
    """
    set_seed(seed)

    # Create dataset configuration
    config = DatasetConfig(
        root_dir=root_dir,
        image_size=image_size,
        seed=seed,
        drop_face_prob=drop_face_prob,
        drop_fingerprint_prob=drop_fingerprint_prob,
        apply_quality_degradation=apply_quality_degradation,
        pairs_per_subject=pairs_per_subject,
    )

    # Create modality simulator for training
    modality_simulator = MissingModalitySimulator(
        drop_face_prob=drop_face_prob,
        drop_fingerprint_prob=drop_fingerprint_prob,
    )

    # Create subject-disjoint splits
    train_dataset, val_dataset, test_dataset = create_subject_disjoint_splits(
        config=config,
        modality_simulator=modality_simulator,
    )

    # Determine pin_memory based on CUDA availability
    should_pin = pin_memory and torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=should_pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=should_pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=should_pin,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "datasets": {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        },
        "train_subjects": train_dataset.subjects,
        "val_subjects": val_dataset.subjects,
        "test_subjects": test_dataset.subjects,
    }


# ---------------------------------------------------------------------------
# Main / Example Usage
# ---------------------------------------------------------------------------

def _create_dummy_dataset(root_dir: str, num_subjects: int = 5, images_per_modality: int = 3) -> None:
    """Create a dummy dataset structure for testing purposes.

    Args:
        root_dir: Root directory to create the dataset in.
        num_subjects: Number of dummy subjects. Defaults to 5.
        images_per_modality: Images per modality per subject. Defaults to 3.
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    for s in range(1, num_subjects + 1):
        subject_dir = root / f"subject_{s:03d}"
        subject_dir.mkdir(exist_ok=True)

        for i in range(1, images_per_modality + 1):
            # Create dummy face images (random RGB)
            face_img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            face_img.save(subject_dir / f"face_{i:03d}.jpg")

            # Create dummy fingerprint images (grayscale saved as RGB)
            fp_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            fp_img = Image.fromarray(fp_array).convert("RGB")
            fp_img.save(subject_dir / f"fingerprint_{i:03d}.jpg")


def main() -> None:
    """Run a complete example of the data loading pipeline.

    Creates a dummy dataset, sets up the data pipeline, and demonstrates
    all key features including pair generation and iteration.
    """
    print("=" * 70)
    print("Multimodal Face + Fingerprint Biometric Data Pipeline Demo")
    print("=" * 70)

    # Create dummy dataset for demonstration
    dummy_root = "/tmp/dummy_biometric_dataset"
    print(f"\n[1] Creating dummy dataset at: {dummy_root}")
    _create_dummy_dataset(dummy_root, num_subjects=10, images_per_modality=4)

    # Create data loaders
    print("\n[2] Creating subject-disjoint train/val/test splits...")
    loaders = get_dataloaders(
        root_dir=dummy_root,
        batch_size=4,
        num_workers=0,  # Use 0 for demo to avoid multiprocessing issues
        image_size=224,
        seed=42,
        drop_face_prob=0.15,
        drop_fingerprint_prob=0.15,
        apply_quality_degradation=True,
    )

    print(f"    Training subjects: {len(loaders['train_subjects'])}")
    print(f"    Validation subjects: {len(loaders['val_subjects'])}")
    print(f"    Test subjects: {len(loaders['test_subjects'])}")

    # Verify subject disjointness
    train_set = set(loaders["train_subjects"])
    val_set = set(loaders["val_subjects"])
    test_set = set(loaders["test_subjects"])
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    print(f"    Subject overlap train/val: {len(overlap_train_val)}")
    print(f"    Subject overlap train/test: {len(overlap_train_test)}")
    print(f"    Subject overlap val/test: {len(overlap_val_test)}")

    # Iterate through training data
    print("\n[3] Iterating through training data...")
    for batch_idx, (face, fingerprint, subject_id, q_face, q_fp) in enumerate(
        loaders["train"]
    ):
        print(f"    Batch {batch_idx}:")
        print(f"      Face shape: {face.shape}, dtype: {face.dtype}")
        print(f"      Fingerprint shape: {fingerprint.shape}, dtype: {fingerprint.dtype}")
        print(f"      Subject IDs: {subject_id.tolist()}")
        print(f"      Face quality: {[f'{q:.3f}' for q in q_face]}")
        print(f"      Fingerprint quality: {[f'{q:.3f}' for q in q_fp]}")
        if batch_idx >= 2:
            break

    # Generate verification pairs
    print("\n[4] Generating verification pairs...")
    train_dataset = loaders["datasets"]["train"]
    pairs = generate_pairs(train_dataset, pairs_per_subject=5, seed=42)
    genuine_count = sum(1 for _, _, label in pairs if label == 1)
    impostor_count = sum(1 for _, _, label in pairs if label == 0)
    print(f"    Total pairs: {len(pairs)}")
    print(f"    Genuine pairs: {genuine_count}")
    print(f"    Impostor pairs: {impostor_count}")

    # Create pair dataset and iterate
    print("\n[5] Iterating through pair dataset...")
    pair_dataset = VerificationPairDataset(train_dataset, pairs)
    pair_loader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
    for batch_idx, (face1, fp1, face2, fp2, labels) in enumerate(pair_loader):
        print(f"    Batch {batch_idx}:")
        print(f"      Face1 shape: {face1.shape}")
        print(f"      Face2 shape: {face2.shape}")
        print(f"      Labels: {labels.tolist()}")
        if batch_idx >= 2:
            break

    # Demonstrate quality estimation
    print("\n[6] Quality estimation demo...")
    estimator = QualityEstimator()
    sample_face = torch.rand(3, 224, 224)
    quality_score = estimator.estimate(sample_face)
    print(f"    Random image quality score: {quality_score:.4f}")

    # Test missing modality simulator
    print("\n[7] Missing modality simulation demo...")
    simulator = MissingModalitySimulator(drop_face_prob=0.5, drop_fingerprint_prob=0.5)
    face_tensor = torch.rand(3, 224, 224)
    fp_tensor = torch.rand(3, 224, 224)
    face_out, fp_out, flags = simulator(face_tensor, fp_tensor)
    print(f"    Face missing: {flags['face_missing']}")
    print(f"    Fingerprint missing: {flags['fingerprint_missing']}")

    # Cleanup
    import shutil
    if Path(dummy_root).exists():
        shutil.rmtree(dummy_root)

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
