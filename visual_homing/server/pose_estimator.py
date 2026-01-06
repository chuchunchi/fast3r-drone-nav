"""SVD Procrustes pose estimation for 3D-3D point alignment."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import roma
except ImportError:
    roma = None

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Result of pose estimation."""

    rotation: torch.Tensor  # 3x3 rotation matrix
    translation: torch.Tensor  # 3D translation vector (meters)
    scale: float  # Scale factor
    confidence: float  # Overall confidence score
    num_inliers: int  # Number of points used for alignment
    success: bool  # Whether estimation was successful


class PoseEstimator:
    """
    SVD Procrustes pose estimator for 3D-3D point alignment.

    Uses roma.rigid_points_registration for efficient GPU-accelerated
    weighted least squares alignment of 3D point clouds.
    """

    def __init__(
        self,
        confidence_percentile: float = 0.5,
        min_points: int = 100,
        metric_scale: float = 1.0,
    ):
        """
        Initialize the pose estimator.

        Args:
            confidence_percentile: Use top N% of points by confidence (0.5 = top 50%).
            min_points: Minimum number of confident points required.
            metric_scale: Scale factor to convert Fast3R units to meters.
        """
        if roma is None:
            raise ImportError(
                "roma library is required for pose estimation. "
                "Install with: pip install roma"
            )

        self.confidence_percentile = confidence_percentile
        self.min_points = min_points
        self.metric_scale = metric_scale

    def estimate_pose(
        self,
        pts3d_source: torch.Tensor,
        pts3d_target: torch.Tensor,
        confidence: torch.Tensor,
        compute_scaling: bool = True,
    ) -> PoseResult:
        """
        Estimate relative pose from source to target using SVD Procrustes.

        The transformation aligns source points to target points:
            pts_target ≈ scale * (pts_source @ R.T) + t

        Args:
            pts3d_source: Source 3D pointmap (H, W, 3) or (N, 3).
            pts3d_target: Target 3D pointmap (H, W, 3) or (N, 3).
            confidence: Confidence weights (H, W) or (N,).
            compute_scaling: Whether to compute scale factor.

        Returns:
            PoseResult with rotation, translation, scale, and metadata.
        """
        # Flatten if needed
        if pts3d_source.dim() == 3:
            pts_source_flat = pts3d_source.reshape(-1, 3)
            pts_target_flat = pts3d_target.reshape(-1, 3)
            conf_flat = confidence.reshape(-1)
        else:
            pts_source_flat = pts3d_source
            pts_target_flat = pts3d_target
            conf_flat = confidence

        # Filter by confidence threshold
        conf_threshold = torch.quantile(conf_flat, self.confidence_percentile)
        mask = conf_flat >= conf_threshold

        # Also filter out invalid points (NaN, Inf, or zero)
        valid_source = torch.isfinite(pts_source_flat).all(dim=1)
        valid_target = torch.isfinite(pts_target_flat).all(dim=1)
        mask = mask & valid_source & valid_target

        num_valid = mask.sum().item()

        if num_valid < self.min_points:
            logger.warning(
                f"Not enough confident points: {num_valid} < {self.min_points}"
            )
            return PoseResult(
                rotation=torch.eye(3, device=pts3d_source.device),
                translation=torch.zeros(3, device=pts3d_source.device),
                scale=1.0,
                confidence=0.0,
                num_inliers=num_valid,
                success=False,
            )

        # Extract valid points and weights
        pts_source_good = pts_source_flat[mask]
        pts_target_good = pts_target_flat[mask]
        weights = conf_flat[mask]

        # Normalize weights
        weights = weights / weights.sum()

        # SVD Procrustes alignment
        # Find R, t, s such that: pts_target ≈ s * (pts_source @ R.T) + t
        try:
            R, t, s = roma.rigid_points_registration(
                pts_source_good,
                pts_target_good,
                weights=weights,
                compute_scaling=compute_scaling,
            )
        except Exception as e:
            logger.error(f"SVD Procrustes failed: {e}")
            return PoseResult(
                rotation=torch.eye(3, device=pts3d_source.device),
                translation=torch.zeros(3, device=pts3d_source.device),
                scale=1.0,
                confidence=0.0,
                num_inliers=num_valid,
                success=False,
            )

        # Convert scale to float
        scale_value = float(s) if torch.is_tensor(s) else s

        # Apply metric scale to translation
        t_meters = t * self.metric_scale * scale_value

        # Compute alignment error for confidence
        pts_aligned = scale_value * (pts_source_good @ R.T) + t
        errors = torch.norm(pts_target_good - pts_aligned, dim=1)
        mean_error = errors.mean().item()
        confidence_score = 1.0 / (1.0 + mean_error)  # Higher is better

        return PoseResult(
            rotation=R,
            translation=t_meters,
            scale=scale_value,
            confidence=confidence_score,
            num_inliers=num_valid,
            success=True,
        )

    def set_metric_scale(self, scale: float) -> None:
        """
        Set the metric scale factor.

        Args:
            scale: Meters per Fast3R unit.
        """
        self.metric_scale = scale

    def compute_scale_factor(
        self,
        pts3d_1: torch.Tensor,
        pts3d_2: torch.Tensor,
        conf: torch.Tensor,
        imu_distance_m: float,
    ) -> float:
        """
        Compute the scale factor from IMU distance.

        During the TEACH phase, we know the true metric distance between
        consecutive keyframes from IMU integration. This function computes
        the scale factor to convert Fast3R's unitless output to meters.

        Args:
            pts3d_1: Pointmap from first image (H, W, 3).
            pts3d_2: Pointmap from second image (H, W, 3).
            conf: Confidence map (H, W).
            imu_distance_m: True distance from IMU integration (meters).

        Returns:
            Scale factor (meters per Fast3R unit).
        """
        # Estimate pose without applying metric scale
        old_scale = self.metric_scale
        self.metric_scale = 1.0

        result = self.estimate_pose(pts3d_1, pts3d_2, conf, compute_scaling=True)

        self.metric_scale = old_scale

        if not result.success:
            logger.warning("Scale calibration failed, using default scale 1.0")
            return 1.0

        # Fast3R translation magnitude
        fast3r_distance = torch.norm(result.translation).item()

        if fast3r_distance < 1e-6:
            logger.warning("Fast3R distance too small, using default scale 1.0")
            return 1.0

        # Scale factor: meters per Fast3R unit
        scale_factor = imu_distance_m / fast3r_distance

        logger.info(
            f"Scale calibration: IMU={imu_distance_m:.2f}m, "
            f"Fast3R={fast3r_distance:.4f}, scale={scale_factor:.4f}"
        )

        return scale_factor


