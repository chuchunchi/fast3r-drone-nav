"""Tests for the SVD Procrustes pose estimator."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.pose_estimator import PoseEstimator, PoseResult


class TestPoseEstimator:
    """Test suite for PoseEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create a pose estimator instance."""
        return PoseEstimator(
            confidence_percentile=0.5,
            min_points=10,
            metric_scale=1.0,
        )

    def test_identity_transformation(self, estimator):
        """Test with identical point clouds (identity transformation)."""
        # Create a simple 3D point cloud
        N = 100
        pts = torch.randn(N, 3)
        conf = torch.ones(N)

        result = estimator.estimate_pose(pts, pts, conf)

        assert result.success
        # Rotation should be close to identity
        assert torch.allclose(result.rotation, torch.eye(3), atol=1e-4)
        # Translation should be close to zero
        assert torch.allclose(result.translation, torch.zeros(3), atol=1e-4)

    def test_known_translation(self, estimator):
        """Test with a known translation."""
        N = 100
        pts_source = torch.randn(N, 3)
        
        # Apply known translation
        t_true = torch.tensor([1.0, 2.0, 3.0])
        pts_target = pts_source + t_true

        conf = torch.ones(N)

        result = estimator.estimate_pose(pts_source, pts_target, conf)

        assert result.success
        # Translation should match
        assert torch.allclose(result.translation, t_true, atol=1e-3)

    def test_known_rotation(self, estimator):
        """Test with a known rotation (90° around Z axis)."""
        N = 100
        pts_source = torch.randn(N, 3)

        # 90° rotation around Z axis
        R_true = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        pts_target = pts_source @ R_true.T

        conf = torch.ones(N)

        result = estimator.estimate_pose(pts_source, pts_target, conf)

        assert result.success
        # Rotation should match
        assert torch.allclose(result.rotation, R_true, atol=1e-3)

    def test_known_scale(self, estimator):
        """Test with a known scale factor."""
        N = 100
        pts_source = torch.randn(N, 3)

        # Apply scale
        s_true = 2.0
        pts_target = pts_source * s_true

        conf = torch.ones(N)

        result = estimator.estimate_pose(
            pts_source, pts_target, conf, compute_scaling=True
        )

        assert result.success
        assert np.isclose(result.scale, s_true, atol=1e-3)

    def test_combined_transformation(self, estimator):
        """Test with rotation + translation + scale."""
        N = 200
        pts_source = torch.randn(N, 3)

        # 45° rotation around Z
        angle = np.pi / 4
        R_true = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        t_true = torch.tensor([0.5, -0.3, 1.0])
        s_true = 1.5

        pts_target = s_true * (pts_source @ R_true.T) + t_true

        conf = torch.ones(N)

        result = estimator.estimate_pose(
            pts_source, pts_target, conf, compute_scaling=True
        )

        assert result.success
        assert torch.allclose(result.rotation, R_true, atol=1e-2)
        assert np.isclose(result.scale, s_true, atol=1e-2)

    def test_insufficient_points(self, estimator):
        """Test failure with too few points."""
        pts = torch.randn(5, 3)  # Less than min_points (10)
        conf = torch.ones(5)

        result = estimator.estimate_pose(pts, pts, conf)

        assert not result.success
        assert result.num_inliers < estimator.min_points

    def test_confidence_weighting(self, estimator):
        """Test that low confidence points are filtered."""
        N = 100
        pts_source = torch.randn(N, 3)
        t_true = torch.tensor([1.0, 0.0, 0.0])
        pts_target = pts_source + t_true

        # Half the points have low confidence
        conf = torch.ones(N)
        conf[:50] = 0.1  # Low confidence

        result = estimator.estimate_pose(pts_source, pts_target, conf)

        assert result.success
        # Should use ~50 inliers (top 50% by confidence)
        assert result.num_inliers >= 45  # Allow some margin

    def test_with_noise(self, estimator):
        """Test robustness to noise."""
        N = 200
        pts_source = torch.randn(N, 3)
        t_true = torch.tensor([2.0, 1.0, 0.5])
        pts_target = pts_source + t_true

        # Add noise
        noise = torch.randn(N, 3) * 0.05
        pts_target_noisy = pts_target + noise

        conf = torch.ones(N)

        result = estimator.estimate_pose(pts_source, pts_target_noisy, conf)

        assert result.success
        # Translation should be approximately correct despite noise
        assert torch.allclose(result.translation, t_true, atol=0.2)

    def test_metric_scale_application(self, estimator):
        """Test that metric scale is applied to translation."""
        estimator.set_metric_scale(2.0)  # 2 meters per unit

        N = 100
        pts_source = torch.randn(N, 3)
        t_fast3r = torch.tensor([1.0, 0.0, 0.0])  # 1 unit
        pts_target = pts_source + t_fast3r

        conf = torch.ones(N)

        result = estimator.estimate_pose(pts_source, pts_target, conf)

        # Translation should be scaled to 2 meters
        assert torch.allclose(result.translation, t_fast3r * 2.0, atol=1e-3)

    def test_scale_factor_computation(self, estimator):
        """Test scale factor computation from IMU distance."""
        N = 100
        pts1 = torch.randn(N, 3)
        pts2 = pts1 + torch.tensor([1.0, 0.0, 0.0])  # 1 unit apart

        conf = torch.ones(N)
        imu_distance = 3.0  # 3 meters

        scale = estimator.compute_scale_factor(pts1, pts2, conf, imu_distance)

        # Scale should be ~3.0 (3 meters / 1 unit)
        assert np.isclose(scale, 3.0, atol=0.1)

    def test_2d_pointmap_input(self, estimator):
        """Test with 2D pointmap input (H, W, 3)."""
        H, W = 10, 15
        pts_source = torch.randn(H, W, 3)
        t_true = torch.tensor([0.5, 0.5, 0.5])
        pts_target = pts_source + t_true

        conf = torch.ones(H, W)

        result = estimator.estimate_pose(pts_source, pts_target, conf)

        assert result.success
        assert torch.allclose(result.translation, t_true, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


