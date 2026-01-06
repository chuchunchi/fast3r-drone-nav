"""Tests for coordinate frame utilities."""

import math

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.coordinate_utils import (
    camera_to_body_translation,
    body_to_virtualstick,
    fast3r_to_dji_command,
    extract_yaw_error,
    extract_euler_angles,
    rotation_matrix_from_euler,
    create_hover_command,
)


class TestTranslationConversion:
    """Tests for translation coordinate conversions."""

    def test_forward_movement(self):
        """Camera Z=forward should map to body X=forward."""
        t_cam = np.array([0.0, 0.0, 2.0])  # 2m forward in camera frame
        t_body = camera_to_body_translation(t_cam)

        # Body frame: X=forward
        assert np.isclose(t_body[0], 2.0)
        assert np.isclose(t_body[1], 0.0)
        assert np.isclose(t_body[2], 0.0)

    def test_lateral_movement(self):
        """Camera X=right should map to body Y=right."""
        t_cam = np.array([1.5, 0.0, 0.0])  # 1.5m right in camera frame
        t_body = camera_to_body_translation(t_cam)

        # Body frame: Y=right
        assert np.isclose(t_body[1], 1.5)

    def test_vertical_movement(self):
        """Camera Y=down should map to body Z=down."""
        t_cam = np.array([0.0, 1.0, 0.0])  # 1m down in camera frame
        t_body = camera_to_body_translation(t_cam)

        # Body frame: Z=down
        assert np.isclose(t_body[2], 1.0)


class TestVirtualStickConversion:
    """Tests for body to VirtualStick conversion."""

    def test_forward_error(self):
        """Body forward should give positive forward error."""
        t_body = np.array([2.0, 0.0, 0.0])  # 2m forward
        vs = body_to_virtualstick(t_body)

        assert vs["forward"] == 2.0
        assert vs["lateral"] == 0.0
        assert vs["vertical"] == 0.0

    def test_vertical_inversion(self):
        """Body Z=down should give negative vertical (up) in VS."""
        t_body = np.array([0.0, 0.0, 1.0])  # 1m down in body frame
        vs = body_to_virtualstick(t_body)

        # VS vertical is up-positive, so down becomes negative
        assert vs["vertical"] == -1.0


class TestCommandGeneration:
    """Tests for DJI command generation."""

    @pytest.fixture
    def pid_gains(self):
        return {
            "kp_forward": 0.5,
            "kp_lateral": 0.5,
            "kp_vertical": 0.3,
            "kp_yaw": 1.0,
        }

    def test_forward_command(self, pid_gains):
        """Target ahead should generate forward pitch velocity."""
        t_cam = np.array([0.0, 0.0, 2.0])
        cmd = fast3r_to_dji_command(t_cam, 0.0, pid_gains)

        assert cmd["pitch_velocity"] > 0  # Forward
        assert cmd["roll_velocity"] == 0.0
        assert cmd["vertical_velocity"] == 0.0
        assert cmd["yaw_rate"] == 0.0

    def test_lateral_command(self, pid_gains):
        """Target to right should generate positive roll velocity."""
        t_cam = np.array([2.0, 0.0, 0.0])
        cmd = fast3r_to_dji_command(t_cam, 0.0, pid_gains)

        assert cmd["roll_velocity"] > 0  # Right

    def test_vertical_command(self, pid_gains):
        """Target above (negative Y in camera) should give positive vertical."""
        t_cam = np.array([0.0, -1.0, 0.0])  # 1m up in camera frame
        cmd = fast3r_to_dji_command(t_cam, 0.0, pid_gains)

        assert cmd["vertical_velocity"] > 0  # Up

    def test_yaw_command(self, pid_gains):
        """Positive yaw error should give positive yaw rate (clockwise)."""
        t_cam = np.array([0.0, 0.0, 1.0])
        cmd = fast3r_to_dji_command(t_cam, 15.0, pid_gains)

        assert cmd["yaw_rate"] > 0  # Clockwise

    def test_velocity_limits(self, pid_gains):
        """Commands should be clamped to velocity limits."""
        t_cam = np.array([100.0, 100.0, 100.0])  # Large errors
        limits = {
            "max_forward": 2.0,
            "max_lateral": 2.0,
            "max_vertical": 1.0,
            "max_yaw": 30.0,
        }
        cmd = fast3r_to_dji_command(t_cam, 100.0, pid_gains, limits)

        assert abs(cmd["pitch_velocity"]) <= 2.0
        assert abs(cmd["roll_velocity"]) <= 2.0
        assert abs(cmd["vertical_velocity"]) <= 1.0
        assert abs(cmd["yaw_rate"]) <= 30.0


class TestYawExtraction:
    """Tests for yaw angle extraction."""

    def test_identity(self):
        """Identity rotation should give zero yaw."""
        R = torch.eye(3)
        yaw = extract_yaw_error(R)
        assert np.isclose(yaw, 0.0, atol=1e-5)

    def test_yaw_rotation_around_y(self):
        """Test yaw extraction for rotation around Y axis (camera frame)."""
        # In camera frame, yaw is rotation around Y axis (down)
        # Create a rotation around Y by 45 degrees
        angle = np.pi / 4  # 45 degrees
        R = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ], dtype=torch.float32)
        
        yaw = extract_yaw_error(R)
        # extract_yaw_error uses atan2(R[0,2], R[2,2]) for camera frame
        # For this rotation: R[0,2] = sin(45°), R[2,2] = cos(45°)
        # So yaw = atan2(sin(45°), cos(45°)) = 45°
        assert abs(yaw - 45.0) < 1.0  # Should be approximately 45 degrees


class TestEulerAngles:
    """Tests for Euler angle extraction and creation."""

    def test_roundtrip(self):
        """Euler angles should survive roundtrip conversion."""
        roll_in, pitch_in, yaw_in = 15.0, -10.0, 45.0

        R = rotation_matrix_from_euler(roll_in, pitch_in, yaw_in)
        R_tensor = torch.tensor(R, dtype=torch.float32)

        roll_out, pitch_out, yaw_out = extract_euler_angles(R_tensor)

        assert np.isclose(roll_in, roll_out, atol=1.0)
        assert np.isclose(pitch_in, pitch_out, atol=1.0)
        assert np.isclose(yaw_in, yaw_out, atol=1.0)

    def test_identity_matrix(self):
        """Identity should give zero angles."""
        R = rotation_matrix_from_euler(0, 0, 0)
        assert np.allclose(R, np.eye(3), atol=1e-10)


class TestHoverCommand:
    """Tests for hover command."""

    def test_hover_zeros(self):
        """Hover command should have all zeros."""
        cmd = create_hover_command()

        assert cmd["pitch_velocity"] == 0.0
        assert cmd["roll_velocity"] == 0.0
        assert cmd["vertical_velocity"] == 0.0
        assert cmd["yaw_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


