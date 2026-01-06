"""Tests for configuration."""

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config, default_config


class TestConfig:
    """Test configuration class."""

    def test_default_config_exists(self):
        """Test default config instance exists."""
        assert default_config is not None
        assert isinstance(default_config, Config)

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()

        # Fast3R settings
        assert config.fast3r_checkpoint == "jedyang97/Fast3R_ViT_Large_512"
        assert config.device == "cuda"

        # Image settings
        assert config.image_size == (512, 384)
        assert config.jpeg_quality == 80

        # Keyframe settings
        assert config.keyframe_interval_m == 2.0
        assert config.keyframe_interval_s == 3.0

        # Homing settings
        assert config.waypoint_threshold_m == 0.8
        assert config.confidence_threshold == 0.3

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = Config(
            device="cpu",
            keyframe_interval_m=5.0,
            max_forward_velocity=3.0,
        )

        assert config.device == "cpu"
        assert config.keyframe_interval_m == 5.0
        assert config.max_forward_velocity == 3.0

        # Other values should remain default
        assert config.keyframe_interval_s == 3.0

    def test_pid_gains(self):
        """Test PID gain defaults."""
        config = Config()

        # Forward PID
        assert config.pid_forward_kp == 0.5
        assert config.pid_forward_ki == 0.01
        assert config.pid_forward_kd == 0.1

        # Lateral PID
        assert config.pid_lateral_kp == 0.5

        # Vertical PID (different gains)
        assert config.pid_vertical_kp == 0.3
        assert config.pid_vertical_kd == 0.05

        # Yaw PID
        assert config.pid_yaw_kp == 1.0
        assert config.pid_yaw_ki == 0.0

    def test_velocity_limits(self):
        """Test velocity limit defaults."""
        config = Config()

        assert config.max_forward_velocity == 2.0
        assert config.max_lateral_velocity == 2.0
        assert config.max_vertical_velocity == 1.0
        assert config.max_yaw_rate == 30.0

    def test_safety_settings(self):
        """Test safety setting defaults."""
        config = Config()

        assert config.min_confidence == 0.3
        assert config.max_path_deviation_factor == 1.5
        assert config.min_battery_percent == 25.0
        assert config.max_height_deviation_m == 2.0

    def test_communication_settings(self):
        """Test communication setting defaults."""
        config = Config()

        assert config.websocket_port == 8765
        assert config.frame_timeout_ms == 500
        assert config.command_timeout_ms == 1000
        assert config.abort_timeout_ms == 2000
        assert config.target_frame_rate_hz == 10.0

    def test_image_size_tuple(self):
        """Test image size is a tuple of (width, height)."""
        config = Config()

        width, height = config.image_size
        assert width == 512
        assert height == 384
        assert width / height == 4 / 3  # 4:3 aspect ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

