"""Configuration constants for the Visual Homing System."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """Configuration for the Visual Homing System."""

    # Fast3R Model Settings
    fast3r_checkpoint: str = "jedyang97/Fast3R_ViT_Large_512"
    device: str = "cuda"
    dtype: str = "float32"

    # Image Settings
    image_size: Tuple[int, int] = (512, 384)  # Width x Height (4:3 aspect ratio)
    jpeg_quality: int = 80

    # Keyframe Settings
    keyframe_interval_m: float = 2.0  # Push keyframe every 2 meters
    keyframe_interval_s: float = 3.0  # OR every 3 seconds (whichever first)

    # Homing Settings
    waypoint_threshold_m: float = 0.8  # Pop keyframe when within 0.8m
    confidence_threshold: float = 0.3  # Minimum confidence for valid pose

    # PID Gains
    pid_forward_kp: float = 0.5
    pid_forward_ki: float = 0.01
    pid_forward_kd: float = 0.1

    pid_lateral_kp: float = 0.5
    pid_lateral_ki: float = 0.01
    pid_lateral_kd: float = 0.1

    pid_vertical_kp: float = 0.3
    pid_vertical_ki: float = 0.01
    pid_vertical_kd: float = 0.05

    pid_yaw_kp: float = 1.0
    pid_yaw_ki: float = 0.0
    pid_yaw_kd: float = 0.2

    # Velocity Limits (m/s and deg/s)
    max_forward_velocity: float = 2.0
    max_lateral_velocity: float = 2.0
    max_vertical_velocity: float = 1.0
    max_yaw_rate: float = 30.0

    # Safety Settings
    min_confidence: float = 0.3
    max_path_deviation_factor: float = 1.5  # Abort if > 1.5x outbound distance
    min_battery_percent: float = 25.0
    max_height_deviation_m: float = 2.0

    # Communication Settings
    websocket_port: int = 8765
    frame_timeout_ms: int = 500
    command_timeout_ms: int = 1000
    abort_timeout_ms: int = 2000

    # Frame Rate
    target_frame_rate_hz: float = 10.0


# Default configuration instance
default_config = Config()


