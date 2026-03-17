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
    keyframe_interval_m: float = 2  # Push keyframe every 2 meters
    keyframe_interval_s: float = 4.0  # OR every 3 seconds (whichever first)

    # Homing Settings
    waypoint_threshold_m: float = 1  # Pop keyframe when within 1.5m (increased for faster transitions)
    waypoint_confirm_frames: int = 1   # Must be within threshold for N consecutive frames
    confidence_threshold: float = 0.3  # Minimum confidence for valid pose
    max_frames_without_progress: int = 100  # Skip keyframe after N frames without progress (increased)
    progress_threshold_m: float = 0.05  # Minimum progress required (meters) - more lenient

    # PID Gains
    pid_forward_kp: float = 5.0
    pid_forward_ki: float = 0.1
    pid_forward_kd: float = 0.5

    pid_lateral_kp: float = 5.0
    pid_lateral_ki: float = 0.1
    pid_lateral_kd: float = 0.5

    pid_vertical_kp: float = 0.0  # Disabled for safety
    pid_vertical_ki: float = 0.0  # Disabled for safety
    pid_vertical_kd: float = 0.0  # Disabled for safety

    pid_yaw_kp: float = 0.6  # Disabled for experiment (was 1.0)
    pid_yaw_ki: float = 0.0
    pid_yaw_kd: float = 0.01

    # Fixed velocity for time-based control (m/s)
    fixed_flight_velocity: float = 0.4  # Optimized velocity for faster flight while maintaining stability

    # Velocity Limits (m/s and deg/s) - used as safety clamps
    max_forward_velocity: float = 1.0
    max_lateral_velocity: float = 1.0
    max_vertical_velocity: float = 0.3
    max_yaw_rate: float = 10.0
    max_total_velocity: float = 1.2  # Total velocity vector magnitude limit (m/s)

    # Safety Settings
    min_confidence: float = 0.3
    max_path_deviation_factor: float = 1.5  # Abort if > 1.5x outbound distance
    min_battery_percent: float = 25.0
    max_height_deviation_m: float = 2.0

    # Command Rate Limiting (Simple Safety)
    # Process frames continuously but limit command update rate with sleep
    command_update_interval_s: float = 0.5  # Sleep time between commands
    command_duration_s: float = 0.5        # How long each command executes on drone (increased for longer movements)

    # Communication Settings
    websocket_port: int = 8765
    frame_timeout_ms: int = 500
    command_timeout_ms: int = 1000
    abort_timeout_ms: int = 2000

    # Frame Rate
    target_frame_rate_hz: float = 10.0

    # Video Recording Settings
    video_recording_enabled: bool = True
    video_output_dir: str = "./videos"
    video_fps: float = 10.0


# Default configuration instance
default_config = Config()


