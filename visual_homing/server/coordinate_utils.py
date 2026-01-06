"""Coordinate frame utilities for converting between Fast3R and DJI frames."""

import math
from typing import Dict, Tuple

import numpy as np
import torch


def extract_yaw_error(R: torch.Tensor) -> float:
    """
    Extract yaw angle from rotation matrix.

    For camera frame (Y-down, Z-forward), yaw is rotation around the Y axis.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Yaw error in degrees.
    """
    # For camera frame where Z is forward:
    # Yaw = atan2(R[0,2], R[2,2])
    # This gives the rotation needed to align Z axes
    yaw_rad = torch.atan2(R[0, 2], R[2, 2])
    return float(yaw_rad) * 180.0 / math.pi


def extract_euler_angles(R: torch.Tensor) -> Tuple[float, float, float]:
    """
    Extract Euler angles (roll, pitch, yaw) from rotation matrix.

    Uses ZYX convention (yaw-pitch-roll) common in aerospace.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Tuple of (roll, pitch, yaw) in degrees.
    """
    # Handle gimbal lock
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    # Convert to degrees
    return (
        float(roll) * 180.0 / math.pi,
        float(pitch) * 180.0 / math.pi,
        float(yaw) * 180.0 / math.pi,
    )


def camera_to_body_translation(t_cam: np.ndarray) -> np.ndarray:
    """
    Convert translation from camera frame to DJI body frame.

    Camera frame (OpenCV/Fast3R):
        X = right, Y = down, Z = forward

    DJI Body frame:
        X = forward, Y = right, Z = down

    Args:
        t_cam: Translation in camera frame [tx, ty, tz].

    Returns:
        Translation in body frame [tx, ty, tz].
    """
    # Camera [X, Y, Z] -> Body [Z, X, Y]
    return np.array([t_cam[2], t_cam[0], t_cam[1]])


def body_to_virtualstick(t_body: np.ndarray) -> Dict[str, float]:
    """
    Convert body frame translation to VirtualStick errors.

    DJI VirtualStick in BODY mode uses:
        pitch_velocity → X axis (forward +)
        roll_velocity → Y axis (right +)
        vertical_velocity → Z axis (up +, INVERTED from body frame)

    Args:
        t_body: Translation in body frame [forward, right, down].

    Returns:
        Dictionary with error terms for each axis.
    """
    return {
        "forward": t_body[0],  # Forward distance
        "lateral": t_body[1],  # Right offset
        "vertical": -t_body[2],  # Up offset (inverted from down)
    }


def fast3r_to_dji_command(
    t_cam: np.ndarray,
    yaw_error_deg: float,
    pid_gains: Dict[str, float],
    velocity_limits: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Convert Fast3R pose error (camera frame) to DJI VirtualStick commands.

    Args:
        t_cam: Translation in camera frame [tx, ty, tz] in meters.
               tx = right offset, ty = down offset, tz = forward distance
        yaw_error_deg: Yaw error in degrees.
        pid_gains: Dictionary with 'kp_forward', 'kp_lateral', 'kp_vertical', 'kp_yaw'.
        velocity_limits: Optional limits. Defaults to safe values.

    Returns:
        Dictionary with DJI VirtualStick velocity commands.
    """
    if velocity_limits is None:
        velocity_limits = {
            "max_forward": 2.0,
            "max_lateral": 2.0,
            "max_vertical": 1.0,
            "max_yaw": 30.0,
        }

    # Map camera frame axes to DJI body frame
    # Camera Z (forward) → DJI pitch (forward velocity)
    # Camera X (right) → DJI roll (right velocity)
    # Camera Y (down) → DJI vertical (INVERTED: up velocity)

    error_forward = t_cam[2]  # Camera Z = how far forward to go
    error_lateral = t_cam[0]  # Camera X = how far right to go
    error_vertical = -t_cam[1]  # Camera Y inverted = how far up to go

    # Apply proportional control (simplified PID)
    cmd = {
        "pitch_velocity": pid_gains["kp_forward"] * error_forward,
        "roll_velocity": pid_gains["kp_lateral"] * error_lateral,
        "vertical_velocity": pid_gains["kp_vertical"] * error_vertical,
        "yaw_rate": pid_gains["kp_yaw"] * yaw_error_deg,
    }

    # Clamp to safe limits
    cmd["pitch_velocity"] = np.clip(
        cmd["pitch_velocity"],
        -velocity_limits["max_forward"],
        velocity_limits["max_forward"],
    )
    cmd["roll_velocity"] = np.clip(
        cmd["roll_velocity"],
        -velocity_limits["max_lateral"],
        velocity_limits["max_lateral"],
    )
    cmd["vertical_velocity"] = np.clip(
        cmd["vertical_velocity"],
        -velocity_limits["max_vertical"],
        velocity_limits["max_vertical"],
    )
    cmd["yaw_rate"] = np.clip(
        cmd["yaw_rate"],
        -velocity_limits["max_yaw"],
        velocity_limits["max_yaw"],
    )

    return cmd


def create_hover_command() -> Dict[str, float]:
    """Create a hover command (zero velocities)."""
    return {
        "pitch_velocity": 0.0,
        "roll_velocity": 0.0,
        "vertical_velocity": 0.0,
        "yaw_rate": 0.0,
    }


def rotation_matrix_from_euler(
    roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (ZYX convention).

    Args:
        roll: Roll angle in degrees.
        pitch: Pitch angle in degrees.
        yaw: Yaw angle in degrees.

    Returns:
        3x3 rotation matrix.
    """
    # Convert to radians
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r), math.cos(r)],
    ])

    Ry = np.array([
        [math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)],
    ])

    Rz = np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y), math.cos(y), 0],
        [0, 0, 1],
    ])

    # ZYX order: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


