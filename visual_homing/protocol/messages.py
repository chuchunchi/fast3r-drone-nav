"""Message definitions for client-server communication."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class MessageType(IntEnum):
    """Binary message types."""

    FRAME_TELEMETRY = 0x01  # Frame + telemetry from client
    CONTROL_COMMAND = 0x02  # Control command from server
    STATE_UPDATE = 0x03  # State update from server
    HEARTBEAT = 0x04  # Heartbeat
    EMERGENCY_STOP = 0x05  # Emergency stop
    START_RECORDING = 0x10  # Start recording command
    STOP_RECORDING = 0x11  # Stop recording command
    START_HOMING = 0x12  # Start homing command
    RESET = 0x13  # Reset system


@dataclass
class Telemetry:
    """Telemetry data from the drone."""

    velocity_x: float  # Forward velocity (m/s)
    velocity_y: float  # Right velocity (m/s)
    velocity_z: float  # Down velocity (m/s)
    yaw: float  # Yaw angle (degrees)
    pitch: float  # Pitch angle (degrees)
    roll: float  # Roll angle (degrees)
    height: float  # Height above ground (meters)


@dataclass
class FrameMessage:
    """Frame + telemetry message from client."""

    message_type: MessageType
    frame_id: int  # Sequential frame ID
    timestamp_ms: int  # Epoch milliseconds
    telemetry: Telemetry
    image_data: bytes  # JPEG compressed image


@dataclass
class ControlCommand:
    """Velocity control command."""

    pitch_velocity: float  # Forward/backward (m/s)
    roll_velocity: float  # Right/left (m/s)
    vertical_velocity: float  # Up/down (m/s)
    yaw_rate: float  # Rotation rate (deg/s)


@dataclass
class StatusMessage:
    """System status message."""

    state: str  # Current state name
    keyframes_remaining: int  # Keyframes left to navigate
    target_distance_m: float  # Distance to current target
    confidence: float  # Pose estimation confidence
    total_keyframes: int  # Total keyframes recorded
    total_distance_m: float  # Total path distance


@dataclass
class ControlResponse:
    """Response from server to client."""

    message_type: MessageType
    frame_id: int  # Corresponding frame ID
    timestamp_ms: int  # Server timestamp
    command: ControlCommand
    status: StatusMessage


