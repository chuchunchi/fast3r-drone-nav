"""Keyframe stack management for teach-and-repeat navigation."""

import logging
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Telemetry:
    """Telemetry data from the drone."""

    timestamp_ms: int  # Epoch milliseconds
    velocity_x: float  # Forward velocity (m/s)
    velocity_y: float  # Right velocity (m/s)
    velocity_z: float  # Down velocity (m/s)
    yaw: float  # Yaw angle (degrees)
    pitch: float  # Pitch angle (degrees)
    roll: float  # Roll angle (degrees)
    height: float  # Height above ground (meters)


@dataclass
class Keyframe:
    """A keyframe in the navigation stack."""

    index: int  # Keyframe index
    image: np.ndarray  # RGB image (H, W, 3)
    timestamp_ms: int  # Capture timestamp
    cumulative_distance: float  # Distance from start (meters)
    telemetry: Optional[Telemetry] = None  # Associated telemetry
    pointmap: Optional[torch.Tensor] = None  # Cached 3D pointmap
    confidence: Optional[torch.Tensor] = None  # Cached confidence map
    scale_factor: Optional[float] = None  # Scale factor to next keyframe


class KeyframeStackManager:
    """
    Manages keyframe stack for teach-and-repeat navigation.

    During TEACH phase:
    - Receives frames and telemetry
    - Pushes keyframes based on distance or time thresholds
    - Tracks cumulative distance from IMU integration

    During REPEAT phase:
    - Provides target keyframes for homing
    - Supports stack pop when waypoint is reached
    """

    def __init__(
        self,
        keyframe_interval_m: float = 2.0,
        keyframe_interval_s: float = 3.0,
    ):
        """
        Initialize keyframe stack manager.

        Args:
            keyframe_interval_m: Push keyframe every N meters.
            keyframe_interval_s: Push keyframe every N seconds (whichever first).
        """
        self.keyframe_interval_m = keyframe_interval_m
        self.keyframe_interval_s = keyframe_interval_s

        # Stack storage
        self.stack: List[Keyframe] = []

        # Distance tracking
        self.cumulative_distance: float = 0.0
        self.last_telemetry_time: Optional[int] = None

        # Scale calibration
        self.global_scale_factor: float = 1.0
        self._scale_factors: List[float] = []

    def process_frame(
        self,
        frame: np.ndarray,
        telemetry: Telemetry,
        force_keyframe: bool = False,
    ) -> Optional[Keyframe]:
        """
        Process an incoming frame during TEACH phase.

        Args:
            frame: RGB image (H, W, 3).
            telemetry: Current telemetry data.
            force_keyframe: Force pushing a keyframe.

        Returns:
            New Keyframe if one was pushed, None otherwise.
        """
        # Update cumulative distance from IMU velocity
        if self.last_telemetry_time is not None:
            dt = (telemetry.timestamp_ms - self.last_telemetry_time) / 1000.0
            if dt > 0:
                velocity_magnitude = sqrt(
                    telemetry.velocity_x ** 2
                    + telemetry.velocity_y ** 2
                    + telemetry.velocity_z ** 2
                )
                self.cumulative_distance += velocity_magnitude * dt

        self.last_telemetry_time = telemetry.timestamp_ms

        # First frame always becomes a keyframe
        if len(self.stack) == 0:
            return self._push_keyframe(frame, telemetry)

        # Check if we should push a new keyframe
        should_push = force_keyframe

        if not should_push:
            distance_since_last = (
                self.cumulative_distance - self.stack[-1].cumulative_distance
            )
            if distance_since_last >= self.keyframe_interval_m:
                should_push = True

        if not should_push:
            time_since_last_ms = telemetry.timestamp_ms - self.stack[-1].timestamp_ms
            if time_since_last_ms >= self.keyframe_interval_s * 1000:
                should_push = True

        if should_push:
            return self._push_keyframe(frame, telemetry)

        return None

    def _push_keyframe(
        self, frame: np.ndarray, telemetry: Telemetry
    ) -> Keyframe:
        """Push a new keyframe onto the stack."""
        keyframe = Keyframe(
            index=len(self.stack),
            image=frame.copy(),
            timestamp_ms=telemetry.timestamp_ms,
            cumulative_distance=self.cumulative_distance,
            telemetry=telemetry,
        )
        self.stack.append(keyframe)

        logger.info(
            f"Pushed keyframe {keyframe.index} at distance {self.cumulative_distance:.2f}m"
        )

        return keyframe

    def get_target_keyframe(self, index: int = -1) -> Optional[Keyframe]:
        """
        Get a keyframe from the stack.

        Args:
            index: Stack index (-1 for top/most recent).

        Returns:
            Keyframe or None if stack is empty.
        """
        if not self.stack:
            return None

        if index < 0:
            index = len(self.stack) + index

        if 0 <= index < len(self.stack):
            return self.stack[index]

        return None

    def pop_keyframe(self) -> Optional[Keyframe]:
        """
        Pop the top keyframe from the stack.

        Returns:
            Popped keyframe or None if stack is empty.
        """
        if not self.stack:
            return None

        keyframe = self.stack.pop()
        logger.info(
            f"Popped keyframe {keyframe.index}, {len(self.stack)} remaining"
        )
        return keyframe

    def get_stack_size(self) -> int:
        """Get number of keyframes in stack."""
        return len(self.stack)

    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self.stack) == 0

    def get_total_distance(self) -> float:
        """Get total recorded distance in meters."""
        return self.cumulative_distance

    def set_scale_factor(self, keyframe_idx: int, scale: float) -> None:
        """
        Set scale factor for a keyframe.

        Args:
            keyframe_idx: Keyframe index.
            scale: Scale factor (meters per Fast3R unit).
        """
        if 0 <= keyframe_idx < len(self.stack):
            self.stack[keyframe_idx].scale_factor = scale
            self._scale_factors.append(scale)

    def compute_global_scale(self) -> float:
        """
        Compute global scale factor from all keyframe pairs.

        Returns:
            Median scale factor.
        """
        if not self._scale_factors:
            return 1.0

        self.global_scale_factor = float(np.median(self._scale_factors))
        logger.info(f"Global scale factor: {self.global_scale_factor:.4f}")
        return self.global_scale_factor

    def get_inter_keyframe_distance(self, idx1: int, idx2: int) -> float:
        """
        Get IMU-based distance between two keyframes.

        Args:
            idx1: First keyframe index.
            idx2: Second keyframe index.

        Returns:
            Distance in meters.
        """
        if not (0 <= idx1 < len(self.stack) and 0 <= idx2 < len(self.stack)):
            return 0.0

        return abs(
            self.stack[idx2].cumulative_distance
            - self.stack[idx1].cumulative_distance
        )

    def cache_pointmap(
        self,
        keyframe_idx: int,
        pointmap: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        """
        Cache computed pointmap for a keyframe.

        Args:
            keyframe_idx: Keyframe index.
            pointmap: 3D pointmap tensor.
            confidence: Confidence tensor.
        """
        if 0 <= keyframe_idx < len(self.stack):
            self.stack[keyframe_idx].pointmap = pointmap
            self.stack[keyframe_idx].confidence = confidence

    def clear(self) -> None:
        """Clear all keyframes and reset state."""
        self.stack.clear()
        self.cumulative_distance = 0.0
        self.last_telemetry_time = None
        self.global_scale_factor = 1.0
        self._scale_factors.clear()
        logger.info("Keyframe stack cleared")

    def get_keyframe_images(self) -> List[np.ndarray]:
        """Get all keyframe images."""
        return [kf.image for kf in self.stack]

    def get_keyframe_distances(self) -> List[float]:
        """Get cumulative distances for all keyframes."""
        return [kf.cumulative_distance for kf in self.stack]

    def __len__(self) -> int:
        return len(self.stack)

    def __iter__(self):
        return iter(self.stack)

    def __getitem__(self, idx: int) -> Keyframe:
        return self.stack[idx]


