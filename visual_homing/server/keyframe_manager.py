"""Keyframe stack management for teach-and-repeat navigation."""

import json
import logging
import time
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
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

    # Velocity constraints for robust distance tracking
    MIN_VELOCITY_THRESHOLD: float = 0.05  # Ignore velocities below this (noise)
    MAX_VELOCITY_THRESHOLD: float = 10.0  # Reject velocities above this (unrealistic)
    MAX_DT_SECONDS: float = 0.5  # Max time gap to integrate (reject gaps > 500ms)
    
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
        self._last_velocity: float = 0.0  # For smoothing/debugging

        # Scale calibration
        self.global_scale_factor: float = 1.0
        self._scale_factors: List[float] = []
        
        # Quality metrics
        self._velocity_samples: List[float] = []
        self._rejected_samples: int = 0

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
        # Update cumulative distance from IMU velocity with robustness checks
        if self.last_telemetry_time is not None:
            dt = (telemetry.timestamp_ms - self.last_telemetry_time) / 1000.0
            
            # Compute velocity magnitude
            velocity_magnitude = sqrt(
                telemetry.velocity_x ** 2
                + telemetry.velocity_y ** 2
                + telemetry.velocity_z ** 2
            )
            
            # Apply robustness checks
            valid_sample = True
            
            # Check 1: Reject unrealistic time gaps (missed frames, reconnection)
            if dt <= 0 or dt > self.MAX_DT_SECONDS:
                logger.debug(f"Rejected dt={dt:.3f}s (outside valid range)")
                valid_sample = False
            
            # Check 2: Reject unrealistically high velocities
            if velocity_magnitude > self.MAX_VELOCITY_THRESHOLD:
                logger.warning(
                    f"Rejected velocity={velocity_magnitude:.2f}m/s "
                    f"(exceeds {self.MAX_VELOCITY_THRESHOLD}m/s)"
                )
                self._rejected_samples += 1
                valid_sample = False
            
            # Check 3: Filter out noise when nearly stationary
            if velocity_magnitude < self.MIN_VELOCITY_THRESHOLD:
                velocity_magnitude = 0.0  # Treat as stationary
            
            # Integrate if valid
            if valid_sample and dt > 0:
                distance_delta = velocity_magnitude * dt
                self.cumulative_distance += distance_delta
                self._last_velocity = velocity_magnitude
                
                # Track for quality metrics
                if velocity_magnitude > 0:
                    self._velocity_samples.append(velocity_magnitude)

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
        self._last_velocity = 0.0
        self.global_scale_factor = 1.0
        self._scale_factors.clear()
        self._velocity_samples.clear()
        self._rejected_samples = 0
        logger.info("Keyframe stack cleared")
    
    def get_velocity_stats(self) -> dict:
        """
        Get velocity statistics for quality assessment.
        
        Returns:
            Dictionary with velocity statistics.
        """
        if not self._velocity_samples:
            return {
                "mean_velocity": 0.0,
                "min_velocity": 0.0,
                "max_velocity": 0.0,
                "std_velocity": 0.0,
                "num_samples": 0,
                "rejected_samples": self._rejected_samples,
                "quality": "no_data",
            }
        
        samples = np.array(self._velocity_samples)
        mean_vel = float(np.mean(samples))
        std_vel = float(np.std(samples))
        
        # Assess quality based on velocity consistency
        # Lower std relative to mean = more consistent = better
        cv = std_vel / mean_vel if mean_vel > 0 else float('inf')  # Coefficient of variation
        
        if cv < 0.3 and 0.3 <= mean_vel <= 3.0:
            quality = "excellent"
        elif cv < 0.5 and 0.2 <= mean_vel <= 4.0:
            quality = "good"
        elif cv < 0.8:
            quality = "acceptable"
        else:
            quality = "poor"
        
        return {
            "mean_velocity": mean_vel,
            "min_velocity": float(np.min(samples)),
            "max_velocity": float(np.max(samples)),
            "std_velocity": std_vel,
            "coefficient_of_variation": cv,
            "num_samples": len(samples),
            "rejected_samples": self._rejected_samples,
            "quality": quality,
        }

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

    def save_metadata(self, output_dir: Path) -> None:
        """
        Save keyframe metadata including metric scale to JSON file.

        Args:
            output_dir: Directory to save metadata (e.g., 'videos/teach_target')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "num_keyframes": len(self.stack),
            "total_distance_m": self.cumulative_distance,
            "global_scale_factor": self.global_scale_factor,
            "keyframe_interval_m": self.keyframe_interval_m,
            "keyframe_interval_s": self.keyframe_interval_s,
            "keyframes": [
                {
                    "index": kf.index,
                    "cumulative_distance_m": kf.cumulative_distance,
                    "timestamp_ms": kf.timestamp_ms,
                    "scale_factor": kf.scale_factor,
                }
                for kf in self.stack
            ],
            "velocity_stats": self.get_velocity_stats(),
        }

        metadata_path = output_dir / "keyframe_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved keyframe metadata to {metadata_path}")
        logger.info(f"  Keyframes: {len(self.stack)}")
        logger.info(f"  Total distance: {self.cumulative_distance:.2f}m")
        logger.info(f"  Global scale factor: {self.global_scale_factor:.4f}")

    @staticmethod
    def load_metadata(input_dir: Path) -> dict:
        """
        Load keyframe metadata from JSON file.

        Args:
            input_dir: Directory containing metadata file

        Returns:
            Metadata dictionary with scale factor and other info
        """
        input_dir = Path(input_dir)
        metadata_path = input_dir / "keyframe_metadata.json"

        if not metadata_path.exists():
            logger.warning(f"No metadata file found at {metadata_path}")
            return {"global_scale_factor": 1.0}

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded keyframe metadata from {metadata_path}")
        logger.info(f"  Global scale factor: {metadata.get('global_scale_factor', 1.0):.4f}")

        return metadata


