"""Main homing controller integrating all components."""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config, default_config
from .coordinate_utils import (
    create_hover_command,
    fast3r_to_dji_command,
    reproject_pose_error_gimbal_aware,
)
from .fast3r_engine import Fast3REngine
from .keyframe_manager import Keyframe, KeyframeStackManager, Telemetry
from .pid_controller import MultiAxisPIDController
from .pose_estimator import PoseEstimator, PoseResult
from .state_machine import StateMachine, SystemState

logger = logging.getLogger(__name__)


@dataclass
class HomingResult:
    """Result from a homing iteration."""

    state: str  # Current state name
    command: Dict[str, float]  # Velocity command
    target_distance_m: float  # Distance to current target
    keyframes_remaining: int  # Number of keyframes left
    confidence: float  # Pose estimation confidence
    pose_result: Optional[PoseResult] = None  # Full pose result
    low_confidence: bool = False  # True if confidence below threshold
    inference_time_ms: float = 0.0  # Time for Fast3R inference


@dataclass
class HomingStats:
    """Statistics for homing phase."""
    
    total_frames: int = 0
    successful_poses: int = 0
    low_confidence_frames: int = 0
    failed_poses: int = 0
    waypoints_reached: int = 0
    total_inference_time_ms: float = 0.0
    command_history: List[Dict[str, float]] = field(default_factory=list)
    
    @property
    def avg_inference_time_ms(self) -> float:
        return self.total_inference_time_ms / max(1, self.total_frames)
    
    @property
    def success_rate(self) -> float:
        return self.successful_poses / max(1, self.total_frames)


class HomingController:
    """
    Main controller for visual homing.

    Integrates:
    - Fast3R inference for 3D reconstruction
    - SVD Procrustes for pose estimation
    - Time-based control (fixed velocity, calculated duration)
    - Keyframe stack management

    Operates in two phases:
    1. TEACH: Record keyframes with IMU-based distance tracking
    2. REPEAT: Navigate back using visual matching

    Control approach:
    - Uses fixed velocity for stable, predictable flight
    - Calculates duration based on distance: duration = distance / velocity
    - Simple proportional control for yaw alignment
    """

    def __init__(
        self,
        fast3r_engine: Optional[Fast3REngine] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize homing controller.

        Args:
            fast3r_engine: Pre-initialized Fast3R engine.
            config: Configuration object.
        """
        self.config = config or default_config

        # Core components
        self.fast3r = fast3r_engine or Fast3REngine(config=self.config)
        self.pose_estimator = PoseEstimator(
            confidence_percentile=0.5,
            min_points=100,
        )
        self.keyframe_manager = KeyframeStackManager(
            keyframe_interval_m=self.config.keyframe_interval_m,
            keyframe_interval_s=self.config.keyframe_interval_s,
        )
        self.state_machine = StateMachine()

        # PID controller
        self.pid = MultiAxisPIDController(
            forward_gains=(
                self.config.pid_forward_kp,
                self.config.pid_forward_ki,
                self.config.pid_forward_kd,
            ),
            lateral_gains=(
                self.config.pid_lateral_kp,
                self.config.pid_lateral_ki,
                self.config.pid_lateral_kd,
            ),
            vertical_gains=(
                self.config.pid_vertical_kp,
                self.config.pid_vertical_ki,
                self.config.pid_vertical_kd,
            ),
            yaw_gains=(
                self.config.pid_yaw_kp,
                self.config.pid_yaw_ki,
                self.config.pid_yaw_kd,
            ),
            velocity_limits={
                "forward": self.config.max_forward_velocity,
                "lateral": self.config.max_lateral_velocity,
                "vertical": self.config.max_vertical_velocity,
                "yaw": self.config.max_yaw_rate,
            },
        )

        # Homing state
        self.target_idx: int = -1
        self.metric_scale: float = 1.0
        self.gimbal_pitch_deg: float = 0.0

        # Rate limiting for safety
        self._last_command_time: float = 0.0

        # Waypoint confirmation state
        self._waypoint_confirm_count: int = 0

        # Statistics and debugging
        self.stats = HomingStats()
        self._consecutive_failures: int = 0
        self._max_consecutive_failures: int = 10  # Skip keyframe after 10 failures

        # Last pose result for debugging
        self._last_pose_result: Optional[PoseResult] = None
        self._last_translation: Optional[np.ndarray] = None

        # Stuck detection - skip keyframe if distance doesn't decrease
        self._last_distance: float = float('inf')
        self._frames_without_progress: int = 0

        # Cached target encoding for pairwise inference
        self._cached_target_encoding: Optional[tuple] = None
        self._cached_target_idx: int = -1

    def initialize(self) -> None:
        """Initialize the controller (load model, etc.)."""
        if not self.fast3r.is_loaded():
            self.fast3r.load_model()

    # =========================================================================
    # TEACH Phase Methods
    # =========================================================================

    def start_recording(self) -> bool:
        """Start recording keyframes."""
        if self.state_machine.start_recording():
            self.keyframe_manager.clear()
            return True
        return False

    def process_teach_frame(
        self,
        frame: np.ndarray,
        telemetry: Telemetry,
    ) -> Optional[Keyframe]:
        """
        Process a frame during TEACH phase.

        Args:
            frame: RGB image.
            telemetry: Current telemetry.

        Returns:
            New keyframe if one was pushed.
        """
        if not self.state_machine.is_recording():
            logger.warning("Not in RECORDING state")
            return None

        return self.keyframe_manager.process_frame(frame, telemetry)

    def stop_recording(self, save_metadata_to: Optional[str] = None) -> bool:
        """
        Stop recording and compute scale factors.

        Args:
            save_metadata_to: Optional directory to save keyframe metadata (e.g., 'videos/teach_target')

        Returns:
            True if successful.
        """
        if not self.state_machine.stop_recording():
            return False

        # Compute scale factors between consecutive keyframes
        self._calibrate_scale()

        # Save metadata if requested
        if save_metadata_to:
            from pathlib import Path
            self.keyframe_manager.save_metadata(Path(save_metadata_to))

        return True

    def _calibrate_scale(self) -> None:
        """Calibrate metric scale from IMU distances and Fast3R."""
        if len(self.keyframe_manager) < 2:
            logger.warning("Not enough keyframes for scale calibration")
            return

        logger.info("Calibrating scale factors...")

        for i in range(1, len(self.keyframe_manager)):
            kf_prev = self.keyframe_manager[i - 1]
            kf_curr = self.keyframe_manager[i]

            # Get IMU distance
            imu_distance = self.keyframe_manager.get_inter_keyframe_distance(
                i - 1, i
            )

            if imu_distance < 0.1:
                logger.warning(f"Keyframe pair {i-1}-{i} has very small distance")
                continue

            # Run Fast3R on pair
            result = self.fast3r.infer_pair(kf_prev.image, kf_curr.image)

            # Compute scale factor
            scale = self.pose_estimator.compute_scale_factor(
                result["pts3d_1"],
                result["pts3d_2"],
                result["conf_1"],
                imu_distance,
            )

            self.keyframe_manager.set_scale_factor(i, scale)

        # Compute global scale
        self.metric_scale = self.keyframe_manager.compute_global_scale()
        self.pose_estimator.set_metric_scale(self.metric_scale)

    # =========================================================================
    # REPEAT Phase Methods
    # =========================================================================

    def start_homing(self) -> bool:
        """Start homing phase."""
        if not self.state_machine.start_homing():
            return False

        # Start from last keyframe
        self.target_idx = len(self.keyframe_manager) - 1
        self.pid.reset()
        
        # Reset homing-specific state
        self.stats = HomingStats()
        self._consecutive_failures = 0
        self._last_pose_result = None
        self._last_translation = None
        self._last_command_time = 0.0
        self._waypoint_confirm_count = 0
        self._last_distance = float('inf')
        self._frames_without_progress = 0

        # Pre-encode the first target keyframe
        self._cache_target_encoding()

        logger.info(f"Starting homing with {self.target_idx + 1} keyframes")
        logger.info(f"Metric scale: {self.metric_scale:.4f}")
        return True

    def should_compute_new_command(self) -> bool:
        """
        Check if enough time has passed to compute a new command.

        Returns:
            True if a new command should be computed, False if should send hover.
        """
        if self._last_command_time == 0.0:
            return True  # First command

        elapsed = time.time() - self._last_command_time
        return elapsed >= self.config.command_update_interval_s

    def _clamp_total_velocity(self, command: Dict[str, float]) -> Dict[str, float]:
        """
        Clamp total translational velocity magnitude to max_total_velocity.

        Prevents dangerous diagonal speeds where individual axis limits
        would allow e.g. forward=1.0 + lateral=1.0 = 1.41 m/s total.
        """
        vx = command["pitch_velocity"]
        vy = command["roll_velocity"]
        vz = command["vertical_velocity"]
        magnitude = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        max_total = self.config.max_total_velocity
        if magnitude > max_total and magnitude > 0:
            scale = max_total / magnitude
            command["pitch_velocity"] *= scale
            command["roll_velocity"] *= scale
            command["vertical_velocity"] *= scale
            logger.debug(
                f"Velocity magnitude {magnitude:.2f} m/s exceeded limit "
                f"{max_total:.2f} m/s, scaled down by {scale:.2f}"
            )

        return command

    def process_homing_frame(
        self,
        live_frame: np.ndarray,
        telemetry: Telemetry,
    ) -> HomingResult:
        """
        Process a frame during HOMING phase with simple rate limiting.

        Each iteration:
        1. Computes fresh command from current frame
        2. Adds command duration for drone execution
        3. Sleeps for command_update_interval_s before returning

        Args:
            live_frame: Current camera frame (RGB).
            telemetry: Current telemetry.

        Returns:
            HomingResult with command and status.
        """
        self.stats.total_frames += 1

        # Check if homing is complete
        if self.target_idx < 0:
            self.state_machine.complete_homing()
            return HomingResult(
                state="COMPLETED",
                command=create_hover_command(),
                target_distance_m=0.0,
                keyframes_remaining=0,
                confidence=1.0,
            )

        if not self.state_machine.is_homing():
            return HomingResult(
                state=self.state_machine.state.name,
                command=create_hover_command(),
                target_distance_m=0.0,
                keyframes_remaining=0,
                confidence=0.0,
            )

        # Get target keyframe
        target_keyframe = self.keyframe_manager[self.target_idx]

        # Run Fast3R inference with timing (cached when possible)
        t_start = time.time()
        if (
            self._cached_target_encoding is not None
            and self._cached_target_idx == self.target_idx
        ):
            result = self.fast3r.infer_pair_cached(
                live_frame, self._cached_target_encoding
            )
        else:
            self._cache_target_encoding()
            assert self._cached_target_encoding is not None
            result = self.fast3r.infer_pair_cached(
                live_frame, self._cached_target_encoding
            )
        inference_time_ms = (time.time() - t_start) * 1000
        self.stats.total_inference_time_ms += inference_time_ms

        # Compute relative pose
        pose_result = self.pose_estimator.estimate_pose(
            result["pts3d_1"],  # Live frame points
            result["pts3d_2"],  # Target frame points
            result["conf_1"],  # Use live frame confidence
        )
        
        self._last_pose_result = pose_result

        # Handle pose estimation failure
        if not pose_result.success:
            self.stats.failed_poses += 1
            self._consecutive_failures += 1
            logger.warning(
                f"Pose estimation failed (attempt {self._consecutive_failures}), hovering"
            )
            
            # Skip to next keyframe if too many consecutive failures
            if self._consecutive_failures >= self._max_consecutive_failures:
                logger.warning(
                    f"Too many failures on keyframe {self.target_idx}, skipping"
                )
                self._advance_to_next_keyframe()
            
            return HomingResult(
                state="HOMING",
                command=create_hover_command(),
                target_distance_m=float("inf"),
                keyframes_remaining=self.target_idx + 1,
                confidence=pose_result.confidence,
                pose_result=pose_result,
                inference_time_ms=inference_time_ms,
            )

        # Extract pose error in camera frame
        t_cam = pose_result.translation.cpu().numpy()
        R = pose_result.rotation
        self._last_translation = t_cam.copy()

        (
            error_forward,
            error_lateral,
            error_yaw,
            distance_to_target,
        ) = reproject_pose_error_gimbal_aware(
            t_cam=t_cam,
            R_cam=R,
            gimbal_pitch_deg=self.gimbal_pitch_deg,
        )
        
        # Check for low confidence
        low_confidence = pose_result.confidence < self.config.min_confidence
        if low_confidence:
            self.stats.low_confidence_frames += 1
            logger.debug(
                f"Low confidence ({pose_result.confidence:.3f}), "
                f"reducing velocity"
            )
            # Reduce velocities when confidence is low
            velocity_scale = pose_result.confidence / self.config.min_confidence
            velocity_scale = max(0.3, min(1.0, velocity_scale))  # Clamp to [0.3, 1.0]
        else:
            self.stats.successful_poses += 1
            self._consecutive_failures = 0  # Reset failure counter on success
            velocity_scale = 1.0

        # Check if we've reached the waypoint (with confirmation)
        if distance_to_target < self.config.waypoint_threshold_m and not low_confidence:
            self._waypoint_confirm_count += 1
            logger.info(
                f"Waypoint proximity: {distance_to_target:.2f}m < "
                f"{self.config.waypoint_threshold_m}m "
                f"(confirm {self._waypoint_confirm_count}/{self.config.waypoint_confirm_frames})"
            )
            if self._waypoint_confirm_count >= self.config.waypoint_confirm_frames:
                self._advance_to_next_keyframe()
                self.stats.waypoints_reached += 1
        else:
            # Reset confirmation counter if we're not within threshold
            if self._waypoint_confirm_count > 0:
                logger.debug(f"Waypoint confirmation reset: distance={distance_to_target:.2f}m")
            self._waypoint_confirm_count = 0

        # Stuck detection: check if making progress toward target
        # Use a more lenient check: if distance improved OR stayed roughly same (within 20%)
        distance_tolerance = self._last_distance * 0.2  # Allow 20% variation

        if distance_to_target < self._last_distance - self.config.progress_threshold_m:
            # Clear progress, reset counter
            self._frames_without_progress = 0
            self._last_distance = distance_to_target
            logger.debug(f"Progress: distance reduced to {distance_to_target:.2f}m")
        elif distance_to_target <= self._last_distance + distance_tolerance:
            # Distance stayed roughly the same or slightly worse, but acceptable
            # This happens with single-axis control when addressing one error at a time
            # Don't reset counter, but don't increment aggressively
            if self._frames_without_progress < self.config.max_frames_without_progress // 2:
                # Only increment slowly in the first half
                self._frames_without_progress += 1
        else:
            # Distance significantly worse
            self._frames_without_progress += 1
            logger.debug(
                f"No progress: dist={distance_to_target:.2f}m, "
                f"last={self._last_distance:.2f}m, "
                f"frames={self._frames_without_progress}/{self.config.max_frames_without_progress}"
            )

            # Skip keyframe if stuck too long
            if self._frames_without_progress >= self.config.max_frames_without_progress:
                logger.warning(
                    f"Stuck on keyframe {self.target_idx} for {self._frames_without_progress} frames, "
                    f"distance not decreasing (current={distance_to_target:.2f}m, "
                    f"best={self._last_distance:.2f}m). Skipping keyframe."
                )
                self._advance_to_next_keyframe()
                return HomingResult(
                    state="HOMING",
                    command=create_hover_command(),
                    target_distance_m=distance_to_target,
                    keyframes_remaining=self.target_idx + 1,
                    confidence=pose_result.confidence,
                    pose_result=pose_result,
                    low_confidence=low_confidence,
                    inference_time_ms=inference_time_ms,
                )

        # Single-axis control: move only in one direction at a time
        # This provides more stable and predictable flight
        # SAFETY: Vertical control disabled - only forward and lateral movement

        # Find the axis with largest error (excluding vertical for safety)
        errors_abs = [abs(error_forward), abs(error_lateral)]
        max_error_idx = np.argmax(errors_abs)
        max_error = errors_abs[max_error_idx]

        # Define minimum error threshold for movement
        min_error_threshold = 0.1  # 10cm

        if max_error > min_error_threshold:
            # Move only on the axis with largest error
            velocity_vector = np.zeros(3)

            # Set velocity for the dominant axis
            fixed_vel = self.config.fixed_flight_velocity * velocity_scale

            if max_error_idx == 0:  # Forward/backward
                velocity_vector[0] = fixed_vel if error_forward > 0 else -fixed_vel
                axis_name = "forward" if error_forward > 0 else "backward"
            elif max_error_idx == 1:  # Lateral (left/right)
                velocity_vector[1] = fixed_vel if error_lateral > 0 else -fixed_vel
                axis_name = "right" if error_lateral > 0 else "left"
            # Vertical control removed for safety

            # Calculate duration to reach target on this axis
            # duration = distance / velocity
            calculated_duration = max_error / fixed_vel
            # Clamp duration to reasonable range
            duration = max(0.5, min(calculated_duration, self.config.command_duration_s))

            logger.debug(f"Single-axis: moving {axis_name}, error={max_error:.2f}m, dur={duration:.1f}s")
        else:
            # All errors below threshold, hover
            velocity_vector = np.zeros(3)
            duration = 0.5
            logger.debug("All axes below threshold, hovering")

        # Simple proportional control for yaw
        yaw_rate = self.config.pid_yaw_kp * error_yaw
        yaw_rate = np.clip(yaw_rate, -self.config.max_yaw_rate, self.config.max_yaw_rate)

        # Build command (DJI coordinate system)
        # SAFETY: vertical_velocity always 0 for safety
        command = {
            "pitch_velocity": float(velocity_vector[0]),  # forward
            "roll_velocity": float(velocity_vector[1]),   # lateral
            "vertical_velocity": 0.0,  # DISABLED FOR SAFETY
            "yaw_rate": float(yaw_rate),
        }

        # Apply safety clamps to individual axes
        command["pitch_velocity"] = np.clip(
            command["pitch_velocity"],
            -self.config.max_forward_velocity,
            self.config.max_forward_velocity
        )
        command["roll_velocity"] = np.clip(
            command["roll_velocity"],
            -self.config.max_lateral_velocity,
            self.config.max_lateral_velocity
        )
        command["vertical_velocity"] = np.clip(
            command["vertical_velocity"],
            -self.config.max_vertical_velocity,
            self.config.max_vertical_velocity
        )

        # Clamp total translational velocity magnitude
        command = self._clamp_total_velocity(command)

        # Add calculated duration for drone to execute this command
        command["duration_s"] = duration

        # Store command in history (for debugging/visualization)
        self.stats.command_history.append(command.copy())

        # Log
        logger.info(
            f"Frame {self.stats.total_frames}: "
            f"target={self.target_idx}, dist={distance_to_target:.2f}m, "
            f"conf={pose_result.confidence:.2f}, "
            f"cmd=[fwd={command['pitch_velocity']:.2f}, lat={command['roll_velocity']:.2f}, "
            f"vert={command['vertical_velocity']:.2f}, yaw={command['yaw_rate']:.1f}, "
            f"dur={command['duration_s']:.1f}s]"
        )

        # Update last command time for rate limiting (handled by caller)
        self._last_command_time = time.time()

        return HomingResult(
            state="HOMING",
            command=command,
            target_distance_m=distance_to_target,
            keyframes_remaining=self.target_idx + 1,
            confidence=pose_result.confidence,
            pose_result=pose_result,
            low_confidence=low_confidence,
            inference_time_ms=inference_time_ms,
        )

    def _cache_target_encoding(self) -> None:
        """Encode and cache the current target keyframe."""
        if self.target_idx < 0:
            self._cached_target_encoding = None
            self._cached_target_idx = -1
            return

        target_kf = self.keyframe_manager[self.target_idx]
        self._cached_target_encoding = self.fast3r.encode_target(
            target_kf.image
        )
        self._cached_target_idx = self.target_idx
        logger.debug(f"Cached encoder output for target keyframe {self.target_idx}")

    def _advance_to_next_keyframe(self) -> None:
        """Advance to the next keyframe in the stack."""
        logger.info(
            f"Reached keyframe {self.target_idx}, "
            f"advancing to {self.target_idx - 1}"
        )

        self.target_idx -= 1
        self.pid.reset_position()  # Reset position PIDs, keep yaw

        # Reset waypoint confirmation for new target
        self._waypoint_confirm_count = 0

        # Reset stuck detection for new target
        self._last_distance = float('inf')
        self._frames_without_progress = 0
        self._consecutive_failures = 0

        # Pre-encode the new target keyframe
        self._cache_target_encoding()

        if self.target_idx < 0:
            logger.info("All keyframes reached, homing complete!")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_state(self) -> SystemState:
        """Get current system state."""
        return self.state_machine.state

    def set_gimbal_pitch_deg(self, pitch_deg: float) -> None:
        """Set fixed gimbal pitch for homing (degrees, downward-positive)."""
        self.gimbal_pitch_deg = float(np.clip(pitch_deg, 0.0, 89.9))

    def get_keyframe_count(self) -> int:
        """Get number of recorded keyframes."""
        return len(self.keyframe_manager)

    def get_total_distance(self) -> float:
        """Get total recorded distance in meters."""
        return self.keyframe_manager.get_total_distance()

    def get_target_keyframe(self) -> Optional[Keyframe]:
        """Get current target keyframe."""
        if self.target_idx >= 0:
            return self.keyframe_manager[self.target_idx]
        return None

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.state_machine.reset()
        self.keyframe_manager.clear()
        self.pid.reset()
        self.target_idx = -1
        self.metric_scale = 1.0
        self.gimbal_pitch_deg = 0.0
        self.stats = HomingStats()
        self._consecutive_failures = 0
        self._last_pose_result = None
        self._last_translation = None
        self._last_command_time = 0.0
        self._waypoint_confirm_count = 0
        self._last_distance = float('inf')
        self._frames_without_progress = 0
        self._cached_target_encoding = None
        self._cached_target_idx = -1
        logger.info("Homing controller reset")

    def emergency_stop(self) -> Dict[str, float]:
        """Immediate stop - return hover command."""
        self.pid.reset()
        self._cached_target_encoding = None
        self._cached_target_idx = -1
        return create_hover_command()
    
    def get_stats(self) -> HomingStats:
        """Get homing statistics."""
        return self.stats
    
    def get_last_pose_result(self) -> Optional[PoseResult]:
        """Get the last pose estimation result for debugging."""
        return self._last_pose_result
    
    def get_last_translation(self) -> Optional[np.ndarray]:
        """Get the last translation vector for debugging."""
        return self._last_translation
    
    def get_debug_info(self) -> Dict:
        """Get debug information about current state."""
        info = {
            "state": self.state_machine.state.name,
            "target_idx": self.target_idx,
            "waypoint_confirm_count": self._waypoint_confirm_count,
            "keyframe_count": len(self.keyframe_manager),
            "total_distance_m": self.keyframe_manager.get_total_distance(),
            "metric_scale": self.metric_scale,
            "consecutive_failures": self._consecutive_failures,
            "stats": {
                "total_frames": self.stats.total_frames,
                "successful_poses": self.stats.successful_poses,
                "failed_poses": self.stats.failed_poses,
                "low_confidence_frames": self.stats.low_confidence_frames,
                "waypoints_reached": self.stats.waypoints_reached,
                "avg_inference_time_ms": self.stats.avg_inference_time_ms,
                "success_rate": self.stats.success_rate,
            },
        }
        if self._last_translation is not None:
            info["last_translation"] = self._last_translation.tolist()
        if self._last_pose_result is not None:
            info["last_confidence"] = self._last_pose_result.confidence
        return info


