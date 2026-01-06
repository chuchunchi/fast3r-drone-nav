"""Main homing controller integrating all components."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import Config, default_config
from .coordinate_utils import create_hover_command, extract_yaw_error, fast3r_to_dji_command
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


class HomingController:
    """
    Main controller for visual homing.

    Integrates:
    - Fast3R inference for 3D reconstruction
    - SVD Procrustes for pose estimation
    - PID control for velocity generation
    - Keyframe stack management

    Operates in two phases:
    1. TEACH: Record keyframes with IMU-based distance tracking
    2. REPEAT: Navigate back using visual matching
    """

    def __init__(
        self,
        fast3r_engine: Optional[Fast3REngine] = None,
        config: Config = None,
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

    def stop_recording(self) -> bool:
        """
        Stop recording and compute scale factors.

        Returns:
            True if successful.
        """
        if not self.state_machine.stop_recording():
            return False

        # Compute scale factors between consecutive keyframes
        self._calibrate_scale()

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

        logger.info(f"Starting homing with {self.target_idx + 1} keyframes")
        return True

    def process_homing_frame(
        self,
        live_frame: np.ndarray,
        telemetry: Telemetry,
    ) -> HomingResult:
        """
        Process a frame during HOMING phase.

        Args:
            live_frame: Current camera frame (RGB).
            telemetry: Current telemetry.

        Returns:
            HomingResult with command and status.
        """
        # Check if homing is complete
        if self.target_idx < 0:
            if self.state_machine.complete_homing():
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

        # Run Fast3R inference
        result = self.fast3r.infer_pair(live_frame, target_keyframe.image)

        # Compute relative pose
        pose_result = self.pose_estimator.estimate_pose(
            result["pts3d_1"],  # Live frame points
            result["pts3d_2"],  # Target frame points
            result["conf_1"],  # Use live frame confidence
        )

        if not pose_result.success:
            logger.warning("Pose estimation failed, hovering")
            return HomingResult(
                state="HOMING",
                command=create_hover_command(),
                target_distance_m=float("inf"),
                keyframes_remaining=self.target_idx + 1,
                confidence=pose_result.confidence,
                pose_result=pose_result,
            )

        # Extract pose error in camera frame
        t_cam = pose_result.translation.cpu().numpy()
        R = pose_result.rotation

        # Camera frame: X=right, Y=down, Z=forward
        error_forward = float(t_cam[2])
        error_lateral = float(t_cam[0])
        error_vertical = -float(t_cam[1])  # Invert: down→up
        error_yaw = extract_yaw_error(R)

        # Compute distance to target
        distance_to_target = np.linalg.norm(t_cam)

        # Check if we've reached the waypoint
        if distance_to_target < self.config.waypoint_threshold_m:
            self._advance_to_next_keyframe()

        # PID control
        command = self.pid.compute(
            error_forward,
            error_lateral,
            error_vertical,
            error_yaw,
        )

        return HomingResult(
            state="HOMING",
            command=command,
            target_distance_m=distance_to_target,
            keyframes_remaining=self.target_idx + 1,
            confidence=pose_result.confidence,
            pose_result=pose_result,
        )

    def _advance_to_next_keyframe(self) -> None:
        """Advance to the next keyframe in the stack."""
        logger.info(
            f"Reached keyframe {self.target_idx}, "
            f"advancing to {self.target_idx - 1}"
        )

        self.target_idx -= 1
        self.pid.reset_position()  # Reset position PIDs, keep yaw

        if self.target_idx < 0:
            logger.info("All keyframes reached, homing complete!")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_state(self) -> SystemState:
        """Get current system state."""
        return self.state_machine.state

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
        logger.info("Homing controller reset")

    def emergency_stop(self) -> Dict[str, float]:
        """Immediate stop - return hover command."""
        self.pid.reset()
        return create_hover_command()


