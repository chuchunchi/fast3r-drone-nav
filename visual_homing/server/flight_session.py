"""Flight session management for organized data storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FlightSession:
    """
    Manages a single flight session with organized storage.

    Creates a folder structure:
        flight_YYYYMMDD_HHMMSS/
            ├── teach.mp4
            ├── homing.mp4
            ├── keyframes/
            │   ├── frame_0001.png
            │   └── ...
            ├── metadata.json
            └── flight_log.txt
    """

    def __init__(self, base_dir: str = "videos"):
        """
        Initialize flight session.

        Args:
            base_dir: Base directory for all flight sessions
        """
        self.base_dir = Path(base_dir)
        self.session_dir: Optional[Path] = None
        self.keyframes_dir: Optional[Path] = None
        self.session_name: Optional[str] = None

        # Paths for videos
        self.teach_video_path: Optional[Path] = None
        self.homing_video_path: Optional[Path] = None

        # Metadata collection
        self.metadata = {
            "session_start": None,
            "session_end": None,
            "teach_phase": {},
            "homing_phase": {},
            "config": {},
        }

        # Flight log for human-readable summary
        self.flight_log_lines = []

    def start_session(self) -> Path:
        """
        Start a new flight session.

        Creates the session folder structure.

        Returns:
            Path to session directory
        """
        # Create session name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"flight_{timestamp}"

        # Create session directory
        self.session_dir = self.base_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create keyframes subdirectory
        self.keyframes_dir = self.session_dir / "keyframes"
        self.keyframes_dir.mkdir(exist_ok=True)

        # Set video paths
        self.teach_video_path = self.session_dir / "teach.mp4"
        self.homing_video_path = self.session_dir / "homing.mp4"

        # Initialize metadata
        self.metadata["session_start"] = datetime.now().isoformat()
        self.metadata["session_name"] = self.session_name

        # Log
        logger.info(f"Started flight session: {self.session_dir}")
        self.log(f"Flight session started: {self.session_name}")

        return self.session_dir

    def save_keyframe_image(self, image: np.ndarray, keyframe_index: int) -> Path:
        """
        Save a keyframe image to the session folder.

        Args:
            image: RGB image array
            keyframe_index: Keyframe index (0-based)

        Returns:
            Path to saved image
        """
        if not self.keyframes_dir:
            raise RuntimeError("Session not started. Call start_session() first.")

        # Save as PNG for lossless quality
        filename = f"frame_{keyframe_index:04d}.png"
        filepath = self.keyframes_dir / filename

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), image_bgr)

        logger.debug(f"Saved keyframe {keyframe_index} to {filepath}")
        return filepath

    def save_teach_metadata(
        self,
        num_keyframes: int,
        total_distance_m: float,
        global_scale_factor: float,
        keyframe_distances: list,
        velocity_stats: dict,
    ) -> None:
        """
        Save teaching phase metadata.

        Args:
            num_keyframes: Number of keyframes recorded
            total_distance_m: Total distance traveled
            global_scale_factor: Computed metric scale
            keyframe_distances: List of cumulative distances for each keyframe
            velocity_stats: Velocity statistics from IMU
        """
        self.metadata["teach_phase"] = {
            "num_keyframes": num_keyframes,
            "total_distance_m": total_distance_m,
            "global_scale_factor": global_scale_factor,
            "keyframe_distances_m": keyframe_distances,
            "velocity_stats": velocity_stats,
            "teach_video": str(self.teach_video_path.name) if self.teach_video_path else None,
        }

        self.log(f"Teaching phase completed:")
        self.log(f"  Keyframes: {num_keyframes}")
        self.log(f"  Distance: {total_distance_m:.2f}m")
        self.log(f"  Metric scale: {global_scale_factor:.4f}")

    def save_homing_metadata(
        self,
        total_frames: int,
        successful_poses: int,
        failed_poses: int,
        waypoints_reached: int,
        avg_inference_time_ms: float,
        success_rate: float,
    ) -> None:
        """
        Save homing phase metadata.

        Args:
            total_frames: Total frames processed
            successful_poses: Number of successful pose estimations
            failed_poses: Number of failed pose estimations
            waypoints_reached: Number of waypoints reached
            avg_inference_time_ms: Average Fast3R inference time
            success_rate: Pose estimation success rate
        """
        self.metadata["homing_phase"] = {
            "total_frames": total_frames,
            "successful_poses": successful_poses,
            "failed_poses": failed_poses,
            "waypoints_reached": waypoints_reached,
            "avg_inference_time_ms": avg_inference_time_ms,
            "success_rate": success_rate,
            "homing_video": str(self.homing_video_path.name) if self.homing_video_path else None,
        }

        self.log(f"Homing phase completed:")
        self.log(f"  Frames processed: {total_frames}")
        self.log(f"  Success rate: {success_rate:.1%}")
        self.log(f"  Waypoints reached: {waypoints_reached}")
        self.log(f"  Avg inference: {avg_inference_time_ms:.1f}ms")

    def save_config(self, config_dict: dict) -> None:
        """
        Save configuration used for this flight.

        Args:
            config_dict: Configuration dictionary
        """
        self.metadata["config"] = config_dict

    def log(self, message: str) -> None:
        """
        Add a line to the flight log.

        Args:
            message: Log message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self.flight_log_lines.append(log_line)

    def finalize_session(self) -> None:
        """
        Finalize the session and save all metadata.

        Writes:
        - metadata.json: Complete session data
        - flight_log.txt: Human-readable log
        """
        if not self.session_dir:
            logger.warning("No session to finalize")
            return

        self.metadata["session_end"] = datetime.now().isoformat()

        # Save metadata.json
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")

        # Save flight_log.txt
        log_path = self.session_dir / "flight_log.txt"
        with open(log_path, "w") as f:
            f.write(f"Flight Session: {self.session_name}\n")
            f.write("=" * 60 + "\n\n")
            for line in self.flight_log_lines:
                f.write(line + "\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Session saved to: {self.session_dir}\n")

        logger.info(f"Saved flight log to {log_path}")
        logger.info(f"Session finalized: {self.session_dir}")

    def get_keyframes_folder(self) -> Optional[Path]:
        """Get path to keyframes folder."""
        return self.keyframes_dir

    def get_session_folder(self) -> Optional[Path]:
        """Get path to session folder."""
        return self.session_dir

    @staticmethod
    def load_session_metadata(session_dir: Path) -> dict:
        """
        Load metadata from a saved session.

        Args:
            session_dir: Path to session directory

        Returns:
            Metadata dictionary
        """
        metadata_path = Path(session_dir) / "metadata.json"

        if not metadata_path.exists():
            logger.warning(f"No metadata file found at {metadata_path}")
            return {}

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loaded session metadata from {metadata_path}")
        return metadata
