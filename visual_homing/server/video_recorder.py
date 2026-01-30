"""
Simple Video Recorder for Visual Homing System.

Collects frames in memory and saves to video file when recording stops.
Non-blocking during operation - video encoding only happens at the end.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SimpleVideoRecorder:
    """
    Simple video recorder that collects frames in memory and saves on stop.
    
    Usage:
        recorder = SimpleVideoRecorder(output_dir="./videos")
        
        recorder.start("teach")
        for frame in frames:
            recorder.add_frame(frame)
        video_path = recorder.stop()  # Saves video here
    """
    
    def __init__(
        self,
        output_dir: str = "./videos",
        fps: float = 10.0,
        max_frames: int = 6000,  # ~10 minutes at 10fps
    ):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.max_frames = max_frames
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._recording = False
        self._phase: Optional[str] = None
        self._frames: List[np.ndarray] = []
    
    def start(self, phase: str) -> None:
        """Start collecting frames for a phase."""
        if self._recording:
            logger.warning("Already recording, stopping previous")
            self.stop()
        
        self._recording = True
        self._phase = phase
        self._frames = []
        logger.info(f"Started collecting frames for {phase}")
    
    def add_frame(self, image: np.ndarray) -> bool:
        """
        Add a frame (non-blocking, just appends to list).
        
        Args:
            image: RGB image array.
            
        Returns:
            True if added, False if not recording or limit reached.
        """
        if not self._recording:
            return False
        
        if len(self._frames) >= self.max_frames:
            if len(self._frames) == self.max_frames:
                logger.warning(f"Max frames ({self.max_frames}) reached, dropping new frames")
            return False
        
        # Store a copy
        self._frames.append(image.copy())
        return True
    
    def stop(self) -> Optional[Path]:
        """
        Stop recording and save video.
        
        Returns:
            Path to saved video, or None if no frames.
        """
        if not self._recording:
            return None
        
        self._recording = False
        
        if not self._frames:
            logger.info(f"No frames collected for {self._phase}")
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"{self._phase}_{timestamp}.mp4"
        
        # Save video
        logger.info(f"Saving {len(self._frames)} frames to {video_path}...")
        self._save_video(video_path)
        
        # Clear frames to free memory
        frame_count = len(self._frames)
        self._frames = []
        
        logger.info(f"Video saved: {video_path} ({frame_count} frames)")
        return video_path
    
    def _save_video(self, path: Path) -> None:
        """Write all frames to video file."""
        if not self._frames:
            return
        
        h, w = self._frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        
        for frame in self._frames:
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        
        writer.release()
    
    @property
    def is_recording(self) -> bool:
        return self._recording
    
    @property
    def frame_count(self) -> int:
        return len(self._frames)


class DualPhaseVideoRecorder:
    """
    Records videos for both TEACH and HOMING phases.
    
    Usage:
        recorder = DualPhaseVideoRecorder(output_dir="./videos")
        
        # TEACH phase
        recorder.start_teach_recording()
        recorder.add_teach_frame(image)
        recorder.stop_teach_recording()  # Saves teach video
        
        # HOMING phase  
        recorder.start_homing_recording()
        recorder.add_homing_frame(image)
        recorder.stop_homing_recording()  # Saves homing video
        
        # On shutdown (saves any unsaved recordings)
        recorder.shutdown()
    """
    
    def __init__(
        self,
        output_dir: str = "./videos",
        fps: float = 10.0,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._output_dir = output_dir
        
        if enabled:
            self._teach_recorder = SimpleVideoRecorder(output_dir=output_dir, fps=fps)
            self._homing_recorder = SimpleVideoRecorder(output_dir=output_dir, fps=fps)
        else:
            self._teach_recorder = None
            self._homing_recorder = None
        
        self._teach_video_path: Optional[Path] = None
        self._homing_video_path: Optional[Path] = None
    
    # === TEACH phase ===
    
    def start_teach_recording(self) -> None:
        """Start recording for TEACH phase."""
        if self._teach_recorder:
            self._teach_recorder.start("teach")
    
    def stop_teach_recording(self) -> Optional[Path]:
        """Stop and save TEACH phase video."""
        if self._teach_recorder:
            self._teach_video_path = self._teach_recorder.stop()
            return self._teach_video_path
        return None
    
    def add_teach_frame(self, image: np.ndarray) -> bool:
        """Add frame during TEACH phase."""
        if self._teach_recorder and self._teach_recorder.is_recording:
            return self._teach_recorder.add_frame(image)
        return False
    
    # === HOMING phase ===
    
    def start_homing_recording(self) -> None:
        """Start recording for HOMING phase."""
        if self._homing_recorder:
            self._homing_recorder.start("homing")
    
    def stop_homing_recording(self) -> Optional[Path]:
        """Stop and save HOMING phase video."""
        if self._homing_recorder:
            self._homing_video_path = self._homing_recorder.stop()
            return self._homing_video_path
        return None
    
    def add_homing_frame(self, image: np.ndarray) -> bool:
        """Add frame during HOMING phase."""
        if self._homing_recorder and self._homing_recorder.is_recording:
            return self._homing_recorder.add_frame(image)
        return False
    
    # === Utility ===
    
    def shutdown(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Shutdown and save any unsaved recordings.
        
        Returns:
            Tuple of (teach_video_path, homing_video_path).
        """
        # Save any ongoing recordings
        if self._teach_recorder and self._teach_recorder.is_recording:
            self._teach_video_path = self._teach_recorder.stop()
        if self._homing_recorder and self._homing_recorder.is_recording:
            self._homing_video_path = self._homing_recorder.stop()
        
        return (self._teach_video_path, self._homing_video_path)
    
    @property
    def is_recording(self) -> bool:
        """Check if any phase is recording."""
        teach = self._teach_recorder.is_recording if self._teach_recorder else False
        homing = self._homing_recorder.is_recording if self._homing_recorder else False
        return teach or homing
    
    def get_stats(self) -> dict:
        """Get recording statistics."""
        return {
            "enabled": self.enabled,
            "teach_recording": self._teach_recorder.is_recording if self._teach_recorder else False,
            "teach_frames": self._teach_recorder.frame_count if self._teach_recorder else 0,
            "teach_video": str(self._teach_video_path) if self._teach_video_path else None,
            "homing_recording": self._homing_recorder.is_recording if self._homing_recorder else False,
            "homing_frames": self._homing_recorder.frame_count if self._homing_recorder else 0,
            "homing_video": str(self._homing_video_path) if self._homing_video_path else None,
        }


# Keep old name as alias for compatibility
AsyncVideoRecorder = SimpleVideoRecorder
