#!/usr/bin/env python3
"""
Visual Homing Server - Main Entry Point

Run this script to start the WebSocket server for communication with
the Android/DJI client.

Usage:
    # Production mode (loads Fast3R model):
    python run_server.py

    # Mock mode (for Android development/testing without GPU):
    python run_server.py --mock

    # Custom host/port:
    python run_server.py --host 0.0.0.0 --port 8765

    # Verbose logging:
    python run_server.py --mock -v
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_homing.protocol.messages import FrameMessage
from visual_homing.server.config import Config
from visual_homing.server.state_machine import StateMachine, SystemState
from visual_homing.server.video_recorder import DualPhaseVideoRecorder
from visual_homing.server.websocket_server import WebSocketServer

logger = logging.getLogger(__name__)


class MockFrameProcessor:
    """
    Mock frame processor for testing without Fast3R.
    
    Use this during Android app development to test communication
    without requiring GPU/model loading.
    """

    def __init__(self, save_sample_image: bool = True, config: Optional[Config] = None):
        self.config = config or Config()
        self.state_machine = StateMachine()
        self.frame_count = 0
        self.keyframe_count = 0
        self.total_distance = 0.0
        self.last_telemetry_time = None
        self.save_sample_image = save_sample_image
        self.image_validated = False
        self.gimbal_pitch_deg = 0.0
        
        # Initialize video recorder
        self.video_recorder = DualPhaseVideoRecorder(
            output_dir=self.config.video_output_dir,
            fps=self.config.video_fps,
            enabled=self.config.video_recording_enabled,
        )
        if self.config.video_recording_enabled:
            logger.info(f"[Mock] Video recording enabled, output dir: {self.config.video_output_dir}")

    def _validate_image(self, frame: FrameMessage) -> bool:
        """Validate and optionally save received image."""
        try:
            import cv2
            import numpy as np
            
            # Decode JPEG
            img_array = np.frombuffer(frame.image_data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"❌ Frame {frame.frame_id}: Failed to decode JPEG image!")
                return False
            
            height, width, channels = image.shape
            jpeg_size_kb = len(frame.image_data) / 1024
            
            logger.info(
                f"✅ Frame {frame.frame_id}: Image decoded successfully! "
                f"Size: {width}x{height}, Channels: {channels}, JPEG: {jpeg_size_kb:.1f}KB"
            )
            
            # Save first image to disk for visual inspection
            if self.save_sample_image and not self.image_validated:
                sample_path = Path(__file__).parent / "sample_frame.jpg"
                cv2.imwrite(str(sample_path), image)
                logger.info(f"📸 Sample image saved to: {sample_path}")
            
            return True
            
        except ImportError:
            logger.warning("OpenCV not available - skipping image validation")
            return True
        except Exception as e:
            logger.error(f"❌ Image validation error: {e}")
            return False

    def process_frame(self, frame: FrameMessage) -> dict:
        """Process frame and return mock response."""
        import cv2
        import numpy as np
        
        self.frame_count += 1

        # Validate first 3 images to ensure protocol is working
        if self.frame_count <= 3 or (not self.image_validated and self.frame_count % 100 == 0):
            if self._validate_image(frame):
                self.image_validated = True

        # Simulate distance tracking from telemetry
        if self.last_telemetry_time is not None:
            dt = (frame.timestamp_ms - self.last_telemetry_time) / 1000.0
            speed = (
                frame.telemetry.velocity_x ** 2 +
                frame.telemetry.velocity_y ** 2 +
                frame.telemetry.velocity_z ** 2
            ) ** 0.5
            self.total_distance += speed * dt
        self.last_telemetry_time = frame.timestamp_ms

        # Decode image for video recording
        if self.video_recorder.is_recording:
            try:
                img_array = np.frombuffer(frame.image_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.state_machine.is_recording():
                        self.video_recorder.add_teach_frame(
                            image_rgb, frame.timestamp_ms, frame.frame_id
                        )
                    elif self.state_machine.state == SystemState.HOMING:
                        self.video_recorder.add_homing_frame(
                            image_rgb, frame.timestamp_ms, frame.frame_id
                        )
            except Exception as e:
                logger.debug(f"Video frame recording error: {e}")

        # Simulate keyframe capture during recording
        if self.state_machine.is_recording():
            if self.frame_count % 30 == 0:  # Every ~3 seconds at 10Hz
                self.keyframe_count += 1
                logger.info(f"Keyframe {self.keyframe_count} captured")

        # Log every 50th frame
        if self.frame_count % 50 == 0:
            logger.info(
                f"Frame {frame.frame_id}: "
                f"state={self.state_machine.state.name}, "
                f"keyframes={self.keyframe_count}, "
                f"distance={self.total_distance:.1f}m"
            )

        # Build response based on current state
        state = self.state_machine.state

        if state == SystemState.HOMING:
            # Simulate approaching target
            target_distance = max(0, 5.0 - (self.frame_count % 50) * 0.1)
            return {
                "command": {
                    "pitch_velocity": 0.3 if target_distance > 0.5 else 0.0,
                    "roll_velocity": 0.0,
                    "vertical_velocity": 0.0,
                    "yaw_rate": 0.0,
                },
                "status": {
                    "state": state.name,
                    "keyframes_remaining": max(0, self.keyframe_count - (self.frame_count // 50)),
                    "target_distance_m": target_distance,
                    "confidence": 0.85,
                    "total_keyframes": self.keyframe_count,
                    "total_distance_m": self.total_distance,
                },
            }
        else:
            # Hover command for other states
            return {
                "command": {
                    "pitch_velocity": 0.0,
                    "roll_velocity": 0.0,
                    "vertical_velocity": 0.0,
                    "yaw_rate": 0.0,
                },
                "status": {
                    "state": state.name,
                    "keyframes_remaining": self.keyframe_count,
                    "target_distance_m": 0.0,
                    "confidence": 0.0,
                    "total_keyframes": self.keyframe_count,
                    "total_distance_m": self.total_distance,
                },
            }

    def _command_result(self, cmd_type: str, ok: bool, **extra) -> dict:
        """Create text command result payload."""
        return {
            "type": "command_result",
            "command": cmd_type,
            "ok": ok,
            **extra,
        }

    def _validate_gimbal_pitch(self, value: Any) -> Optional[float]:
        """Validate gimbal pitch; return float if valid else None."""
        try:
            pitch = float(value)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= pitch < 90.0):
            return None
        return pitch

    def handle_command(self, cmd_type: str, data: dict) -> Optional[dict]:
        """Handle state transition commands."""
        logger.info(f"Command received: {cmd_type}")

        if cmd_type == "init_gimbal_config":
            current_state = self.state_machine.state
            if current_state not in (SystemState.IDLE, SystemState.ARMED):
                return self._command_result(
                    cmd_type,
                    False,
                    reason="invalid_state",
                    allowed_states=["IDLE", "ARMED"],
                    current_state=current_state.name,
                )

            pitch = self._validate_gimbal_pitch(data.get("gimbal_pitch_deg"))
            if pitch is None:
                return self._command_result(
                    cmd_type,
                    False,
                    reason="out_of_range",
                    allowed_range_deg=[0.0, 89.9],
                    received_gimbal_pitch_deg=data.get("gimbal_pitch_deg"),
                )

            self.gimbal_pitch_deg = pitch
            logger.info(f"[Mock] Gimbal pitch configured: {pitch:.1f} deg")
            return self._command_result(
                cmd_type,
                True,
                gimbal_pitch_deg=self.gimbal_pitch_deg,
            )

        if cmd_type == "start_recording":
            if self.state_machine.start_recording():
                logger.info("Started recording")
                self.keyframe_count = 0
                self.total_distance = 0.0
                # Start teach video recording
                self.video_recorder.start_teach_recording()
            else:
                logger.warning("Cannot start recording from current state")

        elif cmd_type == "stop_recording":
            if self.state_machine.stop_recording():
                logger.info(f"Stopped recording: {self.keyframe_count} keyframes")
                # Stop and save teach video
                video_path = self.video_recorder.stop_teach_recording()
                if video_path:
                    logger.info(f"Teach video saved: {video_path}")
            else:
                logger.warning("Cannot stop recording from current state")

        elif cmd_type == "start_homing":
            if self.state_machine.start_homing():
                logger.info("Started homing")
                # Start homing video recording
                self.video_recorder.start_homing_recording()
            else:
                logger.warning("Cannot start homing from current state")

        elif cmd_type == "stop_homing":
            # Stop and save homing video
            video_path = self.video_recorder.stop_homing_recording()
            if video_path:
                logger.info(f"Homing video saved: {video_path}")

        elif cmd_type == "reset":
            # Stop any active recording before reset
            self.video_recorder.stop_teach_recording()
            self.video_recorder.stop_homing_recording()
            self.state_machine.reset()
            self.frame_count = 0
            self.keyframe_count = 0
            self.total_distance = 0.0
            self.gimbal_pitch_deg = 0.0
            logger.info("System reset")

        return None

    def get_state(self) -> SystemState:
        """Get current system state."""
        return self.state_machine.state

    def shutdown(self) -> None:
        """Shutdown processor and finalize video recordings."""
        logger.info("[Mock] Shutting down frame processor...")
        
        # Finalize any active video recordings
        teach_path, homing_path = self.video_recorder.shutdown()
        
        if teach_path:
            logger.info(f"[Mock] Teach video saved: {teach_path}")
        if homing_path:
            logger.info(f"[Mock] Homing video saved: {homing_path}")
        
        logger.info("[Mock] Frame processor shutdown complete")


class ProductionFrameProcessor:
    """
    Production frame processor using actual Fast3R model.

    This integrates with the full HomingController for real operation.
    """

    def __init__(self, config: Optional[Config] = None):
        from visual_homing.server.homing_controller import HomingController
        from visual_homing.server.flight_session import FlightSession

        self.config = config or Config()
        self.controller = HomingController(config=self.config)
        self.controller.initialize()
        self.gimbal_pitch_deg = 0.0
        self.controller.set_gimbal_pitch_deg(self.gimbal_pitch_deg)
        logger.info("Fast3R model loaded and ready")

        # Initialize flight session manager
        self.flight_session = FlightSession(base_dir=self.config.video_output_dir)

        # Initialize video recorder (will be reconfigured when session starts)
        self.video_recorder = None

        if self.config.video_recording_enabled:
            logger.info(f"Flight sessions will be saved to: {self.config.video_output_dir}")

    def _command_result(self, cmd_type: str, ok: bool, **extra) -> dict:
        """Create text command result payload."""
        return {
            "type": "command_result",
            "command": cmd_type,
            "ok": ok,
            **extra,
        }

    def _validate_gimbal_pitch(self, value: Any) -> Optional[float]:
        """Validate gimbal pitch; return float if valid else None."""
        try:
            pitch = float(value)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= pitch < 90.0):
            return None
        return pitch

    def _build_session_config(self) -> dict:
        """Build configuration snapshot for flight metadata."""
        return {
            "keyframe_interval_m": self.config.keyframe_interval_m,
            "keyframe_interval_s": self.config.keyframe_interval_s,
            "waypoint_threshold_m": self.config.waypoint_threshold_m,
            "fixed_flight_velocity": self.config.fixed_flight_velocity,
            "min_confidence": self.config.min_confidence,
            "gimbal_pitch_deg": self.gimbal_pitch_deg,
        }

    def process_frame(self, frame: FrameMessage) -> dict:
        """Process frame through actual homing controller."""
        import cv2
        import numpy as np
        from visual_homing.server.keyframe_manager import Telemetry

        # Decode JPEG image
        img_array = np.frombuffer(frame.image_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode image")
            return self._hover_response()

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create telemetry object
        telemetry = Telemetry(
            velocity_x=frame.telemetry.velocity_x,
            velocity_y=frame.telemetry.velocity_y,
            velocity_z=frame.telemetry.velocity_z,
            yaw=frame.telemetry.yaw,
            pitch=frame.telemetry.pitch,
            roll=frame.telemetry.roll,
            height=frame.telemetry.height,
            timestamp_ms=frame.timestamp_ms,
        )

        state = self.controller.get_state()

        if state == SystemState.RECORDING:
            # Add frame to teach video (non-blocking)
            if self.video_recorder:
                self.video_recorder.add_teach_frame(
                    image, frame.timestamp_ms, frame.frame_id
                )

            # Process frame and save keyframe image if captured
            keyframe = self.controller.process_teach_frame(image, telemetry)
            if keyframe:
                # Save keyframe image to session folder
                self.flight_session.save_keyframe_image(image, keyframe.index)
                logger.info(f"Keyframe {keyframe.index} captured and saved")

            return self._status_response()

        elif state == SystemState.HOMING:
            # Add frame to homing video (non-blocking)
            if self.video_recorder:
                self.video_recorder.add_homing_frame(
                    image, frame.timestamp_ms, frame.frame_id
                )

            # Check if enough time has passed to compute new command
            if self.controller.should_compute_new_command():
                # Compute new command
                result = self.controller.process_homing_frame(image, telemetry)
                logger.debug(f"Computed new command at frame {frame.frame_id}")
                return {
                    "command": result.command,
                    "status": {
                        "state": result.state,
                        "keyframes_remaining": result.keyframes_remaining,
                        "target_distance_m": result.target_distance_m,
                        "confidence": result.confidence,
                        "total_keyframes": self.controller.get_keyframe_count(),
                        "total_distance_m": self.controller.get_total_distance(),
                    },
                }
            else:
                # Rate limited - send hover command
                logger.debug(f"Rate limited at frame {frame.frame_id}, sending hover")
                return {
                    "command": {
                        "pitch_velocity": 0.0,
                        "roll_velocity": 0.0,
                        "vertical_velocity": 0.0,
                        "yaw_rate": 0.0,
                        "duration_s": self.config.command_duration_s,
                    },
                    "status": {
                        "state": state.name,
                        "keyframes_remaining": self.controller.target_idx + 1 if self.controller.target_idx >= 0 else 0,
                        "target_distance_m": 0.0,
                        "confidence": 0.0,
                        "total_keyframes": self.controller.get_keyframe_count(),
                        "total_distance_m": self.controller.get_total_distance(),
                    },
                }

        else:
            return self._status_response()

    def handle_command(self, cmd_type: str, data: dict) -> Optional[dict]:
        """Handle state transition commands."""
        logger.info(f"Command received: {cmd_type}")

        if cmd_type == "init_gimbal_config":
            current_state = self.controller.get_state()
            if current_state not in (SystemState.IDLE, SystemState.ARMED):
                return self._command_result(
                    cmd_type,
                    False,
                    reason="invalid_state",
                    allowed_states=["IDLE", "ARMED"],
                    current_state=current_state.name,
                )

            pitch = self._validate_gimbal_pitch(data.get("gimbal_pitch_deg"))
            if pitch is None:
                return self._command_result(
                    cmd_type,
                    False,
                    reason="out_of_range",
                    allowed_range_deg=[0.0, 89.9],
                    received_gimbal_pitch_deg=data.get("gimbal_pitch_deg"),
                )

            self.gimbal_pitch_deg = pitch
            self.controller.set_gimbal_pitch_deg(self.gimbal_pitch_deg)
            logger.info(f"Gimbal pitch configured: {self.gimbal_pitch_deg:.1f} deg")

            # Keep session metadata in sync when available.
            if self.flight_session.get_session_folder():
                self.flight_session.save_config(self._build_session_config())

            return self._command_result(
                cmd_type,
                True,
                gimbal_pitch_deg=self.gimbal_pitch_deg,
            )

        if cmd_type == "start_recording":
            if self.controller.start_recording():
                # Start a new flight session
                session_dir = self.flight_session.start_session()
                logger.info(f"Started new flight session: {session_dir}")

                # Create video recorder for this session
                if self.config.video_recording_enabled:
                    self.video_recorder = DualPhaseVideoRecorder(
                        output_dir=str(session_dir),
                        fps=self.config.video_fps,
                        enabled=True,
                    )
                    self.video_recorder.start_teach_recording()

                # Save config to session
                self.flight_session.save_config(self._build_session_config())

        elif cmd_type == "stop_recording":
            # Stop controller recording (computes scale)
            self.controller.stop_recording()

            # Stop and save teach video
            if self.video_recorder:
                video_path = self.video_recorder.stop_teach_recording()
                if video_path:
                    logger.info(f"Teach video saved: {video_path}")

            # Save teach metadata to session
            self.flight_session.save_teach_metadata(
                num_keyframes=self.controller.get_keyframe_count(),
                total_distance_m=self.controller.get_total_distance(),
                global_scale_factor=self.controller.metric_scale,
                keyframe_distances=self.controller.keyframe_manager.get_keyframe_distances(),
                velocity_stats=self.controller.keyframe_manager.get_velocity_stats(),
            )

            logger.info(f"Teaching phase metadata saved")

        elif cmd_type == "start_homing":
            if self.controller.start_homing():
                # Start homing video recording
                if self.video_recorder:
                    self.video_recorder.start_homing_recording()

        elif cmd_type == "stop_homing":
            # Save homing metadata
            stats = self.controller.get_stats()
            self.flight_session.save_homing_metadata(
                total_frames=stats.total_frames,
                successful_poses=stats.successful_poses,
                failed_poses=stats.failed_poses,
                waypoints_reached=stats.waypoints_reached,
                avg_inference_time_ms=stats.avg_inference_time_ms,
                success_rate=stats.success_rate,
            )

            # Stop and save homing video
            if self.video_recorder:
                video_path = self.video_recorder.stop_homing_recording()
                if video_path:
                    logger.info(f"Homing video saved: {video_path}")

            # Finalize session
            self.flight_session.finalize_session()
            logger.info(f"Flight session finalized: {self.flight_session.get_session_folder()}")

        elif cmd_type == "reset":
            # Stop any active recording before reset
            if self.video_recorder:
                self.video_recorder.stop_teach_recording()
                self.video_recorder.stop_homing_recording()

            # Finalize current session if any
            if self.flight_session.get_session_folder():
                self.flight_session.finalize_session()

            self.controller.reset()
            self.gimbal_pitch_deg = 0.0
            self.controller.set_gimbal_pitch_deg(self.gimbal_pitch_deg)

        return None

    def get_state(self) -> SystemState:
        """Get current system state."""
        return self.controller.get_state()

    def shutdown(self) -> None:
        """Shutdown processor and finalize session."""
        logger.info("Shutting down frame processor...")

        # Finalize any active video recordings
        if self.video_recorder:
            teach_path, homing_path = self.video_recorder.shutdown()

            if teach_path:
                logger.info(f"Teach video saved: {teach_path}")
            if homing_path:
                logger.info(f"Homing video saved: {homing_path}")

        # Finalize flight session if active
        if self.flight_session.get_session_folder():
            self.flight_session.finalize_session()
            logger.info(f"Session finalized: {self.flight_session.get_session_folder()}")

        logger.info("Frame processor shutdown complete")

    def _hover_response(self) -> dict:
        """Return hover command."""
        return {
            "command": {
                "pitch_velocity": 0.0,
                "roll_velocity": 0.0,
                "vertical_velocity": 0.0,
                "yaw_rate": 0.0,
            },
            "status": {
                "state": self.controller.get_state().name,
                "keyframes_remaining": 0,
                "target_distance_m": 0.0,
                "confidence": 0.0,
                "total_keyframes": self.controller.get_keyframe_count(),
                "total_distance_m": self.controller.get_total_distance(),
            },
        }

    def _status_response(self) -> dict:
        """Return current status with hover command."""
        return self._hover_response()


async def run_server(
    host: str,
    port: int,
    mock: bool = False,
    config: Optional[Config] = None,
) -> None:
    """Run the Visual Homing server."""

    # Create processor
    if mock:
        logger.info("Starting in MOCK mode (no Fast3R model)")
        processor = MockFrameProcessor(config=config)
    else:
        logger.info("Starting in PRODUCTION mode (loading Fast3R model...)")
        processor = ProductionFrameProcessor(config)

    # Create and configure server
    server = WebSocketServer(host=host, port=port, config=config)
    server.set_frame_callback(processor.process_frame)
    server.set_command_callback(processor.handle_command)
    server.set_state_callback(processor.get_state)

    # Start server
    await server.start()

    print()
    print("=" * 60)
    print("Visual Homing Server")
    print("=" * 60)
    print(f"Mode:      {'MOCK (no GPU)' if mock else 'PRODUCTION'}")
    print(f"Listening: ws://{host}:{port}")
    print(f"State:     {processor.get_state().name}")
    print()
    print("Waiting for Android client to connect...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Handle shutdown gracefully
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Status printing loop
    try:
        while not stop_event.is_set():
            await asyncio.sleep(10)
            stats = server.get_stats()
            if stats["client_connected"]:
                client = stats.get("client", {})
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Client: {client.get('client_id', '?')}, "
                    f"Frames: {client.get('frames_received', 0)}, "
                    f"State: {processor.get_state().name}, "
                    f"Latency: {stats['avg_server_latency_ms']:.1f}ms"
                )
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for client...")
    except asyncio.CancelledError:
        pass
    finally:
        print("\nShutting down server...")
        
        # Shutdown processor (finalize videos)
        if hasattr(processor, 'shutdown'):
            processor.shutdown()
        
        await server.stop()
        print("Server stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Visual Homing WebSocket Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server in mock mode (for Android development):
  python run_server.py --mock

  # Start server in production mode:
  python run_server.py

  # Custom port:
  python run_server.py --mock --port 9000

  # Verbose logging:
  python run_server.py --mock -v
        """,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without Fast3R model (for testing)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port number (default: 8765)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress noisy loggers
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # Run server
    try:
        asyncio.run(run_server(
            host=args.host,
            port=args.port,
            mock=args.mock,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

