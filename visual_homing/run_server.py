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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_homing.protocol.messages import FrameMessage
from visual_homing.server.config import Config
from visual_homing.server.state_machine import StateMachine, SystemState
from visual_homing.server.websocket_server import WebSocketServer

logger = logging.getLogger(__name__)


class MockFrameProcessor:
    """
    Mock frame processor for testing without Fast3R.
    
    Use this during Android app development to test communication
    without requiring GPU/model loading.
    """

    def __init__(self):
        self.state_machine = StateMachine()
        self.frame_count = 0
        self.keyframe_count = 0
        self.total_distance = 0.0
        self.last_telemetry_time = None

    def process_frame(self, frame: FrameMessage) -> dict:
        """Process frame and return mock response."""
        self.frame_count += 1

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

    def handle_command(self, cmd_type: str, data: dict) -> None:
        """Handle state transition commands."""
        logger.info(f"Command received: {cmd_type}")

        if cmd_type == "start_recording":
            if self.state_machine.start_recording():
                logger.info("Started recording")
                self.keyframe_count = 0
                self.total_distance = 0.0
            else:
                logger.warning("Cannot start recording from current state")

        elif cmd_type == "stop_recording":
            if self.state_machine.stop_recording():
                logger.info(f"Stopped recording: {self.keyframe_count} keyframes")
            else:
                logger.warning("Cannot stop recording from current state")

        elif cmd_type == "start_homing":
            if self.state_machine.start_homing():
                logger.info("Started homing")
            else:
                logger.warning("Cannot start homing from current state")

        elif cmd_type == "reset":
            self.state_machine.reset()
            self.frame_count = 0
            self.keyframe_count = 0
            self.total_distance = 0.0
            logger.info("System reset")

    def get_state(self) -> SystemState:
        """Get current system state."""
        return self.state_machine.state


class ProductionFrameProcessor:
    """
    Production frame processor using actual Fast3R model.
    
    This integrates with the full HomingController for real operation.
    """

    def __init__(self, config: Config = None):
        from visual_homing.server.homing_controller import HomingController

        self.config = config or Config()
        self.controller = HomingController(config=self.config)
        self.controller.initialize()
        logger.info("Fast3R model loaded and ready")

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
            keyframe = self.controller.process_teach_frame(image, telemetry)
            if keyframe:
                logger.info(f"Keyframe captured: {self.controller.get_keyframe_count()}")
            return self._status_response()

        elif state == SystemState.HOMING:
            result = self.controller.process_homing_frame(image, telemetry)
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
            return self._status_response()

    def handle_command(self, cmd_type: str, data: dict) -> None:
        """Handle state transition commands."""
        logger.info(f"Command received: {cmd_type}")

        if cmd_type == "start_recording":
            self.controller.start_recording()
        elif cmd_type == "stop_recording":
            self.controller.stop_recording()
        elif cmd_type == "start_homing":
            self.controller.start_homing()
        elif cmd_type == "reset":
            self.controller.reset()

    def get_state(self) -> SystemState:
        """Get current system state."""
        return self.controller.get_state()

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
    config: Config = None,
) -> None:
    """Run the Visual Homing server."""

    # Create processor
    if mock:
        logger.info("Starting in MOCK mode (no Fast3R model)")
        processor = MockFrameProcessor()
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

