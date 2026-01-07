"""WebSocket server for Visual Homing System.

This module implements the async WebSocket server that:
- Receives binary frame+telemetry data from a single client (Android/DJI)
- Processes frames through the homing controller
- Sends JSON control commands back to the client
- Manages connection lifecycle and heartbeats

Note: This server supports only ONE client at a time (single drone operation).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import websockets
from websockets.asyncio.server import serve, ServerConnection

from ..protocol.binary_codec import BinaryCodec
from ..protocol.messages import (
    ControlCommand,
    FrameMessage,
    MessageType,
    StatusMessage,
    Telemetry,
)
from .config import Config, default_config
from .state_machine import SystemState

logger = logging.getLogger(__name__)


@dataclass
class ClientState:
    """Tracks state for the connected client."""

    websocket: ServerConnection
    client_id: str
    connected_at: float = field(default_factory=time.time)
    last_frame_time: float = field(default_factory=time.time)
    last_command_time: float = field(default_factory=time.time)
    frames_received: int = 0
    commands_sent: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average server processing latency."""
        if self.commands_sent == 0:
            return 0.0
        return self.total_latency_ms / self.commands_sent


class WebSocketServer:
    """
    Async WebSocket server for Visual Homing communication.

    Supports a single client connection (one drone at a time).

    Handles:
    - Binary frame+telemetry reception (upstream)
    - JSON control command transmission (downstream)
    - Connection lifecycle and heartbeat monitoring
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        config: Optional[Config] = None,
    ):
        """
        Initialize WebSocket server.

        Args:
            host: Host address to bind to.
            port: Port number for WebSocket connections.
            config: Configuration object.
        """
        self.host = host
        self.port = port
        self.config = config or default_config

        # Single client connection (only one drone at a time)
        self._client: Optional[ClientState] = None

        # Callbacks for frame processing
        self._frame_callback: Optional[
            Callable[[FrameMessage], Optional[dict]]
        ] = None
        self._command_callback: Optional[Callable[[str, dict], None]] = None
        self._state_callback: Optional[Callable[[], SystemState]] = None

        # Server state
        self._server: Optional[websockets.WebSocketServer] = None
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Latency tracking
        self._latency_samples: list = []
        self._max_latency_samples = 100

    def set_frame_callback(
        self,
        callback: Callable[[FrameMessage], Optional[dict]],
    ) -> None:
        """
        Set callback for processing received frames.

        The callback receives a FrameMessage and should return a dict
        with control command and status, or None to skip response.

        Args:
            callback: Function(FrameMessage) -> Optional[dict]
        """
        self._frame_callback = callback

    def set_command_callback(
        self,
        callback: Callable[[str, dict], None],
    ) -> None:
        """
        Set callback for command events (for state machine control).

        Args:
            callback: Function(command_type: str, data: dict)
        """
        self._command_callback = callback

    def set_state_callback(
        self,
        callback: Callable[[], SystemState],
    ) -> None:
        """
        Set callback to get current system state.

        Args:
            callback: Function() -> SystemState
        """
        self._state_callback = callback

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._running = True

        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message size
        )

        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close client connection if active
        if self._client:
            try:
                await self._client.websocket.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
            self._client = None

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server stopped")

    async def _handle_connection(
        self,
        websocket: ServerConnection,
    ) -> None:
        """
        Handle a client connection.

        Only one client can be connected at a time. If a new client
        connects while one is already connected, the old one is disconnected.

        Args:
            websocket: Client WebSocket connection.
        """
        remote = websocket.remote_address
        client_id = f"{remote[0]}:{remote[1]}" if remote else "unknown"

        # Reject or disconnect existing client if one is connected
        if self._client is not None:
            logger.warning(
                f"New client {client_id} connecting, disconnecting existing client"
            )
            try:
                await self._client.websocket.close()
            except Exception:
                pass

        # Register new client
        self._client = ClientState(websocket=websocket, client_id=client_id)
        logger.info(f"Client connected: {client_id}")

        try:
            async for message in websocket:
                await self._process_message(message)
        except websockets.ConnectionClosed as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if self._client and self._client.client_id == client_id:
                logger.info(
                    f"Client {client_id} removed. "
                    f"Stats: {self._client.frames_received} frames, "
                    f"avg latency: {self._client.avg_latency_ms:.1f}ms"
                )
                self._client = None

    async def _process_message(self, message: bytes) -> None:
        """
        Process an incoming message from the client.

        Args:
            message: Raw message bytes.
        """
        if not self._client:
            return

        receive_time = time.time()

        # Handle text messages (commands)
        if isinstance(message, str):
            await self._handle_text_message(message)
            return

        # Handle binary messages (frame + telemetry)
        if len(message) < 1:
            logger.warning("Empty message from client")
            return

        message_type = message[0]

        if message_type == MessageType.FRAME_TELEMETRY:
            await self._handle_frame_message(message, receive_time)
        elif message_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat()
        else:
            logger.warning(f"Unknown message type 0x{message_type:02x}")

    async def _handle_frame_message(
        self,
        message: bytes,
        receive_time: float,
    ) -> None:
        """Handle a frame+telemetry message."""
        if not self._client:
            return

        try:
            # Decode binary message
            frame_msg = BinaryCodec.decode_frame_message(message)
            self._client.frames_received += 1
            self._client.last_frame_time = receive_time

            logger.debug(
                f"Frame {frame_msg.frame_id}: "
                f"image={len(frame_msg.image_data)} bytes"
            )

            # Process frame through callback
            if self._frame_callback:
                response_data = self._frame_callback(frame_msg)

                if response_data:
                    # Send control response
                    await self._send_control_response(
                        frame_msg.frame_id,
                        response_data,
                        receive_time,
                    )

        except ValueError as e:
            logger.error(f"Failed to decode frame: {e}")

    async def _handle_text_message(self, message: str) -> None:
        """Handle a text (JSON) message - typically commands."""
        try:
            data = json.loads(message)
            command_type = data.get("type", "")

            logger.info(f"Command received: {command_type}")

            if self._command_callback:
                self._command_callback(command_type, data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client: {e}")

    async def _handle_heartbeat(self) -> None:
        """Handle heartbeat message."""
        if not self._client:
            return

        self._client.last_frame_time = time.time()

        # Send heartbeat response
        response = {
            "type": "heartbeat_ack",
            "timestamp_ms": int(time.time() * 1000),
            "state": self._get_current_state(),
        }

        try:
            await self._client.websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Failed to send heartbeat ack: {e}")

    async def _send_control_response(
        self,
        frame_id: int,
        response_data: dict,
        receive_time: float,
    ) -> None:
        """
        Send control response to the client.

        Args:
            frame_id: Corresponding frame ID.
            response_data: Response data with command and status.
            receive_time: When the frame was received.
        """
        if not self._client:
            return

        send_time = time.time()
        server_latency_ms = (send_time - receive_time) * 1000

        # Build response
        command = response_data.get("command", {})
        status = response_data.get("status", {})

        response = BinaryCodec.encode_control_response(
            frame_id=frame_id,
            timestamp_ms=int(send_time * 1000),
            command=ControlCommand(
                pitch_velocity=command.get("pitch_velocity", 0.0),
                roll_velocity=command.get("roll_velocity", 0.0),
                vertical_velocity=command.get("vertical_velocity", 0.0),
                yaw_rate=command.get("yaw_rate", 0.0),
            ),
            status=StatusMessage(
                state=status.get("state", "UNKNOWN"),
                keyframes_remaining=status.get("keyframes_remaining", 0),
                target_distance_m=status.get("target_distance_m", 0.0),
                confidence=status.get("confidence", 0.0),
                total_keyframes=status.get("total_keyframes", 0),
                total_distance_m=status.get("total_distance_m", 0.0),
            ),
        )

        # Add server processing latency to response
        response["server_latency_ms"] = server_latency_ms

        try:
            await self._client.websocket.send(json.dumps(response))
            self._client.commands_sent += 1
            self._client.last_command_time = send_time
            self._client.total_latency_ms += server_latency_ms

            # Track latency samples
            self._latency_samples.append(server_latency_ms)
            if len(self._latency_samples) > self._max_latency_samples:
                self._latency_samples.pop(0)

        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    async def _heartbeat_monitor(self) -> None:
        """Monitor client connection for timeouts."""
        while self._running:
            await asyncio.sleep(0.5)  # Check every 500ms

            if not self._client:
                continue

            current_time = time.time()
            time_since_frame = (current_time - self._client.last_frame_time) * 1000

            # Warning threshold
            if time_since_frame > self.config.frame_timeout_ms:
                logger.warning(
                    f"Client {self._client.client_id} frame timeout: "
                    f"{time_since_frame:.0f}ms since last frame"
                )

            # Abort threshold - send hover command
            if time_since_frame > self.config.abort_timeout_ms:
                logger.error(
                    f"Client {self._client.client_id} abort timeout: "
                    f"sending hover command"
                )
                await self._send_hover_command()

    async def _send_hover_command(self) -> None:
        """Send emergency hover command to the client."""
        if not self._client:
            return

        response = {
            "type": "control",
            "frame_id": -1,
            "timestamp_ms": int(time.time() * 1000),
            "command": {
                "pitch_velocity": 0.0,
                "roll_velocity": 0.0,
                "vertical_velocity": 0.0,
                "yaw_rate": 0.0,
            },
            "status": {
                "state": "TIMEOUT",
                "keyframes_remaining": 0,
                "target_distance_m": 0.0,
                "confidence": 0.0,
                "total_keyframes": 0,
                "total_distance_m": 0.0,
            },
            "emergency": True,
        }

        try:
            await self._client.websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Failed to send hover command: {e}")

    def _get_current_state(self) -> str:
        """Get current system state name."""
        if self._state_callback:
            return self._state_callback().name
        return "UNKNOWN"

    # =========================================================================
    # Send Methods
    # =========================================================================

    async def send_state_update(self, state: str, **kwargs) -> None:
        """
        Send state update to the client.

        Args:
            state: New state name.
            **kwargs: Additional state data.
        """
        if not self._client:
            return

        message = {
            "type": "state_update",
            "timestamp_ms": int(time.time() * 1000),
            "state": state,
            **kwargs,
        }

        try:
            await self._client.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send state update: {e}")

    async def send_emergency_stop(self) -> None:
        """Send emergency stop to the client."""
        if not self._client:
            return

        message = {
            "type": "emergency_stop",
            "timestamp_ms": int(time.time() * 1000),
        }

        try:
            await self._client.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send emergency stop: {e}")

    # Keep broadcast methods as aliases for backward compatibility
    async def broadcast_state_update(self, state: str, **kwargs) -> None:
        """Alias for send_state_update (single client)."""
        await self.send_state_update(state, **kwargs)

    async def broadcast_emergency_stop(self) -> None:
        """Alias for send_emergency_stop (single client)."""
        await self.send_emergency_stop()

    # =========================================================================
    # Statistics Methods
    # =========================================================================

    def get_stats(self) -> dict:
        """Get server statistics."""
        stats = {
            "client_connected": self._client is not None,
            "avg_server_latency_ms": self.get_avg_latency(),
            "min_server_latency_ms": min(self._latency_samples) if self._latency_samples else 0,
            "max_server_latency_ms": max(self._latency_samples) if self._latency_samples else 0,
        }

        if self._client:
            stats["client"] = {
                "client_id": self._client.client_id,
                "frames_received": self._client.frames_received,
                "commands_sent": self._client.commands_sent,
                "avg_latency_ms": self._client.avg_latency_ms,
                "connected_seconds": time.time() - self._client.connected_at,
            }

        return stats

    def get_avg_latency(self) -> float:
        """Get average server processing latency in ms."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    @property
    def client_count(self) -> int:
        """Number of connected clients (0 or 1)."""
        return 1 if self._client else 0

    @property
    def is_connected(self) -> bool:
        """Whether a client is connected."""
        return self._client is not None

    @property
    def is_running(self) -> bool:
        """Whether server is running."""
        return self._running


async def run_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    frame_callback: Optional[Callable] = None,
) -> None:
    """
    Run the WebSocket server.

    Args:
        host: Host address.
        port: Port number.
        frame_callback: Callback for processing frames.
    """
    server = WebSocketServer(host=host, port=port)

    if frame_callback:
        server.set_frame_callback(frame_callback)

    await server.start()

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Simple echo callback for testing
    def echo_callback(frame: FrameMessage) -> Dict:
        return {
            "command": {
                "pitch_velocity": 0.0,
                "roll_velocity": 0.0,
                "vertical_velocity": 0.0,
                "yaw_rate": 0.0,
            },
            "status": {
                "state": "IDLE",
                "keyframes_remaining": 0,
                "target_distance_m": 0.0,
                "confidence": 0.0,
                "total_keyframes": 0,
                "total_distance_m": 0.0,
            },
        }

    asyncio.run(run_server(frame_callback=echo_callback))

