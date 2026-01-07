"""Test WebSocket client that simulates Android app.

This client is used to test the WebSocket server by:
- Sending binary frame+telemetry messages
- Receiving JSON control responses
- Measuring round-trip latency
"""

import asyncio
import io
import json
import logging
import struct
import time
from dataclasses import dataclass
from typing import Callable, Optional

import websockets
from PIL import Image
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.protocol.binary_codec import BinaryCodec
from visual_homing.protocol.messages import (
    FrameMessage,
    MessageType,
    Telemetry,
)

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    samples: list
    min_ms: float = 0.0
    max_ms: float = 0.0
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    def compute(self) -> None:
        """Compute statistics from samples."""
        if not self.samples:
            return

        sorted_samples = sorted(self.samples)
        self.min_ms = sorted_samples[0]
        self.max_ms = sorted_samples[-1]
        self.avg_ms = sum(self.samples) / len(self.samples)

        n = len(sorted_samples)
        self.p50_ms = sorted_samples[n // 2]
        self.p95_ms = sorted_samples[int(n * 0.95)]
        self.p99_ms = sorted_samples[int(n * 0.99)]

    def __str__(self) -> str:
        return (
            f"Latency: min={self.min_ms:.1f}ms, avg={self.avg_ms:.1f}ms, "
            f"max={self.max_ms:.1f}ms, p50={self.p50_ms:.1f}ms, "
            f"p95={self.p95_ms:.1f}ms, p99={self.p99_ms:.1f}ms"
        )


class TestWebSocketClient:
    """
    Test WebSocket client simulating Android app.

    Sends simulated frames and telemetry, receives control responses.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        frame_rate_hz: float = 10.0,
        image_size: tuple = (512, 384),
        jpeg_quality: int = 80,
    ):
        """
        Initialize test client.

        Args:
            server_url: WebSocket server URL.
            frame_rate_hz: Target frame rate.
            image_size: Image dimensions (width, height).
            jpeg_quality: JPEG compression quality.
        """
        self.server_url = server_url
        self.frame_rate_hz = frame_rate_hz
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality

        self._websocket = None
        self._running = False
        self._frame_id = 0

        # Statistics
        self._frames_sent = 0
        self._responses_received = 0
        self._latency_samples = []
        self._send_times = {}

        # Response callback
        self._response_callback: Optional[Callable] = None

    def set_response_callback(self, callback: Callable) -> None:
        """Set callback for received responses."""
        self._response_callback = callback

    async def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns:
            True if connected successfully.
        """
        try:
            self._websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            )
            logger.info(f"Connected to {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._running = False
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected from server")

    async def send_frame(
        self,
        image: Optional[np.ndarray] = None,
        telemetry: Optional[Telemetry] = None,
    ) -> int:
        """
        Send a frame+telemetry message.

        Args:
            image: RGB image array (H, W, 3). If None, generates test image.
            telemetry: Telemetry data. If None, uses default values.

        Returns:
            Frame ID.
        """
        if not self._websocket:
            raise RuntimeError("Not connected")

        # Generate test image if not provided
        if image is None:
            image = self._generate_test_image()

        # Compress to JPEG
        image_data = self._compress_jpeg(image)

        # Create telemetry if not provided
        if telemetry is None:
            telemetry = Telemetry(
                velocity_x=0.0,
                velocity_y=0.0,
                velocity_z=0.0,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                height=10.0,
            )

        # Create frame message
        self._frame_id += 1
        frame_msg = FrameMessage(
            message_type=MessageType.FRAME_TELEMETRY,
            frame_id=self._frame_id,
            timestamp_ms=int(time.time() * 1000),
            telemetry=telemetry,
            image_data=image_data,
        )

        # Encode and send
        binary_data = BinaryCodec.encode_frame_message(frame_msg)

        send_time = time.time()
        self._send_times[self._frame_id] = send_time

        await self._websocket.send(binary_data)
        self._frames_sent += 1

        logger.debug(
            f"Sent frame {self._frame_id}: {len(image_data)} bytes"
        )

        return self._frame_id

    async def receive_response(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Receive a response from server.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Response dictionary or None if timeout.
        """
        if not self._websocket:
            raise RuntimeError("Not connected")

        try:
            message = await asyncio.wait_for(
                self._websocket.recv(),
                timeout=timeout,
            )

            receive_time = time.time()

            # Parse JSON response
            response = json.loads(message)
            self._responses_received += 1

            # Calculate round-trip latency
            frame_id = response.get("frame_id", -1)
            if frame_id in self._send_times:
                rtt_ms = (receive_time - self._send_times[frame_id]) * 1000
                self._latency_samples.append(rtt_ms)
                del self._send_times[frame_id]

                logger.debug(
                    f"Response for frame {frame_id}: RTT={rtt_ms:.1f}ms"
                )

            if self._response_callback:
                self._response_callback(response)

            return response

        except asyncio.TimeoutError:
            logger.warning("Receive timeout")
            return None
        except Exception as e:
            logger.error(f"Receive error: {e}")
            return None

    async def send_command(self, command_type: str, **kwargs) -> None:
        """
        Send a text command to server.

        Args:
            command_type: Command type (e.g., "start_recording").
            **kwargs: Additional command data.
        """
        if not self._websocket:
            raise RuntimeError("Not connected")

        message = {
            "type": command_type,
            "timestamp_ms": int(time.time() * 1000),
            **kwargs,
        }

        await self._websocket.send(json.dumps(message))
        logger.info(f"Sent command: {command_type}")

    async def run_streaming_test(
        self,
        duration_seconds: float = 10.0,
        frame_rate_hz: Optional[float] = None,
    ) -> LatencyStats:
        """
        Run a streaming test sending frames at specified rate.

        Args:
            duration_seconds: Test duration.
            frame_rate_hz: Target frame rate (uses default if None).

        Returns:
            Latency statistics.
        """
        if not self._websocket:
            raise RuntimeError("Not connected")

        rate = frame_rate_hz or self.frame_rate_hz
        interval = 1.0 / rate

        self._running = True
        start_time = time.time()

        # Start receiver task
        receive_task = asyncio.create_task(self._receive_loop())

        logger.info(f"Starting streaming test: {duration_seconds}s at {rate}Hz")

        try:
            while self._running and (time.time() - start_time) < duration_seconds:
                loop_start = time.time()

                await self.send_frame()

                # Sleep to maintain frame rate
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

        finally:
            self._running = False
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

        # Wait a bit for final responses
        await asyncio.sleep(0.5)

        # Compute statistics
        stats = LatencyStats(samples=self._latency_samples.copy())
        stats.compute()

        logger.info(
            f"Test complete: {self._frames_sent} sent, "
            f"{self._responses_received} received"
        )
        logger.info(str(stats))

        return stats

    async def _receive_loop(self) -> None:
        """Background loop to receive responses."""
        while self._running:
            try:
                await self.receive_response(timeout=0.5)
            except Exception as e:
                if self._running:
                    logger.error(f"Receive loop error: {e}")
                break

    def _generate_test_image(self) -> np.ndarray:
        """Generate a test image with timestamp."""
        width, height = self.image_size

        # Create gradient image
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)

        # RGB channels
        r = xx.astype(np.uint8)
        g = yy.astype(np.uint8)
        b = ((xx + yy) // 2).astype(np.uint8)

        image = np.stack([r, g, b], axis=-1)

        # Add some variation with frame ID
        noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
        image = np.clip(image.astype(np.int16) + noise - 10, 0, 255).astype(np.uint8)

        return image

    def _compress_jpeg(self, image: np.ndarray) -> bytes:
        """Compress image to JPEG."""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        return buffer.getvalue()

    def get_stats(self) -> dict:
        """Get client statistics."""
        stats = LatencyStats(samples=self._latency_samples.copy())
        stats.compute()

        return {
            "frames_sent": self._frames_sent,
            "responses_received": self._responses_received,
            "latency": {
                "min_ms": stats.min_ms,
                "avg_ms": stats.avg_ms,
                "max_ms": stats.max_ms,
                "p50_ms": stats.p50_ms,
                "p95_ms": stats.p95_ms,
                "p99_ms": stats.p99_ms,
            },
        }


async def run_latency_test(
    server_url: str = "ws://localhost:8765",
    duration_seconds: float = 10.0,
    frame_rate_hz: float = 10.0,
) -> LatencyStats:
    """
    Run a latency test against the server.

    Args:
        server_url: WebSocket server URL.
        duration_seconds: Test duration.
        frame_rate_hz: Target frame rate.

    Returns:
        Latency statistics.
    """
    client = TestWebSocketClient(
        server_url=server_url,
        frame_rate_hz=frame_rate_hz,
    )

    if not await client.connect():
        raise RuntimeError("Failed to connect to server")

    try:
        stats = await client.run_streaming_test(
            duration_seconds=duration_seconds,
            frame_rate_hz=frame_rate_hz,
        )
        return stats
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket latency test client")
    parser.add_argument(
        "--url",
        default="ws://localhost:8765",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Frame rate in Hz",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"\n{'='*60}")
    print("WebSocket Latency Test Client")
    print(f"{'='*60}")
    print(f"Server URL: {args.url}")
    print(f"Duration: {args.duration}s")
    print(f"Frame Rate: {args.rate}Hz")
    print(f"{'='*60}\n")

    try:
        stats = asyncio.run(
            run_latency_test(
                server_url=args.url,
                duration_seconds=args.duration,
                frame_rate_hz=args.rate,
            )
        )

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Round-trip latency:")
        print(f"  Min:  {stats.min_ms:.1f} ms")
        print(f"  Avg:  {stats.avg_ms:.1f} ms")
        print(f"  Max:  {stats.max_ms:.1f} ms")
        print(f"  P50:  {stats.p50_ms:.1f} ms")
        print(f"  P95:  {stats.p95_ms:.1f} ms")
        print(f"  P99:  {stats.p99_ms:.1f} ms")
        print(f"{'='*60}\n")

        # Check if latency meets requirement
        if stats.avg_ms < 100:
            print("✓ PASS: Average latency < 100ms")
        else:
            print("✗ FAIL: Average latency >= 100ms")

    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\nTest failed: {e}")

