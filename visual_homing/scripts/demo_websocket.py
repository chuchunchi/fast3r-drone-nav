#!/usr/bin/env python3
"""Demo script to run WebSocket server and test latency.

This script starts the WebSocket server and runs a latency test
using the simulated Android client.

Usage:
    python visual_homing/scripts/demo_websocket.py --mode server
    python visual_homing/scripts/demo_websocket.py --mode client --url ws://localhost:8765
    python visual_homing/scripts/demo_websocket.py --mode both  # Runs server and client together
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.protocol.messages import FrameMessage
from visual_homing.server.websocket_server import WebSocketServer
from visual_homing.tests.test_websocket_client import (
    LatencyStats,
    TestWebSocketClient,
)

logger = logging.getLogger(__name__)


class DemoFrameProcessor:
    """Simple frame processor for demo purposes."""

    def __init__(self):
        self.frame_count = 0
        self.state = "IDLE"
        self.keyframes = 0

    def process_frame(self, frame: FrameMessage) -> dict:
        """Process a frame and return control response."""
        self.frame_count += 1

        # Log every 10th frame
        if self.frame_count % 10 == 0:
            logger.info(
                f"Processed {self.frame_count} frames, "
                f"latest: id={frame.frame_id}, "
                f"image={len(frame.image_data)} bytes"
            )

        # Return hover command
        return {
            "command": {
                "pitch_velocity": 0.0,
                "roll_velocity": 0.0,
                "vertical_velocity": 0.0,
                "yaw_rate": 0.0,
            },
            "status": {
                "state": self.state,
                "keyframes_remaining": self.keyframes,
                "target_distance_m": 0.0,
                "confidence": 0.0,
                "total_keyframes": self.keyframes,
                "total_distance_m": 0.0,
            },
        }

    def handle_command(self, cmd_type: str, data: dict) -> None:
        """Handle incoming commands."""
        logger.info(f"Received command: {cmd_type}")

        if cmd_type == "start_recording":
            self.state = "RECORDING"
            self.keyframes = 0
        elif cmd_type == "stop_recording":
            self.state = "ARMED"
        elif cmd_type == "start_homing":
            self.state = "HOMING"
        elif cmd_type == "reset":
            self.state = "IDLE"
            self.keyframes = 0


async def run_server(host: str, port: int) -> None:
    """Run the WebSocket server."""
    processor = DemoFrameProcessor()

    server = WebSocketServer(host=host, port=port)
    server.set_frame_callback(processor.process_frame)
    server.set_command_callback(processor.handle_command)

    await server.start()

    print(f"\n{'='*60}")
    print("Visual Homing WebSocket Server")
    print(f"{'='*60}")
    print(f"Listening on: ws://{host}:{port}")
    print("Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        while True:
            await asyncio.sleep(5)
            stats = server.get_stats()
            if stats["client_connected"]:
                client = stats.get("client", {})
                print(
                    f"Client connected: {client.get('client_id', 'unknown')}, "
                    f"Frames: {client.get('frames_received', 0)}, "
                    f"Avg latency: {stats['avg_server_latency_ms']:.1f}ms"
                )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await server.stop()


async def run_client(
    url: str,
    duration: float,
    rate: float,
) -> LatencyStats:
    """Run the test client."""
    client = TestWebSocketClient(server_url=url, frame_rate_hz=rate)

    if not await client.connect():
        raise RuntimeError("Failed to connect to server")

    print(f"\n{'='*60}")
    print("WebSocket Latency Test Client")
    print(f"{'='*60}")
    print(f"Server URL: {url}")
    print(f"Duration: {duration}s")
    print(f"Frame Rate: {rate}Hz")
    print(f"{'='*60}\n")

    try:
        stats = await client.run_streaming_test(
            duration_seconds=duration,
            frame_rate_hz=rate,
        )

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Frames sent: {client._frames_sent}")
        print(f"Responses received: {client._responses_received}")
        print(f"\nRound-trip latency:")
        print(f"  Min:  {stats.min_ms:.1f} ms")
        print(f"  Avg:  {stats.avg_ms:.1f} ms")
        print(f"  Max:  {stats.max_ms:.1f} ms")
        print(f"  P50:  {stats.p50_ms:.1f} ms")
        print(f"  P95:  {stats.p95_ms:.1f} ms")
        print(f"  P99:  {stats.p99_ms:.1f} ms")
        print(f"{'='*60}\n")

        # Check if latency meets requirement
        if stats.avg_ms < 100:
            print("✓ PASS: Average latency < 100ms (requirement met)")
        else:
            print("✗ FAIL: Average latency >= 100ms (requirement not met)")

        return stats

    finally:
        await client.disconnect()


async def run_both(
    host: str,
    port: int,
    duration: float,
    rate: float,
) -> LatencyStats:
    """Run server and client together for testing."""
    processor = DemoFrameProcessor()

    server = WebSocketServer(host=host, port=port)
    server.set_frame_callback(processor.process_frame)
    server.set_command_callback(processor.handle_command)

    await server.start()

    print(f"\n{'='*60}")
    print("Visual Homing WebSocket Demo")
    print(f"{'='*60}")
    print(f"Server running on: ws://{host}:{port}")
    print(f"Testing for {duration}s at {rate}Hz")
    print(f"{'='*60}\n")

    # Wait a bit for server to be ready
    await asyncio.sleep(0.5)

    client = TestWebSocketClient(
        server_url=f"ws://{host}:{port}",
        frame_rate_hz=rate,
    )

    try:
        if not await client.connect():
            raise RuntimeError("Failed to connect to server")

        # Run streaming test
        stats = await client.run_streaming_test(
            duration_seconds=duration,
            frame_rate_hz=rate,
        )

        # Get server stats
        server_stats = server.get_stats()

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"\nClient Statistics:")
        print(f"  Frames sent: {client._frames_sent}")
        print(f"  Responses received: {client._responses_received}")
        print(f"\nServer Statistics:")
        print(f"  Avg processing latency: {server_stats['avg_server_latency_ms']:.1f}ms")
        print(f"  Min processing latency: {server_stats['min_server_latency_ms']:.1f}ms")
        print(f"  Max processing latency: {server_stats['max_server_latency_ms']:.1f}ms")
        print(f"\nRound-trip Latency (Client Measured):")
        print(f"  Min:  {stats.min_ms:.1f} ms")
        print(f"  Avg:  {stats.avg_ms:.1f} ms")
        print(f"  Max:  {stats.max_ms:.1f} ms")
        print(f"  P50:  {stats.p50_ms:.1f} ms")
        print(f"  P95:  {stats.p95_ms:.1f} ms")
        print(f"  P99:  {stats.p99_ms:.1f} ms")
        print(f"{'='*60}\n")

        # Check if latency meets requirement
        if stats.avg_ms < 100:
            print("✓ PASS: Average round-trip latency < 100ms")
            print("  Bidirectional communication working correctly!")
        else:
            print("✗ FAIL: Average round-trip latency >= 100ms")

        return stats

    finally:
        await client.disconnect()
        await server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Visual Homing WebSocket Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run server only:
  python demo_websocket.py --mode server

  # Run client only (connect to existing server):
  python demo_websocket.py --mode client --url ws://localhost:8765

  # Run both server and client for testing:
  python demo_websocket.py --mode both --duration 10 --rate 10
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["server", "client", "both"],
        default="both",
        help="Run mode: server, client, or both",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port",
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8765",
        help="WebSocket URL (for client mode)",
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

    try:
        if args.mode == "server":
            asyncio.run(run_server(args.host, args.port))
        elif args.mode == "client":
            asyncio.run(run_client(args.url, args.duration, args.rate))
        elif args.mode == "both":
            asyncio.run(run_both(args.host, args.port, args.duration, args.rate))
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

