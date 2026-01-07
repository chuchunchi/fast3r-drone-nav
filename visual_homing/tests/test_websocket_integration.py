"""Integration tests for WebSocket server and client."""

import asyncio
import json
import time
import pytest
import pytest_asyncio
import sys
from pathlib import Path

# Enable pytest-asyncio auto mode
pytestmark = pytest.mark.asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.protocol.binary_codec import BinaryCodec
from visual_homing.protocol.messages import (
    ControlCommand,
    FrameMessage,
    MessageType,
    StatusMessage,
    Telemetry,
)
from visual_homing.server.websocket_server import WebSocketServer


class TestWebSocketServerBasic:
    """Basic WebSocket server tests."""

    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        """Test server can start and stop."""
        server = WebSocketServer(port=8766)

        await server.start()
        assert server.is_running

        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_server_client_connection(self):
        """Test client can connect to server."""
        import websockets

        server = WebSocketServer(port=8767)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8767") as ws:
                assert server.client_count == 1
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_binary_frame_roundtrip(self):
        """Test sending binary frame and receiving response."""
        import websockets

        responses_received = []

        def frame_callback(frame: FrameMessage):
            return {
                "command": {
                    "pitch_velocity": 0.5,
                    "roll_velocity": -0.2,
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

        server = WebSocketServer(port=8768)
        server.set_frame_callback(frame_callback)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8768") as ws:
                # Create and send frame
                telemetry = Telemetry(
                    velocity_x=1.0,
                    velocity_y=0.0,
                    velocity_z=0.0,
                    yaw=0.0,
                    pitch=0.0,
                    roll=0.0,
                    height=10.0,
                )

                frame = FrameMessage(
                    message_type=MessageType.FRAME_TELEMETRY,
                    frame_id=1,
                    timestamp_ms=int(time.time() * 1000),
                    telemetry=telemetry,
                    image_data=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
                )

                binary_data = BinaryCodec.encode_frame_message(frame)
                await ws.send(binary_data)

                # Receive response
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                response_data = json.loads(response)

                assert response_data["type"] == "control"
                assert response_data["frame_id"] == 1
                assert response_data["command"]["pitch_velocity"] == 0.5
                assert response_data["command"]["roll_velocity"] == -0.2

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_frames(self):
        """Test sending multiple frames and receiving responses."""
        import websockets

        frame_count = 10
        responses = []

        def frame_callback(frame: FrameMessage):
            return {
                "command": {
                    "pitch_velocity": frame.frame_id * 0.1,
                    "roll_velocity": 0.0,
                    "vertical_velocity": 0.0,
                    "yaw_rate": 0.0,
                },
                "status": {
                    "state": "RECORDING",
                    "keyframes_remaining": frame.frame_id,
                    "target_distance_m": 0.0,
                    "confidence": 0.0,
                    "total_keyframes": 0,
                    "total_distance_m": 0.0,
                },
            }

        server = WebSocketServer(port=8769)
        server.set_frame_callback(frame_callback)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8769") as ws:
                telemetry = Telemetry(
                    velocity_x=0.0,
                    velocity_y=0.0,
                    velocity_z=0.0,
                    yaw=0.0,
                    pitch=0.0,
                    roll=0.0,
                    height=10.0,
                )

                for i in range(frame_count):
                    frame = FrameMessage(
                        message_type=MessageType.FRAME_TELEMETRY,
                        frame_id=i + 1,
                        timestamp_ms=int(time.time() * 1000),
                        telemetry=telemetry,
                        image_data=b"\xff\xd8\xff\xe0" + bytes([i] * 50),
                    )

                    await ws.send(BinaryCodec.encode_frame_message(frame))

                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    responses.append(json.loads(response))

                assert len(responses) == frame_count

                # Verify frame IDs match
                for i, resp in enumerate(responses):
                    assert resp["frame_id"] == i + 1
                    assert abs(resp["command"]["pitch_velocity"] - (i + 1) * 0.1) < 0.01

        finally:
            await server.stop()


class TestWebSocketLatency:
    """Latency measurement tests."""

    @pytest.mark.asyncio
    async def test_server_processing_latency(self):
        """Test server adds latency measurement to response."""
        import websockets

        def frame_callback(frame: FrameMessage):
            # Simulate some processing time
            time.sleep(0.005)  # 5ms processing
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

        server = WebSocketServer(port=8770)
        server.set_frame_callback(frame_callback)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8770") as ws:
                telemetry = Telemetry(
                    velocity_x=0.0,
                    velocity_y=0.0,
                    velocity_z=0.0,
                    yaw=0.0,
                    pitch=0.0,
                    roll=0.0,
                    height=10.0,
                )

                frame = FrameMessage(
                    message_type=MessageType.FRAME_TELEMETRY,
                    frame_id=1,
                    timestamp_ms=int(time.time() * 1000),
                    telemetry=telemetry,
                    image_data=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
                )

                await ws.send(BinaryCodec.encode_frame_message(frame))

                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                response_data = json.loads(response)

                # Check server latency is included
                assert "server_latency_ms" in response_data
                assert response_data["server_latency_ms"] >= 5.0  # At least 5ms

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_roundtrip_latency_under_100ms(self):
        """Test that round-trip latency is under 100ms."""
        import websockets

        def fast_callback(frame: FrameMessage):
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

        server = WebSocketServer(port=8771)
        server.set_frame_callback(fast_callback)
        await server.start()

        latencies = []

        try:
            async with websockets.connect("ws://localhost:8771") as ws:
                telemetry = Telemetry(
                    velocity_x=0.0,
                    velocity_y=0.0,
                    velocity_z=0.0,
                    yaw=0.0,
                    pitch=0.0,
                    roll=0.0,
                    height=10.0,
                )

                for i in range(20):
                    frame = FrameMessage(
                        message_type=MessageType.FRAME_TELEMETRY,
                        frame_id=i + 1,
                        timestamp_ms=int(time.time() * 1000),
                        telemetry=telemetry,
                        image_data=b"\xff\xd8\xff\xe0" + b"\x00" * 1000,
                    )

                    send_time = time.time()
                    await ws.send(BinaryCodec.encode_frame_message(frame))
                    response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    receive_time = time.time()

                    latency_ms = (receive_time - send_time) * 1000
                    latencies.append(latency_ms)

                avg_latency = sum(latencies) / len(latencies)
                print(f"\nAverage round-trip latency: {avg_latency:.1f}ms")
                print(f"Min: {min(latencies):.1f}ms, Max: {max(latencies):.1f}ms")

                # Assert average is under 100ms
                assert avg_latency < 100, f"Average latency {avg_latency:.1f}ms >= 100ms"

        finally:
            await server.stop()


class TestWebSocketCommands:
    """Test text command handling."""

    @pytest.mark.asyncio
    async def test_command_callback(self):
        """Test that text commands trigger callback."""
        import websockets

        received_commands = []

        def command_callback(cmd_type: str, data: dict):
            received_commands.append((cmd_type, data))

        server = WebSocketServer(port=8772)
        server.set_command_callback(command_callback)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8772") as ws:
                # Send start recording command
                await ws.send(json.dumps({
                    "type": "start_recording",
                    "timestamp_ms": int(time.time() * 1000),
                }))

                await asyncio.sleep(0.1)

                assert len(received_commands) == 1
                assert received_commands[0][0] == "start_recording"

        finally:
            await server.stop()


class TestWebSocketBroadcast:
    """Test broadcast functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_state_update(self):
        """Test broadcasting state updates to all clients."""
        import websockets

        server = WebSocketServer(port=8773)
        await server.start()

        messages_received = []

        try:
            async with websockets.connect("ws://localhost:8773") as ws:
                # Start receiving in background
                async def receive():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        messages_received.append(json.loads(msg))
                    except asyncio.TimeoutError:
                        pass

                receive_task = asyncio.create_task(receive())

                # Broadcast state update
                await server.broadcast_state_update("RECORDING", total_keyframes=5)

                await receive_task

                assert len(messages_received) == 1
                assert messages_received[0]["type"] == "state_update"
                assert messages_received[0]["state"] == "RECORDING"
                assert messages_received[0]["total_keyframes"] == 5

        finally:
            await server.stop()


class TestBinaryCodecThroughWebSocket:
    """Test binary codec through actual WebSocket transmission."""

    @pytest.mark.asyncio
    async def test_large_image_transmission(self):
        """Test transmitting large JPEG images."""
        import websockets

        # Simulate ~50KB JPEG
        large_image = bytes(range(256)) * 200  # ~51KB

        received_sizes = []

        def frame_callback(frame: FrameMessage):
            received_sizes.append(len(frame.image_data))
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

        server = WebSocketServer(port=8774)
        server.set_frame_callback(frame_callback)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8774") as ws:
                telemetry = Telemetry(
                    velocity_x=0.0,
                    velocity_y=0.0,
                    velocity_z=0.0,
                    yaw=0.0,
                    pitch=0.0,
                    roll=0.0,
                    height=10.0,
                )

                frame = FrameMessage(
                    message_type=MessageType.FRAME_TELEMETRY,
                    frame_id=1,
                    timestamp_ms=int(time.time() * 1000),
                    telemetry=telemetry,
                    image_data=large_image,
                )

                await ws.send(BinaryCodec.encode_frame_message(frame))
                await asyncio.wait_for(ws.recv(), timeout=2.0)

                assert len(received_sizes) == 1
                assert received_sizes[0] == len(large_image)

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_telemetry_values_preserved(self):
        """Test that telemetry values are preserved through transmission."""
        import websockets

        received_telemetry = []

        def frame_callback(frame: FrameMessage):
            received_telemetry.append(frame.telemetry)
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

        server = WebSocketServer(port=8775)
        server.set_frame_callback(frame_callback)
        await server.start()

        try:
            async with websockets.connect("ws://localhost:8775") as ws:
                telemetry = Telemetry(
                    velocity_x=1.23,
                    velocity_y=-0.45,
                    velocity_z=0.67,
                    yaw=45.5,
                    pitch=5.2,
                    roll=-2.1,
                    height=15.3,
                )

                frame = FrameMessage(
                    message_type=MessageType.FRAME_TELEMETRY,
                    frame_id=1,
                    timestamp_ms=int(time.time() * 1000),
                    telemetry=telemetry,
                    image_data=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
                )

                await ws.send(BinaryCodec.encode_frame_message(frame))
                await asyncio.wait_for(ws.recv(), timeout=2.0)

                assert len(received_telemetry) == 1
                recv = received_telemetry[0]

                assert abs(recv.velocity_x - 1.23) < 0.001
                assert abs(recv.velocity_y - (-0.45)) < 0.001
                assert abs(recv.velocity_z - 0.67) < 0.001
                assert abs(recv.yaw - 45.5) < 0.001
                assert abs(recv.pitch - 5.2) < 0.001
                assert abs(recv.roll - (-2.1)) < 0.001
                assert abs(recv.height - 15.3) < 0.001

        finally:
            await server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

