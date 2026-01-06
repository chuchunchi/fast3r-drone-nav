"""Tests for communication protocol."""

import pytest
import struct

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.protocol.messages import (
    MessageType,
    Telemetry,
    FrameMessage,
    ControlCommand,
    StatusMessage,
)
from visual_homing.protocol.binary_codec import BinaryCodec


class TestMessageTypes:
    """Test message type enum."""

    def test_frame_telemetry_value(self):
        """Test FRAME_TELEMETRY has correct value."""
        assert MessageType.FRAME_TELEMETRY == 0x01

    def test_control_command_value(self):
        """Test CONTROL_COMMAND has correct value."""
        assert MessageType.CONTROL_COMMAND == 0x02

    def test_all_types_unique(self):
        """Test all message types are unique."""
        values = [m.value for m in MessageType]
        assert len(values) == len(set(values))


class TestTelemetry:
    """Test telemetry dataclass."""

    def test_telemetry_creation(self):
        """Test creating telemetry object."""
        telem = Telemetry(
            velocity_x=1.0,
            velocity_y=0.5,
            velocity_z=-0.2,
            yaw=45.0,
            pitch=5.0,
            roll=-2.0,
            height=10.0,
        )

        assert telem.velocity_x == 1.0
        assert telem.velocity_y == 0.5
        assert telem.velocity_z == -0.2
        assert telem.yaw == 45.0
        assert telem.height == 10.0


class TestFrameMessage:
    """Test frame message dataclass."""

    def test_frame_message_creation(self):
        """Test creating frame message."""
        telem = Telemetry(
            velocity_x=1.0,
            velocity_y=0.0,
            velocity_z=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            height=5.0,
        )

        msg = FrameMessage(
            message_type=MessageType.FRAME_TELEMETRY,
            frame_id=12345,
            timestamp_ms=1704412800000,
            telemetry=telem,
            image_data=b"\xff\xd8\xff\xe0",  # JPEG magic bytes
        )

        assert msg.frame_id == 12345
        assert msg.timestamp_ms == 1704412800000
        assert len(msg.image_data) == 4


class TestControlCommand:
    """Test control command dataclass."""

    def test_control_command_creation(self):
        """Test creating control command."""
        cmd = ControlCommand(
            pitch_velocity=0.5,
            roll_velocity=-0.2,
            vertical_velocity=0.1,
            yaw_rate=10.0,
        )

        assert cmd.pitch_velocity == 0.5
        assert cmd.roll_velocity == -0.2
        assert cmd.vertical_velocity == 0.1
        assert cmd.yaw_rate == 10.0


class TestBinaryCodec:
    """Test binary codec."""

    def test_header_size(self):
        """Test header size constant."""
        # Should be 45 bytes based on spec
        assert BinaryCodec.HEADER_SIZE == 45

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding produces same data."""
        telem = Telemetry(
            velocity_x=1.5,
            velocity_y=-0.5,
            velocity_z=0.2,
            yaw=45.0,
            pitch=5.0,
            roll=-2.0,
            height=10.5,
        )

        original_msg = FrameMessage(
            message_type=MessageType.FRAME_TELEMETRY,
            frame_id=42,
            timestamp_ms=1704412800000,
            telemetry=telem,
            image_data=b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09",
        )

        # Encode
        binary_data = BinaryCodec.encode_frame_message(original_msg)

        # Decode
        decoded_msg = BinaryCodec.decode_frame_message(binary_data)

        # Verify
        assert decoded_msg.message_type == original_msg.message_type
        assert decoded_msg.frame_id == original_msg.frame_id
        assert decoded_msg.timestamp_ms == original_msg.timestamp_ms
        assert abs(decoded_msg.telemetry.velocity_x - telem.velocity_x) < 0.0001
        assert abs(decoded_msg.telemetry.velocity_y - telem.velocity_y) < 0.0001
        assert abs(decoded_msg.telemetry.velocity_z - telem.velocity_z) < 0.0001
        assert abs(decoded_msg.telemetry.yaw - telem.yaw) < 0.0001
        assert abs(decoded_msg.telemetry.height - telem.height) < 0.0001
        assert decoded_msg.image_data == original_msg.image_data

    def test_encode_large_image(self):
        """Test encoding with large image data."""
        telem = Telemetry(
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            height=0.0,
        )

        # Simulate a ~30KB JPEG
        image_data = bytes(range(256)) * 120  # ~30KB

        msg = FrameMessage(
            message_type=MessageType.FRAME_TELEMETRY,
            frame_id=1,
            timestamp_ms=0,
            telemetry=telem,
            image_data=image_data,
        )

        binary = BinaryCodec.encode_frame_message(msg)

        # Should be header + image
        assert len(binary) == BinaryCodec.HEADER_SIZE + len(image_data)

        # Roundtrip
        decoded = BinaryCodec.decode_frame_message(binary)
        assert decoded.image_data == image_data

    def test_decode_too_short_data(self):
        """Test decoding fails with too short data."""
        short_data = b"\x01\x02\x03"  # Way too short

        with pytest.raises(ValueError, match="Data too short"):
            BinaryCodec.decode_frame_message(short_data)

    def test_decode_image_size_mismatch(self):
        """Test decoding fails when image size doesn't match."""
        # Create valid header but wrong image size
        header = struct.pack(
            BinaryCodec.HEADER_FORMAT,
            0x01,  # message_type
            1,  # frame_id
            1000,  # timestamp
            0.0, 0.0, 0.0,  # velocities
            0.0, 0.0, 0.0,  # attitude
            0.0,  # height
            100,  # image_size claims 100 bytes
        )
        # But only provide 10 bytes of image
        data = header + b"\x00" * 10

        with pytest.raises(ValueError, match="Image data mismatch"):
            BinaryCodec.decode_frame_message(data)

    def test_control_response_encoding(self):
        """Test encoding control response to JSON dict."""
        cmd = ControlCommand(
            pitch_velocity=0.5,
            roll_velocity=-0.2,
            vertical_velocity=0.0,
            yaw_rate=5.0,
        )

        status = StatusMessage(
            state="HOMING",
            keyframes_remaining=5,
            target_distance_m=2.3,
            confidence=0.85,
            total_keyframes=10,
            total_distance_m=15.0,
        )

        result = BinaryCodec.encode_control_response(
            frame_id=100,
            timestamp_ms=2000,
            command=cmd,
            status=status,
        )

        assert result["type"] == "control"
        assert result["frame_id"] == 100
        assert result["timestamp_ms"] == 2000
        assert result["command"]["pitch_velocity"] == 0.5
        assert result["command"]["roll_velocity"] == -0.2
        assert result["status"]["state"] == "HOMING"
        assert result["status"]["keyframes_remaining"] == 5
        assert result["status"]["confidence"] == 0.85

    def test_control_response_roundtrip(self):
        """Test control response encode/decode roundtrip."""
        cmd = ControlCommand(
            pitch_velocity=1.0,
            roll_velocity=0.5,
            vertical_velocity=-0.3,
            yaw_rate=15.0,
        )

        status = StatusMessage(
            state="ARMED",
            keyframes_remaining=3,
            target_distance_m=1.5,
            confidence=0.9,
            total_keyframes=8,
            total_distance_m=12.0,
        )

        encoded = BinaryCodec.encode_control_response(
            frame_id=50,
            timestamp_ms=5000,
            command=cmd,
            status=status,
        )

        decoded = BinaryCodec.decode_control_response(encoded)

        assert decoded.frame_id == 50
        assert decoded.timestamp_ms == 5000
        assert decoded.command.pitch_velocity == 1.0
        assert decoded.command.roll_velocity == 0.5
        assert decoded.status.state == "ARMED"
        assert decoded.status.keyframes_remaining == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

