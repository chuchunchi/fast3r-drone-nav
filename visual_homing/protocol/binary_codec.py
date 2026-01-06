"""Binary encoding/decoding for WebSocket communication."""

import struct
from typing import Tuple

from .messages import (
    ControlCommand,
    ControlResponse,
    FrameMessage,
    MessageType,
    StatusMessage,
    Telemetry,
)


class BinaryCodec:
    """
    Binary codec for efficient WebSocket communication.

    Frame format (client -> server):
    ┌────────────────────────────────────────────────────────────────┐
    │ Byte Offset │ Size    │ Field            │ Description         │
    ├─────────────┼─────────┼──────────────────┼─────────────────────┤
    │ 0           │ 1       │ message_type     │ 0x01 = Frame+Telem  │
    │ 1           │ 4       │ frame_id         │ uint32, sequential  │
    │ 5           │ 8       │ timestamp_ms     │ uint64, epoch ms    │
    │ 13          │ 4       │ velocity_x       │ float32, m/s        │
    │ 17          │ 4       │ velocity_y       │ float32, m/s        │
    │ 21          │ 4       │ velocity_z       │ float32, m/s        │
    │ 25          │ 4       │ yaw              │ float32, degrees    │
    │ 29          │ 4       │ pitch            │ float32, degrees    │
    │ 33          │ 4       │ roll             │ float32, degrees    │
    │ 37          │ 4       │ height           │ float32, meters     │
    │ 41          │ 4       │ image_size       │ uint32, JPEG bytes  │
    │ 45          │ N       │ image_data       │ JPEG bytes          │
    └────────────────────────────────────────────────────────────────┘
    """

    # Header format: type(1) + frame_id(4) + timestamp(8) + 7*float(28) + image_size(4)
    HEADER_FORMAT = "<B I Q 7f I"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 45 bytes

    @staticmethod
    def encode_frame_message(msg: FrameMessage) -> bytes:
        """
        Encode a frame message to binary.

        Args:
            msg: FrameMessage to encode.

        Returns:
            Binary data.
        """
        header = struct.pack(
            BinaryCodec.HEADER_FORMAT,
            msg.message_type,
            msg.frame_id,
            msg.timestamp_ms,
            msg.telemetry.velocity_x,
            msg.telemetry.velocity_y,
            msg.telemetry.velocity_z,
            msg.telemetry.yaw,
            msg.telemetry.pitch,
            msg.telemetry.roll,
            msg.telemetry.height,
            len(msg.image_data),
        )
        return header + msg.image_data

    @staticmethod
    def decode_frame_message(data: bytes) -> FrameMessage:
        """
        Decode a binary frame message.

        Args:
            data: Binary data.

        Returns:
            FrameMessage.
        """
        if len(data) < BinaryCodec.HEADER_SIZE:
            raise ValueError(
                f"Data too short: {len(data)} < {BinaryCodec.HEADER_SIZE}"
            )

        header = struct.unpack(BinaryCodec.HEADER_FORMAT, data[: BinaryCodec.HEADER_SIZE])

        (
            message_type,
            frame_id,
            timestamp_ms,
            vx,
            vy,
            vz,
            yaw,
            pitch,
            roll,
            height,
            image_size,
        ) = header

        image_data = data[BinaryCodec.HEADER_SIZE : BinaryCodec.HEADER_SIZE + image_size]

        if len(image_data) != image_size:
            raise ValueError(
                f"Image data mismatch: {len(image_data)} != {image_size}"
            )

        return FrameMessage(
            message_type=MessageType(message_type),
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            telemetry=Telemetry(
                velocity_x=vx,
                velocity_y=vy,
                velocity_z=vz,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                height=height,
            ),
            image_data=image_data,
        )

    @staticmethod
    def encode_control_response(
        frame_id: int,
        timestamp_ms: int,
        command: ControlCommand,
        status: StatusMessage,
    ) -> dict:
        """
        Encode control response as JSON-serializable dict.

        The downstream response uses JSON for flexibility.

        Args:
            frame_id: Corresponding frame ID.
            timestamp_ms: Server timestamp.
            command: Control command.
            status: Status message.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "type": "control",
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "command": {
                "pitch_velocity": command.pitch_velocity,
                "roll_velocity": command.roll_velocity,
                "vertical_velocity": command.vertical_velocity,
                "yaw_rate": command.yaw_rate,
            },
            "status": {
                "state": status.state,
                "keyframes_remaining": status.keyframes_remaining,
                "target_distance_m": status.target_distance_m,
                "confidence": status.confidence,
                "total_keyframes": status.total_keyframes,
                "total_distance_m": status.total_distance_m,
            },
        }

    @staticmethod
    def decode_control_response(data: dict) -> ControlResponse:
        """
        Decode control response from JSON dict.

        Args:
            data: JSON dictionary.

        Returns:
            ControlResponse.
        """
        cmd = data["command"]
        status = data["status"]

        return ControlResponse(
            message_type=MessageType.CONTROL_COMMAND,
            frame_id=data["frame_id"],
            timestamp_ms=data["timestamp_ms"],
            command=ControlCommand(
                pitch_velocity=cmd["pitch_velocity"],
                roll_velocity=cmd["roll_velocity"],
                vertical_velocity=cmd["vertical_velocity"],
                yaw_rate=cmd["yaw_rate"],
            ),
            status=StatusMessage(
                state=status["state"],
                keyframes_remaining=status["keyframes_remaining"],
                target_distance_m=status["target_distance_m"],
                confidence=status["confidence"],
                total_keyframes=status.get("total_keyframes", 0),
                total_distance_m=status.get("total_distance_m", 0.0),
            ),
        )


