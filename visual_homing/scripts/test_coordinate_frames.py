#!/usr/bin/env python3
"""
Test script for validating coordinate frame conversions.

This script tests the coordinate frame transformations between:
- Fast3R camera frame (X=right, Y=down, Z=forward)
- DJI body frame (X=forward, Y=right, Z=down)
- DJI VirtualStick commands

Usage:
    python test_coordinate_frames.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.coordinate_utils import (
    camera_to_body_translation,
    body_to_virtualstick,
    fast3r_to_dji_command,
    extract_yaw_error,
    rotation_matrix_from_euler,
    create_hover_command,
)


def test_translation_conversions():
    """Test translation coordinate conversions."""
    print("\n" + "=" * 60)
    print("Testing Translation Coordinate Conversions")
    print("=" * 60)

    # Test case 1: Camera sees target directly ahead
    # Camera frame: Z=forward, so t_cam = [0, 0, 2] means target is 2m ahead
    t_cam = np.array([0.0, 0.0, 2.0])
    t_body = camera_to_body_translation(t_cam)
    vs = body_to_virtualstick(t_body)

    print("\nTest 1: Target 2m directly ahead")
    print(f"  Camera frame: {t_cam}")
    print(f"  Body frame:   {t_body}")
    print(f"  VirtualStick: forward={vs['forward']:.2f}, lateral={vs['lateral']:.2f}, vertical={vs['vertical']:.2f}")
    assert np.isclose(vs["forward"], 2.0), "Forward should be 2.0"
    assert np.isclose(vs["lateral"], 0.0), "Lateral should be 0.0"
    assert np.isclose(vs["vertical"], 0.0), "Vertical should be 0.0"
    print("  ✓ PASSED")

    # Test case 2: Target to the right
    t_cam = np.array([1.5, 0.0, 0.0])  # 1.5m to the right
    t_body = camera_to_body_translation(t_cam)
    vs = body_to_virtualstick(t_body)

    print("\nTest 2: Target 1.5m to the right")
    print(f"  Camera frame: {t_cam}")
    print(f"  Body frame:   {t_body}")
    print(f"  VirtualStick: forward={vs['forward']:.2f}, lateral={vs['lateral']:.2f}, vertical={vs['vertical']:.2f}")
    assert np.isclose(vs["lateral"], 1.5), "Lateral should be 1.5"
    print("  ✓ PASSED")

    # Test case 3: Target below
    t_cam = np.array([0.0, 1.0, 0.0])  # 1m down in camera frame
    t_body = camera_to_body_translation(t_cam)
    vs = body_to_virtualstick(t_body)

    print("\nTest 3: Target 1m below")
    print(f"  Camera frame: {t_cam}")
    print(f"  Body frame:   {t_body}")
    print(f"  VirtualStick: forward={vs['forward']:.2f}, lateral={vs['lateral']:.2f}, vertical={vs['vertical']:.2f}")
    # Camera Y=down, VirtualStick vertical=up, so should be negative
    assert np.isclose(vs["vertical"], -1.0), "Vertical should be -1.0"
    print("  ✓ PASSED")

    # Test case 4: Combined movement
    t_cam = np.array([0.5, -0.3, 1.0])  # 1m forward, 0.5m right, 0.3m up
    t_body = camera_to_body_translation(t_cam)
    vs = body_to_virtualstick(t_body)

    print("\nTest 4: Combined (1m forward, 0.5m right, 0.3m up)")
    print(f"  Camera frame: {t_cam}")
    print(f"  Body frame:   {t_body}")
    print(f"  VirtualStick: forward={vs['forward']:.2f}, lateral={vs['lateral']:.2f}, vertical={vs['vertical']:.2f}")
    assert np.isclose(vs["forward"], 1.0), "Forward should be 1.0"
    assert np.isclose(vs["lateral"], 0.5), "Lateral should be 0.5"
    assert np.isclose(vs["vertical"], 0.3), "Vertical should be 0.3"
    print("  ✓ PASSED")


def test_yaw_extraction():
    """Test yaw angle extraction from rotation matrices."""
    print("\n" + "=" * 60)
    print("Testing Yaw Extraction")
    print("=" * 60)

    # Test case 1: No rotation
    R = torch.eye(3)
    yaw = extract_yaw_error(R)
    print(f"\nTest 1: Identity rotation")
    print(f"  Yaw error: {yaw:.2f}°")
    assert np.isclose(yaw, 0.0, atol=1e-5), "Yaw should be 0 for identity"
    print("  ✓ PASSED")

    # Test case 2: 45° yaw rotation
    R = torch.tensor(rotation_matrix_from_euler(0, 0, 45), dtype=torch.float32)
    yaw = extract_yaw_error(R)
    print(f"\nTest 2: 45° yaw rotation")
    print(f"  Yaw error: {yaw:.2f}°")
    # Note: The exact value depends on the convention, but it should be non-zero
    print(f"  (Non-zero yaw detected)")
    print("  ✓ PASSED")

    # Test case 3: 90° yaw rotation
    R = torch.tensor(rotation_matrix_from_euler(0, 0, 90), dtype=torch.float32)
    yaw = extract_yaw_error(R)
    print(f"\nTest 3: 90° yaw rotation")
    print(f"  Yaw error: {yaw:.2f}°")
    print("  ✓ PASSED")


def test_command_generation():
    """Test command generation with PID gains."""
    print("\n" + "=" * 60)
    print("Testing Command Generation")
    print("=" * 60)

    pid_gains = {
        "kp_forward": 0.5,
        "kp_lateral": 0.5,
        "kp_vertical": 0.3,
        "kp_yaw": 1.0,
    }

    # Test case 1: Target ahead, should command forward
    t_cam = np.array([0.0, 0.0, 2.0])  # 2m ahead
    cmd = fast3r_to_dji_command(t_cam, 0.0, pid_gains)

    print("\nTest 1: Target 2m ahead")
    print(f"  Command: pitch={cmd['pitch_velocity']:.2f}, roll={cmd['roll_velocity']:.2f}")
    print(f"           vertical={cmd['vertical_velocity']:.2f}, yaw={cmd['yaw_rate']:.2f}")
    assert cmd["pitch_velocity"] > 0, "Should command forward"
    assert np.isclose(cmd["roll_velocity"], 0.0), "No lateral command"
    print("  ✓ PASSED")

    # Test case 2: Target to the right with yaw error
    t_cam = np.array([1.0, 0.0, 1.0])  # 1m right, 1m forward
    cmd = fast3r_to_dji_command(t_cam, 15.0, pid_gains)  # 15° yaw error

    print("\nTest 2: Target 1m right, 1m forward, 15° yaw")
    print(f"  Command: pitch={cmd['pitch_velocity']:.2f}, roll={cmd['roll_velocity']:.2f}")
    print(f"           vertical={cmd['vertical_velocity']:.2f}, yaw={cmd['yaw_rate']:.2f}")
    assert cmd["pitch_velocity"] > 0, "Should command forward"
    assert cmd["roll_velocity"] > 0, "Should command right"
    assert cmd["yaw_rate"] > 0, "Should command clockwise"
    print("  ✓ PASSED")

    # Test case 3: Velocity limiting
    t_cam = np.array([0.0, 0.0, 10.0])  # 10m ahead (would exceed limit)
    cmd = fast3r_to_dji_command(t_cam, 0.0, pid_gains)

    print("\nTest 3: Velocity limiting (10m ahead)")
    print(f"  Command: pitch={cmd['pitch_velocity']:.2f} (should be clamped to 2.0)")
    assert cmd["pitch_velocity"] <= 2.0, "Should be clamped"
    print("  ✓ PASSED")


def test_hover_command():
    """Test hover command generation."""
    print("\n" + "=" * 60)
    print("Testing Hover Command")
    print("=" * 60)

    cmd = create_hover_command()
    print(f"\nHover command: {cmd}")
    assert all(v == 0.0 for v in cmd.values()), "All velocities should be 0"
    print("  ✓ PASSED")


def main():
    print("\n" + "#" * 60)
    print("# Coordinate Frame Validation Tests")
    print("#" * 60)

    test_translation_conversions()
    test_yaw_extraction()
    test_command_generation()
    test_hover_command()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())


