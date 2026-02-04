#!/usr/bin/env python3
"""
Real-time monitor for homing controller state.

This connects to the running server and displays what the controller sees.

Usage:
    python monitor_homing.py
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.homing_controller import HomingController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_debug_info(controller: HomingController):
    """Print current controller debug state."""
    info = controller.get_debug_info()

    print("\n" + "="*60)
    print("HOMING CONTROLLER STATUS")
    print("="*60)

    print(f"\nState: {info['state']}")
    print(f"Target Keyframe: {info['target_idx']}")
    print(f"Waypoint Confirmation: {info['waypoint_confirm_count']}/2")
    print(f"Total Keyframes: {info['keyframe_count']}")
    print(f"Metric Scale: {info['metric_scale']:.4f}")
    print(f"Consecutive Failures: {info['consecutive_failures']}")

    stats = info['stats']
    print(f"\nStatistics:")
    print(f"  Total Frames: {stats['total_frames']}")
    print(f"  Successful: {stats['successful_poses']}")
    print(f"  Failed: {stats['failed_poses']}")
    print(f"  Low Confidence: {stats['low_confidence_frames']}")
    print(f"  Waypoints Reached: {stats['waypoints_reached']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Avg Inference: {stats['avg_inference_time_ms']:.1f}ms")

    if 'last_translation' in info:
        t = info['last_translation']
        print(f"\nLast Pose Estimate:")
        print(f"  Forward (Z): {t[2]:+.3f}m")
        print(f"  Lateral (X): {t[0]:+.3f}m")
        print(f"  Vertical (Y): {-t[1]:+.3f}m")
        distance = (t[0]**2 + t[1]**2 + t[2]**2)**0.5
        print(f"  Distance: {distance:.3f}m")
        print(f"  Confidence: {info.get('last_confidence', 0):.3f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    print("This is a template for monitoring a running controller.")
    print("In actual use, you would:")
    print("1. Connect to the running server")
    print("2. Query controller.get_debug_info() periodically")
    print("3. Display the information")
    print("\nTo use this with your server, modify run_server.py to expose")
    print("the controller instance via a monitoring endpoint.")
