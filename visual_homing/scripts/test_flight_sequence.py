#!/usr/bin/env python3
"""
Test script for analyzing a flight sequence.

Analyzes consecutive image pairs from a flight sequence and displays
the estimated relative poses, validating the system with real data.

Uses Fast3R's built-in PnP-based camera pose estimation for accurate results.

Usage:
    python test_flight_sequence.py --folder demo_examples/target1
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.fast3r_engine import Fast3REngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """Extract Euler angles (roll, pitch, yaw) from rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze flight sequence")
    parser.add_argument(
        "--folder",
        type=str,
        default="demo_examples/target1",
        help="Folder containing flight images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    args = parser.parse_args()

    # Find all images
    folder = Path(args.folder)
    images = sorted(folder.glob("*.jpg"))

    if len(images) < 2:
        logger.error(f"Need at least 2 images, found {len(images)}")
        return 1

    logger.info(f"Found {len(images)} images in {folder}")
    for img in images:
        logger.info(f"  - {img.name}")

    # Initialize engine
    logger.info("\nInitializing Fast3R engine...")
    config = Config(device=args.device)
    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    # Analyze consecutive pairs
    print("\n" + "=" * 80)
    print("FLIGHT SEQUENCE ANALYSIS (Using PnP-based Pose Estimation)")
    print("=" * 80)
    print("\nExpected movements (based on your description):")
    print("  - 100 → 400: Forward flight")
    print("  - 500 → 700: Leftward flight (strafing, no rotation)")
    print("\nCoordinate frame: X=right, Y=down, Z=forward (camera frame)")
    print("-" * 80)

    cumulative_forward = 0.0
    cumulative_lateral = 0.0
    cumulative_vertical = 0.0

    for i in range(len(images) - 1):
        img1_path = images[i]
        img2_path = images[i + 1]

        img1 = load_image(str(img1_path))
        img2 = load_image(str(img2_path))

        # Run inference with pose estimation
        result = fast3r.infer_pair_with_pose(img1, img2, niter_PnP=10)

        R_rel = result["R_rel"]
        t_rel = result["t_rel"]

        # In camera frame: X=right, Y=down, Z=forward
        # t_rel represents where camera1 origin is in camera2's frame
        # So if drone moved left, camera1 is to the RIGHT of camera2 -> positive X
        # We want to show drone motion, so we might need to negate
        
        # The relative pose tells us: cam2 = R_rel @ cam1 + t_rel
        # Meaning t_rel is cam1's origin expressed in cam2's frame
        # If drone moved LEFT from frame1 to frame2, then cam1 is to the RIGHT of cam2
        # So positive X in t_rel means drone moved LEFT
        
        # To get drone motion (from cam1 pose to cam2 pose), we need the inverse interpretation:
        dx = -t_rel[0]  # Negate: positive X in t_rel means drone went left, we report as negative
        dy = -t_rel[1]  # Negate similarly
        dz = -t_rel[2]  # Negate similarly

        cumulative_forward += dz
        cumulative_lateral += dx
        cumulative_vertical += dy

        # Get Euler angles
        roll, pitch, yaw = rotation_matrix_to_euler(R_rel)

        # Determine movement description
        movements = []
        if abs(dz) > 0.001:
            movements.append(f"{'Forward' if dz > 0 else 'Backward'}: {abs(dz):.4f}")
        if abs(dx) > 0.001:
            movements.append(f"{'Right' if dx > 0 else 'Left'}: {abs(dx):.4f}")
        if abs(dy) > 0.001:
            movements.append(f"{'Down' if dy > 0 else 'Up'}: {abs(dy):.4f}")

        distance = np.linalg.norm(t_rel)

        print(f"\n{img1_path.stem} → {img2_path.stem}:")
        print(f"  Translation: X={dx:+.4f}, Y={dy:+.4f}, Z={dz:+.4f}")
        print(f"  Distance: {distance:.4f} (Fast3R units)")
        print(f"  Rotation: roll={roll:+.1f}°, pitch={pitch:+.1f}°, yaw={yaw:+.1f}°")
        print(f"  Movement: {', '.join(movements) if movements else 'Minimal'}")

    print("\n" + "-" * 80)
    print("CUMULATIVE MOTION SUMMARY")
    print("-" * 80)
    print(f"  Total Forward (Z):  {cumulative_forward:+.4f}")
    print(f"  Total Lateral (X):  {cumulative_lateral:+.4f}")
    print(f"  Total Vertical (Y): {cumulative_vertical:+.4f}")
    print("\nInterpretation:")
    if cumulative_forward > 0:
        print(f"  - Net forward movement: {cumulative_forward:.4f}")
    else:
        print(f"  - Net backward movement: {-cumulative_forward:.4f}")
    if cumulative_lateral > 0:
        print(f"  - Net rightward movement: {cumulative_lateral:.4f}")
    else:
        print(f"  - Net leftward movement: {-cumulative_lateral:.4f}")
    print("\nNote: Values are in Fast3R's internal scale.")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

