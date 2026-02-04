#!/usr/bin/env python3
"""
Debug script to visualize pose estimation during homing.

This script reads the server logs or connects to live flight and shows:
1. Current pose estimate (translation + rotation)
2. Target direction visualization
3. PID command output
4. Whether pose estimation is successful

Usage:
    # Visualize from saved keyframes and current position
    python debug_pose_estimation.py --live-image path/to/current.jpg --keyframe-folder path/to/keyframes

    # Show detailed info about what controller sees
    python debug_pose_estimation.py --controller-debug
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.coordinate_utils import extract_yaw_error
from visual_homing.server.fast3r_engine import Fast3REngine
from visual_homing.server.pid_controller import MultiAxisPIDController
from visual_homing.server.pose_estimator import PoseEstimator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load image as RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_pose_estimate(
    live_image: np.ndarray,
    target_image: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    confidence: float,
    command: dict,
    distance: float,
):
    """Create visualization of pose estimate and control command."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Live image
    axes[0, 0].imshow(live_image)
    axes[0, 0].set_title("Live Image (Current Position)")
    axes[0, 0].axis("off")

    # Target image
    axes[0, 1].imshow(target_image)
    axes[0, 1].set_title("Target Keyframe")
    axes[0, 1].axis("off")

    # Translation visualization (bird's eye view)
    ax = axes[0, 2]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Forward (m)")
    ax.set_title(f"Pose Estimate (Distance: {distance:.2f}m)")

    # Draw drone position (at origin)
    ax.plot(0, 0, "bo", markersize=15, label="Drone (current)")
    ax.arrow(0, 0, 0, 0.5, head_width=0.2, head_length=0.2, fc="blue", ec="blue")

    # Draw target position (camera frame: X=right, Y=down, Z=forward)
    target_x = translation[0]  # Lateral (right)
    target_z = translation[2]  # Forward
    ax.plot(target_x, target_z, "ro", markersize=15, label="Target")

    # Draw arrow from drone to target
    ax.arrow(0, 0, target_x, target_z, head_width=0.2, head_length=0.2,
             fc="green", ec="green", linestyle="--", alpha=0.7, label="Required motion")

    ax.legend()

    # Translation details
    ax = axes[1, 0]
    ax.axis("off")
    ax.text(0.1, 0.9, "Translation (Camera Frame):", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.1, 0.75, f"Forward (Z): {translation[2]:+.3f}m", fontsize=10,
            transform=ax.transAxes, color="blue" if abs(translation[2]) > 0.1 else "gray")
    ax.text(0.1, 0.65, f"Lateral (X): {translation[0]:+.3f}m", fontsize=10,
            transform=ax.transAxes, color="blue" if abs(translation[0]) > 0.1 else "gray")
    ax.text(0.1, 0.55, f"Vertical (Y): {translation[1]:+.3f}m", fontsize=10,
            transform=ax.transAxes, color="blue" if abs(translation[1]) > 0.1 else "gray")
    ax.text(0.1, 0.45, f"Distance: {distance:.3f}m", fontsize=10,
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.1, 0.35, f"Confidence: {confidence:.3f}", fontsize=10,
            transform=ax.transAxes,
            color="green" if confidence > 0.5 else ("orange" if confidence > 0.3 else "red"))

    yaw_error = extract_yaw_error(rotation)
    ax.text(0.1, 0.20, f"Yaw Error: {yaw_error:+.1f}°", fontsize=10,
            transform=ax.transAxes, color="blue" if abs(yaw_error) > 5 else "gray")

    # Command output
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.1, 0.9, "Control Command:", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.1, 0.75, f"Forward: {command['pitch_velocity']:+.3f} m/s", fontsize=10,
            transform=ax.transAxes,
            color="red" if abs(command['pitch_velocity']) > 0.01 else "gray")
    ax.text(0.1, 0.65, f"Lateral: {command['roll_velocity']:+.3f} m/s", fontsize=10,
            transform=ax.transAxes,
            color="red" if abs(command['roll_velocity']) > 0.01 else "gray")
    ax.text(0.1, 0.55, f"Vertical: {command['vertical_velocity']:+.3f} m/s", fontsize=10,
            transform=ax.transAxes,
            color="red" if abs(command['vertical_velocity']) > 0.01 else "gray")
    ax.text(0.1, 0.45, f"Yaw Rate: {command['yaw_rate']:+.3f} °/s", fontsize=10,
            transform=ax.transAxes,
            color="red" if abs(command['yaw_rate']) > 1 else "gray")

    # Velocity magnitude
    vel_mag = np.sqrt(
        command['pitch_velocity']**2 +
        command['roll_velocity']**2 +
        command['vertical_velocity']**2
    )
    ax.text(0.1, 0.30, f"Total velocity: {vel_mag:.3f} m/s", fontsize=10,
            transform=ax.transAxes, fontweight="bold",
            color="red" if vel_mag > 0.01 else "gray")

    # Interpretation
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0.1, 0.9, "Interpretation:", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    y_pos = 0.75

    # Check if drone should move
    if distance < 0.5:
        ax.text(0.1, y_pos, "✓ Near target (< 0.5m)", fontsize=10,
                transform=ax.transAxes, color="green")
        y_pos -= 0.1
    elif distance > 5.0:
        ax.text(0.1, y_pos, "⚠ Very far from target (> 5m)", fontsize=10,
                transform=ax.transAxes, color="orange")
        y_pos -= 0.1
    else:
        ax.text(0.1, y_pos, f"● Target is {distance:.2f}m away", fontsize=10,
                transform=ax.transAxes, color="blue")
        y_pos -= 0.1

    # Check command magnitude
    if vel_mag < 0.01:
        ax.text(0.1, y_pos, "⚠ Command is HOVER (no movement!)", fontsize=10,
                transform=ax.transAxes, color="red", fontweight="bold")
        y_pos -= 0.1
        ax.text(0.1, y_pos, "  Possible reasons:", fontsize=9,
                transform=ax.transAxes, color="red")
        y_pos -= 0.08
        ax.text(0.1, y_pos, "  - Errors too small for PID", fontsize=9,
                transform=ax.transAxes, color="red")
        y_pos -= 0.08
        ax.text(0.1, y_pos, "  - Already at target", fontsize=9,
                transform=ax.transAxes, color="red")
        y_pos -= 0.08
    else:
        ax.text(0.1, y_pos, f"✓ Velocity command: {vel_mag:.3f} m/s", fontsize=10,
                transform=ax.transAxes, color="green")
        y_pos -= 0.1

    # Check confidence
    if confidence < 0.3:
        ax.text(0.1, y_pos, "⚠ Low confidence pose estimate", fontsize=10,
                transform=ax.transAxes, color="red")
        y_pos -= 0.1
    elif confidence < 0.5:
        ax.text(0.1, y_pos, "⚠ Moderate confidence", fontsize=10,
                transform=ax.transAxes, color="orange")
        y_pos -= 0.1
    else:
        ax.text(0.1, y_pos, "✓ Good confidence", fontsize=10,
                transform=ax.transAxes, color="green")
        y_pos -= 0.1

    # Direction recommendation
    if abs(translation[2]) > 0.1:
        direction = "forward" if translation[2] > 0 else "backward"
        ax.text(0.1, y_pos, f"→ Should move {direction}", fontsize=10,
                transform=ax.transAxes, color="blue")
        y_pos -= 0.1

    if abs(translation[0]) > 0.1:
        direction = "right" if translation[0] > 0 else "left"
        ax.text(0.1, y_pos, f"→ Should move {direction}", fontsize=10,
                transform=ax.transAxes, color="blue")
        y_pos -= 0.1

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Debug pose estimation")
    parser.add_argument("--live-image", required=True, help="Current drone position image")
    parser.add_argument("--target-image", required=True, help="Target keyframe image")
    parser.add_argument("--metric-scale", type=float, default=1.0,
                       help="Metric scale factor")
    parser.add_argument("--output", default="pose_debug.png",
                       help="Output visualization path")

    args = parser.parse_args()

    # Load images
    logger.info(f"Loading live image: {args.live_image}")
    live_img = load_image(args.live_image)
    logger.info(f"Loading target image: {args.target_image}")
    target_img = load_image(args.target_image)

    # Initialize components
    config = Config()
    logger.info("Loading Fast3R model...")
    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    pose_estimator = PoseEstimator(
        confidence_percentile=0.5,
        min_points=100,
        metric_scale=args.metric_scale,
    )

    pid = MultiAxisPIDController(
        forward_gains=(config.pid_forward_kp, config.pid_forward_ki, config.pid_forward_kd),
        lateral_gains=(config.pid_lateral_kp, config.pid_lateral_ki, config.pid_lateral_kd),
        vertical_gains=(config.pid_vertical_kp, config.pid_vertical_ki, config.pid_vertical_kd),
        yaw_gains=(config.pid_yaw_kp, config.pid_yaw_ki, config.pid_yaw_kd),
        velocity_limits={
            "forward": config.max_forward_velocity,
            "lateral": config.max_lateral_velocity,
            "vertical": config.max_vertical_velocity,
            "yaw": config.max_yaw_rate,
        },
    )

    # Run inference
    logger.info("Running Fast3R inference...")
    result = fast3r.infer_pair(live_img, target_img)

    # Estimate pose
    logger.info("Estimating pose...")
    pose_result = pose_estimator.estimate_pose(
        result["pts3d_1"],
        result["pts3d_2"],
        result["conf_1"],
    )

    if not pose_result.success:
        logger.error("❌ Pose estimation FAILED!")
        logger.error(f"   Confidence: {pose_result.confidence:.3f}")
        logger.error(f"   Inliers: {pose_result.num_inliers}")
        sys.exit(1)

    logger.info(f"✓ Pose estimation successful (confidence: {pose_result.confidence:.3f})")

    # Extract pose
    t_cam = pose_result.translation.cpu().numpy()
    R = pose_result.rotation

    error_forward = float(t_cam[2])
    error_lateral = float(t_cam[0])
    error_vertical = -float(t_cam[1])
    error_yaw = extract_yaw_error(R)
    distance = float(np.linalg.norm(t_cam))

    logger.info(f"   Forward: {error_forward:+.3f}m")
    logger.info(f"   Lateral: {error_lateral:+.3f}m")
    logger.info(f"   Vertical: {error_vertical:+.3f}m")
    logger.info(f"   Yaw: {error_yaw:+.1f}°")
    logger.info(f"   Distance: {distance:.3f}m")

    # Compute command
    command = pid.compute(error_forward, error_lateral, error_vertical, error_yaw)

    logger.info("Control command:")
    logger.info(f"   Forward velocity: {command['pitch_velocity']:+.3f} m/s")
    logger.info(f"   Lateral velocity: {command['roll_velocity']:+.3f} m/s")
    logger.info(f"   Vertical velocity: {command['vertical_velocity']:+.3f} m/s")
    logger.info(f"   Yaw rate: {command['yaw_rate']:+.3f} °/s")

    # Create visualization
    fig = visualize_pose_estimate(
        live_img, target_img, t_cam, R,
        pose_result.confidence, command, distance
    )

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Visualization saved to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
