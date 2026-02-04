#!/usr/bin/env python3
"""
Test what commands are generated at different frame separations.

This compares frames that are N steps apart to see how the controller
responds when the drone is at different distances.

Usage:
    python test_frame_separations.py --folder path/to/images --max-separation 20
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


def test_frame_pair(fast3r, pose_estimator, pid, live_img, target_img, separation):
    """Test a single frame pair and return results."""
    # Run inference
    result = fast3r.infer_pair(live_img, target_img)

    # Estimate pose
    pose_result = pose_estimator.estimate_pose(
        result["pts3d_1"],
        result["pts3d_2"],
        result["conf_1"],
    )

    if not pose_result.success:
        return None

    # Extract pose
    t_cam = pose_result.translation.cpu().numpy()
    R = pose_result.rotation

    error_forward = float(t_cam[2])
    error_lateral = float(t_cam[0])
    error_vertical = -float(t_cam[1])
    error_yaw = extract_yaw_error(R)
    distance = float(np.linalg.norm(t_cam))

    # Compute command
    command = pid.compute(error_forward, error_lateral, error_vertical, error_yaw)

    vel_magnitude = np.sqrt(
        command['pitch_velocity']**2 +
        command['roll_velocity']**2 +
        command['vertical_velocity']**2
    )

    return {
        'separation': separation,
        'distance': distance,
        'confidence': pose_result.confidence,
        'error_forward': error_forward,
        'error_lateral': error_lateral,
        'error_vertical': error_vertical,
        'error_yaw': error_yaw,
        'cmd_forward': command['pitch_velocity'],
        'cmd_lateral': command['roll_velocity'],
        'cmd_vertical': command['vertical_velocity'],
        'cmd_yaw': command['yaw_rate'],
        'vel_magnitude': vel_magnitude,
    }


def main():
    parser = argparse.ArgumentParser(description="Test frame separations")
    parser.add_argument("--folder", required=True, help="Folder with image sequence")
    parser.add_argument("--max-separation", type=int, default=20,
                       help="Maximum frame separation to test")
    parser.add_argument("--base-frame", type=int, default=50,
                       help="Base frame index to compare from")
    parser.add_argument("--metric-scale", type=float, default=1.0)

    args = parser.parse_args()

    # Find images
    folder = Path(args.folder)
    image_paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

    if len(image_paths) < args.max_separation + 5:
        logger.error(f"Need at least {args.max_separation + 5} images")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images")
    logger.info(f"Testing separations 1 to {args.max_separation} frames")

    # Initialize
    config = Config()
    logger.info(f"Config: kp={config.pid_forward_kp}, max_vel={config.max_forward_velocity} m/s")

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

    # Test different separations
    base_idx = min(args.base_frame, len(image_paths) - args.max_separation - 1)
    live_img = load_image(str(image_paths[base_idx]))

    logger.info(f"Using frame {base_idx} as 'current position'")
    logger.info("Comparing to earlier frames (going backwards)\n")

    results = []

    for sep in range(1, args.max_separation + 1):
        target_idx = base_idx - sep  # Go backwards
        if target_idx < 0:
            break

        target_img = load_image(str(image_paths[target_idx]))

        logger.info(f"Separation {sep}: Frame {base_idx} → Frame {target_idx}")

        result = test_frame_pair(fast3r, pose_estimator, pid, live_img, target_img, sep)

        if result:
            logger.info(f"  Distance: {result['distance']:.3f}m, "
                       f"Velocity: {result['vel_magnitude']:.3f} m/s, "
                       f"Conf: {result['confidence']:.3f}")
            results.append(result)
        else:
            logger.warning(f"  Pose estimation failed")

    if not results:
        logger.error("No successful results!")
        sys.exit(1)

    # Print summary
    print("\n" + "="*70)
    print("FRAME SEPARATION ANALYSIS")
    print("="*70)
    print(f"{'Sep':<5} {'Dist(m)':<10} {'Vel(m/s)':<10} {'Fwd Cmd':<10} {'Lat Cmd':<10} {'Conf':<8}")
    print("-"*70)

    for r in results:
        print(f"{r['separation']:<5} {r['distance']:<10.3f} {r['vel_magnitude']:<10.3f} "
              f"{r['cmd_forward']:<10.3f} {r['cmd_lateral']:<10.3f} {r['confidence']:<8.3f}")

    print("="*70)

    # Check for issues
    print("\n⚠ Analysis:")

    avg_vel = np.mean([r['vel_magnitude'] for r in results])
    max_vel = np.max([r['vel_magnitude'] for r in results])

    print(f"  Average velocity command: {avg_vel:.3f} m/s")
    print(f"  Maximum velocity command: {max_vel:.3f} m/s")

    if avg_vel < 0.1:
        print(f"  ⚠ PROBLEM: Average velocity is very low!")
        print(f"     → PID gains are too small OR velocity limits are too restrictive")
        print(f"     → Current: kp={config.pid_forward_kp}, max_vel={config.max_forward_velocity}")
        print(f"     → Try: kp=0.8, max_vel=0.8")

    if max_vel < config.max_forward_velocity * 0.3:
        print(f"  ⚠ PROBLEM: Never using more than 30% of max velocity")
        print(f"     → PID gains are too conservative")

    # Check relationship between distance and velocity
    far_results = [r for r in results if r['distance'] > 1.0]
    if far_results:
        far_avg_vel = np.mean([r['vel_magnitude'] for r in far_results])
        print(f"\n  When distance > 1.0m:")
        print(f"    Average velocity: {far_avg_vel:.3f} m/s")
        if far_avg_vel < 0.2:
            print(f"    ⚠ This is too slow! Drone will barely move.")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    separations = [r['separation'] for r in results]
    distances = [r['distance'] for r in results]
    velocities = [r['vel_magnitude'] for r in results]
    forward_cmds = [r['cmd_forward'] for r in results]
    lateral_cmds = [r['cmd_lateral'] for r in results]

    # Plot 1: Distance vs Separation
    axes[0, 0].plot(separations, distances, 'b-o', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Frame Separation', fontsize=12)
    axes[0, 0].set_ylabel('Distance (m)', fontsize=12)
    axes[0, 0].set_title('Distance vs Frame Separation', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Velocity vs Separation
    axes[0, 1].plot(separations, velocities, 'r-o', linewidth=2, markersize=5)
    axes[0, 1].axhline(y=config.max_total_velocity, color='orange', linestyle='--',
                      linewidth=2, label=f'Max limit ({config.max_total_velocity} m/s)')
    axes[0, 1].set_xlabel('Frame Separation', fontsize=12)
    axes[0, 1].set_ylabel('Velocity Command (m/s)', fontsize=12)
    axes[0, 1].set_title('Velocity Command vs Frame Separation', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Velocity vs Distance
    axes[1, 0].scatter(distances, velocities, c=separations, cmap='viridis', s=100, alpha=0.7)
    axes[1, 0].plot(distances, velocities, 'k--', alpha=0.3)
    axes[1, 0].set_xlabel('Distance (m)', fontsize=12)
    axes[1, 0].set_ylabel('Velocity Command (m/s)', fontsize=12)
    axes[1, 0].set_title('Velocity vs Distance (color = separation)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Forward vs Lateral commands
    axes[1, 1].scatter(forward_cmds, lateral_cmds, c=distances, cmap='plasma', s=100, alpha=0.7)
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 1].axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    axes[1, 1].set_xlabel('Forward Command (m/s)', fontsize=12)
    axes[1, 1].set_ylabel('Lateral Command (m/s)', fontsize=12)
    axes[1, 1].set_title('Command Direction (color = distance)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Frame Separation')
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Distance (m)')

    plt.tight_layout()
    plt.savefig('frame_separation_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("\n✓ Visualization saved to: frame_separation_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
