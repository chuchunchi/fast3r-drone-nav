#!/usr/bin/env python3
"""
Simulate full homing sequence with recorded images.

This processes a complete teach sequence in reverse (simulating homing)
and shows how pose estimates and commands evolve over the trajectory.

Usage:
    python simulate_full_homing.py --folder path/to/recorded_sequence --output homing_analysis.png
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

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


def simulate_homing_sequence(image_paths, config, metric_scale=1.0):
    """
    Simulate homing through a sequence of images.

    Returns:
        List of dicts with pose estimates and commands for each step.
    """
    logger.info("Loading Fast3R model...")
    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    pose_estimator = PoseEstimator(
        confidence_percentile=0.5,
        min_points=100,
        metric_scale=metric_scale,
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

    # Load all keyframes
    keyframes = [load_image(str(p)) for p in image_paths]
    num_keyframes = len(keyframes)

    results = []

    logger.info(f"Simulating homing with {num_keyframes} keyframes...")
    logger.info("Strategy: Start from last frame, compare to earlier keyframes (moving backwards)")

    # Start from the last frame (end position)
    # Compare to earlier keyframes going backwards (simulating return journey)
    live_frame_idx = num_keyframes - 1  # Start at the end
    target_idx = num_keyframes - 1  # Target is also the last keyframe initially

    step_idx = 0

    # Process frames moving backwards through the sequence
    while live_frame_idx >= 0 and target_idx >= 0:
        # Live image is current position
        live_img = keyframes[live_frame_idx]

        # Target is the keyframe we're trying to reach (going backwards)
        target_img = keyframes[target_idx]

        logger.info(f"Step {step_idx}: Live frame {live_frame_idx} → Target KF{target_idx}")

        # Run inference
        result = fast3r.infer_pair(live_img, target_img)

        # Estimate pose
        pose_result = pose_estimator.estimate_pose(
            result["pts3d_1"],
            result["pts3d_2"],
            result["conf_1"],
        )

        if not pose_result.success:
            logger.warning(f"  ❌ Pose estimation failed")
            results.append({
                'step': step_idx,
                'target_idx': target_idx,
                'success': False,
                'distance': float('inf'),
            })
            continue

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

        logger.info(f"  Distance: {distance:.3f}m, Conf: {pose_result.confidence:.3f}, "
                   f"Vel: {vel_magnitude:.3f} m/s")

        results.append({
            'step': step_idx,
            'target_idx': target_idx,
            'live_image': live_img,
            'target_image': target_img,
            'success': True,
            'translation': t_cam,
            'rotation': R,
            'yaw_error': error_yaw,
            'distance': distance,
            'confidence': pose_result.confidence,
            'command': command,
            'vel_magnitude': vel_magnitude,
            'error_forward': error_forward,
            'error_lateral': error_lateral,
            'error_vertical': error_vertical,
        })

        # Check if reached waypoint (or if we're at the same frame)
        if distance < config.waypoint_threshold_m or live_frame_idx == target_idx:
            if live_frame_idx != target_idx:
                logger.info(f"  ✓ Reached keyframe {target_idx}")
            target_idx -= 1  # Move to next (earlier) keyframe
            pid.reset_position()

        # Move to previous live frame (one step back in sequence)
        live_frame_idx -= 1
        step_idx += 1

        # Safety: stop if we're going backwards forever
        if step_idx > num_keyframes * 2:
            logger.warning("Too many steps, stopping simulation")
            break

    return results


def visualize_sequence_results(results, output_path):
    """Create comprehensive visualization of homing sequence."""

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Filter successful results
    valid = [r for r in results if r['success']]

    if not valid:
        logger.error("No valid results to visualize!")
        return

    steps = [r['step'] for r in valid]

    # Plot 1: Distance over time
    ax1 = fig.add_subplot(gs[0, :])
    distances = [r['distance'] for r in valid]
    ax1.plot(steps, distances, 'b-o', linewidth=2, markersize=5, label='Distance to target')
    ax1.axhline(y=0.5, color='g', linestyle='--', linewidth=2, label='Waypoint threshold (0.5m)')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Distance (m)', fontsize=12)
    ax1.set_title('Distance to Target Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Velocity commands
    ax2 = fig.add_subplot(gs[1, :])
    forward_vels = [r['command']['pitch_velocity'] for r in valid]
    lateral_vels = [r['command']['roll_velocity'] for r in valid]
    vertical_vels = [r['command']['vertical_velocity'] for r in valid]
    total_vels = [r['vel_magnitude'] for r in valid]

    ax2.plot(steps, forward_vels, 'r-', linewidth=2, label='Forward', alpha=0.7)
    ax2.plot(steps, lateral_vels, 'g-', linewidth=2, label='Lateral', alpha=0.7)
    ax2.plot(steps, vertical_vels, 'b-', linewidth=2, label='Vertical', alpha=0.7)
    ax2.plot(steps, total_vels, 'k-', linewidth=3, label='Total magnitude', alpha=0.9)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Control Commands Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Translation errors
    ax3 = fig.add_subplot(gs[2, 0])
    forward_errors = [r['error_forward'] for r in valid]
    ax3.plot(steps, forward_errors, 'r-o', linewidth=2, markersize=4)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_ylabel('Error (m)', fontsize=10)
    ax3.set_title('Forward Error (Z)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    lateral_errors = [r['error_lateral'] for r in valid]
    ax4.plot(steps, lateral_errors, 'g-o', linewidth=2, markersize=4)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('Error (m)', fontsize=10)
    ax4.set_title('Lateral Error (X)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 2])
    vertical_errors = [r['error_vertical'] for r in valid]
    ax5.plot(steps, vertical_errors, 'b-o', linewidth=2, markersize=4)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Step', fontsize=10)
    ax5.set_ylabel('Error (m)', fontsize=10)
    ax5.set_title('Vertical Error (-Y)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 4: Confidence
    ax6 = fig.add_subplot(gs[3, 0])
    confidences = [r['confidence'] for r in valid]
    ax6.plot(steps, confidences, 'purple', linewidth=2, marker='o', markersize=4)
    ax6.axhline(y=0.3, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Min threshold')
    ax6.set_xlabel('Step', fontsize=10)
    ax6.set_ylabel('Confidence', fontsize=10)
    ax6.set_title('Pose Confidence', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 1.1])
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Plot 5: Trajectory (bird's eye view)
    ax7 = fig.add_subplot(gs[3, 1:])

    # Accumulate positions (approximate)
    positions = [(0, 0)]  # Start at origin
    for r in valid:
        prev = positions[-1]
        # Move by velocity command (approximate)
        dx = r['command']['roll_velocity'] * 0.1  # Assume 0.1s timestep
        dy = r['command']['pitch_velocity'] * 0.1
        positions.append((prev[0] + dx, prev[1] + dy))

    xs, ys = zip(*positions)
    ax7.plot(ys, xs, 'b-o', linewidth=2, markersize=5, alpha=0.7, label='Trajectory')
    ax7.plot(ys[0], xs[0], 'go', markersize=15, label='Start')
    ax7.plot(ys[-1], xs[-1], 'ro', markersize=15, label='End')

    ax7.set_xlabel('Forward (m)', fontsize=10)
    ax7.set_ylabel('Lateral (m)', fontsize=10)
    ax7.set_title('Approximate Trajectory (Bird\'s Eye View)', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')

    plt.suptitle('Homing Sequence Analysis', fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Visualization saved to: {output_path}")


def print_summary(results):
    """Print summary statistics."""
    valid = [r for r in results if r['success']]

    print("\n" + "="*60)
    print("HOMING SEQUENCE SUMMARY")
    print("="*60)

    print(f"Total steps: {len(results)}")
    print(f"Successful: {len(valid)}/{len(results)}")

    if valid:
        distances = [r['distance'] for r in valid]
        vels = [r['vel_magnitude'] for r in valid]
        confs = [r['confidence'] for r in valid]

        print(f"\nDistance:")
        print(f"  Initial: {distances[0]:.3f}m")
        print(f"  Final: {distances[-1]:.3f}m")
        print(f"  Average: {np.mean(distances):.3f}m")

        print(f"\nVelocity commands:")
        print(f"  Average: {np.mean(vels):.3f} m/s")
        print(f"  Max: {np.max(vels):.3f} m/s")
        print(f"  Min: {np.min(vels):.3f} m/s")

        print(f"\nConfidence:")
        print(f"  Average: {np.mean(confs):.3f}")
        print(f"  Min: {np.min(confs):.3f}")

        # Identify potential issues
        print(f"\n⚠ Potential Issues:")
        low_vel_count = sum(1 for v in vels if v < 0.05)
        if low_vel_count > len(vels) * 0.5:
            print(f"  ⚠ {low_vel_count}/{len(vels)} steps have very low velocity (< 0.05 m/s)")
            print(f"     → PID gains might be too small!")

        large_dist_low_vel = sum(1 for r in valid if r['distance'] > 1.0 and r['vel_magnitude'] < 0.1)
        if large_dist_low_vel > 0:
            print(f"  ⚠ {large_dist_low_vel} steps have large distance (>1m) but low velocity (<0.1 m/s)")
            print(f"     → PID is not responding aggressively enough!")

        if not any(r['distance'] < 0.5 for r in valid):
            print(f"  ⚠ Never got within waypoint threshold (0.5m)")
            print(f"     → May never reach target!")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Simulate full homing sequence")
    parser.add_argument("--folder", required=True, help="Folder with recorded image sequence")
    parser.add_argument("--metric-scale", type=float, default=1.0, help="Metric scale factor")
    parser.add_argument("--output", default="homing_analysis.png", help="Output visualization")

    args = parser.parse_args()

    # Find images
    folder = Path(args.folder)
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    image_paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

    if len(image_paths) < 2:
        logger.error(f"Need at least 2 images, found {len(image_paths)}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images in {folder}")

    # Load config
    config = Config()
    logger.info(f"Config: kp={config.pid_forward_kp}, max_vel={config.max_forward_velocity} m/s")

    # Simulate homing
    results = simulate_homing_sequence(image_paths, config, args.metric_scale)

    # Print summary
    print_summary(results)

    # Visualize
    visualize_sequence_results(results, args.output)

    print(f"\n✓ Analysis complete! Check {args.output}")


if __name__ == "__main__":
    main()
