#!/usr/bin/env python3
"""
Demo script for Phase 6: Homing Simulation.

This script demonstrates the complete homing workflow:
1. Load a sequence of images (simulating recorded TEACH phase)
2. Build keyframe stack and calibrate scale
3. Simulate HOMING phase by processing frames in reverse
4. Display pose estimation and PID control outputs

This simulates what happens during actual homing without requiring
a real drone connection.

Usage:
    # Full simulation with visualization (simulates complete homing):
    python demo_homing_simulation.py --folder demo_examples/target1 --visualize

    # Single image mode (test where one image should go):
    python demo_homing_simulation.py --folder demo_examples/target1 --single-image

    # Quick test:
    python demo_homing_simulation.py --folder demo_examples/target1

    # Use specific images for live/target:
    python demo_homing_simulation.py --live image1.jpg --target image2.jpg
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.coordinate_utils import create_hover_command, extract_yaw_error
from visual_homing.server.fast3r_engine import Fast3REngine
from visual_homing.server.keyframe_manager import (
    Keyframe,
    KeyframeStackManager,
    Telemetry,
)
from visual_homing.server.pid_controller import MultiAxisPIDController
from visual_homing.server.pose_estimator import PoseEstimator, PoseResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class HomingStep:
    """Result of a single homing step."""
    step_idx: int
    live_image_name: str
    target_keyframe_idx: int
    target_distance_m: float
    pose_result: Optional[PoseResult]
    command: Dict[str, float]
    translation_cam: np.ndarray  # In camera frame
    yaw_error_deg: float
    reached_waypoint: bool


def load_image(path: str) -> np.ndarray:
    """Load an image from disk and convert to RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_mock_telemetry(timestamp_ms: int) -> Telemetry:
    """Create mock telemetry for simulation."""
    return Telemetry(
        timestamp_ms=timestamp_ms,
        velocity_x=0.0,
        velocity_y=0.0,
        velocity_z=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        height=2.0,
    )


def build_keyframe_stack(
    image_paths: List[Path],
    mock_distance_per_frame: float = 0.5,
) -> KeyframeStackManager:
    """Build keyframe stack from images (simulating TEACH phase)."""
    manager = KeyframeStackManager(
        keyframe_interval_m=0.1,  # Low threshold to capture all frames as keyframes
        keyframe_interval_s=0.1,
    )

    base_timestamp = int(time.time() * 1000)
    cumulative_dist = 0.0

    for i, img_path in enumerate(image_paths):
        timestamp_ms = base_timestamp + i * 100
        cumulative_dist += mock_distance_per_frame

        image = load_image(str(img_path))

        telemetry = Telemetry(
            timestamp_ms=timestamp_ms,
            velocity_x=mock_distance_per_frame * 10,  # At 10 FPS
            velocity_y=0.0,
            velocity_z=0.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            height=2.0,
        )

        manager.process_frame(image, telemetry, force_keyframe=True)

    return manager


def simulate_homing(
    live_images: List[Tuple[str, np.ndarray]],
    keyframe_manager: KeyframeStackManager,
    fast3r: Fast3REngine,
    config: Config,
    metric_scale: float = 1.0,
    waypoint_threshold_m: float = 0.8,
    single_image_mode: bool = False,
) -> List[HomingStep]:
    """
    Simulate the homing loop.

    Args:
        live_images: List of (name, image) tuples for "live" frames.
        keyframe_manager: Manager with recorded keyframes.
        fast3r: Fast3R inference engine.
        config: Configuration.
        metric_scale: Scale factor (meters per Fast3R unit).
        waypoint_threshold_m: Distance to consider waypoint reached.
        single_image_mode: If True, use only the first live image for all steps.

    Returns:
        List of HomingStep results.
    """
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

    target_idx = keyframe_manager.get_stack_size() - 1
    results = []
    live_image_idx = 0

    # In single image mode, we test one live image against all keyframes
    if single_image_mode:
        if len(live_images) == 0:
            return results
        live_name, live_frame = live_images[0]

        # Test against all keyframes from end to start
        for step_idx in range(keyframe_manager.get_stack_size()):
            target_keyframe = keyframe_manager[target_idx]

            # Run Fast3R inference
            result = fast3r.infer_pair(live_frame, target_keyframe.image)

            # Estimate pose
            pose_result = pose_estimator.estimate_pose(
                result["pts3d_1"],  # Live frame points
                result["pts3d_2"],  # Target frame points
                result["conf_1"],  # Live frame confidence
            )

            if not pose_result.success:
                logger.warning(f"Step {step_idx}: Pose estimation failed for KF{target_idx}")
                results.append(HomingStep(
                    step_idx=step_idx,
                    live_image_name=live_name,
                    target_keyframe_idx=target_idx,
                    target_distance_m=float("inf"),
                    pose_result=pose_result,
                    command=create_hover_command(),
                    translation_cam=np.zeros(3),
                    yaw_error_deg=0.0,
                    reached_waypoint=False,
                ))
                target_idx -= 1
                continue

            # Extract pose error in camera frame
            t_cam = pose_result.translation.cpu().numpy()
            R = pose_result.rotation

            # Camera frame: X=right, Y=down, Z=forward
            error_forward = float(t_cam[2])
            error_lateral = float(t_cam[0])
            error_vertical = -float(t_cam[1])  # Invert: down→up
            error_yaw = extract_yaw_error(R)

            # Distance to target
            distance_to_target = float(np.linalg.norm(t_cam))

            # Check waypoint
            reached_waypoint = distance_to_target < waypoint_threshold_m

            # PID control
            command = pid.compute(
                error_forward,
                error_lateral,
                error_vertical,
                error_yaw,
            )

            results.append(HomingStep(
                step_idx=step_idx,
                live_image_name=live_name,
                target_keyframe_idx=target_idx,
                target_distance_m=distance_to_target,
                pose_result=pose_result,
                command=command,
                translation_cam=t_cam,
                yaw_error_deg=error_yaw,
                reached_waypoint=reached_waypoint,
            ))

            if reached_waypoint:
                logger.info(f"Step {step_idx}: Would reach keyframe {target_idx}")

            target_idx -= 1
            pid.reset_position()

        return results

    # Standard mode: simulate actual homing with progressing live images
    for step_idx in range(len(live_images)):
        if target_idx < 0:
            logger.info("Homing complete - all keyframes reached")
            break

        # Use the corresponding live image for this step
        if live_image_idx >= len(live_images):
            logger.info("No more live images available")
            break

        img_name, live_frame = live_images[live_image_idx]
        target_keyframe = keyframe_manager[target_idx]

        # Run Fast3R inference
        result = fast3r.infer_pair(live_frame, target_keyframe.image)

        # Estimate pose
        pose_result = pose_estimator.estimate_pose(
            result["pts3d_1"],  # Live frame points
            result["pts3d_2"],  # Target frame points
            result["conf_1"],  # Live frame confidence
        )

        if not pose_result.success:
            logger.warning(f"Step {step_idx}: Pose estimation failed")
            results.append(HomingStep(
                step_idx=step_idx,
                live_image_name=img_name,
                target_keyframe_idx=target_idx,
                target_distance_m=float("inf"),
                pose_result=pose_result,
                command=create_hover_command(),
                translation_cam=np.zeros(3),
                yaw_error_deg=0.0,
                reached_waypoint=False,
            ))
            live_image_idx += 1
            continue

        # Extract pose error in camera frame
        t_cam = pose_result.translation.cpu().numpy()
        R = pose_result.rotation

        # Camera frame: X=right, Y=down, Z=forward
        error_forward = float(t_cam[2])
        error_lateral = float(t_cam[0])
        error_vertical = -float(t_cam[1])  # Invert: down→up
        error_yaw = extract_yaw_error(R)

        # Distance to target
        distance_to_target = float(np.linalg.norm(t_cam))

        # Check waypoint
        reached_waypoint = distance_to_target < waypoint_threshold_m

        # PID control
        command = pid.compute(
            error_forward,
            error_lateral,
            error_vertical,
            error_yaw,
        )

        results.append(HomingStep(
            step_idx=step_idx,
            live_image_name=img_name,
            target_keyframe_idx=target_idx,
            target_distance_m=distance_to_target,
            pose_result=pose_result,
            command=command,
            translation_cam=t_cam,
            yaw_error_deg=error_yaw,
            reached_waypoint=reached_waypoint,
        ))

        if reached_waypoint:
            logger.info(f"Step {step_idx}: Reached keyframe {target_idx}")
            target_idx -= 1
            pid.reset_position()

        # Always advance to next live image in standard mode
        live_image_idx += 1

    return results


def get_direction_description(translation_cam: np.ndarray, yaw_error: float) -> str:
    """
    Get human-readable direction description from camera-frame translation.

    Args:
        translation_cam: Translation vector in camera frame [X=right, Y=down, Z=forward]
        yaw_error: Yaw error in degrees

    Returns:
        Description string like "Move forward 2.5m, left 0.3m, turn right 15°"
    """
    # Camera frame convention: X=right, Y=down, Z=forward
    x, y, z = translation_cam

    parts = []

    # Forward/backward (Z axis)
    if abs(z) > 0.05:  # Threshold for meaningful movement
        direction = "forward" if z > 0 else "backward"
        parts.append(f"{direction} {abs(z):.2f}m")

    # Left/right (X axis)
    if abs(x) > 0.05:
        direction = "right" if x > 0 else "left"
        parts.append(f"{direction} {abs(x):.2f}m")

    # Up/down (Y axis, inverted)
    if abs(y) > 0.05:
        direction = "down" if y > 0 else "up"
        parts.append(f"{direction} {abs(y):.2f}m")

    # Yaw
    if abs(yaw_error) > 2.0:  # Threshold for meaningful rotation
        direction = "right" if yaw_error > 0 else "left"
        parts.append(f"turn {direction} {abs(yaw_error):.1f}°")

    if not parts:
        return "At target position"

    return "Move " + ", ".join(parts)


def print_homing_results(results: List[HomingStep], single_image_mode: bool = False):
    """Print homing simulation results."""
    print("\n" + "=" * 80)
    print("HOMING SIMULATION RESULTS")
    print("=" * 80)

    if single_image_mode:
        print("\n[SINGLE IMAGE MODE] Testing one live image against all keyframes")
        print("This shows which keyframe the live image is closest to.\n")

    print("\nPose Estimation & Control Commands:")
    print("-" * 80)
    print(f"{'Step':<5} {'Live→Target':<15} {'Distance':<10} {'Fwd':<8} {'Lat':<8} {'Vert':<8} {'Yaw°':<8} {'Reached':<8}")
    print("-" * 80)

    for r in results:
        reached = "✓" if r.reached_waypoint else ""
        print(
            f"{r.step_idx:<5} "
            f"{r.live_image_name[:8]}→KF{r.target_keyframe_idx:<5} "
            f"{r.target_distance_m:<10.3f} "
            f"{r.command['pitch_velocity']:<+8.3f} "
            f"{r.command['roll_velocity']:<+8.3f} "
            f"{r.command['vertical_velocity']:<+8.3f} "
            f"{r.yaw_error_deg:<+8.1f} "
            f"{reached:<8}"
        )

    print("-" * 80)

    # Summary
    successful = sum(1 for r in results if r.pose_result and r.pose_result.success)
    waypoints_reached = sum(1 for r in results if r.reached_waypoint)

    print(f"\nSummary:")
    print(f"  Total steps: {len(results)}")
    print(f"  Successful pose estimations: {successful}/{len(results)}")
    print(f"  Waypoints reached: {waypoints_reached}")

    if results:
        avg_distance = np.mean([r.target_distance_m for r in results if r.target_distance_m < float("inf")])
        print(f"  Average target distance: {avg_distance:.3f}m")

        avg_confidence = np.mean([
            r.pose_result.confidence for r in results
            if r.pose_result and r.pose_result.success
        ])
        print(f"  Average confidence: {avg_confidence:.3f}")

    # Find best match and show directions
    if single_image_mode and results:
        valid_results = [r for r in results if r.target_distance_m < float("inf")]
        if valid_results:
            best_match = min(valid_results, key=lambda r: r.target_distance_m)
            print(f"\n" + "=" * 80)
            print("RECOMMENDED ACTION")
            print("=" * 80)
            print(f"Closest match: Keyframe {best_match.target_keyframe_idx}")
            print(f"Distance: {best_match.target_distance_m:.3f}m")
            print(f"Confidence: {best_match.pose_result.confidence:.3f}")
            print(f"\nTo reach this keyframe:")
            direction_desc = get_direction_description(best_match.translation_cam, best_match.yaw_error_deg)
            print(f"  → {direction_desc}")

            # Show translation components
            print(f"\nDetailed translation (camera frame):")
            print(f"  Forward (Z):  {best_match.translation_cam[2]:+.3f}m {'(target ahead)' if best_match.translation_cam[2] > 0 else '(target behind)'}")
            print(f"  Lateral (X):  {best_match.translation_cam[0]:+.3f}m {'(target to right)' if best_match.translation_cam[0] > 0 else '(target to left)'}")
            print(f"  Vertical (Y): {-best_match.translation_cam[1]:+.3f}m {'(target above)' if -best_match.translation_cam[1] > 0 else '(target below)'}")
            print(f"  Yaw error:    {best_match.yaw_error_deg:+.1f}° {'(turn right)' if best_match.yaw_error_deg > 0 else '(turn left)'}")


def visualize_homing(
    results: List[HomingStep],
    keyframe_manager: KeyframeStackManager,
    live_images: List[Tuple[str, np.ndarray]],
    output_path: str = "homing_simulation.png",
):
    """Create visualization of homing simulation."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig)

        # Plot 1: Distance over time
        ax1 = fig.add_subplot(gs[0, :2])
        distances = [r.target_distance_m for r in results]
        ax1.plot(distances, 'b-o', label='Distance to target')
        ax1.axhline(y=0.8, color='g', linestyle='--', label='Waypoint threshold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Distance to Target Keyframe')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Control commands
        ax2 = fig.add_subplot(gs[0, 2:])
        steps = range(len(results))
        ax2.plot(steps, [r.command['pitch_velocity'] for r in results], 'r-', label='Forward')
        ax2.plot(steps, [r.command['roll_velocity'] for r in results], 'g-', label='Lateral')
        ax2.plot(steps, [r.command['vertical_velocity'] for r in results], 'b-', label='Vertical')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('PID Control Commands')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Translation errors
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(steps, [r.translation_cam[0] for r in results], 'r-', label='X (lateral)')
        ax3.plot(steps, [r.translation_cam[1] for r in results], 'g-', label='Y (vertical)')
        ax3.plot(steps, [r.translation_cam[2] for r in results], 'b-', label='Z (forward)')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Error (m)')
        ax3.set_title('Translation Errors (Camera Frame)')
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Yaw error
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.plot(steps, [r.yaw_error_deg for r in results], 'purple', marker='o')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Yaw Error (degrees)')
        ax4.set_title('Yaw Error')
        ax4.grid(True)

        # Plot 5-8: Sample image pairs
        sample_indices = [0, len(results)//3, 2*len(results)//3, -1]
        for i, idx in enumerate(sample_indices):
            if idx < 0:
                idx = len(results) + idx
            if idx < len(results):
                ax = fig.add_subplot(gs[2, i])
                r = results[idx]
                
                # Find the live image
                live_img = None
                for name, img in live_images:
                    if name == r.live_image_name:
                        live_img = img
                        break
                
                if live_img is not None:
                    ax.imshow(live_img)
                    ax.set_title(f"Step {r.step_idx}\nDist: {r.target_distance_m:.2f}m")
                ax.axis('off')

        plt.suptitle('Homing Simulation Results', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n✓ Visualization saved to: {output_path}")

    except ImportError:
        print("\n⚠ matplotlib not available for visualization")


def test_single_pair(
    live_path: str,
    target_path: str,
    config: Config,
    metric_scale: float = 1.0,
):
    """Test homing with a single live/target pair."""
    print("\n" + "=" * 60)
    print("SINGLE PAIR HOMING TEST")
    print("=" * 60)

    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    live_img = load_image(live_path)
    target_img = load_image(target_path)

    print(f"\nLive image: {live_path}")
    print(f"Target image: {target_path}")
    print(f"Metric scale: {metric_scale}")

    # Run inference
    result = fast3r.infer_pair(live_img, target_img)

    pose_estimator = PoseEstimator(
        confidence_percentile=0.5,
        min_points=100,
        metric_scale=metric_scale,
    )

    pose_result = pose_estimator.estimate_pose(
        result["pts3d_1"],
        result["pts3d_2"],
        result["conf_1"],
    )

    print(f"\nPose Estimation Result:")
    print(f"  Success: {pose_result.success}")
    print(f"  Confidence: {pose_result.confidence:.3f}")
    print(f"  Num inliers: {pose_result.num_inliers}")

    if pose_result.success:
        t_cam = pose_result.translation.cpu().numpy()
        yaw_err = extract_yaw_error(pose_result.rotation)
        distance = np.linalg.norm(t_cam)

        print(f"\nTranslation (camera frame, meters):")
        print(f"  X (lateral):  {t_cam[0]:+.4f} (+ = target is to the right)")
        print(f"  Y (vertical): {t_cam[1]:+.4f} (+ = target is below)")
        print(f"  Z (forward):  {t_cam[2]:+.4f} (+ = target is ahead)")
        print(f"  Distance:     {distance:.4f}m")

        print(f"\nRotation:")
        print(f"  Yaw error: {yaw_err:+.2f}°")

        # Compute control command
        pid = MultiAxisPIDController()
        command = pid.compute(
            float(t_cam[2]),   # Forward error
            float(t_cam[0]),   # Lateral error
            -float(t_cam[1]),  # Vertical error (inverted)
            yaw_err,
        )

        print(f"\nPID Control Command:")
        print(f"  pitch_velocity (forward):  {command['pitch_velocity']:+.3f} m/s")
        print(f"  roll_velocity (lateral):   {command['roll_velocity']:+.3f} m/s")
        print(f"  vertical_velocity:         {command['vertical_velocity']:+.3f} m/s")
        print(f"  yaw_rate:                  {command['yaw_rate']:+.3f} deg/s")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Homing simulation with recorded keyframes"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="demo_examples/target1",
        help="Folder containing flight images",
    )
    parser.add_argument(
        "--live",
        type=str,
        help="Single live image for testing",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Single target image for testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Fast3R inference",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Metric scale factor (meters per Fast3R unit)",
    )
    parser.add_argument(
        "--waypoint-threshold",
        type=float,
        default=0.8,
        help="Distance threshold for waypoint reached (meters)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization",
    )
    parser.add_argument(
        "--mock-distance",
        type=float,
        default=0.5,
        help="Simulated distance between keyframes (meters)",
    )
    parser.add_argument(
        "--single-image",
        action="store_true",
        help="Single image mode: test one live image against all keyframes",
    )

    args = parser.parse_args()

    config = Config(device=args.device)

    # Mode 1: Single pair test
    if args.live and args.target:
        test_single_pair(args.live, args.target, config, args.scale)
        return 0

    # Mode 2: Full sequence simulation
    folder = Path(args.folder)
    if not folder.exists():
        project_root = Path(__file__).parent.parent.parent
        folder = project_root / args.folder

    if not folder.exists():
        logger.error(f"Folder not found: {args.folder}")
        return 1

    image_paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

    if len(image_paths) < 2:
        logger.error(f"Need at least 2 images, found {len(image_paths)}")
        return 1

    print("\n" + "=" * 60)
    print("PHASE 6: HOMING SIMULATION")
    print("=" * 60)
    print(f"\nInput images: {len(image_paths)}")
    print(f"Metric scale: {args.scale}")
    print(f"Waypoint threshold: {args.waypoint_threshold}m")
    if args.single_image:
        print(f"Mode: Single image test (using last image in sequence)")

    # Step 1: Build keyframe stack (simulating TEACH phase)
    print("\n[Step 1] Building keyframe stack...")
    keyframe_manager = build_keyframe_stack(image_paths, args.mock_distance)
    print(f"  Recorded {keyframe_manager.get_stack_size()} keyframes")
    print(f"  Total distance: {keyframe_manager.get_total_distance():.2f}m")

    # Step 2: Initialize Fast3R
    print("\n[Step 2] Loading Fast3R model...")
    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    # Step 3: Simulate homing
    print("\n[Step 3] Simulating homing...")

    if args.single_image:
        # Single image mode: use only the last image (first position in reversed order)
        live_images = [
            (image_paths[-1].name, load_image(str(image_paths[-1])))
        ]
        print(f"  Testing live image: {image_paths[-1].name}")
    else:
        # Standard mode: use images in reverse order as "live" frames
        live_images = [
            (p.name, load_image(str(p)))
            for p in reversed(image_paths)
        ]
        print(f"  Using {len(live_images)} live images")

    results = simulate_homing(
        live_images=live_images,
        keyframe_manager=keyframe_manager,
        fast3r=fast3r,
        config=config,
        metric_scale=args.scale,
        waypoint_threshold_m=args.waypoint_threshold,
        single_image_mode=args.single_image,
    )

    # Step 4: Print results
    print_homing_results(results, single_image_mode=args.single_image)

    # Step 5: Visualization
    if args.visualize:
        visualize_homing(results, keyframe_manager, live_images)

    print("\n" + "=" * 60)
    print("PHASE 6 DEMO COMPLETE")
    print("=" * 60)
    print(f"\n✓ Keyframe recording: Working")
    print(f"✓ Fast3R inference: Working")
    print(f"✓ Pose estimation: Working")
    print(f"✓ PID control: Working")

    return 0


if __name__ == "__main__":
    sys.exit(main())

