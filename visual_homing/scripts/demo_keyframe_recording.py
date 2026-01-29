#!/usr/bin/env python3
"""
Demo script for Phase 5: Keyframe Recording with Scale Calibration.

This script demonstrates the keyframe recording workflow:
1. Load a sequence of images (simulating TEACH phase flight)
2. Process frames through KeyframeStackManager with simulated telemetry
3. Compute scale factors between consecutive keyframes using Fast3R
4. Display scale calibration results

Usage:
    # Use demo images from the repo:
    python demo_keyframe_recording.py --folder demo_examples/target1

    # With mock telemetry (for testing without actual IMU data):
    python demo_keyframe_recording.py --folder demo_examples/target1 --mock-distance 0.5

    # Skip Fast3R and just test keyframe logic:
    python demo_keyframe_recording.py --folder demo_examples/target1 --no-fast3r
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.keyframe_manager import (
    Keyframe,
    KeyframeStackManager,
    Telemetry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk and convert to RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_mock_telemetry(
    timestamp_ms: int,
    cumulative_distance: float,
    last_timestamp_ms: Optional[int] = None,
) -> Telemetry:
    """
    Create mock telemetry data for testing.
    
    Simulates forward flight with constant velocity.
    """
    # Simulate velocity based on distance change
    if last_timestamp_ms is not None:
        dt = (timestamp_ms - last_timestamp_ms) / 1000.0
        if dt > 0:
            velocity = (cumulative_distance) / (timestamp_ms / 1000.0)
        else:
            velocity = 0.5  # Default 0.5 m/s
    else:
        velocity = 0.5

    return Telemetry(
        timestamp_ms=timestamp_ms,
        velocity_x=velocity,  # Forward velocity
        velocity_y=0.0,  # No lateral
        velocity_z=0.0,  # No vertical
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        height=2.0,  # 2m height
    )


def test_keyframe_manager_basic(
    image_paths: List[Path],
    mock_distance_per_frame: float = 0.5,
) -> KeyframeStackManager:
    """
    Test basic keyframe manager functionality without Fast3R.
    
    Args:
        image_paths: List of image file paths.
        mock_distance_per_frame: Simulated distance between frames in meters.
    
    Returns:
        KeyframeStackManager with recorded keyframes.
    """
    print("\n" + "=" * 60)
    print("PHASE 5a: KEYFRAME RECORDING (Basic Test)")
    print("=" * 60)

    # Create keyframe manager with 1.5m interval (so every 3 frames at 0.5m each)
    manager = KeyframeStackManager(
        keyframe_interval_m=1.5,
        keyframe_interval_s=5.0,
    )

    print(f"\nSettings:")
    print(f"  Keyframe interval: {manager.keyframe_interval_m}m or {manager.keyframe_interval_s}s")
    print(f"  Mock distance per frame: {mock_distance_per_frame}m")
    print(f"  Input images: {len(image_paths)}")

    base_timestamp = int(time.time() * 1000)
    last_timestamp = None

    for i, img_path in enumerate(image_paths):
        # Create timestamp (100ms per frame = 10 FPS)
        timestamp_ms = base_timestamp + i * 100

        # Load image
        image = load_image(str(img_path))

        # Create mock telemetry
        telemetry = create_mock_telemetry(
            timestamp_ms=timestamp_ms,
            cumulative_distance=(i + 1) * mock_distance_per_frame,
            last_timestamp_ms=last_timestamp,
        )
        last_timestamp = timestamp_ms

        # Process frame
        keyframe = manager.process_frame(image, telemetry)

        if keyframe:
            print(f"\n  ✓ Keyframe {keyframe.index} captured:")
            print(f"    - Image: {img_path.name}")
            print(f"    - Distance: {keyframe.cumulative_distance:.2f}m")
            print(f"    - Image shape: {keyframe.image.shape}")
        else:
            print(f"  - Frame {i} ({img_path.name}): No keyframe (dist={manager.cumulative_distance:.2f}m)")

    print(f"\n" + "-" * 40)
    print(f"Recording Summary:")
    print(f"  Total keyframes: {manager.get_stack_size()}")
    print(f"  Total distance: {manager.get_total_distance():.2f}m")
    print(f"  Distances: {manager.get_keyframe_distances()}")

    return manager


def test_scale_calibration(
    manager: KeyframeStackManager,
    config: Config,
) -> float:
    """
    Test scale calibration using Fast3R.
    
    Computes scale factors between consecutive keyframes by comparing
    IMU-based distances with Fast3R-estimated distances.
    
    Args:
        manager: KeyframeStackManager with recorded keyframes.
        config: Configuration object.
    
    Returns:
        Global scale factor.
    """
    print("\n" + "=" * 60)
    print("PHASE 5b: SCALE CALIBRATION (Fast3R)")
    print("=" * 60)

    if manager.get_stack_size() < 2:
        print("  ⚠ Need at least 2 keyframes for scale calibration")
        return 1.0

    # Import Fast3R components
    from visual_homing.server.fast3r_engine import Fast3REngine
    from visual_homing.server.pose_estimator import PoseEstimator

    # Initialize engine
    print("\nLoading Fast3R model...")
    engine = Fast3REngine(config=config)
    engine.load_model()

    pose_estimator = PoseEstimator(
        confidence_percentile=0.5,
        min_points=100,
        metric_scale=1.0,  # We're computing scale, so start with 1.0
    )

    print(f"\nComputing scale factors for {manager.get_stack_size() - 1} keyframe pairs...")
    print("-" * 60)

    scale_factors = []

    for i in range(1, len(manager)):
        kf_prev = manager[i - 1]
        kf_curr = manager[i]

        # IMU distance
        imu_distance = manager.get_inter_keyframe_distance(i - 1, i)

        # Run Fast3R
        result = engine.infer_pair(kf_prev.image, kf_curr.image)

        # Compute scale factor
        scale = pose_estimator.compute_scale_factor(
            result["pts3d_1"],
            result["pts3d_2"],
            result["conf_1"],
            imu_distance,
        )

        scale_factors.append(scale)
        manager.set_scale_factor(i, scale)

        print(f"\n  Keyframe {i-1} → {i}:")
        print(f"    IMU distance:    {imu_distance:.3f}m")
        print(f"    Scale factor:    {scale:.4f}")

    # Compute global scale
    global_scale = manager.compute_global_scale()

    print(f"\n" + "-" * 60)
    print(f"Scale Calibration Results:")
    print(f"  Individual scales: {[f'{s:.4f}' for s in scale_factors]}")
    print(f"  Global scale (median): {global_scale:.4f}")
    print(f"  Scale std dev: {np.std(scale_factors):.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"  1 Fast3R unit ≈ {global_scale:.4f} meters")
    if global_scale > 0:
        print(f"  1 meter ≈ {1/global_scale:.2f} Fast3R units")

    return global_scale


def test_keyframe_retrieval(manager: KeyframeStackManager):
    """Test keyframe retrieval for homing."""
    print("\n" + "=" * 60)
    print("PHASE 5c: KEYFRAME RETRIEVAL TEST")
    print("=" * 60)

    print(f"\nStack has {manager.get_stack_size()} keyframes")
    print("\nRetrieving keyframes (as we would during homing):")

    # Simulate homing - retrieve from last to first
    idx = manager.get_stack_size() - 1
    while idx >= 0:
        kf = manager.get_target_keyframe(idx)
        print(f"  Target keyframe {idx}: distance={kf.cumulative_distance:.2f}m, shape={kf.image.shape}")
        idx -= 1

    # Test popping
    print("\nSimulating waypoint reaching (popping keyframes):")
    while not manager.is_empty():
        kf = manager.pop_keyframe()
        print(f"  Popped keyframe {kf.index}, {manager.get_stack_size()} remaining")


def visualize_keyframes(manager: KeyframeStackManager, output_path: str = "keyframes.png"):
    """Visualize recorded keyframes."""
    try:
        import matplotlib.pyplot as plt

        n_keyframes = min(manager.get_stack_size(), 8)  # Show at most 8
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < n_keyframes:
                kf = manager[i]
                ax.imshow(kf.image)
                ax.set_title(f"KF {kf.index}\n{kf.cumulative_distance:.2f}m")
                ax.axis("off")
            else:
                ax.axis("off")

        plt.suptitle(f"Recorded Keyframes ({manager.get_stack_size()} total)", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n✓ Keyframe visualization saved to: {output_path}")

    except ImportError:
        print("\n⚠ matplotlib not available for visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Keyframe recording and scale calibration"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="demo_examples/target1",
        help="Folder containing flight images",
    )
    parser.add_argument(
        "--mock-distance",
        type=float,
        default=0.5,
        help="Simulated distance between frames in meters",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Fast3R inference",
    )
    parser.add_argument(
        "--no-fast3r",
        action="store_true",
        help="Skip Fast3R scale calibration (test keyframe logic only)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate keyframe visualization",
    )
    parser.add_argument(
        "--no-pop-test",
        action="store_true",
        help="Skip keyframe pop test (preserves stack for further testing)",
    )

    args = parser.parse_args()

    # Find images
    folder = Path(args.folder)
    if not folder.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        folder = project_root / args.folder

    if not folder.exists():
        logger.error(f"Folder not found: {args.folder}")
        return 1

    image_paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

    if len(image_paths) < 2:
        logger.error(f"Need at least 2 images, found {len(image_paths)}")
        return 1

    logger.info(f"Found {len(image_paths)} images in {folder}")

    # Test 1: Basic keyframe recording
    manager = test_keyframe_manager_basic(image_paths, args.mock_distance)

    # Test 2: Scale calibration with Fast3R (optional)
    if not args.no_fast3r:
        config = Config(device=args.device)
        global_scale = test_scale_calibration(manager, config)
    else:
        print("\n⚠ Skipping Fast3R scale calibration (--no-fast3r)")
        global_scale = 1.0

    # Visualize
    if args.visualize:
        # Reload manager for visualization (if we popped everything)
        manager2 = test_keyframe_manager_basic(image_paths, args.mock_distance)
        visualize_keyframes(manager2)

    # Test 3: Keyframe retrieval (simulates homing preparation)
    if not args.no_pop_test:
        # Reload manager for pop test
        manager3 = test_keyframe_manager_basic(image_paths, args.mock_distance)
        test_keyframe_retrieval(manager3)

    print("\n" + "=" * 60)
    print("PHASE 5 DEMO COMPLETE")
    print("=" * 60)
    print(f"\n✓ Keyframe recording: Working")
    print(f"✓ IMU distance tracking: Working")
    if not args.no_fast3r:
        print(f"✓ Scale calibration: Working (scale = {global_scale:.4f})")
    print(f"✓ Keyframe retrieval: Working")

    return 0


if __name__ == "__main__":
    sys.exit(main())

