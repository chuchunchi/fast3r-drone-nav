#!/usr/bin/env python3
"""
Test pairwise keyframe directions.

This simpler script tests consecutive keyframes to understand the
"forward direction" of your recorded trajectory.

Usage:
    python test_pairwise_directions.py --folder videos/teach_target
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
from visual_homing.server.fast3r_engine import Fast3REngine
from visual_homing.server.pose_estimator import PoseEstimator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load image as RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def test_pairwise(image_paths, config, metric_scale=1.0, folder_path=None):
    """Test consecutive pairs to understand forward trajectory."""

    # Try to load metric scale from metadata file (new session format)
    if folder_path:
        from visual_homing.server.flight_session import FlightSession

        # Try new format (metadata.json in session folder)
        metadata = FlightSession.load_session_metadata(folder_path)
        teach_metadata = metadata.get('teach_phase', {})
        loaded_scale = teach_metadata.get('global_scale_factor', None)

        if not loaded_scale or loaded_scale == 1.0:
            # Fallback to old format (keyframe_metadata.json)
            from visual_homing.server.keyframe_manager import KeyframeStackManager
            old_metadata = KeyframeStackManager.load_metadata(folder_path)
            loaded_scale = old_metadata.get('global_scale_factor', None)

        if loaded_scale and loaded_scale != 1.0:
            logger.info(f"✓ Using saved metric scale: {loaded_scale:.4f}")
            metric_scale = loaded_scale
        else:
            logger.warning(f"⚠ No saved scale found, using fallback: {metric_scale:.4f}")

    logger.info("Loading Fast3R model...")
    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    pose_estimator = PoseEstimator(
        confidence_percentile=0.5,
        min_points=100,
        metric_scale=metric_scale,
    )

    # Load all keyframes
    keyframes = [load_image(str(p)) for p in image_paths]
    num_keyframes = len(keyframes)

    logger.info(f"\nTesting {num_keyframes} keyframes pairwise")
    logger.info("="*70)

    results = []

    # Test consecutive pairs
    for i in range(num_keyframes - 1):
        img1 = keyframes[i]
        img2 = keyframes[i + 1]

        logger.info(f"\n{'='*70}")
        logger.info(f"Pair {i+1}: Frame {i} → Frame {i+1}")
        logger.info(f"This represents the FORWARD direction during TEACH phase")

        # Run Fast3R
        result = fast3r.infer_pair(img1, img2)

        # Estimate pose
        pose_result = pose_estimator.estimate_pose(
            result["pts3d_1"],
            result["pts3d_2"],
            result["conf_1"],
        )

        if not pose_result.success:
            logger.warning(f"❌ Pose estimation failed!")
            continue

        # Extract translation
        t_cam = pose_result.translation.cpu().numpy()
        distance = float(np.linalg.norm(t_cam))

        logger.info(f"Translation: X={t_cam[0]:.3f}m (right), Y={t_cam[1]:.3f}m (down), Z={t_cam[2]:.3f}m (forward)")
        logger.info(f"Distance: {distance:.3f}m, Confidence: {pose_result.confidence:.3f}")

        # Interpret direction
        dominant_axis = np.argmax(np.abs(t_cam))
        dominant_value = t_cam[dominant_axis]

        if dominant_axis == 0:  # X axis
            direction = "RIGHT" if dominant_value > 0 else "LEFT"
        elif dominant_axis == 1:  # Y axis
            direction = "DOWN" if dominant_value > 0 else "UP"
        else:  # Z axis
            direction = "FORWARD" if dominant_value > 0 else "BACKWARD"

        logger.info(f"Dominant direction: {direction} ({dominant_value:.3f}m)")

        # During homing, we need to REVERSE this direction
        if dominant_axis == 0:
            reverse_dir = "LEFT" if dominant_value > 0 else "RIGHT"
        elif dominant_axis == 1:
            reverse_dir = "UP" if dominant_value > 0 else "DOWN"
        else:
            reverse_dir = "BACKWARD" if dominant_value > 0 else "FORWARD"

        logger.info(f"→ During HOMING from frame {i+1} to frame {i}, command should be: {reverse_dir}")

        results.append({
            'pair': i + 1,
            'from_idx': i,
            'to_idx': i + 1,
            'translation': t_cam,
            'distance': distance,
            'confidence': pose_result.confidence,
            'dominant_axis': dominant_axis,
            'dominant_direction': direction,
            'reverse_direction': reverse_dir,
        })

    return results


def visualize_trajectory(results, output_path):
    """Visualize the trajectory."""

    if not results:
        logger.error("No results to visualize!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract data
    pairs = [r['pair'] for r in results]
    tx = [r['translation'][0] for r in results]
    ty = [r['translation'][1] for r in results]
    tz = [r['translation'][2] for r in results]
    distances = [r['distance'] for r in results]

    # Plot 1: Translation components
    ax1 = axes[0, 0]
    ax1.plot(pairs, tx, 'r-o', linewidth=2, label='X (right)', markersize=8)
    ax1.plot(pairs, ty, 'g-o', linewidth=2, label='Y (down)', markersize=8)
    ax1.plot(pairs, tz, 'b-o', linewidth=2, label='Z (forward)', markersize=8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Pair Number', fontsize=12)
    ax1.set_ylabel('Translation (m)', fontsize=12)
    ax1.set_title('Translation Components (TEACH Phase)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distance between frames
    ax2 = axes[0, 1]
    ax2.bar(pairs, distances, color='skyblue', edgecolor='navy', alpha=0.7)
    ax2.set_xlabel('Pair Number', fontsize=12)
    ax2.set_ylabel('Distance (m)', fontsize=12)
    ax2.set_title('Inter-Keyframe Distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: 2D trajectory (top-down view: X vs Z)
    ax3 = axes[1, 0]
    cum_x = np.cumsum([0] + tx)
    cum_z = np.cumsum([0] + tz)

    ax3.plot(cum_x, cum_z, 'purple', linewidth=3, marker='o', markersize=8, alpha=0.7)
    ax3.plot(cum_x[0], cum_z[0], 'go', markersize=15, label='Start', zorder=5)
    ax3.plot(cum_x[-1], cum_z[-1], 'ro', markersize=15, label='End', zorder=5)

    # Add arrows to show direction
    for i in range(len(cum_x) - 1):
        dx = cum_x[i+1] - cum_x[i]
        dz = cum_z[i+1] - cum_z[i]
        ax3.annotate('', xy=(cum_x[i+1], cum_z[i+1]), xytext=(cum_x[i], cum_z[i]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='purple', alpha=0.5))

    ax3.set_xlabel('Right (X, m)', fontsize=12)
    ax3.set_ylabel('Forward (Z, m)', fontsize=12)
    ax3.set_title('Trajectory (Bird\'s Eye View)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Plot 4: Direction summary (text)
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "TRAJECTORY SUMMARY\n" + "="*40 + "\n\n"
    summary_text += "TEACH Phase (forward):\n"

    # Count dominant directions
    direction_counts = {}
    for r in results:
        direction = r['dominant_direction']
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    for direction, count in sorted(direction_counts.items(), key=lambda x: -x[1]):
        summary_text += f"  {direction}: {count} segments\n"

    summary_text += f"\nTotal distance: {sum(distances):.2f}m\n"
    summary_text += f"Average step: {np.mean(distances):.2f}m\n"

    summary_text += "\n" + "="*40 + "\n"
    summary_text += "HOMING Phase (reverse):\n"
    summary_text += "Commands should move in OPPOSITE directions\n\n"

    reverse_counts = {}
    for r in results:
        direction = r['reverse_direction']
        reverse_counts[direction] = reverse_counts.get(direction, 0) + 1

    for direction, count in sorted(reverse_counts.items(), key=lambda x: -x[1]):
        summary_text += f"  {direction}: {count} segments\n"

    ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test pairwise keyframe directions"
    )
    parser.add_argument(
        "--folder",
        default="videos/teach_target",
        help="Folder with keyframe images"
    )
    parser.add_argument(
        "--metric-scale",
        type=float,
        default=10.0,
        help="Metric scale factor (fallback if no saved metadata, default=10.0)"
    )
    parser.add_argument(
        "--output",
        default="pairwise_directions.png",
        help="Output visualization path"
    )

    args = parser.parse_args()

    # Find images (check if it's a session folder or direct image folder)
    folder = Path(args.folder)
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    # Check for keyframes subfolder (new session structure)
    keyframes_folder = folder / "keyframes"
    if keyframes_folder.exists():
        logger.info(f"Found session folder, using keyframes from: {keyframes_folder}")
        image_folder = keyframes_folder
    else:
        logger.info(f"Using images directly from: {folder}")
        image_folder = folder

    image_paths = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.png"))

    if len(image_paths) < 2:
        logger.error(f"Need at least 2 images, found {len(image_paths)}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} keyframes")

    # Load config
    config = Config()

    # Test pairwise
    results = test_pairwise(image_paths, config, args.metric_scale, folder)

    # Visualize
    if results:
        visualize_trajectory(results, args.output)
        print(f"\n✓ Analysis complete! Check {args.output}")
    else:
        logger.error("No valid results to visualize")


if __name__ == "__main__":
    main()
