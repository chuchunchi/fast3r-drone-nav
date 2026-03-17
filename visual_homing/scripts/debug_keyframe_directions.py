#!/usr/bin/env python3
"""
Debug keyframe directions for homing control.

This script tests each frame in a sequence against the keyframe stack
and visualizes whether the control commands point in the correct direction.

Usage:
    python debug_keyframe_directions.py --folder videos/teach_target --output keyframe_directions.png
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.coordinate_utils import extract_yaw_error
from visual_homing.server.fast3r_engine import Fast3REngine
from visual_homing.server.pose_estimator import PoseEstimator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_image(path: str) -> np.ndarray:
    """Load image as RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_single_axis_command(t_cam, config):
    """
    Compute single-axis command like in homing_controller.py.

    Returns:
        dict with command and which axis was selected
    """
    error_forward = float(t_cam[2])
    error_lateral = float(t_cam[0])
    error_vertical = -float(t_cam[1])

    # Find axis with largest error
    errors_abs = [abs(error_forward), abs(error_lateral), abs(error_vertical)]
    max_error_idx = np.argmax(errors_abs)
    max_error = errors_abs[max_error_idx]

    min_error_threshold = 0.01  # 1cm (lowered for testing with close keyframes)

    command = {
        'pitch_velocity': 0.0,
        'roll_velocity': 0.0,
        'vertical_velocity': 0.0,
        'selected_axis': 'none',
        'selected_direction': 'none',
    }

    if max_error > min_error_threshold:
        fixed_vel = config.fixed_flight_velocity

        if max_error_idx == 0:  # Forward/backward
            command['pitch_velocity'] = fixed_vel if error_forward > 0 else -fixed_vel
            command['selected_axis'] = 'forward'
            command['selected_direction'] = 'forward' if error_forward > 0 else 'backward'
        elif max_error_idx == 1:  # Lateral
            command['roll_velocity'] = fixed_vel if error_lateral > 0 else -fixed_vel
            command['selected_axis'] = 'lateral'
            command['selected_direction'] = 'right' if error_lateral > 0 else 'left'
        else:  # Vertical
            command['vertical_velocity'] = fixed_vel if error_vertical > 0 else -fixed_vel
            command['selected_axis'] = 'vertical'
            command['selected_direction'] = 'up' if error_vertical > 0 else 'down'

    return command, (error_forward, error_lateral, error_vertical)


def test_keyframe_directions(image_paths, config, metric_scale=1.0, folder_path=None):
    """
    Test each frame against the keyframe stack.

    Returns:
        List of results for visualization
    """
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

    logger.info(f"Testing {num_keyframes} keyframes")
    logger.info("="*60)

    results = []

    # Test each frame as "live" against the keyframe stack
    # Simulate the homing process: start from end, work backwards
    for live_idx in range(num_keyframes - 1, -1, -1):
        live_img = keyframes[live_idx]

        # The target is the previous keyframe in the stack
        # (simulating popping from stack during homing)
        target_idx = live_idx - 1

        if target_idx < 0:
            logger.info(f"\nFrame {live_idx}: Reached start (no more targets)")
            break

        target_img = keyframes[target_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: Live frame {live_idx} → Target frame {target_idx}")
        logger.info(f"Expected: Should command movement BACKWARDS in sequence")

        # Run Fast3R
        result = fast3r.infer_pair(live_img, target_img)

        # Estimate pose
        pose_result = pose_estimator.estimate_pose(
            result["pts3d_1"],
            result["pts3d_2"],
            result["conf_1"],
        )

        if not pose_result.success:
            logger.warning(f"❌ Pose estimation failed!")
            results.append({
                'live_idx': live_idx,
                'target_idx': target_idx,
                'success': False,
            })
            continue

        # Extract translation in camera frame
        t_cam = pose_result.translation.cpu().numpy()
        R = pose_result.rotation
        distance = float(np.linalg.norm(t_cam))

        # Compute single-axis command
        command, errors = compute_single_axis_command(t_cam, config)
        error_forward, error_lateral, error_vertical = errors

        # Log results
        logger.info(f"Translation (camera frame): X={t_cam[0]:.3f}, Y={t_cam[1]:.3f}, Z={t_cam[2]:.3f}")
        logger.info(f"  X=right, Y=down, Z=forward (camera conventions)")
        logger.info(f"Errors: forward={error_forward:.3f}m, lateral={error_lateral:.3f}m, vertical={error_vertical:.3f}m")
        logger.info(f"Distance: {distance:.3f}m, Confidence: {pose_result.confidence:.3f}")
        logger.info(f"Selected axis: {command['selected_axis']} ({command['selected_direction']})")
        logger.info(f"Command: pitch={command['pitch_velocity']:.2f}, roll={command['roll_velocity']:.2f}, vert={command['vertical_velocity']:.2f}")

        # Interpret what this means
        if command['pitch_velocity'] > 0:
            logger.info("  → Drone will move FORWARD")
        elif command['pitch_velocity'] < 0:
            logger.info("  → Drone will move BACKWARD")

        if command['roll_velocity'] > 0:
            logger.info("  → Drone will move RIGHT")
        elif command['roll_velocity'] < 0:
            logger.info("  → Drone will move LEFT")

        if command['vertical_velocity'] > 0:
            logger.info("  → Drone will move UP")
        elif command['vertical_velocity'] < 0:
            logger.info("  → Drone will move DOWN")

        results.append({
            'live_idx': live_idx,
            'target_idx': target_idx,
            'live_image': live_img,
            'target_image': target_img,
            'success': True,
            'translation': t_cam,
            'rotation': R,
            'distance': distance,
            'confidence': pose_result.confidence,
            'error_forward': error_forward,
            'error_lateral': error_lateral,
            'error_vertical': error_vertical,
            'command': command,
        })

    return results


def visualize_directions(results, output_path):
    """Create comprehensive visualization of directions."""

    valid = [r for r in results if r['success']]

    if not valid:
        logger.error("No valid results to visualize!")
        return

    # Create figure with multiple subplots
    n_results = len(valid)
    n_cols = min(4, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))

    for idx, result in enumerate(valid):
        # Create 4 subplots for each result: live image, target image, 3D view, 2D bird's-eye
        base = idx * 4

        # Live image
        ax1 = plt.subplot(n_rows, n_cols * 4, base + 1)
        ax1.imshow(result['live_image'])
        ax1.set_title(f"Live Frame {result['live_idx']}", fontsize=10, fontweight='bold')
        ax1.axis('off')

        # Target image
        ax2 = plt.subplot(n_rows, n_cols * 4, base + 2)
        ax2.imshow(result['target_image'])
        ax2.set_title(f"Target Frame {result['target_idx']}", fontsize=10, fontweight='bold')
        ax2.axis('off')

        # 3D visualization of translation vector
        ax3 = plt.subplot(n_rows, n_cols * 4, base + 3, projection='3d')
        t = result['translation']

        # Draw axes
        ax3.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.3, label='X (right)', arrow_length_ratio=0.3)
        ax3.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.3, label='Y (down)', arrow_length_ratio=0.3)
        ax3.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.3, label='Z (forward)', arrow_length_ratio=0.3)

        # Draw translation vector
        ax3.quiver(0, 0, 0, t[0], t[1], t[2],
                  color='purple', linewidth=3, arrow_length_ratio=0.2, label='Translation')

        ax3.set_xlabel('X (right)', fontsize=8)
        ax3.set_ylabel('Y (down)', fontsize=8)
        ax3.set_zlabel('Z (forward)', fontsize=8)
        ax3.set_title(f"3D Translation\n({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})",
                     fontsize=10, fontweight='bold')
        ax3.legend(fontsize=6)

        # Set equal aspect ratio
        max_range = np.max(np.abs(t)) * 1.2
        ax3.set_xlim([-max_range, max_range])
        ax3.set_ylim([-max_range, max_range])
        ax3.set_zlim([-max_range, max_range])

        # 2D bird's-eye view (looking down)
        ax4 = plt.subplot(n_rows, n_cols * 4, base + 4)

        # Draw coordinate system
        ax4.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.3)
        ax4.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.3)
        ax4.text(1.2, 0, 'Right (X)', fontsize=8, ha='left', va='center')
        ax4.text(0, 1.2, 'Forward (Z)', fontsize=8, ha='center', va='bottom')

        # Draw translation (top-down view: X vs Z)
        ax4.arrow(0, 0, t[0], t[2], head_width=0.15, head_length=0.15,
                 fc='purple', ec='purple', linewidth=3, alpha=0.8)

        # Draw command direction
        cmd = result['command']
        cmd_x = cmd['roll_velocity'] * 0.5  # Scale for visibility
        cmd_z = cmd['pitch_velocity'] * 0.5

        if abs(cmd_x) > 0.01 or abs(cmd_z) > 0.01:
            ax4.arrow(0, 0, cmd_x, cmd_z, head_width=0.12, head_length=0.12,
                     fc='red', ec='red', linewidth=2, linestyle='--', alpha=0.7)

        # Add text
        direction = result['command']['selected_direction']
        ax4.text(0, -max_range * 1.2, f"Command: {direction.upper()}",
                fontsize=10, fontweight='bold', ha='center', color='red')

        ax4.set_xlim([-max_range * 1.5, max_range * 1.5])
        ax4.set_ylim([-max_range * 1.5, max_range * 1.5])
        ax4.set_aspect('equal')
        ax4.set_title(f"Bird's Eye View\nConf={result['confidence']:.2f}",
                     fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Lateral (m)', fontsize=8)
        ax4.set_ylabel('Forward (m)', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug keyframe directions for homing control"
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
        default="keyframe_directions.png",
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

    # Test directions
    results = test_keyframe_directions(image_paths, config, args.metric_scale, folder)

    # Visualize
    visualize_directions(results, args.output)

    print(f"\n✓ Analysis complete! Check {args.output}")


if __name__ == "__main__":
    main()
