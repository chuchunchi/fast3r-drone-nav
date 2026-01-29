"""Visualization utilities for debugging pose estimation and homing."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Optional imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.debug("matplotlib not available - visualization disabled")


def visualize_pointmaps(
    pts3d_1: Union[np.ndarray, torch.Tensor],
    pts3d_2: Union[np.ndarray, torch.Tensor],
    conf_1: Union[np.ndarray, torch.Tensor],
    conf_2: Union[np.ndarray, torch.Tensor],
    image1: Optional[np.ndarray] = None,
    image2: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Fast3R Pointmaps",
    confidence_threshold: float = 0.5,
    subsample: int = 10,
) -> Optional[plt.Figure]:
    """
    Visualize Fast3R pointmaps from an image pair.
    
    Args:
        pts3d_1: Pointmap for image 1 (H, W, 3).
        pts3d_2: Pointmap for image 2 (H, W, 3).
        conf_1: Confidence for image 1 (H, W).
        conf_2: Confidence for image 2 (H, W).
        image1: Optional RGB image 1.
        image2: Optional RGB image 2.
        output_path: Path to save figure.
        title: Figure title.
        confidence_threshold: Percentile threshold for confidence filtering.
        subsample: Subsample factor for point cloud visualization.
    
    Returns:
        matplotlib Figure or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available for visualization")
        return None

    # Convert to numpy if needed
    if torch.is_tensor(pts3d_1):
        pts3d_1 = pts3d_1.cpu().numpy()
    if torch.is_tensor(pts3d_2):
        pts3d_2 = pts3d_2.cpu().numpy()
    if torch.is_tensor(conf_1):
        conf_1 = conf_1.cpu().numpy()
    if torch.is_tensor(conf_2):
        conf_2 = conf_2.cpu().numpy()

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Row 1: Images and confidence maps
    if image1 is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image1)
        ax1.set_title("Image 1")
        ax1.axis("off")

    if image2 is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(image2)
        ax2.set_title("Image 2")
        ax2.axis("off")

    # Confidence map
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(conf_1, cmap="viridis")
    ax3.set_title(f"Confidence Map 1\nRange: [{conf_1.min():.2f}, {conf_1.max():.2f}]")
    plt.colorbar(im, ax=ax3)
    ax3.axis("off")

    # Row 2: 3D pointmaps
    # Flatten and filter by confidence
    pts1_flat = pts3d_1.reshape(-1, 3)
    pts2_flat = pts3d_2.reshape(-1, 3)
    conf1_flat = conf_1.reshape(-1)
    conf2_flat = conf_2.reshape(-1)

    # Filter by confidence threshold (top N%)
    conf_thresh_1 = np.percentile(conf1_flat, confidence_threshold * 100)
    conf_thresh_2 = np.percentile(conf2_flat, confidence_threshold * 100)

    mask1 = conf1_flat >= conf_thresh_1
    mask2 = conf2_flat >= conf_thresh_2

    pts1_good = pts1_flat[mask1][::subsample]
    pts2_good = pts2_flat[mask2][::subsample]

    # 3D scatter plot for image 1
    ax4 = fig.add_subplot(gs[1, 0], projection="3d")
    ax4.scatter(
        pts1_good[:, 0],
        pts1_good[:, 1],
        pts1_good[:, 2],
        c=pts1_good[:, 2],  # Color by depth
        cmap="viridis",
        s=1,
        alpha=0.6,
    )
    ax4.set_xlabel("X (right)")
    ax4.set_ylabel("Y (down)")
    ax4.set_zlabel("Z (forward)")
    ax4.set_title(f"Pointmap 1 ({len(pts1_good)} pts)")

    # 3D scatter plot for image 2
    ax5 = fig.add_subplot(gs[1, 1], projection="3d")
    ax5.scatter(
        pts2_good[:, 0],
        pts2_good[:, 1],
        pts2_good[:, 2],
        c=pts2_good[:, 2],
        cmap="plasma",
        s=1,
        alpha=0.6,
    )
    ax5.set_xlabel("X (right)")
    ax5.set_ylabel("Y (down)")
    ax5.set_zlabel("Z (forward)")
    ax5.set_title(f"Pointmap 2 ({len(pts2_good)} pts)")

    # Combined view (both pointmaps)
    ax6 = fig.add_subplot(gs[1, 2], projection="3d")
    ax6.scatter(
        pts1_good[:, 0],
        pts1_good[:, 1],
        pts1_good[:, 2],
        c="blue",
        s=1,
        alpha=0.5,
        label="Image 1",
    )
    ax6.scatter(
        pts2_good[:, 0],
        pts2_good[:, 1],
        pts2_good[:, 2],
        c="red",
        s=1,
        alpha=0.5,
        label="Image 2",
    )
    ax6.set_xlabel("X (right)")
    ax6.set_ylabel("Y (down)")
    ax6.set_zlabel("Z (forward)")
    ax6.set_title("Combined View")
    ax6.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved visualization to {output_path}")

    return fig


def visualize_pose_estimation(
    translation: Union[np.ndarray, torch.Tensor],
    rotation: Union[np.ndarray, torch.Tensor],
    confidence: float,
    image_live: Optional[np.ndarray] = None,
    image_target: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    metric_scale: float = 1.0,
) -> Optional[plt.Figure]:
    """
    Visualize pose estimation result.
    
    Args:
        translation: Translation vector (3,) in camera frame.
        rotation: Rotation matrix (3, 3).
        confidence: Pose confidence score.
        image_live: Live camera image.
        image_target: Target keyframe image.
        output_path: Path to save figure.
        metric_scale: Scale to convert to meters.
    
    Returns:
        matplotlib Figure or None.
    """
    if not HAS_MATPLOTLIB:
        return None

    if torch.is_tensor(translation):
        translation = translation.cpu().numpy()
    if torch.is_tensor(rotation):
        rotation = rotation.cpu().numpy()

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig)

    # Images
    if image_live is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_live)
        ax1.set_title("Live Frame")
        ax1.axis("off")

    if image_target is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(image_target)
        ax2.set_title("Target Keyframe")
        ax2.axis("off")

    # Translation visualization
    ax3 = fig.add_subplot(gs[0, 2])
    labels = ["X (lateral)", "Y (vertical)", "Z (forward)"]
    colors = ["red", "green", "blue"]
    ax3.barh(labels, translation, color=colors)
    ax3.axvline(x=0, color="black", linewidth=0.5)
    ax3.set_xlabel(f"Translation ({'' if metric_scale == 1 else 'meters'})")
    ax3.set_title(f"Pose Translation\nConfidence: {confidence:.3f}")

    # Add distance annotation
    distance = np.linalg.norm(translation)
    ax3.text(
        0.95, 0.95,
        f"Distance: {distance:.3f}{'m' if metric_scale != 1 else ''}",
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    # Top-down view (X-Z plane)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_aspect("equal")
    ax4.plot(0, 0, "ro", markersize=10, label="Live (origin)")
    ax4.plot(translation[0], translation[2], "g^", markersize=10, label="Target")
    ax4.arrow(
        0, 0, translation[0], translation[2],
        head_width=0.05 * distance if distance > 0 else 0.05,
        head_length=0.03 * distance if distance > 0 else 0.03,
        fc="blue", ec="blue",
    )
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("X (lateral)")
    ax4.set_ylabel("Z (forward)")
    ax4.set_title("Top-Down View (X-Z plane)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Side view (Y-Z plane)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_aspect("equal")
    ax5.plot(0, 0, "ro", markersize=10, label="Live (origin)")
    ax5.plot(translation[2], -translation[1], "g^", markersize=10, label="Target")
    ax5.arrow(
        0, 0, translation[2], -translation[1],
        head_width=0.05 * distance if distance > 0 else 0.05,
        head_length=0.03 * distance if distance > 0 else 0.03,
        fc="blue", ec="blue",
    )
    ax5.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax5.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Z (forward)")
    ax5.set_ylabel("Y (up)")  # Inverted for intuitive view
    ax5.set_title("Side View (Y-Z plane)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Rotation matrix visualization
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(rotation, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax6)
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(["X", "Y", "Z"])
    ax6.set_yticklabels(["X", "Y", "Z"])
    ax6.set_title("Rotation Matrix")

    # Add rotation values as text
    for i in range(3):
        for j in range(3):
            ax6.text(
                j, i, f"{rotation[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(rotation[i, j]) > 0.5 else "black",
            )

    plt.suptitle("Pose Estimation Result", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved pose visualization to {output_path}")

    return fig


def visualize_control_history(
    history: List[Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Control Command History",
) -> Optional[plt.Figure]:
    """
    Visualize control command history over time.
    
    Args:
        history: List of command dicts with pitch_velocity, roll_velocity, etc.
        output_path: Path to save figure.
        title: Figure title.
    
    Returns:
        matplotlib Figure or None.
    """
    if not HAS_MATPLOTLIB:
        return None

    if not history:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    steps = range(len(history))

    # Forward velocity (pitch)
    ax1 = axes[0, 0]
    ax1.plot(steps, [h["pitch_velocity"] for h in history], "b-o", markersize=3)
    ax1.axhline(y=0, color="gray", linestyle="--")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Forward Velocity")
    ax1.grid(True, alpha=0.3)

    # Lateral velocity (roll)
    ax2 = axes[0, 1]
    ax2.plot(steps, [h["roll_velocity"] for h in history], "g-o", markersize=3)
    ax2.axhline(y=0, color="gray", linestyle="--")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Lateral Velocity")
    ax2.grid(True, alpha=0.3)

    # Vertical velocity
    ax3 = axes[1, 0]
    ax3.plot(steps, [h["vertical_velocity"] for h in history], "r-o", markersize=3)
    ax3.axhline(y=0, color="gray", linestyle="--")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Vertical Velocity")
    ax3.grid(True, alpha=0.3)

    # Yaw rate
    ax4 = axes[1, 1]
    ax4.plot(steps, [h["yaw_rate"] for h in history], "m-o", markersize=3)
    ax4.axhline(y=0, color="gray", linestyle="--")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Rate (deg/s)")
    ax4.set_title("Yaw Rate")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved control history visualization to {output_path}")

    return fig


def visualize_keyframe_sequence(
    keyframes: List["Keyframe"],
    max_display: int = 8,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Visualize a sequence of recorded keyframes.
    
    Args:
        keyframes: List of Keyframe objects.
        max_display: Maximum number of keyframes to display.
        output_path: Path to save figure.
    
    Returns:
        matplotlib Figure or None.
    """
    if not HAS_MATPLOTLIB:
        return None

    n_keyframes = min(len(keyframes), max_display)
    cols = min(4, n_keyframes)
    rows = (n_keyframes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_keyframes == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for i, ax in enumerate(axes.flatten()):
        if i < n_keyframes:
            kf = keyframes[i]
            ax.imshow(kf.image)
            ax.set_title(
                f"KF {kf.index}\n"
                f"Dist: {kf.cumulative_distance:.2f}m"
            )
        ax.axis("off")

    plt.suptitle(f"Recorded Keyframes ({len(keyframes)} total)", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved keyframe visualization to {output_path}")

    return fig


def visualize_homing_trajectory(
    translations: List[np.ndarray],
    reached_flags: List[bool],
    keyframe_positions: Optional[List[np.ndarray]] = None,
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Visualize the homing trajectory from translation errors.
    
    Args:
        translations: List of translation vectors (camera frame).
        reached_flags: List of booleans indicating waypoint reached.
        keyframe_positions: Optional list of keyframe positions.
        output_path: Path to save figure.
    
    Returns:
        matplotlib Figure or None.
    """
    if not HAS_MATPLOTLIB:
        return None

    fig = plt.figure(figsize=(12, 5))

    # Top-down view (X-Z)
    ax1 = fig.add_subplot(121)
    ax1.set_aspect("equal")

    # Plot trajectory
    xs = [t[0] for t in translations]
    zs = [t[2] for t in translations]
    ax1.plot(xs, zs, "b-", alpha=0.5, linewidth=1)
    ax1.scatter(xs, zs, c=range(len(xs)), cmap="viridis", s=30, zorder=5)

    # Mark waypoint reached points
    for i, (t, reached) in enumerate(zip(translations, reached_flags)):
        if reached:
            ax1.scatter(t[0], t[2], c="green", s=100, marker="*", zorder=10)

    # Plot origin (drone position)
    ax1.scatter(0, 0, c="red", s=100, marker="o", label="Drone", zorder=10)

    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax1.set_xlabel("X (lateral)")
    ax1.set_ylabel("Z (forward)")
    ax1.set_title("Top-Down View (X-Z)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Distance over time
    ax2 = fig.add_subplot(122)
    distances = [np.linalg.norm(t) for t in translations]
    ax2.plot(distances, "b-o", markersize=4)
    ax2.axhline(y=0.8, color="green", linestyle="--", label="Waypoint threshold")

    # Mark waypoint reached
    for i, reached in enumerate(reached_flags):
        if reached:
            ax2.axvline(x=i, color="red", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance to Target (m)")
    ax2.set_title("Distance Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Homing Trajectory", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved trajectory visualization to {output_path}")

    return fig


# Avoid circular import by using forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .keyframe_manager import Keyframe

