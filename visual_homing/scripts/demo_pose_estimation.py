#!/usr/bin/env python3
"""
Demo script for Phase 1: Pose Estimation from Image Pairs.

This script demonstrates the core functionality of the visual homing system:
1. Load the Fast3R model
2. Run inference on a pair of images
3. Compute relative pose using SVD Procrustes
4. Display the results

Usage:
    python demo_pose_estimation.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
    python demo_pose_estimation.py --demo  # Use demo images from the repo

Deliverable: Python script that takes 2 images, outputs relative pose in meters
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.fast3r_engine import Fast3REngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Pose estimation from image pairs using Fast3R"
    )
    parser.add_argument(
        "--image1",
        type=str,
        help="Path to first image",
    )
    parser.add_argument(
        "--image2",
        type=str,
        help="Path to second image",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo images from the repository",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Metric scale factor (meters per Fast3R unit)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the pointmaps",
    )

    args = parser.parse_args()

    # Determine image paths
    if args.demo:
        # Use images from croco demo if available
        repo_root = Path(__file__).parent.parent.parent
        demo_images = list(repo_root.glob("fast3r/croco/assets/*.png"))
        if len(demo_images) >= 2:
            image1_path = str(demo_images[0])
            image2_path = str(demo_images[1])
            logger.info(f"Using demo images: {demo_images[0].name}, {demo_images[1].name}")
        else:
            logger.error("Demo images not found. Please provide --image1 and --image2")
            return 1
    elif args.image1 and args.image2:
        image1_path = args.image1
        image2_path = args.image2
    else:
        parser.print_help()
        return 1

    # Load images
    logger.info("Loading images...")
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)
    logger.info(f"Image 1 shape: {image1.shape}")
    logger.info(f"Image 2 shape: {image2.shape}")

    # Initialize Fast3R engine
    logger.info("Initializing Fast3R engine...")
    config = Config(device=args.device)
    fast3r = Fast3REngine(config=config)
    fast3r.load_model()

    # Run inference with pose estimation
    logger.info("Running Fast3R inference with PnP pose estimation...")
    result = fast3r.infer_pair_with_pose(image1, image2, niter_PnP=10)

    pts3d_1 = result["pts3d_1"]
    pts3d_2 = result["pts3d_2"]
    conf_1 = result["conf_1"]
    R_rel = result["R_rel"]
    t_rel = result["t_rel"]

    logger.info(f"Pointmap 1 shape: {pts3d_1.shape}")
    logger.info(f"Pointmap 2 shape: {pts3d_2.shape}")
    logger.info(f"Confidence 1 range: [{conf_1.min():.3f}, {conf_1.max():.3f}]")

    # Print results
    print("\n" + "=" * 60)
    print("POSE ESTIMATION RESULTS (PnP-based)")
    print("=" * 60)

    # t_rel is cam1 origin in cam2 frame
    # To get drone motion, we negate (drone moved opposite to where cam1 appears in cam2)
    t_drone = -t_rel

    # Extract Euler angles
    import math
    def rotation_to_euler(R):
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
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    roll, pitch, yaw = rotation_to_euler(R_rel)
    distance = np.linalg.norm(t_rel)

    print(f"\nRelative Translation (drone motion, camera frame):")
    print(f"  X (lateral):  {t_drone[0]:+.4f} (positive=right, negative=left)")
    print(f"  Y (vertical): {t_drone[1]:+.4f} (positive=down, negative=up)")
    print(f"  Z (forward):  {t_drone[2]:+.4f} (positive=forward, negative=backward)")
    print(f"  Distance:     {distance:.4f} (Fast3R units)")

    print(f"\nRelative Rotation (Euler angles):")
    print(f"  Roll:  {roll:+.2f}°")
    print(f"  Pitch: {pitch:+.2f}°")
    print(f"  Yaw:   {yaw:+.2f}°")

    print(f"\nEstimated focal lengths:")
    print(f"  Image 1: {result['focal_1']:.2f} px")
    print(f"  Image 2: {result['focal_2']:.2f} px")

    print(f"\nInterpretation:")
    print(f"  The drone moved {distance:.4f} Fast3R units from image1 to image2")
    if t_drone[2] > 0.001:
        print(f"  - Forward: {t_drone[2]:.4f}")
    elif t_drone[2] < -0.001:
        print(f"  - Backward: {-t_drone[2]:.4f}")
    if t_drone[0] > 0.001:
        print(f"  - Right: {t_drone[0]:.4f}")
    elif t_drone[0] < -0.001:
        print(f"  - Left: {-t_drone[0]:.4f}")
    if t_drone[1] > 0.001:
        print(f"  - Down: {t_drone[1]:.4f}")
    elif t_drone[1] < -0.001:
        print(f"  - Up: {-t_drone[1]:.4f}")

    if args.scale != 1.0:
        print(f"\n  With scale factor {args.scale}:")
        print(f"    Estimated real distance: {distance * args.scale:.4f} meters")

    print("=" * 60 + "\n")

    # Visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(15, 5))

            # Image pair
            ax1 = fig.add_subplot(131)
            ax1.imshow(image1)
            ax1.set_title("Image 1 (Source)")
            ax1.axis("off")

            ax2 = fig.add_subplot(132)
            ax2.imshow(image2)
            ax2.set_title("Image 2 (Target)")
            ax2.axis("off")

            # 3D pointmap
            ax3 = fig.add_subplot(133, projection="3d")

            # Subsample for visualization
            pts = pts3d_1.cpu().numpy().reshape(-1, 3)
            conf = conf_1.cpu().numpy().reshape(-1)
            mask = conf > np.percentile(conf, 50)
            pts_vis = pts[mask][::10]  # Subsample

            ax3.scatter(
                pts_vis[:, 0],
                pts_vis[:, 1],
                pts_vis[:, 2],
                c=pts_vis[:, 2],
                cmap="viridis",
                s=1,
            )
            ax3.set_xlabel("X (right)")
            ax3.set_ylabel("Y (down)")
            ax3.set_zlabel("Z (forward)")
            ax3.set_title("3D Pointmap")

            plt.tight_layout()
            plt.savefig("pose_estimation_demo.png", dpi=150)
            logger.info("Saved visualization to pose_estimation_demo.png")
            plt.show()

        except ImportError:
            logger.warning("matplotlib not available for visualization")

    return 0


if __name__ == "__main__":
    sys.exit(main())

