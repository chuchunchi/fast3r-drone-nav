"""Fast3R inference engine for 3D reconstruction from image pairs."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .config import Config, default_config

logger = logging.getLogger(__name__)


def _compute_relative_pose(pose_c2w_1: np.ndarray, pose_c2w_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose from camera 1 to camera 2.

    Args:
        pose_c2w_1: 4x4 camera-to-world matrix for camera 1
        pose_c2w_2: 4x4 camera-to-world matrix for camera 2

    Returns:
        R_rel: 3x3 rotation matrix (cam1 to cam2)
        t_rel: 3D translation vector (cam1 origin in cam2 frame)
    """
    # World-to-camera for camera 2
    pose_w2c_2 = np.linalg.inv(pose_c2w_2)

    # Relative pose: cam1 -> world -> cam2
    pose_rel = pose_w2c_2 @ pose_c2w_1

    R_rel = pose_rel[:3, :3]
    t_rel = pose_rel[:3, 3]

    return R_rel, t_rel


class Fast3REngine:
    """
    Fast3R inference engine for computing 3D pointmaps from image pairs.

    This class wraps the Fast3R model and provides a simple interface for
    running inference on pairs of images to obtain 3D point clouds in a
    shared reference frame.
    """

    def __init__(
        self,
        checkpoint: str = None,
        device: str = None,
        config: Config = None,
    ):
        """
        Initialize the Fast3R engine.

        Args:
            checkpoint: Path or HuggingFace model ID for Fast3R weights.
            device: Device to run inference on ('cuda', 'cpu').
            config: Configuration object. If None, uses default_config.
        """
        self.config = config or default_config
        self.checkpoint = checkpoint or self.config.fast3r_checkpoint
        self.device = torch.device(device or self.config.device)

        # Determine dtype
        if self.config.dtype == "float16":
            self.dtype = torch.float16
        elif self.config.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.model = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the Fast3R model into memory."""
        if self._loaded:
            logger.info("Model already loaded, skipping...")
            return

        logger.info(f"Loading Fast3R model from {self.checkpoint}...")

        # Import Fast3R components
        from fast3r.models.fast3r import Fast3R

        # Load model
        self.model = Fast3R.from_pretrained(self.checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(f"Fast3R model loaded on {self.device}")

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def _prepare_image(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Prepare a single image for Fast3R inference.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            Dictionary with 'img' tensor and 'true_shape' tensor.
        """
        target_w, target_h = self.config.image_size

        # Resize if needed
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h))

        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dim

        return {
            "img": img_tensor.to(self.device, dtype=self.dtype),
            "true_shape": torch.tensor([[target_h, target_w]]).to(self.device),
        }

    def _prepare_images(
        self, images: List[np.ndarray]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare multiple images for Fast3R inference.

        Args:
            images: List of RGB images as numpy arrays.

        Returns:
            List of prepared image dictionaries.
        """
        return [self._prepare_image(img) for img in images]

    @torch.no_grad()
    def infer_pair(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Run Fast3R inference on a pair of images.

        Args:
            image1: First RGB image as numpy array (H, W, 3), uint8.
            image2: Second RGB image as numpy array (H, W, 3), uint8.

        Returns:
            Dictionary containing:
                - 'pts3d_1': 3D pointmap for image1 in reference frame (H, W, 3)
                - 'pts3d_2': 3D pointmap for image2 in reference frame (H, W, 3)
                - 'conf_1': Confidence map for image1 (H, W)
                - 'conf_2': Confidence map for image2 (H, W)
                - 'preds': Raw predictions from Fast3R
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Import inference function
        from fast3r.dust3r.inference_multiview import inference

        # Prepare images
        imgs = self._prepare_images([image1, image2])

        # Run inference
        # Note: inference() returns a single dict when profiling=False
        output = inference(
            imgs,
            self.model,
            self.device,
            dtype=self.dtype,
            verbose=False,
            profiling=False,
        )

        preds = output["preds"]

        # Extract pointmaps and confidence
        # preds is a list with one entry per image
        # Each entry has 'pts3d_in_other_view' and 'conf'
        # Outputs have batch dimension [B, H, W, 3] - squeeze if batch=1
        pts3d_1 = preds[0]["pts3d_in_other_view"]
        pts3d_2 = preds[1]["pts3d_in_other_view"]
        conf_1 = preds[0]["conf"]
        conf_2 = preds[1]["conf"]

        # Remove batch dimension if present
        if pts3d_1.dim() == 4 and pts3d_1.shape[0] == 1:
            pts3d_1 = pts3d_1.squeeze(0)
            pts3d_2 = pts3d_2.squeeze(0)
            conf_1 = conf_1.squeeze(0)
            conf_2 = conf_2.squeeze(0)

        result = {
            "pts3d_1": pts3d_1,
            "pts3d_2": pts3d_2,
            "conf_1": conf_1,
            "conf_2": conf_2,
            "preds": preds,
        }

        return result

    @torch.no_grad()
    def infer_multiview(
        self, images: List[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Run Fast3R inference on multiple images.

        Args:
            images: List of RGB images as numpy arrays.

        Returns:
            Dictionary containing pointmaps and confidence for all views.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from fast3r.dust3r.inference_multiview import inference

        imgs = self._prepare_images(images)

        # Note: inference() returns a single dict when profiling=False
        output = inference(
            imgs,
            self.model,
            self.device,
            dtype=self.dtype,
            verbose=False,
            profiling=False,
        )

        return output

    @torch.no_grad()
    def infer_pair_with_pose(
        self, image1: np.ndarray, image2: np.ndarray, niter_PnP: int = 10
    ) -> Dict[str, any]:
        """
        Run Fast3R inference and estimate relative camera pose.

        This uses Fast3R's built-in PnP-based pose estimation which
        correctly handles the coordinate frames.

        Args:
            image1: First RGB image as numpy array (H, W, 3), uint8.
            image2: Second RGB image as numpy array (H, W, 3), uint8.
            niter_PnP: Number of PnP iterations for pose refinement.

        Returns:
            Dictionary containing:
                - 'R_rel': 3x3 relative rotation (cam1 to cam2)
                - 't_rel': 3D relative translation (cam1 origin in cam2 frame)
                - 'pose_c2w_1': 4x4 camera-to-world for image1
                - 'pose_c2w_2': 4x4 camera-to-world for image2
                - 'focal_1', 'focal_2': Estimated focal lengths
                - 'pts3d_1', 'pts3d_2': 3D pointmaps
                - 'conf_1', 'conf_2': Confidence maps
                - 'preds': Raw predictions
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from fast3r.dust3r.inference_multiview import inference
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

        # Prepare and run inference
        imgs = self._prepare_images([image1, image2])

        output = inference(
            imgs,
            self.model,
            self.device,
            dtype=self.dtype,
            verbose=False,
            profiling=False,
        )

        preds = output["preds"]

        # Estimate camera poses using Fast3R's built-in method
        poses_c2w, focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            preds, niter_PnP=niter_PnP
        )

        # Extract for batch index 0
        pose_c2w_1 = poses_c2w[0][0]  # First sample, first view
        pose_c2w_2 = poses_c2w[0][1]  # First sample, second view
        focal_1 = focals[0][0]
        focal_2 = focals[0][1]

        # Convert to numpy if needed
        if isinstance(pose_c2w_1, torch.Tensor):
            pose_c2w_1 = pose_c2w_1.cpu().numpy()
        if isinstance(pose_c2w_2, torch.Tensor):
            pose_c2w_2 = pose_c2w_2.cpu().numpy()

        # Compute relative pose
        R_rel, t_rel = _compute_relative_pose(pose_c2w_1, pose_c2w_2)

        # Extract pointmaps
        pts3d_1 = preds[0]["pts3d_in_other_view"]
        pts3d_2 = preds[1]["pts3d_in_other_view"]
        conf_1 = preds[0]["conf"]
        conf_2 = preds[1]["conf"]

        # Remove batch dimension if present
        if pts3d_1.dim() == 4 and pts3d_1.shape[0] == 1:
            pts3d_1 = pts3d_1.squeeze(0)
            pts3d_2 = pts3d_2.squeeze(0)
            conf_1 = conf_1.squeeze(0)
            conf_2 = conf_2.squeeze(0)

        return {
            "R_rel": R_rel,
            "t_rel": t_rel,
            "pose_c2w_1": pose_c2w_1,
            "pose_c2w_2": pose_c2w_2,
            "focal_1": focal_1,
            "focal_2": focal_2,
            "pts3d_1": pts3d_1,
            "pts3d_2": pts3d_2,
            "conf_1": conf_1,
            "conf_2": conf_2,
            "preds": preds,
        }

    # ------------------------------------------------------------------
    # Cached pairwise inference
    # ------------------------------------------------------------------

    @property
    def _autocast_dtype(self) -> torch.dtype:
        """Resolve the effective autocast dtype.

        The standard inference path passes ``self.dtype`` as the *precision*
        arg to ``loss_of_one_batch()``, which only recognises the string
        ``"32"`` and ``torch.bfloat16``.  Every other value falls through
        to the default CUDA autocast dtype (``torch.float16``).
        """
        if self.dtype == torch.bfloat16:
            return torch.bfloat16
        return torch.float16

    @torch.no_grad()
    def encode_target(
        self, image: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-encode a target image for cached pairwise inference.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            ``(encoded_feat, position, true_shape)`` tuple that can be
            passed to :meth:`infer_pair_cached`.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        view = self._prepare_image(image)
        with torch.autocast(
            device_type=self.device.type, dtype=self._autocast_dtype
        ):
            return self.model.encode_image(view)

    @torch.no_grad()
    def infer_pair_cached(
        self,
        live_image: np.ndarray,
        cached_target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run pairwise inference using a pre-encoded target.

        Produces the same output format as :meth:`infer_pair`, but skips
        the encoder for the target image.

        Args:
            live_image:    Current camera frame (H, W, 3), uint8.
            cached_target: Output of :meth:`encode_target` for the
                           reference keyframe.

        Returns:
            Dict with ``pts3d_1``, ``pts3d_2``, ``conf_1``, ``conf_2``,
            and ``preds`` — identical format to :meth:`infer_pair`.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        live_view = self._prepare_image(live_image)
        with torch.autocast(
            device_type=self.device.type, dtype=self._autocast_dtype
        ):
            preds = self.model.forward_pair_cached(live_view, cached_target)

        pts3d_1 = preds[0]["pts3d_in_other_view"]
        pts3d_2 = preds[1]["pts3d_in_other_view"]
        conf_1 = preds[0]["conf"]
        conf_2 = preds[1]["conf"]

        if pts3d_1.dim() == 4 and pts3d_1.shape[0] == 1:
            pts3d_1 = pts3d_1.squeeze(0)
            pts3d_2 = pts3d_2.squeeze(0)
            conf_1 = conf_1.squeeze(0)
            conf_2 = conf_2.squeeze(0)

        return {
            "pts3d_1": pts3d_1,
            "pts3d_2": pts3d_2,
            "conf_1": conf_1,
            "conf_2": conf_2,
            "preds": preds,
        }

    # ------------------------------------------------------------------

    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        return self.device

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"Fast3REngine(checkpoint={self.checkpoint}, device={self.device}, status={status})"

