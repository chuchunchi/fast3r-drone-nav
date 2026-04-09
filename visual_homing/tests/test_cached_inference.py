"""Tests for the pairwise cached inference module.

Covers:
- Cache lifecycle in HomingController (mock-based, no GPU)
- Encoder output equivalence (GPU required)
- Cached forward equivalence (GPU required)
- Engine-level equivalence (GPU required)
- Latency comparison (GPU required)
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.config import Config
from visual_homing.server.homing_controller import HomingController
from visual_homing.server.keyframe_manager import Keyframe, Telemetry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _make_dummy_keyframe(index: int) -> Keyframe:
    """Create a keyframe with a random 512x384 RGB image."""
    return Keyframe(
        index=index,
        image=np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8),
        timestamp_ms=int(time.time() * 1000),
        cumulative_distance=float(index) * 2.0,
    )


def _make_mock_fast3r_engine():
    """Build a mock Fast3REngine whose encode_target returns a fake tuple."""
    engine = MagicMock()
    engine.is_loaded.return_value = True

    fake_feat = torch.randn(1, 768, 1024)
    fake_pos = torch.zeros(1, 768, 2, dtype=torch.long)
    fake_shape = torch.tensor([[384, 512]])
    fake_encoding = (fake_feat, fake_pos, fake_shape)
    engine.encode_target.return_value = fake_encoding

    fake_result = {
        "pts3d_1": torch.randn(384, 512, 3),
        "pts3d_2": torch.randn(384, 512, 3),
        "conf_1": torch.ones(384, 512),
        "conf_2": torch.ones(384, 512),
        "preds": [{}, {}],
    }
    engine.infer_pair.return_value = fake_result
    engine.infer_pair_cached.return_value = fake_result
    return engine


# ===================================================================
# Section 8.4 — Cache Invalidation (mock-based, no GPU)
# ===================================================================


class TestCacheInvalidation:
    """Verify cache lifecycle in HomingController."""

    @pytest.fixture
    def controller(self):
        """Build a HomingController with mocked engine and 3 keyframes."""
        engine = _make_mock_fast3r_engine()
        config = Config(device="cpu")
        ctrl = HomingController(fast3r_engine=engine, config=config)

        # Manually add keyframes (bypass TEACH phase)
        for i in range(3):
            ctrl.keyframe_manager.stack.append(_make_dummy_keyframe(i))

        # Transition to ARMED then HOMING-ready state
        ctrl.state_machine._state = ctrl.state_machine._state  # stay IDLE
        ctrl.state_machine._state = __import__(
            "visual_homing.server.state_machine", fromlist=["SystemState"]
        ).SystemState.ARMED

        return ctrl

    def test_cache_populated_on_start_homing(self, controller):
        controller.start_homing()

        assert controller._cached_target_encoding is not None
        assert controller._cached_target_idx == 2
        controller.fast3r.encode_target.assert_called_once()

    def test_cache_refreshed_on_advance(self, controller):
        controller.start_homing()
        assert controller._cached_target_idx == 2

        controller._advance_to_next_keyframe()
        assert controller._cached_target_idx == 1

        controller._advance_to_next_keyframe()
        assert controller._cached_target_idx == 0

        assert controller.fast3r.encode_target.call_count == 3

    def test_cache_cleared_on_reset(self, controller):
        controller.start_homing()
        assert controller._cached_target_encoding is not None

        controller.reset()
        assert controller._cached_target_encoding is None
        assert controller._cached_target_idx == -1

    def test_cache_cleared_on_emergency_stop(self, controller):
        controller.start_homing()
        assert controller._cached_target_encoding is not None

        controller.emergency_stop()
        assert controller._cached_target_encoding is None
        assert controller._cached_target_idx == -1

    def test_cache_cleared_when_target_becomes_negative(self, controller):
        controller.start_homing()

        # Advance past all keyframes
        controller._advance_to_next_keyframe()  # idx=1
        controller._advance_to_next_keyframe()  # idx=0
        controller._advance_to_next_keyframe()  # idx=-1

        assert controller._cached_target_encoding is None
        assert controller._cached_target_idx == -1


# ===================================================================
# GPU-required tests — model-level
# ===================================================================


def _load_model():
    """Load the Fast3R model for testing."""
    from fast3r.models.fast3r import Fast3R

    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
    model = model.to("cuda")
    model.eval()
    return model


def _make_test_views(device="cuda", dtype=torch.float32):
    """Create two random view dicts for testing."""
    views = []
    for _ in range(2):
        img = torch.randn(1, 3, 384, 512, device=device, dtype=dtype)
        view = {
            "img": img,
            "true_shape": torch.tensor([[384, 512]], device=device),
        }
        views.append(view)
    return views


@requires_cuda
class TestEncoderEquivalence:
    """Section 8.1 — Verify batched vs. individual encoding equivalence."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load_model()

    def test_single_vs_batched_encoding(self, model):
        view_a, view_b = _make_test_views()

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            feats_batched, pos_batched, _ = model._encode_images(
                [view_a, view_b]
            )
            feat_a_batched = feats_batched[0]
            pos_a_batched = pos_batched[0]

            feat_a_solo, pos_a_solo, _ = model.encode_image(view_a)

        assert torch.allclose(
            feat_a_batched, feat_a_solo, atol=1e-2, rtol=1e-3
        ), "Encoder features differ between batched and individual encoding"
        assert torch.equal(
            pos_a_batched, pos_a_solo
        ), "Positional encodings differ between batched and individual encoding"

    def test_encode_image_returns_correct_shapes(self, model):
        view_a, _ = _make_test_views()

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            feat, pos, shape = model.encode_image(view_a)

        assert feat.shape[0] == 1, "Batch dim should be 1"
        assert feat.shape[2] == model.encoder_args["embed_dim"]
        expected_patches = (384 // 16) * (512 // 16)  # 576
        assert feat.shape[1] == expected_patches
        assert pos.shape == (1, expected_patches, 2)
        assert shape.shape == (1, 2)


@requires_cuda
class TestCachedForwardEquivalence:
    """Section 8.2 — Verify forward_pair_cached matches forward."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load_model()

    def test_equivalence_with_deterministic_decoder(self, model):
        """Disable random_image_idx_embedding for deterministic comparison."""
        view_live, view_target = _make_test_views()

        original_flag = model.decoder.random_image_idx_embedding
        model.decoder.random_image_idx_embedding = False

        try:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                results_standard = model.forward([view_live, view_target])

                cached = model.encode_image(view_target)
                results_cached = model.forward_pair_cached(
                    view_live, cached
                )

            for i in range(2):
                for key in results_standard[i]:
                    assert torch.allclose(
                        results_standard[i][key],
                        results_cached[i][key],
                        atol=1e-2,
                    ), f"Mismatch in view {i}, key '{key}'"
        finally:
            model.decoder.random_image_idx_embedding = original_flag

    def test_equivalence_with_fixed_seed(self, model):
        """With random embeddings enabled, seed RNG for equivalence."""
        view_live, view_target = _make_test_views()
        seed = 42

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            torch.manual_seed(seed)
            results_standard = model.forward([view_live, view_target])

            cached = model.encode_image(view_target)
            torch.manual_seed(seed)
            results_cached = model.forward_pair_cached(view_live, cached)

        for i in range(2):
            for key in results_standard[i]:
                assert torch.allclose(
                    results_standard[i][key],
                    results_cached[i][key],
                    atol=1e-2,
                ), f"Mismatch in view {i}, key '{key}' (seeded)"

    def test_output_format(self, model):
        """Verify cached path returns the expected dict structure."""
        view_live, view_target = _make_test_views()

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            cached = model.encode_image(view_target)
            results = model.forward_pair_cached(view_live, cached)

        assert len(results) == 2
        for r in results:
            assert "pts3d_in_other_view" in r
            assert "conf" in r

    def test_local_head_guard(self, model):
        """Verify NotImplementedError when local_head is present."""
        view_live, _ = _make_test_views()
        fake_cached = (
            torch.randn(1, 576, 1024, device="cuda"),
            torch.zeros(1, 576, 2, dtype=torch.long, device="cuda"),
            torch.tensor([[384, 512]], device="cuda"),
        )

        model.local_head = MagicMock()
        try:
            with pytest.raises(NotImplementedError, match="local_head"):
                model.forward_pair_cached(view_live, fake_cached)
        finally:
            model.local_head = None


@requires_cuda
class TestEngineEquivalence:
    """Section 8.3 — Engine-level cached vs. uncached equivalence."""

    @pytest.fixture(scope="class")
    def engine(self):
        from visual_homing.server.fast3r_engine import Fast3REngine

        eng = Fast3REngine()
        eng.load_model()
        return eng

    def test_engine_cached_matches_uncached(self, engine):
        img1 = np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8)

        original_flag = engine.model.decoder.random_image_idx_embedding
        engine.model.decoder.random_image_idx_embedding = False

        try:
            result_standard = engine.infer_pair(img1, img2)
            cached_target = engine.encode_target(img2)
            result_cached = engine.infer_pair_cached(img1, cached_target)

            for key in ["pts3d_1", "pts3d_2", "conf_1", "conf_2"]:
                assert torch.allclose(
                    result_standard[key], result_cached[key], atol=1e-2
                ), f"Engine mismatch in '{key}'"
        finally:
            engine.model.decoder.random_image_idx_embedding = original_flag


# ===================================================================
# Section 8.5 — Latency comparison
# ===================================================================


@requires_cuda
class TestLatency:
    """Measure speedup from cached inference."""

    @pytest.fixture(scope="class")
    def model(self):
        return _load_model()

    def test_cached_is_faster(self, model):
        view_live, view_target = _make_test_views()
        N = 20

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            # Warmup
            for _ in range(3):
                model.forward([view_live, view_target])

            # Uncached
            times_uncached = []
            for _ in range(N):
                torch.cuda.synchronize()
                t0 = time.time()
                model.forward([view_live, view_target])
                torch.cuda.synchronize()
                times_uncached.append(time.time() - t0)

            # Cached
            cached = model.encode_image(view_target)
            times_cached = []
            for _ in range(N):
                torch.cuda.synchronize()
                t0 = time.time()
                model.forward_pair_cached(view_live, cached)
                torch.cuda.synchronize()
                times_cached.append(time.time() - t0)

        mean_uncached = np.mean(times_uncached) * 1000
        mean_cached = np.mean(times_cached) * 1000
        speedup_pct = (1 - mean_cached / mean_uncached) * 100

        print(f"\n  Uncached: {mean_uncached:.1f} ms")
        print(f"  Cached:   {mean_cached:.1f} ms")
        print(f"  Speedup:  {speedup_pct:.1f}%")

        assert speedup_pct > 5, (
            f"Expected >5% speedup from caching, got {speedup_pct:.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
