"""Tests for keyframe manager."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.keyframe_manager import (
    Keyframe,
    KeyframeStackManager,
    Telemetry,
)


def make_telemetry(timestamp_ms: int, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> Telemetry:
    """Create a telemetry object for testing."""
    return Telemetry(
        timestamp_ms=timestamp_ms,
        velocity_x=vx,
        velocity_y=vy,
        velocity_z=vz,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        height=1.0,
    )


def make_frame(h: int = 384, w: int = 512) -> np.ndarray:
    """Create a dummy frame for testing."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestTelemetry:
    """Test telemetry dataclass."""

    def test_telemetry_creation(self):
        """Test creating telemetry object."""
        telem = Telemetry(
            timestamp_ms=1000,
            velocity_x=1.0,
            velocity_y=0.5,
            velocity_z=-0.2,
            yaw=45.0,
            pitch=5.0,
            roll=-2.0,
            height=10.0,
        )

        assert telem.timestamp_ms == 1000
        assert telem.velocity_x == 1.0
        assert telem.height == 10.0


class TestKeyframe:
    """Test keyframe dataclass."""

    def test_keyframe_creation(self):
        """Test creating keyframe object."""
        frame = make_frame()
        kf = Keyframe(
            index=0,
            image=frame,
            timestamp_ms=1000,
            cumulative_distance=5.0,
        )

        assert kf.index == 0
        assert kf.timestamp_ms == 1000
        assert kf.cumulative_distance == 5.0
        assert kf.pointmap is None
        assert kf.scale_factor is None


class TestKeyframeStackManager:
    """Test suite for keyframe stack manager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh keyframe manager."""
        return KeyframeStackManager(
            keyframe_interval_m=2.0,
            keyframe_interval_s=3.0,
        )

    def test_initial_state(self, manager):
        """Test initial state of manager."""
        assert manager.get_stack_size() == 0
        assert manager.is_empty()
        assert manager.cumulative_distance == 0.0

    def test_first_frame_becomes_keyframe(self, manager):
        """Test that first frame always becomes keyframe."""
        frame = make_frame()
        telem = make_telemetry(timestamp_ms=1000)

        kf = manager.process_frame(frame, telem)

        assert kf is not None
        assert kf.index == 0
        assert manager.get_stack_size() == 1

    def test_distance_based_keyframe_push(self, manager):
        """Test keyframe is pushed after moving threshold distance."""
        # First frame
        frame1 = make_frame()
        telem1 = make_telemetry(timestamp_ms=1000, vx=2.0)
        manager.process_frame(frame1, telem1)

        # Move 2m forward (vx=2.0 m/s for 1 second)
        frame2 = make_frame()
        telem2 = make_telemetry(timestamp_ms=2000, vx=2.0)
        kf2 = manager.process_frame(frame2, telem2)

        assert kf2 is not None
        assert kf2.index == 1
        assert manager.get_stack_size() == 2

    def test_time_based_keyframe_push(self, manager):
        """Test keyframe is pushed after threshold time even without movement."""
        # First frame
        frame1 = make_frame()
        telem1 = make_telemetry(timestamp_ms=1000, vx=0.0)
        manager.process_frame(frame1, telem1)

        # Wait 3+ seconds without moving
        frame2 = make_frame()
        telem2 = make_telemetry(timestamp_ms=4500, vx=0.0)  # 3.5 seconds later
        kf2 = manager.process_frame(frame2, telem2)

        assert kf2 is not None
        assert kf2.index == 1

    def test_no_keyframe_before_threshold(self, manager):
        """Test no keyframe pushed before reaching thresholds."""
        # First frame
        frame1 = make_frame()
        telem1 = make_telemetry(timestamp_ms=1000, vx=0.5)
        manager.process_frame(frame1, telem1)

        # Small movement, short time
        frame2 = make_frame()
        telem2 = make_telemetry(timestamp_ms=1500, vx=0.5)  # 0.25m moved, 0.5s elapsed
        kf2 = manager.process_frame(frame2, telem2)

        assert kf2 is None
        assert manager.get_stack_size() == 1

    def test_force_keyframe(self, manager):
        """Test forcing a keyframe regardless of thresholds."""
        frame1 = make_frame()
        telem1 = make_telemetry(timestamp_ms=1000)
        manager.process_frame(frame1, telem1)

        frame2 = make_frame()
        telem2 = make_telemetry(timestamp_ms=1100)  # Very short time
        kf2 = manager.process_frame(frame2, telem2, force_keyframe=True)

        assert kf2 is not None

    def test_cumulative_distance_tracking(self, manager):
        """Test cumulative distance is tracked correctly."""
        # Move at 1 m/s for 5 seconds
        frame1 = make_frame()
        telem1 = make_telemetry(timestamp_ms=0, vx=1.0)
        manager.process_frame(frame1, telem1)

        for i in range(1, 6):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000, vx=1.0)
            manager.process_frame(frame, telem)

        # Should have accumulated ~5m
        assert abs(manager.cumulative_distance - 5.0) < 0.1

    def test_get_target_keyframe(self, manager):
        """Test getting keyframe by index."""
        # Add multiple keyframes
        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 4000, vx=1.0)
            manager.process_frame(frame, telem, force_keyframe=True)

        # Get by positive index
        kf0 = manager.get_target_keyframe(0)
        assert kf0.index == 0

        # Get by negative index
        kf_last = manager.get_target_keyframe(-1)
        assert kf_last.index == 2

    def test_pop_keyframe(self, manager):
        """Test popping keyframes from stack."""
        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 4000)
            manager.process_frame(frame, telem, force_keyframe=True)

        assert manager.get_stack_size() == 3

        kf = manager.pop_keyframe()
        assert kf.index == 2
        assert manager.get_stack_size() == 2

        kf = manager.pop_keyframe()
        assert kf.index == 1
        assert manager.get_stack_size() == 1

    def test_pop_empty_stack(self, manager):
        """Test popping from empty stack returns None."""
        result = manager.pop_keyframe()
        assert result is None

    def test_inter_keyframe_distance(self, manager):
        """Test computing distance between keyframes."""
        # Add keyframes at known distances
        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000, vx=2.0)
            manager.process_frame(frame, telem, force_keyframe=True)

        dist_01 = manager.get_inter_keyframe_distance(0, 1)
        dist_12 = manager.get_inter_keyframe_distance(1, 2)

        # Each interval should be ~2m (2 m/s * 1s)
        assert abs(dist_01 - 2.0) < 0.1
        assert abs(dist_12 - 2.0) < 0.1

    def test_scale_factor_storage(self, manager):
        """Test storing scale factors."""
        frame = make_frame()
        telem = make_telemetry(timestamp_ms=1000)
        manager.process_frame(frame, telem)

        manager.set_scale_factor(0, 2.5)
        assert manager.stack[0].scale_factor == 2.5

    def test_global_scale_computation(self, manager):
        """Test computing global scale from multiple keyframes."""
        for i in range(5):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000)
            manager.process_frame(frame, telem, force_keyframe=True)
            if i > 0:
                manager.set_scale_factor(i, 2.0 + i * 0.1)

        global_scale = manager.compute_global_scale()
        # Median of [2.1, 2.2, 2.3, 2.4] = 2.25
        assert abs(global_scale - 2.25) < 0.1

    def test_clear(self, manager):
        """Test clearing the manager."""
        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000, vx=1.0)
            manager.process_frame(frame, telem, force_keyframe=True)

        assert manager.get_stack_size() == 3
        assert manager.cumulative_distance > 0

        manager.clear()

        assert manager.get_stack_size() == 0
        assert manager.cumulative_distance == 0.0
        assert manager.is_empty()

    def test_iteration(self, manager):
        """Test iterating over keyframes."""
        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000)
            manager.process_frame(frame, telem, force_keyframe=True)

        indices = [kf.index for kf in manager]
        assert indices == [0, 1, 2]

    def test_len(self, manager):
        """Test len() on manager."""
        assert len(manager) == 0

        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000)
            manager.process_frame(frame, telem, force_keyframe=True)

        assert len(manager) == 3

    def test_getitem(self, manager):
        """Test indexing manager."""
        for i in range(3):
            frame = make_frame()
            telem = make_telemetry(timestamp_ms=i * 1000)
            manager.process_frame(frame, telem, force_keyframe=True)

        assert manager[0].index == 0
        assert manager[1].index == 1
        assert manager[-1].index == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

