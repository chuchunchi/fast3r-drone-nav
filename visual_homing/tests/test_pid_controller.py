"""Tests for PID controller."""

import time

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.pid_controller import PIDController, MultiAxisPIDController


class TestPIDController:
    """Test suite for single-axis PID controller."""

    def test_proportional_only(self):
        """Test proportional-only control."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)

        # Error of 1.0 with kp=2.0 should give output 2.0
        output = pid.compute(1.0, dt=0.1)
        assert abs(output - 2.0) < 0.01

        # Error of -0.5 should give -1.0
        pid.reset()
        output = pid.compute(-0.5, dt=0.1)
        assert abs(output - (-1.0)) < 0.01

    def test_integral_accumulation(self):
        """Test that integral term accumulates over time."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)

        # Constant error of 1.0 for 3 time steps
        output1 = pid.compute(1.0, dt=0.1)  # integral = 0.1
        output2 = pid.compute(1.0, dt=0.1)  # integral = 0.2
        output3 = pid.compute(1.0, dt=0.1)  # integral = 0.3

        assert abs(output1 - 0.1) < 0.01
        assert abs(output2 - 0.2) < 0.01
        assert abs(output3 - 0.3) < 0.01

    def test_derivative_response(self):
        """Test derivative term responds to error changes."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0, derivative_filter=1.0)

        # First call establishes baseline
        pid.compute(0.0, dt=0.1)

        # Error jumps from 0 to 1, derivative = (1-0)/0.1 = 10
        output = pid.compute(1.0, dt=0.1)
        assert output > 5.0  # Should be significant positive derivative

    def test_output_limits(self):
        """Test output clamping."""
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0))

        # Large positive error should clamp to 1.0
        output = pid.compute(100.0, dt=0.1)
        assert output == 1.0

        # Large negative error should clamp to -1.0
        pid.reset()
        output = pid.compute(-100.0, dt=0.1)
        assert output == -1.0

    def test_integral_anti_windup(self):
        """Test integral anti-windup limits."""
        pid = PIDController(
            kp=0.0, ki=1.0, kd=0.0, integral_limits=(-0.5, 0.5)
        )

        # Accumulate integral beyond limit
        for _ in range(100):
            pid.compute(1.0, dt=0.1)

        # Integral should be clamped at 0.5
        state = pid.get_state()
        assert state["integral"] == 0.5

    def test_reset(self):
        """Test controller reset."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)

        # Accumulate some state
        pid.compute(1.0, dt=0.1)
        pid.compute(2.0, dt=0.1)

        # Reset and verify clean state
        pid.reset()
        state = pid.get_state()

        assert state["integral"] == 0.0
        assert state["last_error"] is None
        assert state["filtered_derivative"] == 0.0

    def test_zero_error(self):
        """Test response to zero error."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        output = pid.compute(0.0, dt=0.1)
        assert output == 0.0

    def test_gain_update(self):
        """Test dynamic gain update."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)

        output1 = pid.compute(1.0, dt=0.1)
        assert abs(output1 - 1.0) < 0.01

        pid.set_gains(kp=2.0, ki=0.0, kd=0.0)
        pid.reset()

        output2 = pid.compute(1.0, dt=0.1)
        assert abs(output2 - 2.0) < 0.01


class TestMultiAxisPIDController:
    """Test suite for multi-axis PID controller."""

    def test_initialization(self):
        """Test multi-axis controller initialization."""
        pid = MultiAxisPIDController(
            forward_gains=(0.5, 0.01, 0.1),
            lateral_gains=(0.5, 0.01, 0.1),
            vertical_gains=(0.3, 0.01, 0.05),
            yaw_gains=(1.0, 0.0, 0.2),
        )

        assert pid.forward.kp == 0.5
        assert pid.lateral.kp == 0.5
        assert pid.vertical.kp == 0.3
        assert pid.yaw.kp == 1.0

    def test_compute_all_axes(self):
        """Test computing commands for all axes."""
        pid = MultiAxisPIDController(
            forward_gains=(1.0, 0.0, 0.0),
            lateral_gains=(1.0, 0.0, 0.0),
            vertical_gains=(1.0, 0.0, 0.0),
            yaw_gains=(1.0, 0.0, 0.0),
        )

        cmd = pid.compute(
            error_forward=2.0,
            error_lateral=1.0,
            error_vertical=-0.5,
            error_yaw=10.0,
            dt=0.1,
        )

        assert "pitch_velocity" in cmd
        assert "roll_velocity" in cmd
        assert "vertical_velocity" in cmd
        assert "yaw_rate" in cmd

        assert abs(cmd["pitch_velocity"] - 2.0) < 0.01
        assert abs(cmd["roll_velocity"] - 1.0) < 0.01
        assert abs(cmd["vertical_velocity"] - (-0.5)) < 0.01
        assert abs(cmd["yaw_rate"] - 10.0) < 0.01

    def test_velocity_limits(self):
        """Test velocity limits are applied."""
        pid = MultiAxisPIDController(
            forward_gains=(10.0, 0.0, 0.0),
            lateral_gains=(10.0, 0.0, 0.0),
            vertical_gains=(10.0, 0.0, 0.0),
            yaw_gains=(10.0, 0.0, 0.0),
            velocity_limits={
                "forward": 2.0,
                "lateral": 2.0,
                "vertical": 1.0,
                "yaw": 30.0,
            },
        )

        cmd = pid.compute(100.0, 100.0, 100.0, 100.0, dt=0.1)

        assert abs(cmd["pitch_velocity"]) <= 2.0
        assert abs(cmd["roll_velocity"]) <= 2.0
        assert abs(cmd["vertical_velocity"]) <= 1.0
        assert abs(cmd["yaw_rate"]) <= 30.0

    def test_reset_all(self):
        """Test resetting all controllers."""
        pid = MultiAxisPIDController()

        # Accumulate state
        pid.compute(1.0, 1.0, 1.0, 1.0, dt=0.1)
        pid.compute(1.0, 1.0, 1.0, 1.0, dt=0.1)

        pid.reset()

        # Verify all are reset
        assert pid.forward.get_state()["integral"] == 0.0
        assert pid.lateral.get_state()["integral"] == 0.0
        assert pid.vertical.get_state()["integral"] == 0.0
        assert pid.yaw.get_state()["integral"] == 0.0

    def test_reset_position_only(self):
        """Test resetting only position controllers."""
        pid = MultiAxisPIDController(
            forward_gains=(0.0, 1.0, 0.0),
            lateral_gains=(0.0, 1.0, 0.0),
            vertical_gains=(0.0, 1.0, 0.0),
            yaw_gains=(0.0, 1.0, 0.0),
        )

        # Accumulate integral
        for _ in range(5):
            pid.compute(1.0, 1.0, 1.0, 1.0, dt=0.1)

        # Reset position only
        pid.reset_position()

        # Position controllers should be reset
        assert pid.forward.get_state()["integral"] == 0.0
        assert pid.lateral.get_state()["integral"] == 0.0
        assert pid.vertical.get_state()["integral"] == 0.0

        # Yaw should NOT be reset
        assert pid.yaw.get_state()["integral"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

