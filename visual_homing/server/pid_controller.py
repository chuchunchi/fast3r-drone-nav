"""PID Controller for velocity command generation."""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDGains:
    """PID controller gains."""

    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain


class PIDController:
    """
    PID controller for single-axis control.

    Implements a discrete PID controller with anti-windup and
    derivative filtering for smooth velocity command generation.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limits: tuple = (-float("inf"), float("inf")),
        integral_limits: tuple = (-float("inf"), float("inf")),
        derivative_filter: float = 0.1,
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            output_limits: (min, max) output limits.
            integral_limits: (min, max) integral term limits (anti-windup).
            derivative_filter: Low-pass filter coefficient for derivative (0-1).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_filter = derivative_filter

        # State
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._filtered_derivative = 0.0

    def compute(self, error: float, dt: Optional[float] = None) -> float:
        """
        Compute PID output.

        Args:
            error: Current error (setpoint - measurement).
            dt: Time step in seconds. If None, uses actual elapsed time.

        Returns:
            PID output value.
        """
        current_time = time.time()

        # Compute dt
        if dt is None:
            if self._last_time is None:
                dt = 0.1  # Default for first call
            else:
                dt = current_time - self._last_time
        self._last_time = current_time

        # Avoid division by zero
        if dt <= 0:
            dt = 0.001

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = max(
            self.integral_limits[0],
            min(self.integral_limits[1], self._integral),
        )
        i_term = self.ki * self._integral

        # Derivative term with filtering
        if self._last_error is not None:
            raw_derivative = (error - self._last_error) / dt
            # Low-pass filter
            self._filtered_derivative = (
                self.derivative_filter * raw_derivative
                + (1 - self.derivative_filter) * self._filtered_derivative
            )
        else:
            self._filtered_derivative = 0.0
        d_term = self.kd * self._filtered_derivative

        self._last_error = error

        # Compute output
        output = p_term + i_term + d_term

        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        return output

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = None
        self._last_time = None
        self._filtered_derivative = 0.0

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_output_limits(self, min_val: float, max_val: float) -> None:
        """Set output limits."""
        self.output_limits = (min_val, max_val)

    def set_integral_limits(self, min_val: float, max_val: float) -> None:
        """Set integral term limits (anti-windup)."""
        self.integral_limits = (min_val, max_val)

    def get_state(self) -> dict:
        """Get current controller state for debugging."""
        return {
            "integral": self._integral,
            "last_error": self._last_error,
            "filtered_derivative": self._filtered_derivative,
        }


class MultiAxisPIDController:
    """
    Multi-axis PID controller for drone control.

    Manages separate PID controllers for forward, lateral, vertical,
    and yaw axes with coordinated reset and update functionality.
    """

    def __init__(
        self,
        forward_gains: tuple = (0.5, 0.01, 0.1),
        lateral_gains: tuple = (0.5, 0.01, 0.1),
        vertical_gains: tuple = (0.3, 0.01, 0.05),
        yaw_gains: tuple = (1.0, 0.0, 0.2),
        velocity_limits: dict = None,
    ):
        """
        Initialize multi-axis PID controller.

        Args:
            forward_gains: (kp, ki, kd) for forward axis.
            lateral_gains: (kp, ki, kd) for lateral axis.
            vertical_gains: (kp, ki, kd) for vertical axis.
            yaw_gains: (kp, ki, kd) for yaw axis.
            velocity_limits: Dictionary with max velocities.
        """
        if velocity_limits is None:
            velocity_limits = {
                "forward": 2.0,
                "lateral": 2.0,
                "vertical": 1.0,
                "yaw": 30.0,
            }

        self.forward = PIDController(
            kp=forward_gains[0],
            ki=forward_gains[1],
            kd=forward_gains[2],
            output_limits=(-velocity_limits["forward"], velocity_limits["forward"]),
        )

        self.lateral = PIDController(
            kp=lateral_gains[0],
            ki=lateral_gains[1],
            kd=lateral_gains[2],
            output_limits=(-velocity_limits["lateral"], velocity_limits["lateral"]),
        )

        self.vertical = PIDController(
            kp=vertical_gains[0],
            ki=vertical_gains[1],
            kd=vertical_gains[2],
            output_limits=(-velocity_limits["vertical"], velocity_limits["vertical"]),
        )

        self.yaw = PIDController(
            kp=yaw_gains[0],
            ki=yaw_gains[1],
            kd=yaw_gains[2],
            output_limits=(-velocity_limits["yaw"], velocity_limits["yaw"]),
        )

    def compute(
        self,
        error_forward: float,
        error_lateral: float,
        error_vertical: float,
        error_yaw: float,
        dt: Optional[float] = None,
    ) -> dict:
        """
        Compute velocity commands for all axes.

        Args:
            error_forward: Forward error in meters.
            error_lateral: Lateral error in meters.
            error_vertical: Vertical error in meters.
            error_yaw: Yaw error in degrees.
            dt: Time step in seconds.

        Returns:
            Dictionary with velocity commands.
        """
        return {
            "pitch_velocity": self.forward.compute(error_forward, dt),
            "roll_velocity": self.lateral.compute(error_lateral, dt),
            "vertical_velocity": self.vertical.compute(error_vertical, dt),
            "yaw_rate": self.yaw.compute(error_yaw, dt),
        }

    def reset(self) -> None:
        """Reset all controllers."""
        self.forward.reset()
        self.lateral.reset()
        self.vertical.reset()
        self.yaw.reset()

    def reset_position(self) -> None:
        """Reset position controllers (forward, lateral, vertical) but keep yaw."""
        self.forward.reset()
        self.lateral.reset()
        self.vertical.reset()


