"""Tests for state machine."""

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visual_homing.server.state_machine import StateMachine, SystemState


class TestStateMachine:
    """Test suite for the state machine."""

    @pytest.fixture
    def sm(self):
        """Create a fresh state machine."""
        return StateMachine()

    def test_initial_state(self, sm):
        """Test that state machine starts in IDLE."""
        assert sm.state == SystemState.IDLE
        assert sm.is_idle()

    def test_idle_to_recording(self, sm):
        """Test transition from IDLE to RECORDING."""
        assert sm.can_transition_to(SystemState.RECORDING)
        assert sm.start_recording()
        assert sm.state == SystemState.RECORDING
        assert sm.is_recording()

    def test_recording_to_armed(self, sm):
        """Test transition from RECORDING to ARMED."""
        sm.start_recording()
        assert sm.can_transition_to(SystemState.ARMED)
        assert sm.stop_recording()
        assert sm.state == SystemState.ARMED
        assert sm.is_armed()

    def test_armed_to_homing(self, sm):
        """Test transition from ARMED to HOMING."""
        sm.start_recording()
        sm.stop_recording()
        assert sm.can_transition_to(SystemState.HOMING)
        assert sm.start_homing()
        assert sm.state == SystemState.HOMING
        assert sm.is_homing()

    def test_homing_to_completed(self, sm):
        """Test transition from HOMING to COMPLETED."""
        sm.start_recording()
        sm.stop_recording()
        sm.start_homing()
        assert sm.can_transition_to(SystemState.COMPLETED)
        assert sm.complete_homing()
        assert sm.state == SystemState.COMPLETED
        assert sm.is_completed()

    def test_full_workflow(self, sm):
        """Test complete workflow: IDLE → RECORDING → ARMED → HOMING → COMPLETED."""
        assert sm.is_idle()

        # Start recording
        assert sm.start_recording()
        assert sm.is_recording()

        # Stop recording (arm)
        assert sm.stop_recording()
        assert sm.is_armed()

        # Start homing
        assert sm.start_homing()
        assert sm.is_homing()

        # Complete homing
        assert sm.complete_homing()
        assert sm.is_completed()

    def test_invalid_transition_idle_to_armed(self, sm):
        """Test invalid direct transition from IDLE to ARMED."""
        assert not sm.can_transition_to(SystemState.ARMED)
        assert not sm.transition_to(SystemState.ARMED)
        assert sm.is_idle()  # State unchanged

    def test_invalid_transition_idle_to_homing(self, sm):
        """Test invalid direct transition from IDLE to HOMING."""
        assert not sm.can_transition_to(SystemState.HOMING)
        assert not sm.transition_to(SystemState.HOMING)
        assert sm.is_idle()

    def test_invalid_transition_recording_to_homing(self, sm):
        """Test invalid direct transition from RECORDING to HOMING."""
        sm.start_recording()
        assert not sm.can_transition_to(SystemState.HOMING)
        assert not sm.transition_to(SystemState.HOMING)
        assert sm.is_recording()

    def test_reset_from_any_state(self, sm):
        """Test that reset works from any state."""
        # Reset from IDLE
        sm.reset()
        assert sm.is_idle()

        # Reset from RECORDING
        sm.start_recording()
        sm.reset()
        assert sm.is_idle()

        # Reset from ARMED
        sm.start_recording()
        sm.stop_recording()
        sm.reset()
        assert sm.is_idle()

        # Reset from HOMING
        sm.start_recording()
        sm.stop_recording()
        sm.start_homing()
        sm.reset()
        assert sm.is_idle()

        # Reset from COMPLETED
        sm.start_recording()
        sm.stop_recording()
        sm.start_homing()
        sm.complete_homing()
        sm.reset()
        assert sm.is_idle()

    def test_error_state(self, sm):
        """Test transition to ERROR state."""
        sm.start_recording()
        assert sm.set_error()
        assert sm.is_error()
        assert sm.state == SystemState.ERROR

    def test_reset_from_error(self, sm):
        """Test reset from ERROR state."""
        sm.start_recording()
        sm.set_error()
        assert sm.is_error()
        sm.reset()
        assert sm.is_idle()

    def test_state_callbacks(self, sm):
        """Test state change callbacks."""
        callback_data = {"called": False, "old_state": None, "new_state": None}

        def on_recording(old_state, new_state):
            callback_data["called"] = True
            callback_data["old_state"] = old_state
            callback_data["new_state"] = new_state

        sm.on_state(SystemState.RECORDING, on_recording)
        sm.start_recording()

        assert callback_data["called"]
        assert callback_data["old_state"] == SystemState.IDLE
        assert callback_data["new_state"] == SystemState.RECORDING

    def test_any_state_callback(self, sm):
        """Test callback for any state change."""
        transitions = []

        def on_any_change(old_state, new_state):
            transitions.append((old_state, new_state))

        sm.on_any_state_change(on_any_change)

        sm.start_recording()
        sm.stop_recording()
        sm.start_homing()

        assert len(transitions) == 3
        assert transitions[0] == (SystemState.IDLE, SystemState.RECORDING)
        assert transitions[1] == (SystemState.RECORDING, SystemState.ARMED)
        assert transitions[2] == (SystemState.ARMED, SystemState.HOMING)

    def test_convenience_methods(self, sm):
        """Test all convenience state check methods."""
        assert sm.is_idle()
        assert not sm.is_recording()
        assert not sm.is_armed()
        assert not sm.is_homing()
        assert not sm.is_completed()
        assert not sm.is_error()

        sm.start_recording()
        assert not sm.is_idle()
        assert sm.is_recording()

        sm.stop_recording()
        assert sm.is_armed()

        sm.start_homing()
        assert sm.is_homing()

        sm.complete_homing()
        assert sm.is_completed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

