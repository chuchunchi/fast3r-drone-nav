"""State machine for system phase management."""

import logging
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System states for the visual homing workflow."""

    IDLE = auto()  # Waiting for user to start recording
    RECORDING = auto()  # Recording keyframes during TEACH phase
    ARMED = auto()  # Recording complete, ready to home
    HOMING = auto()  # Actively navigating back to start
    COMPLETED = auto()  # Successfully returned to start
    ERROR = auto()  # Error state, requires user intervention


class StateTransition:
    """Represents a valid state transition."""

    def __init__(
        self,
        from_state: SystemState,
        to_state: SystemState,
        condition: Optional[Callable[[], bool]] = None,
        on_transition: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize state transition.

        Args:
            from_state: Source state.
            to_state: Target state.
            condition: Optional condition that must be True for transition.
            on_transition: Optional callback executed on transition.
        """
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.on_transition = on_transition


class StateMachine:
    """
    State machine for managing system phases.

    Handles transitions between IDLE, RECORDING, ARMED, HOMING, and COMPLETED
    states with validation and callbacks.
    """

    def __init__(self):
        """Initialize state machine in IDLE state."""
        self._state = SystemState.IDLE
        self._transitions: Dict[SystemState, List[StateTransition]] = {}
        self._state_callbacks: Dict[SystemState, List[Callable]] = {}
        self._any_state_callbacks: List[Callable] = []

        # Define valid transitions
        self._setup_transitions()

    def _setup_transitions(self) -> None:
        """Setup valid state transitions."""
        # IDLE -> RECORDING: User starts recording
        self._add_transition(SystemState.IDLE, SystemState.RECORDING)

        # RECORDING -> ARMED: User stops recording
        self._add_transition(SystemState.RECORDING, SystemState.ARMED)

        # RECORDING -> RECORDING: Keyframe pushed (stays in recording)
        # This is implicit - no state change

        # ARMED -> HOMING: User triggers return
        self._add_transition(SystemState.ARMED, SystemState.HOMING)

        # HOMING -> HOMING: Next keyframe (stays in homing)
        # This is implicit - no state change

        # HOMING -> COMPLETED: Stack empty (home reached)
        self._add_transition(SystemState.HOMING, SystemState.COMPLETED)

        # Any state -> IDLE: Reset
        for state in SystemState:
            if state != SystemState.IDLE:
                self._add_transition(state, SystemState.IDLE)

        # Any state -> ERROR: Error occurred
        for state in SystemState:
            if state != SystemState.ERROR:
                self._add_transition(state, SystemState.ERROR)

    def _add_transition(
        self,
        from_state: SystemState,
        to_state: SystemState,
        condition: Optional[Callable[[], bool]] = None,
        on_transition: Optional[Callable[[], None]] = None,
    ) -> None:
        """Add a valid transition."""
        if from_state not in self._transitions:
            self._transitions[from_state] = []

        self._transitions[from_state].append(
            StateTransition(from_state, to_state, condition, on_transition)
        )

    @property
    def state(self) -> SystemState:
        """Get current state."""
        return self._state

    def can_transition_to(self, target: SystemState) -> bool:
        """Check if transition to target state is valid."""
        if self._state not in self._transitions:
            return False

        for transition in self._transitions[self._state]:
            if transition.to_state == target:
                if transition.condition is None or transition.condition():
                    return True
        return False

    def transition_to(self, target: SystemState) -> bool:
        """
        Attempt to transition to target state.

        Args:
            target: Target state.

        Returns:
            True if transition succeeded, False otherwise.
        """
        if not self.can_transition_to(target):
            logger.warning(
                f"Invalid transition: {self._state.name} -> {target.name}"
            )
            return False

        # Find and execute transition
        for transition in self._transitions[self._state]:
            if transition.to_state == target:
                if transition.condition is None or transition.condition():
                    old_state = self._state
                    self._state = target

                    # Execute transition callback
                    if transition.on_transition:
                        transition.on_transition()

                    # Execute state entry callbacks
                    self._notify_state_change(old_state, target)

                    logger.info(f"State transition: {old_state.name} -> {target.name}")
                    return True

        return False

    def _notify_state_change(
        self, old_state: SystemState, new_state: SystemState
    ) -> None:
        """Notify callbacks of state change."""
        # Call specific state callbacks
        if new_state in self._state_callbacks:
            for callback in self._state_callbacks[new_state]:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

        # Call any-state callbacks
        for callback in self._any_state_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def on_state(
        self, state: SystemState, callback: Callable[[SystemState, SystemState], None]
    ) -> None:
        """
        Register callback for entering a specific state.

        Args:
            state: State to watch.
            callback: Function(old_state, new_state) to call.
        """
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)

    def on_any_state_change(
        self, callback: Callable[[SystemState, SystemState], None]
    ) -> None:
        """
        Register callback for any state change.

        Args:
            callback: Function(old_state, new_state) to call.
        """
        self._any_state_callbacks.append(callback)

    def reset(self) -> None:
        """Reset state machine to IDLE."""
        old_state = self._state
        self._state = SystemState.IDLE
        self._notify_state_change(old_state, SystemState.IDLE)
        logger.info("State machine reset to IDLE")

    # Convenience methods for common transitions

    def start_recording(self) -> bool:
        """Start recording (IDLE -> RECORDING)."""
        return self.transition_to(SystemState.RECORDING)

    def stop_recording(self) -> bool:
        """Stop recording (RECORDING -> ARMED)."""
        return self.transition_to(SystemState.ARMED)

    def start_homing(self) -> bool:
        """Start homing (ARMED -> HOMING)."""
        return self.transition_to(SystemState.HOMING)

    def complete_homing(self) -> bool:
        """Complete homing (HOMING -> COMPLETED)."""
        return self.transition_to(SystemState.COMPLETED)

    def set_error(self) -> bool:
        """Set error state."""
        return self.transition_to(SystemState.ERROR)

    def is_idle(self) -> bool:
        """Check if in IDLE state."""
        return self._state == SystemState.IDLE

    def is_recording(self) -> bool:
        """Check if in RECORDING state."""
        return self._state == SystemState.RECORDING

    def is_armed(self) -> bool:
        """Check if in ARMED state."""
        return self._state == SystemState.ARMED

    def is_homing(self) -> bool:
        """Check if in HOMING state."""
        return self._state == SystemState.HOMING

    def is_completed(self) -> bool:
        """Check if in COMPLETED state."""
        return self._state == SystemState.COMPLETED

    def is_error(self) -> bool:
        """Check if in ERROR state."""
        return self._state == SystemState.ERROR


