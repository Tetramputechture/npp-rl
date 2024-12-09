from collections import deque
from npp_rl.environments.constants import TIMESTEP
import numpy as np


class ActionTracker:
    """Tracks action history and durations for the agent."""

    def __init__(self, history_size: int = 4):
        """Initialize action tracker.

        Args:
            history_size: Number of previous actions to track
        """
        self.history_size = history_size
        self.action_history = deque(maxlen=history_size)
        self.action_duration = deque(maxlen=history_size)
        self.current_action = None
        self.current_duration = 0.0

    def update(self, action: int) -> None:
        """Update action history and durations.

        Args:
            action: Current action taken by the agent
        """
        if action == self.current_action:
            # If same action, increase duration
            self.current_duration += TIMESTEP
        else:
            # If new action, store previous action and duration
            if self.current_action is not None:
                self.action_history.append(self.current_action)
                self.action_duration.append(self.current_duration)

            # Start tracking new action
            self.current_action = action
            self.current_duration = TIMESTEP

    def get_features(self) -> np.ndarray:
        """Get action history and duration features.

        Returns:
            Array containing action history and durations, normalized
        """
        # Fill history if needed
        while len(self.action_history) < self.history_size:
            self.action_history.append(0)  # NOOP
            self.action_duration.append(0.0)

        # Normalize actions to [0, 1]
        # 6 possible actions (0-5)
        action_features = np.array(list(self.action_history)) / 5.0

        # Normalize durations to [0, 1] assuming max duration of 10 seconds
        duration_features = np.array(list(self.action_duration)) / 10.0
        duration_features = np.clip(duration_features, 0, 1)

        return np.concatenate([action_features, duration_features])

    def reset(self) -> None:
        """Reset tracker state."""
        self.action_history.clear()
        self.action_duration.clear()
        self.current_action = None
        self.current_duration = 0.0
