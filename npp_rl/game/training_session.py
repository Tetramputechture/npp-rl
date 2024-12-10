"""Training session tracker for monitoring training progress."""


class TrainingSession:
    """Tracks information about the current training session.

    This class maintains telemetry data about the ongoing training session,
    including episode count and current reward.
    It does not affect the training process itself, serving purely as
    a monitoring tool.
    """

    def __init__(self):
        self._is_training = False
        self._episode_count = 0
        self._current_episode_reward = 0.0
        self._best_reward = 0.0

    @property
    def is_training(self) -> bool:
        """Whether a training session is currently ongoing."""
        return self._is_training

    @property
    def episode_count(self) -> int:
        """Number of episodes completed in current session."""
        return self._episode_count

    @property
    def current_episode_reward(self) -> float:
        """Total reward accumulated in current episode."""
        return self._current_episode_reward

    @property
    def best_reward(self) -> float:
        """Best reward accumulated in current session."""
        return self._best_reward

    def start_session(self):
        """Start a new training session."""
        self._is_training = True
        self._episode_count = 0
        self._current_episode_reward = 0.0

    def end_session(self):
        """End the current training session."""
        self._is_training = False

    def increment_episode(self):
        """Increment the episode counter and reset episode reward."""
        self._episode_count += 1
        self._current_episode_reward = 0.0

    def add_reward(self, reward: float):
        """Add reward to current episode total."""
        self._current_episode_reward += reward

    def update_best_reward(self, reward: float):
        """Update the best reward if the current reward is higher.

        Args:
            reward (float): The reward to compare against the best reward
        """
        if reward > self._best_reward:
            self._best_reward = reward


# Global training session instance
training_session = TrainingSession()
