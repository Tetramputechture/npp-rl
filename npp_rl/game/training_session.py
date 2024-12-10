"""Training session tracker for monitoring training progress."""
from typing import List, Tuple, Deque
from collections import deque


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
        self._reward_history: List[float] = []
        self._current_episode_positions: List[Tuple[float, float]] = []
        self._episode_paths: Deque[List[Tuple[float, float]]] = deque(
            maxlen=10)  # Keep last 10 episodes

        # Current episode step data
        self._current_episode_actions: List[int] = []
        self._current_episode_step_rewards: List[float] = []

        # Historical step data
        self._historical_actions: List[List[int]] = []
        self._historical_step_rewards: List[List[float]] = []

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

    @property
    def reward_history(self) -> List[float]:
        """History of episode rewards."""
        return self._reward_history

    @property
    def current_episode_positions(self) -> List[Tuple[float, float]]:
        """Current episode's position history."""
        return self._current_episode_positions

    @property
    def episode_paths(self) -> List[List[Tuple[float, float]]]:
        """Last 10 episodes' position paths."""
        return list(self._episode_paths)

    @property
    def current_episode_actions(self) -> List[int]:
        """Current episode's action history."""
        return self._current_episode_actions

    @property
    def current_episode_step_rewards(self) -> List[float]:
        """Current episode's per-step reward history."""
        return self._current_episode_step_rewards

    @property
    def historical_actions(self) -> List[List[int]]:
        """History of actions taken during training for all episodes."""
        return self._historical_actions

    @property
    def historical_step_rewards(self) -> List[List[float]]:
        """History of per-step rewards for all episodes."""
        return self._historical_step_rewards

    def start_session(self):
        """Start a new training session."""
        self._is_training = True
        self._episode_count = 0
        self._current_episode_reward = 0.0
        self._reward_history = []
        self._current_episode_positions = []
        self._episode_paths.clear()
        self._current_episode_actions = []
        self._current_episode_step_rewards = []
        self._historical_actions = []
        self._historical_step_rewards = []

    def end_session(self):
        """End the current training session."""
        self._is_training = False

    def increment_episode(self):
        """Increment the episode counter and reset episode data."""
        self._episode_count += 1
        self._reward_history.append(self._current_episode_reward)

        # Store current episode path and start new one
        if self._current_episode_positions:
            self._episode_paths.append(self._current_episode_positions)

        # Store current episode step data
        if self._current_episode_actions:
            self._historical_actions.append(self._current_episode_actions)
            self._historical_step_rewards.append(
                self._current_episode_step_rewards)

        # Reset current episode data
        self._current_episode_positions = []
        self._current_episode_reward = 0.0
        self._current_episode_actions = []
        self._current_episode_step_rewards = []

    def add_reward(self, reward: float):
        """Add reward to current episode total and step history.

        Args:
            reward (float): The reward received for the current step
        """
        self._current_episode_reward += reward
        self._current_episode_step_rewards.append(reward)

    def update_best_reward(self, reward: float):
        """Update the best reward if the current reward is higher.

        Args:
            reward (float): The reward to compare against the best reward
        """
        if reward > self._best_reward:
            self._best_reward = reward

    def add_position(self, x: float, y: float):
        """Add a position to the position history.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self._current_episode_positions.append((x, y))

    def add_action(self, action: int):
        """Add an action to the current episode's action history.

        Args:
            action (int): The action taken by the agent
        """
        self._current_episode_actions.append(action)


# Global training session instance
training_session = TrainingSession()
