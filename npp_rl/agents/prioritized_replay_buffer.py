import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from collections import deque
import random


class PrioritizedReplayBuffer:
    """A prioritized experience replay buffer specifically designed for PPO."""

    def __init__(self,
                 buffer_size: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        Initialize the buffer.

        Args:
            buffer_size: Maximum size of buffer
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta: Importance sampling factor
            beta_increment: How much to increase beta over time
            epsilon: Small constant to avoid zero priorities
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

        # Main storage
        self.observations = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_observations = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)

        # Additional PPO-specific storage
        self.log_probs = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)
        self.advantages = deque(maxlen=buffer_size)
        self.returns = deque(maxlen=buffer_size)

        self._next_idx = 0
        self.size = 0

    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            done: bool,
            log_prob: float,
            value: float,
            advantage: float = 0.0,
            returns: float = 0.0) -> None:
        """Add a new experience to memory."""

        # Use max priority for new experiences
        priority = self.max_priority ** self.alpha

        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.priorities.append(priority)

        # PPO-specific data
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.advantages.append(advantage)
        self.returns.append(returns)

        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences with importance sampling weights.

        Returns:
            Tuple containing:
            - Dictionary of sampled transitions
            - Array of indices that were sampled
            - Array of importance sampling weights
        """
        if self.size < batch_size:
            raise ValueError("Not enough experiences in buffer")

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Gather sampled experiences
        batch = {
            'observations': np.array([self.observations[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_observations': np.array([self.next_observations[i] for i in indices]),
            'dones': np.array([self.dones[i] for i in indices]),
            'log_probs': np.array([self.log_probs[i] for i in indices]),
            'values': np.array([self.values[i] for i in indices]),
            'advantages': np.array([self.advantages[i] for i in indices]),
            'returns': np.array([self.returns[i] for i in indices])
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def clear(self) -> None:
        """Clear the buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.dones.clear()
        self.priorities.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
        self.size = 0
        self._next_idx = 0
        self.max_priority = 1.0
