"""Dueling architecture for value function decomposition.

Implements the dueling architecture from "Dueling Network Architectures for Deep
Reinforcement Learning" (Wang et al., 2016) which decomposes Q-values into:
- State value V(s): How good is this state regardless of action?
- Action advantages A(s,a): How much better is action a than average?
"""

import torch
import torch.nn as nn


class DuelingValueHead(nn.Module):
    """Dueling architecture: V(s) + (A(s,a) - mean(A(s,*))).

    Decomposes Q-values into:
    - State value V(s): How good is this state?
    - Action advantages A(s,a): How much better is action a than average?

    Benefits:
    - Learns state values even without taking actions
    - Faster convergence in many environments
    - Better gradient flow for value function
    - More stable value estimation when many actions have similar values

    For PPO value function:
    - We return V(s) + mean(A(s,*)) as the state value estimate
    - The advantage stream helps learn action-dependent value differences
    - Even though PPO doesn't use Q-values directly, the decomposition
      helps the value network learn more efficiently
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_actions: int = 6,
        hidden_dim: int = 256,
    ):
        """Initialize dueling value head.

        Args:
            feature_dim: Dimension of input features from MLP extractor
            num_actions: Number of actions (used for advantage stream)
            hidden_dim: Hidden layer dimension for both streams
        """
        super().__init__()

        self.num_actions = num_actions

        # State value stream: V(s)
        # Learns how good the state is regardless of action
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single scalar value
        )

        # Advantage stream: A(s,a)
        # Learns how much better each action is than average
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),  # One advantage per action
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute value using dueling architecture.

        Returns V(s) + mean(A(s,*)) as the state value estimate.

        The mean advantage is used because:
        1. It centers the advantages around zero
        2. Provides a stable baseline for value estimation
        3. In PPO, we only need the state value, not Q-values

        Args:
            features: Value latent features [batch, feature_dim]

        Returns:
            Value estimate [batch, 1]
        """
        # Compute state value
        state_value = self.value_stream(features)  # [batch, 1]

        # Compute action advantages
        advantages = self.advantage_stream(features)  # [batch, num_actions]

        # Combine: V(s) + mean(A(s,*))
        # We use mean instead of max for PPO value function
        # This gives us a more stable baseline estimate
        mean_advantage = advantages.mean(dim=1, keepdim=True)  # [batch, 1]

        # Return combined value (uses mean advantage as baseline)
        value = state_value + mean_advantage  # [batch, 1]

        return value

    def get_value_and_advantages(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get separate state value and advantages for analysis.

        This can be useful for debugging or logging to understand
        how the dueling architecture is learning.

        Args:
            features: Value latent features [batch, feature_dim]

        Returns:
            Tuple of (state_value [batch, 1], advantages [batch, num_actions])
        """
        state_value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return state_value, advantages
