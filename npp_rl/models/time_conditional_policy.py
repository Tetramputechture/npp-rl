"""Time-conditional policy gating for urgency-aware decision making.

Modulates policy behavior based on time remaining to encourage appropriate
risk-taking and exploration based on episode progress.
"""

import torch
import torch.nn as nn
from typing import Optional


class TimeConditionalGating(nn.Module):
    """Gate policy outputs based on time urgency.

    This module modulates policy features based on time remaining in the episode.
    The idea is to make the policy urgency-aware:
    - Early in episode (lots of time): Conservative, prefer safe paths
    - Late in episode (low time): Aggressive, take risks to complete quickly

    This is implemented through learned gating that conditions on normalized
    time remaining.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        urgency_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        """Initialize time-conditional gating.

        Args:
            feature_dim: Dimension of policy features
            urgency_embed_dim: Dimension of urgency embedding
            dropout: Dropout rate
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.urgency_embed_dim = urgency_embed_dim

        # Urgency encoder: time_remaining → urgency_embedding
        # Takes normalized time remaining [0, 1] and encodes it
        self.urgency_encoder = nn.Sequential(
            nn.Linear(1, urgency_embed_dim),
            nn.LayerNorm(urgency_embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(urgency_embed_dim, feature_dim),
            nn.Tanh(),  # Output in [-1, 1] for gating
        )

        # Gating mechanism
        # Takes both policy features and urgency encoding
        # Outputs gate values in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )

    def forward(
        self,
        policy_features: torch.Tensor,
        time_remaining_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Apply time-conditional gating to policy features.

        Args:
            policy_features: Policy latent features [batch, feature_dim]
            time_remaining_normalized: Normalized time remaining [batch, 1]
                Values in [0, 1] where 1 = full time, 0 = no time left

        Returns:
            Gated policy features [batch, feature_dim]
        """
        # Encode urgency from time remaining
        urgency = self.urgency_encoder(
            time_remaining_normalized
        )  # [batch, feature_dim]

        # Compute gate values based on both policy features and urgency
        gate_input = torch.cat([policy_features, urgency], dim=-1)
        gate_values = self.gate(gate_input)  # [batch, feature_dim]

        # Apply gating with residual connection
        # gated = features * gate + urgency * (1 - gate)
        # This allows urgency signal to modulate or override features
        gated_features = policy_features * gate_values + urgency * (1.0 - gate_values)

        return gated_features

    def get_urgency_embedding(
        self, time_remaining_normalized: torch.Tensor
    ) -> torch.Tensor:
        """Get urgency embedding for visualization or analysis.

        Args:
            time_remaining_normalized: Normalized time remaining [batch, 1]

        Returns:
            Urgency embedding [batch, feature_dim]
        """
        return self.urgency_encoder(time_remaining_normalized)

    def get_gate_values(
        self,
        policy_features: torch.Tensor,
        time_remaining_normalized: torch.Tensor,
    ) -> torch.Tensor:
        """Get gate values for visualization or analysis.

        Args:
            policy_features: Policy latent features [batch, feature_dim]
            time_remaining_normalized: Normalized time remaining [batch, 1]

        Returns:
            Gate values [batch, feature_dim]
        """
        urgency = self.urgency_encoder(time_remaining_normalized)
        gate_input = torch.cat([policy_features, urgency], dim=-1)
        return self.gate(gate_input)


class TimeAwarePolicy(nn.Module):
    """Policy wrapper with time-conditional gating.

    Wraps a base policy to add time-awareness through gating mechanism.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        feature_dim: int = 256,
        enable_time_gating: bool = True,
    ):
        """Initialize time-aware policy.

        Args:
            base_policy: Base policy network
            feature_dim: Dimension of policy features
            enable_time_gating: Whether to enable time gating
        """
        super().__init__()

        self.base_policy = base_policy
        self.enable_time_gating = enable_time_gating

        if enable_time_gating:
            self.time_gating = TimeConditionalGating(feature_dim=feature_dim)
        else:
            self.time_gating = None

    def forward(
        self,
        observations: torch.Tensor,
        time_remaining: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional time gating.

        Args:
            observations: Input observations
            time_remaining: Normalized time remaining [batch, 1]
                If None, time gating is not applied

        Returns:
            Action logits [batch, num_actions]
        """
        # Get base policy features
        policy_features = self.base_policy.get_policy_latent(observations)

        # Apply time gating if enabled and time provided
        if self.time_gating is not None and time_remaining is not None:
            policy_features = self.time_gating(policy_features, time_remaining)

        # Get action logits
        action_logits = self.base_policy.action_net(policy_features)

        return action_logits


def normalize_time_remaining(
    current_step: torch.Tensor,
    max_steps: int = 20000,
) -> torch.Tensor:
    """Normalize time remaining to [0, 1] range.

    Args:
        current_step: Current step number [batch]
        max_steps: Maximum steps per episode

    Returns:
        Normalized time remaining [batch, 1]
        - 1.0 = full time remaining (step 0)
        - 0.0 = no time remaining (step max_steps)
    """
    time_remaining = 1.0 - (current_step.float() / max_steps)
    return time_remaining.clamp(0.0, 1.0).unsqueeze(-1)


def compute_urgency_score(
    time_remaining_normalized: torch.Tensor,
    urgency_threshold: float = 0.3,
) -> torch.Tensor:
    """Compute urgency score from time remaining.

    Urgency increases non-linearly as time runs out.

    Args:
        time_remaining_normalized: Normalized time remaining [batch, 1]
        urgency_threshold: Threshold below which urgency is high

    Returns:
        Urgency scores [batch, 1]
        - 0.0 = low urgency (lots of time)
        - 1.0 = high urgency (very little time)
    """
    # Inverse of time remaining, scaled
    urgency = 1.0 - time_remaining_normalized

    # Apply non-linear transformation to emphasize low-time urgency
    # When time_remaining > threshold: gradual urgency increase
    # When time_remaining < threshold: rapid urgency increase
    urgency = torch.where(
        time_remaining_normalized > urgency_threshold,
        urgency * 0.5,  # Low urgency when plenty of time
        urgency * 1.5,  # High urgency when time is low
    )

    return urgency.clamp(0.0, 1.0)
