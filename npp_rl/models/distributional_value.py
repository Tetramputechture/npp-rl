"""Distributional value function using quantile regression.

Implements QR-DQN style distributional value estimation for better
uncertainty quantification and more robust value learning.
"""

import torch
import torch.nn as nn
from typing import Tuple


class QuantileValueHead(nn.Module):
    """Quantile Regression DQN value head.
    
    Instead of outputting a single value V(s), this head outputs a distribution
    over values using N quantiles (default N=51). This allows the network to
    capture uncertainty in returns and is more robust to outliers than MSE loss.
    
    Reference:
        Dabney et al. (2018): "Distributional Reinforcement Learning with Quantile Regression"
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_quantiles: int = 51,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize quantile value head.
        
        Args:
            feature_dim: Dimension of input value features
            num_quantiles: Number of quantiles to predict (default: 51)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_quantiles = num_quantiles
        self.hidden_dim = hidden_dim
        
        # Value network outputting multiple quantiles
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_quantiles),  # Output one value per quantile
        )
        
        # Quantile midpoints τ (tau): [0.01, 0.03, ..., 0.99]
        # These define which quantiles we're predicting
        quantile_tau = torch.linspace(
            1.0 / (2 * num_quantiles),
            1.0 - 1.0 / (2 * num_quantiles),
            num_quantiles,
        )
        self.register_buffer("quantile_tau", quantile_tau)
    
    def forward(self, value_features: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute quantile values.
        
        Args:
            value_features: Value network features [batch, feature_dim]
        
        Returns:
            Quantile values [batch, num_quantiles]
        """
        quantile_values = self.value_net(value_features)
        return quantile_values
    
    def get_value_estimate(self, quantile_values: torch.Tensor) -> torch.Tensor:
        """Get point value estimate from quantile distribution.
        
        For policy updates and evaluation, we use the mean of quantiles
        as the value estimate V(s).
        
        Args:
            quantile_values: Quantile values [batch, num_quantiles]
        
        Returns:
            Value estimates [batch, 1]
        """
        return quantile_values.mean(dim=-1, keepdim=True)
    
    def get_value_statistics(
        self, quantile_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get value statistics from quantile distribution.
        
        Args:
            quantile_values: Quantile values [batch, num_quantiles]
        
        Returns:
            Tuple of (mean, std, median)
        """
        mean = quantile_values.mean(dim=-1)
        std = quantile_values.std(dim=-1)
        median = quantile_values.median(dim=-1).values
        
        return mean, std, median


def quantile_huber_loss(
    quantile_values: torch.Tensor,
    target_values: torch.Tensor,
    quantile_tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss from QR-DQN paper.
    
    This loss function encourages different quantiles to capture different
    parts of the return distribution through asymmetric weighting.
    
    For each quantile τ:
    - If TD error > 0: weight by τ
    - If TD error < 0: weight by (1 - τ)
    
    Args:
        quantile_values: Predicted quantile values [batch, num_quantiles]
        target_values: Target quantile values [batch, num_quantiles]
        quantile_tau: Quantile midpoints [num_quantiles]
        kappa: Huber loss threshold (default: 1.0)
    
    Returns:
        Scalar loss value
    """
    # Compute TD errors for each quantile
    # Shape: [batch, num_quantiles, 1]
    td_errors = target_values.unsqueeze(-1) - quantile_values.unsqueeze(-2)
    
    # Huber loss element: L2 for small errors, L1 for large errors
    # This makes the loss more robust to outliers
    huber_loss = torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa)
    )
    
    # Quantile weighting: asymmetric loss based on quantile position
    # If error is negative (underestimate), weight by (1-τ)
    # If error is positive (overestimate), weight by τ
    # Shape: [1, num_quantiles, 1]
    quantile_tau_expanded = quantile_tau.reshape(1, -1, 1)
    
    # Indicator: 1 if td_error < 0, else 0
    indicator = (td_errors < 0).float()
    
    # Asymmetric weight
    quantile_weight = torch.abs(quantile_tau_expanded - indicator)
    
    # Combine Huber loss with quantile weighting
    # Average over all quantiles and samples
    loss = (quantile_weight * huber_loss).mean()
    
    return loss


def compute_quantile_targets(
    next_quantile_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute quantile regression targets for TD learning.
    
    Target for each quantile: r + γ * Q_next(quantile)
    
    Args:
        next_quantile_values: Next state quantile values [batch, num_quantiles]
        rewards: Rewards [batch]
        dones: Done flags [batch]
        gamma: Discount factor
    
    Returns:
        Target quantile values [batch, num_quantiles]
    """
    # Expand rewards to match quantile dimension
    rewards_expanded = rewards.unsqueeze(-1)  # [batch, 1]
    dones_expanded = dones.unsqueeze(-1)  # [batch, 1]
    
    # Compute targets: r + γ * (1 - done) * Q_next
    targets = rewards_expanded + gamma * (1.0 - dones_expanded) * next_quantile_values
    
    return targets


class DistributionalValueFunction(nn.Module):
    """Complete distributional value function with quantile regression.
    
    This combines the value network trunk with the quantile head to create
    a complete distributional value function for PPO.
    """
    
    def __init__(
        self,
        value_network: nn.Module,
        feature_dim: int = 256,
        num_quantiles: int = 51,
        hidden_dim: int = 512,
    ):
        """Initialize distributional value function.
        
        Args:
            value_network: Value network trunk (e.g., from DeepResNetMLPExtractor)
            feature_dim: Dimension of value network output
            num_quantiles: Number of quantiles
            hidden_dim: Hidden dimension for quantile head
        """
        super().__init__()
        
        self.value_network = value_network
        self.quantile_head = QuantileValueHead(
            feature_dim=feature_dim,
            num_quantiles=num_quantiles,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get value distribution and point estimate.
        
        Args:
            observations: Input observations
        
        Returns:
            Tuple of (value_estimate, quantile_values)
        """
        # Get value features
        value_features = self.value_network(observations)
        
        # Get quantile distribution
        quantile_values = self.quantile_head(value_features)
        
        # Get point estimate (mean of quantiles)
        value_estimate = self.quantile_head.get_value_estimate(quantile_values)
        
        return value_estimate, quantile_values
    
    def compute_loss(
        self,
        observations: torch.Tensor,
        target_values: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Compute quantile regression loss.
        
        Args:
            observations: Input observations
            target_values: Target quantile values [batch, num_quantiles]
            kappa: Huber loss threshold
        
        Returns:
            Scalar loss value
        """
        # Get predicted quantiles
        value_features = self.value_network(observations)
        quantile_values = self.quantile_head(value_features)
        
        # Compute quantile Huber loss
        loss = quantile_huber_loss(
            quantile_values,
            target_values,
            self.quantile_head.quantile_tau,
            kappa=kappa,
        )
        
        return loss

