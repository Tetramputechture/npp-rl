"""
Reachability-Aware Intrinsic Curiosity Module (ICM)

This module provides a clean implementation that integrates with nclone's
existing reachability systems for enhanced exploration in RL training.

Key Design Principles:
1. Use nclone's ExplorationRewardCalculator as the base exploration system
2. Enhance it with reachability awareness rather than replacing it
3. Leverage nclone's OpenCV-based reachability analysis and frontier detection
4. Avoid duplicating exploration tracking logic
5. Maintain performance requirements (<1ms computation)

Integration with nclone:
- ReachabilitySystem for multi-tier reachability analysis
- CompactReachabilityFeatures for 64-dimensional feature encoding
- FrontierDetector for boundary detection and classification
- ExplorationRewardCalculator for multi-scale exploration tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, Optional
from collections import deque

from .reachability_exploration import ReachabilityAwareExplorationCalculator


class ICMNetwork(nn.Module):
    """
    Reachability-aware ICM that integrates with nclone's reachability systems.

    This implementation:
    - Uses standard ICM forward/inverse models for base curiosity
    - Integrates with nclone's reachability analysis for modulation
    - Leverages existing ExplorationRewardCalculator logic
    - Avoids duplicating exploration tracking functionality
    """

    def __init__(
        self,
        feature_dim: int = 512,
        action_dim: int = 6,
        hidden_dim: int = 256,
        eta: float = 0.01,
        lambda_inv: float = 0.1,
        lambda_fwd: float = 0.9,
        debug: bool = False,
    ):
        """
        Initialize reachability-aware ICM.

        Args:
            feature_dim: Dimension of feature representations
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            eta: Intrinsic reward scaling factor
            lambda_inv: Weight for inverse model loss
            lambda_fwd: Weight for forward model loss
            debug: Enable debug output
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.lambda_inv = lambda_inv
        self.lambda_fwd = lambda_fwd
        self.debug = debug

        # Standard ICM components
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.reachability_calculator = ReachabilityAwareExplorationCalculator(
            debug=debug
        )

        # Performance tracking
        self.computation_times = deque(maxlen=100)

    def forward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Forward pass through ICM models."""
        batch_size = features_current.shape[0]

        # Inverse model: predict action from state transition
        inverse_input = torch.cat([features_current, features_next], dim=1)
        predicted_actions = self.inverse_model(inverse_input)

        # Forward model: predict next state from current state and action
        actions_one_hot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        forward_input = torch.cat([features_current, actions_one_hot], dim=1)
        predicted_next_features = self.forward_model(forward_input)

        return predicted_actions, predicted_next_features

    def compute_intrinsic_reward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor,
        observations: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward with optional reachability modulation.

        Args:
            features_current: Current state features
            features_next: Next state features
            actions: Actions taken
            observations: Environment observations for reachability analysis

        Returns:
            Intrinsic reward tensor
        """
        start_time = time.time()

        # Compute base ICM curiosity (forward model prediction error)
        with torch.no_grad():
            _, predicted_next_features = self.forward(
                features_current, features_next, actions
            )
            prediction_error = F.mse_loss(
                predicted_next_features, features_next, reduction="none"
            )
            base_curiosity = prediction_error.mean(dim=1) * self.eta

        # Skip if no level data to avoid expensive failed computations
        if observations is not None and observations.get("level_data") is not None:
            modulated_curiosity = self._apply_reachability_modulation(
                base_curiosity, observations
            )
        else:
            modulated_curiosity = base_curiosity

        # Track performance
        computation_time = (time.time() - start_time) * 1000
        self.computation_times.append(computation_time)

        return modulated_curiosity

    def _apply_reachability_modulation(
        self, base_curiosity: torch.Tensor, observations: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply reachability-aware modulation using nclone systems."""
        batch_size = base_curiosity.shape[0]
        modulated_curiosity = base_curiosity.clone()

        for i in range(batch_size):
            # Extract single observation
            obs_i = self._extract_single_observation(observations, i)

            # Get reachability analysis from nclone
            reward_info = (
                self.reachability_calculator.calculate_reachability_aware_reward(
                    player_x=obs_i.get("player_x", 0.0),
                    player_y=obs_i.get("player_y", 0.0),
                    level_data=obs_i.get("level_data"),
                    switch_states=obs_i.get("switch_states", {}),
                )
            )

            # Apply modulation factor
            modulation_factor = reward_info.get("reachability_modulation", 1.0)
            modulated_curiosity[i] *= modulation_factor

        return modulated_curiosity

    def _extract_single_observation(
        self, observations: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """Extract single observation from batch."""
        single_obs = {}
        for key, value in observations.items():
            if isinstance(value, (list, np.ndarray, torch.Tensor)):
                if len(value) > index:
                    single_obs[key] = value[index]
                else:
                    single_obs[key] = value[0] if len(value) > 0 else None
            else:
                single_obs[key] = value
        return single_obs

    def compute_losses(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute ICM losses."""
        predicted_actions, predicted_next_features = self.forward(
            features_current, features_next, actions
        )

        # Inverse model loss
        inverse_loss = F.cross_entropy(predicted_actions, actions.long())

        # Forward model loss
        forward_loss = F.mse_loss(predicted_next_features, features_next)

        # Total loss
        total_loss = self.lambda_inv * inverse_loss + self.lambda_fwd * forward_loss

        return {
            "total_loss": total_loss,
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
        }

    def get_reachability_info(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get reachability information using nclone systems."""
        if self.reachability_calculator is None:
            return {"available": False}

        # Extract first observation
        obs = self._extract_single_observation(observations, 0)

        # Get compact features
        compact_features = self.reachability_calculator.extract_compact_features(
            level_data=obs.get("level_data"),
            player_position=(obs.get("player_x", 0.0), obs.get("player_y", 0.0)),
            switch_states=obs.get("switch_states", {}),
        )

        # Get frontier information
        frontiers = self.reachability_calculator.get_frontier_information(
            level_data=obs.get("level_data"),
            player_position=(obs.get("player_x", 0.0), obs.get("player_y", 0.0)),
            switch_states=obs.get("switch_states", {}),
        )

        return {
            "available": True,
            "compact_features": compact_features,
            "frontiers": frontiers,
            "num_frontiers": len(frontiers),
        }

    def reset(self):
        """Reset for new episode."""
        if self.reachability_calculator is not None:
            self.reachability_calculator.reset()
        self.computation_times.clear()

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if len(self.computation_times) == 0:
            return {"avg_time_ms": 0.0, "max_time_ms": 0.0, "min_time_ms": 0.0}

        times = np.array(self.computation_times)
        return {
            "avg_time_ms": float(np.mean(times)),
            "max_time_ms": float(np.max(times)),
            "min_time_ms": float(np.min(times)),
            "p95_time_ms": float(np.percentile(times, 95)),
            "samples": len(times),
        }


class ICMTrainer:
    """
    Trainer for reachability-aware ICM.

    This trainer integrates with nclone's exploration systems and provides
    a clean interface for RL training loops.
    """

    def __init__(
        self, icm_network: ICMNetwork, learning_rate: float = 1e-3, device: str = "cpu"
    ):
        """
        Initialize ICM trainer.

        Args:
            icm_network: ICM network to train
            learning_rate: Learning rate for optimizer
            device: Device to run on
        """
        self.icm_network = icm_network.to(device)
        self.optimizer = torch.optim.Adam(icm_network.parameters(), lr=learning_rate)
        self.device = device

    def update(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor,
        observations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Update ICM with a batch of transitions.

        Args:
            features_current: Current state features
            features_next: Next state features
            actions: Actions taken
            observations: Environment observations

        Returns:
            Training statistics
        """
        # Move to device
        features_current = features_current.to(self.device)
        features_next = features_next.to(self.device)
        actions = actions.to(self.device)

        # Compute losses
        losses = self.icm_network.compute_losses(
            features_current, features_next, actions
        )

        # Update parameters
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()

        # Return statistics
        stats = {
            "total_loss": losses["total_loss"].item(),
            "inverse_loss": losses["inverse_loss"].item(),
            "forward_loss": losses["forward_loss"].item(),
        }

        # Add performance stats
        perf_stats = self.icm_network.get_performance_stats()
        stats.update(perf_stats)

        return stats

    def get_intrinsic_reward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor,
        observations: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Get intrinsic rewards for a batch of transitions."""
        features_current = features_current.to(self.device)
        features_next = features_next.to(self.device)
        actions = actions.to(self.device)

        return self.icm_network.compute_intrinsic_reward(
            features_current, features_next, actions, observations
        )

    def reset(self):
        """Reset for new episode."""
        self.icm_network.reset()
