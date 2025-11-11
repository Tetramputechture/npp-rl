"""
Utility functions for intrinsic motivation modules with reachability awareness.

This module provides utilities for extracting reachability information from
environment observations and integrating it with intrinsic curiosity modules.

Updated to use real nclone integration instead of placeholder implementations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union

# Import real nclone integration
from .reachability_exploration import (
    extract_reachability_info_from_observations as _real_extract_reachability_info,
)


def extract_features_from_policy(
    policy,
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    detach: bool = True,
) -> torch.Tensor:
    """
    Extract features from a policy's feature extractor.

    Args:
        policy: SB3 policy with features_extractor
        observations: Observations to extract features from
        detach: Whether to detach features from computation graph

    Returns:
        Feature tensor [batch_size, feature_dim]
    """
    if hasattr(policy, "features_extractor"):
        features = policy.features_extractor(observations)
        if detach:
            features = features.detach()
        return features
    else:
        raise ValueError("Policy does not have a features_extractor")


def clip_intrinsic_reward(
    intrinsic_reward: np.ndarray, clip_max: float = 1.0, clip_min: float = 0.0
) -> np.ndarray:
    """
    Clip intrinsic rewards to prevent instability.

    Args:
        intrinsic_reward: Intrinsic rewards to clip
        clip_max: Maximum reward value
        clip_min: Minimum reward value

    Returns:
        Clipped intrinsic rewards
    """
    return np.clip(intrinsic_reward, clip_min, clip_max)


def normalize_features(features: torch.Tensor, method: str = "l2") -> torch.Tensor:
    """
    Normalize feature vectors.

    Args:
        features: Feature tensor to normalize
        method: Normalization method ('l2', 'batch', or 'none')

    Returns:
        Normalized features
    """
    if method == "l2":
        return torch.nn.functional.normalize(features, p=2, dim=1)
    elif method == "batch":
        return (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    elif method == "none":
        return features
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_feature_statistics(features: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about feature representations.

    Args:
        features: Feature tensor [batch_size, feature_dim]

    Returns:
        Dictionary of feature statistics
    """
    with torch.no_grad():
        stats = {
            "mean_norm": torch.norm(features, dim=1).mean().item(),
            "std_norm": torch.norm(features, dim=1).std().item(),
            "mean_feature": features.mean().item(),
            "std_feature": features.std().item(),
            "max_feature": features.max().item(),
            "min_feature": features.min().item(),
        }
    return stats


def create_icm_config(
    feature_dim: int = 512,
    action_dim: int = 6,
    hidden_dim: int = 256,
    eta: float = 0.01,
    lambda_inv: float = 0.1,
    lambda_fwd: float = 0.9,
    learning_rate: float = 1e-3,
    alpha: float = 0.1,
    r_int_clip: float = 1.0,
    share_backbone: bool = True,
) -> Dict[str, Any]:
    """
    Create a default ICM configuration.

    Args:
        feature_dim: Dimension of feature representations
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer dimension
        eta: Scaling factor for intrinsic reward
        lambda_inv: Weight for inverse model loss
        lambda_fwd: Weight for forward model loss
        learning_rate: Learning rate for ICM optimizer
        alpha: Weight for combining intrinsic and extrinsic rewards
        r_int_clip: Maximum intrinsic reward value
        share_backbone: Whether to share backbone with policy

    Returns:
        ICM configuration dictionary
    """
    return {
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "eta": eta,
        "lambda_inv": lambda_inv,
        "lambda_fwd": lambda_fwd,
        "learning_rate": learning_rate,
        "alpha": alpha,
        "r_int_clip": r_int_clip,
        "share_backbone": share_backbone,
    }


class RewardCombiner:
    """
    Combines extrinsic and intrinsic rewards with optional annealing.
    """

    def __init__(
        self,
        alpha_start: float = 0.1,
        alpha_end: float = 0.01,
        anneal_steps: Optional[int] = None,
        anneal_method: str = "linear",
    ):
        """
        Initialize reward combiner.

        Args:
            alpha_start: Initial weight for intrinsic rewards
            alpha_end: Final weight for intrinsic rewards
            anneal_steps: Number of steps to anneal over (None for no annealing)
            anneal_method: Annealing method ('linear' or 'exponential')
        """
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.anneal_steps = anneal_steps
        self.anneal_method = anneal_method
        self.current_step = 0

    def get_alpha(self) -> float:
        """Get current alpha value."""
        if self.anneal_steps is None:
            return self.alpha_start

        progress = min(self.current_step / self.anneal_steps, 1.0)

        if self.anneal_method == "linear":
            alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        elif self.anneal_method == "exponential":
            alpha = self.alpha_start * (self.alpha_end / self.alpha_start) ** progress
        else:
            raise ValueError(f"Unknown annealing method: {self.anneal_method}")

        return alpha

    def combine_rewards(
        self, extrinsic_rewards: np.ndarray, intrinsic_rewards: np.ndarray
    ) -> np.ndarray:
        """
        Combine extrinsic and intrinsic rewards.

        Args:
            extrinsic_rewards: External environment rewards
            intrinsic_rewards: ICM intrinsic rewards

        Returns:
            Combined rewards
        """
        alpha = self.get_alpha()
        combined = extrinsic_rewards + alpha * intrinsic_rewards
        self.current_step += len(extrinsic_rewards)
        return combined

    def reset(self):
        """Reset the step counter."""
        self.current_step = 0


def extract_reachability_info_from_observations(
    observations: Dict[str, Any],
    batch_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract reachability information from environment observations using real nclone systems.

    This function now uses the real nclone reachability analysis instead of placeholder
    implementations, providing accurate reachability information for ICM modulation.

    Args:
        observations: Dictionary of observations from environment
        batch_size: Expected batch size (for validation)

    Returns:
        Dictionary containing reachability information or None if not available
    """
    # Use real nclone integration
    return _real_extract_reachability_info(observations)


def create_reachability_aware_icm_config(
    feature_dim: int = 512,
    action_dim: int = 6,
    hidden_dim: int = 256,
    eta: float = 0.01,
    lambda_inv: float = 0.1,
    lambda_fwd: float = 0.9,
    learning_rate: float = 1e-3,
    alpha: float = 0.1,
    r_int_clip: float = 1.0,
    reachability_dim: int = 8,
    reachability_scale_factor: float = 2.0,
    frontier_boost_factor: float = 3.0,
    strategic_weight_factor: float = 1.5,
    unreachable_penalty: float = 0.1,
) -> Dict[str, Any]:
    """
    Create a reachability-aware ICM configuration.

    Args:
        feature_dim: Dimension of feature representations
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer dimension
        eta: Scaling factor for intrinsic reward
        lambda_inv: Weight for inverse model loss
        lambda_fwd: Weight for forward model loss
        learning_rate: Learning rate for ICM optimizer
        alpha: Weight for combining intrinsic and extrinsic rewards
        r_int_clip: Maximum intrinsic reward value
        reachability_dim: Dimension of reachability features
        reachability_scale_factor: Boost factor for reachable areas
        frontier_boost_factor: Extra boost for newly accessible areas
        strategic_weight_factor: Weight for objective-proximate areas
        unreachable_penalty: Penalty for confirmed unreachable areas

    Returns:
        Enhanced ICM configuration dictionary
    """
    config = create_icm_config(
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        eta=eta,
        lambda_inv=lambda_inv,
        lambda_fwd=lambda_fwd,
        learning_rate=learning_rate,
        alpha=alpha,
        r_int_clip=r_int_clip,
    )

    # Add reachability-specific parameters
    config.update(
        {
            "reachability_dim": reachability_dim,
            "reachability_scale_factor": reachability_scale_factor,
            "frontier_boost_factor": frontier_boost_factor,
            "strategic_weight_factor": strategic_weight_factor,
            "unreachable_penalty": unreachable_penalty,
        }
    )

    return config
