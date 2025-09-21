"""
Utility functions for intrinsic motivation modules with reachability awareness.

This module provides utilities for extracting reachability information from
environment observations and integrating it with intrinsic curiosity modules.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Set


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
    Extract reachability information from environment observations for ICM modulation.
    
    This function processes observations from the nclone environment to extract
    reachability information needed for reachability-aware curiosity computation.
    
    Args:
        observations: Dictionary of observations from environment
        batch_size: Expected batch size (for validation)
        
    Returns:
        Dictionary containing reachability information or None if not available
        
    The returned dictionary contains:
        - current_positions: List of (x, y) current positions
        - target_positions: List of (x, y) target positions (approximated from movement)
        - reachable_positions: List of sets of reachable grid positions
        - door_positions: List of door positions
        - switch_positions: List of switch positions
        - exit_position: Exit position (if known)
    """
    if "reachability_features" not in observations:
        return None
    
    # Extract basic position information
    current_positions = []
    target_positions = []
    reachable_positions = []
    
    # Get player positions
    if "player_x" in observations and "player_y" in observations:
        player_x = observations["player_x"]
        player_y = observations["player_y"]
        
        # Handle both single values and arrays
        if isinstance(player_x, (int, float)):
            current_positions = [(float(player_x), float(player_y))]
        else:
            current_positions = [(float(x), float(y)) for x, y in zip(player_x, player_y)]
    
    # For target positions, we approximate based on current position + small offset
    # In practice, this would be the position the agent is trying to reach
    for pos in current_positions:
        # Simple approximation: target is slightly ahead in movement direction
        target_x = pos[0] + np.random.uniform(-24, 24)  # Within one cell
        target_y = pos[1] + np.random.uniform(-24, 24)
        target_positions.append((target_x, target_y))
    
    # Extract reachable positions from reachability features
    # The 64-dim reachability features encode spatial accessibility information
    reachability_features = observations["reachability_features"]
    
    if isinstance(reachability_features, np.ndarray):
        if len(reachability_features.shape) == 1:
            # Single observation
            reachable_set = _decode_reachability_features(reachability_features)
            reachable_positions = [reachable_set]
        else:
            # Batch of observations
            reachable_positions = [
                _decode_reachability_features(features) 
                for features in reachability_features
            ]
    
    # Extract entity positions (doors, switches, exit) if available
    door_positions = _extract_entity_positions(observations, "door")
    switch_positions = _extract_entity_positions(observations, "switch")
    exit_position = _extract_entity_positions(observations, "exit", single=True)
    
    return {
        "current_positions": current_positions,
        "target_positions": target_positions,
        "reachable_positions": reachable_positions,
        "door_positions": door_positions,
        "switch_positions": switch_positions,
        "exit_position": exit_position,
    }


def _decode_reachability_features(features: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Decode 64-dimensional reachability features into set of reachable grid positions.
    
    This is a simplified decoder that approximates reachable positions from
    the compact feature representation. In practice, this would use the actual
    decoding logic from the nclone reachability system.
    
    Args:
        features: 64-dimensional reachability feature vector
        
    Returns:
        Set of reachable grid positions (x, y)
    """
    # Simplified decoding: use feature values to determine reachable areas
    # This is a placeholder - actual implementation would use proper decoding
    reachable_positions = set()
    
    # Grid dimensions (44x25 cells)
    grid_width, grid_height = 44, 25
    
    # Use feature values to determine reachability in different grid regions
    # Divide features into spatial regions
    features_per_region = len(features) // 16  # 16 regions for 64 features
    
    for region_idx in range(16):
        region_x = region_idx % 4
        region_y = region_idx // 4
        
        # Get features for this region
        start_idx = region_idx * features_per_region
        end_idx = start_idx + features_per_region
        region_features = features[start_idx:end_idx]
        
        # If region has high feature values, mark as reachable
        if np.mean(region_features) > 0.5:  # Threshold for reachability
            # Add positions in this region
            region_width = grid_width // 4
            region_height = grid_height // 4
            
            for x in range(region_x * region_width, (region_x + 1) * region_width):
                for y in range(region_y * region_height, (region_y + 1) * region_height):
                    if x < grid_width and y < grid_height:
                        reachable_positions.add((x, y))
    
    return reachable_positions


def _extract_entity_positions(
    observations: Dict[str, Any], 
    entity_type: str, 
    single: bool = False
) -> Union[List[Tuple[float, float]], Tuple[float, float], None]:
    """
    Extract entity positions from observations.
    
    Args:
        observations: Environment observations
        entity_type: Type of entity to extract ("door", "switch", "exit")
        single: Whether to return single position or list
        
    Returns:
        Entity positions or None if not available
    """
    # This is a placeholder implementation
    # In practice, this would extract actual entity positions from observations
    
    if entity_type == "door":
        # Mock door positions
        return [(200.0, 300.0), (400.0, 500.0)]
    elif entity_type == "switch":
        # Mock switch positions
        return [(150.0, 250.0)]
    elif entity_type == "exit":
        # Mock exit position
        return (800.0, 100.0) if single else [(800.0, 100.0)]
    
    return None if single else []


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
    enable_reachability_awareness: bool = True,
    reachability_dim: int = 64,
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
        enable_reachability_awareness: Whether to enable reachability modulation
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
    config.update({
        "enable_reachability_awareness": enable_reachability_awareness,
        "reachability_dim": reachability_dim,
        "reachability_scale_factor": reachability_scale_factor,
        "frontier_boost_factor": frontier_boost_factor,
        "strategic_weight_factor": strategic_weight_factor,
        "unreachable_penalty": unreachable_penalty,
    })
    
    return config
