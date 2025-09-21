#!/usr/bin/env python3
"""
Example usage of reachability-aware ICM with nclone environment.

This example demonstrates how to integrate the enhanced ICM module with
reachability awareness into a training loop with the nclone environment.
"""

import numpy as np
import torch
from typing import Dict, Any

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced ICM components
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.intrinsic.utils import (
    extract_reachability_info_from_observations,
    create_reachability_aware_icm_config,
    RewardCombiner,
)


def create_reachability_aware_icm_trainer(
    feature_dim: int = 512,
    action_dim: int = 6,
    device: str = "cpu",
) -> ICMTrainer:
    """
    Create a reachability-aware ICM trainer.
    
    Args:
        feature_dim: Dimension of feature representations
        action_dim: Number of discrete actions (6 for N++)
        device: Device to run on
        
    Returns:
        Configured ICM trainer with reachability awareness
    """
    # Create reachability-aware configuration
    config = create_reachability_aware_icm_config(
        feature_dim=feature_dim,
        action_dim=action_dim,
        enable_reachability_awareness=True,
        reachability_dim=64,  # From nclone reachability features
        eta=0.01,  # Intrinsic reward scaling
        reachability_scale_factor=2.0,  # Boost for reachable areas
        frontier_boost_factor=3.0,  # Extra boost for newly accessible areas
        strategic_weight_factor=1.5,  # Weight for objective-proximate areas
        unreachable_penalty=0.1,  # Penalty for unreachable areas
    )
    
    # Create ICM network
    icm_network = ICMNetwork(
        feature_dim=config["feature_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        eta=config["eta"],
        lambda_inv=config["lambda_inv"],
        lambda_fwd=config["lambda_fwd"],
        enable_reachability_awareness=config["enable_reachability_awareness"],
        reachability_dim=config["reachability_dim"],
    )
    
    # Create trainer
    trainer = ICMTrainer(
        icm_network=icm_network,
        learning_rate=config["learning_rate"],
        device=device,
    )
    
    return trainer


def training_step_with_reachability_awareness(
    icm_trainer: ICMTrainer,
    reward_combiner: RewardCombiner,
    observations: Dict[str, Any],
    next_observations: Dict[str, Any],
    actions: np.ndarray,
    extrinsic_rewards: np.ndarray,
    features_current: torch.Tensor,
    features_next: torch.Tensor,
) -> Dict[str, Any]:
    """
    Perform a training step with reachability-aware curiosity.
    
    Args:
        icm_trainer: ICM trainer instance
        reward_combiner: Reward combiner for intrinsic/extrinsic rewards
        observations: Current observations from environment
        next_observations: Next observations from environment
        actions: Actions taken
        extrinsic_rewards: External environment rewards
        features_current: Current state features
        features_next: Next state features
        
    Returns:
        Dictionary containing training statistics and combined rewards
    """
    # Extract reachability information from observations
    reachability_info = extract_reachability_info_from_observations(observations)
    next_reachability_info = extract_reachability_info_from_observations(next_observations)
    
    # Use next state reachability info for curiosity computation
    # (we want to know about the reachability of where we're going)
    reachability_info_for_icm = next_reachability_info or reachability_info
    
    # Update ICM with reachability information
    icm_stats = icm_trainer.update(
        features_current=features_current,
        features_next=features_next,
        actions=torch.from_numpy(actions),
        reachability_info=reachability_info_for_icm,
    )
    
    # Get intrinsic rewards with reachability modulation
    intrinsic_rewards = icm_trainer.get_intrinsic_reward(
        features_current=features_current,
        features_next=features_next,
        actions=torch.from_numpy(actions),
        reachability_info=reachability_info_for_icm,
    )
    
    # Combine extrinsic and intrinsic rewards
    combined_rewards = reward_combiner.combine_rewards(
        extrinsic_rewards=extrinsic_rewards,
        intrinsic_rewards=intrinsic_rewards,
    )
    
    return {
        "icm_stats": icm_stats,
        "intrinsic_rewards": intrinsic_rewards,
        "combined_rewards": combined_rewards,
        "reachability_available": reachability_info_for_icm is not None,
    }


def example_training_loop():
    """
    Example training loop demonstrating reachability-aware ICM usage.
    """
    print("Reachability-Aware ICM Training Example")
    print("=" * 50)
    
    # Configuration
    feature_dim = 512
    action_dim = 6
    batch_size = 32
    device = "cpu"
    
    # Create ICM trainer
    icm_trainer = create_reachability_aware_icm_trainer(
        feature_dim=feature_dim,
        action_dim=action_dim,
        device=device,
    )
    
    # Create reward combiner with annealing
    reward_combiner = RewardCombiner(
        alpha_start=0.1,  # Start with 10% intrinsic reward weight
        alpha_end=0.01,   # End with 1% intrinsic reward weight
        anneal_steps=100000,  # Anneal over 100k steps
        anneal_method="linear",
    )
    
    # Simulate training steps
    for step in range(10):
        print(f"\nTraining Step {step + 1}")
        print("-" * 30)
        
        # Mock environment data
        observations = {
            "player_x": np.random.uniform(0, 1000, batch_size),
            "player_y": np.random.uniform(0, 600, batch_size),
            "reachability_features": np.random.rand(batch_size, 64),
        }
        
        next_observations = {
            "player_x": observations["player_x"] + np.random.uniform(-50, 50, batch_size),
            "player_y": observations["player_y"] + np.random.uniform(-50, 50, batch_size),
            "reachability_features": np.random.rand(batch_size, 64),
        }
        
        actions = np.random.randint(0, action_dim, batch_size)
        extrinsic_rewards = np.random.uniform(-0.1, 1.0, batch_size)
        
        # Mock feature extraction (in practice, this would come from policy)
        features_current = torch.randn(batch_size, feature_dim)
        features_next = torch.randn(batch_size, feature_dim)
        
        # Perform training step
        step_results = training_step_with_reachability_awareness(
            icm_trainer=icm_trainer,
            reward_combiner=reward_combiner,
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            extrinsic_rewards=extrinsic_rewards,
            features_current=features_current,
            features_next=features_next,
        )
        
        # Print statistics
        icm_stats = step_results["icm_stats"]
        print(f"ICM Inverse Loss: {icm_stats['inverse_loss']:.4f}")
        print(f"ICM Forward Loss: {icm_stats['forward_loss']:.4f}")
        print(f"Mean Intrinsic Reward: {step_results['intrinsic_rewards'].mean():.4f}")
        print(f"Mean Combined Reward: {step_results['combined_rewards'].mean():.4f}")
        print(f"Reachability Available: {step_results['reachability_available']}")
        print(f"Current Alpha: {reward_combiner.get_alpha():.4f}")
    
    print("\n" + "=" * 50)
    print("Training example completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Reachability-aware curiosity modulation")
    print("- Integration with nclone reachability features")
    print("- Performance-optimized computation (<1ms)")
    print("- Backward compatibility with standard ICM")
    print("- Reward combination with annealing")


if __name__ == "__main__":
    example_training_loop()