"""Utilities for creating and managing policy networks.

Provides functions for creating standalone policy networks from architecture
configurations, independent of full PPO models, for use in behavioral cloning
and other pretraining scenarios.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.feature_extractors.configurable_extractor import (
    ConfigurableMultimodalExtractor,
)

logger = logging.getLogger(__name__)


def create_observation_space_from_config(
    architecture_config: ArchitectureConfig,
) -> spaces.Dict:
    """Create observation space based on architecture configuration.
    
    Args:
        architecture_config: Architecture configuration
        
    Returns:
        Gymnasium Dict observation space
    """
    from nclone.gym_environment.npp_environment import NppEnvironment
    from nclone.gym_environment.config import EnvironmentConfig
    
    # Create temporary environment to get observation space
    config = EnvironmentConfig.for_training()
    env = NppEnvironment(config=config)
    obs_space = env.observation_space
    env.close()
    
    return obs_space


def create_policy_network(
    observation_space: spaces.Dict,
    action_space: spaces.Discrete,
    architecture_config: ArchitectureConfig,
    features_dim: int = 512,
    net_arch: Optional[list] = None,
) -> nn.Module:
    """Create a policy network from architecture configuration.
    
    This creates a standalone policy network that can be used for behavioral
    cloning or other pretraining tasks, without requiring a full PPO model.
    
    Args:
        observation_space: Observation space
        action_space: Action space
        architecture_config: Architecture configuration
        features_dim: Dimension of feature extractor output
        net_arch: Network architecture for policy head (default: [256, 256])
        
    Returns:
        Policy network module
    """
    if net_arch is None:
        net_arch = [256, 256]
    
    # Create feature extractor
    feature_extractor = ConfigurableMultimodalExtractor(
        observation_space=observation_space,
        architecture_config=architecture_config,
        features_dim=features_dim,
    )
    
    # Create policy head (MLP that outputs action logits)
    policy_layers = []
    in_dim = features_dim
    
    for hidden_dim in net_arch:
        policy_layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        ])
        in_dim = hidden_dim
    
    # Final layer outputs action logits
    policy_layers.append(nn.Linear(in_dim, action_space.n))
    
    policy_head = nn.Sequential(*policy_layers)
    
    # Combine into full policy network
    class PolicyNetwork(nn.Module):
        """Combined feature extractor and policy head."""
        
        def __init__(self, feature_extractor, policy_head, features_dim):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.policy_head = policy_head
            self.features_dim = features_dim
        
        def forward(self, observations):
            """Forward pass through policy network.
            
            Args:
                observations: Dictionary of observations
                
            Returns:
                Action logits
            """
            features = self.feature_extractor(observations)
            logits = self.policy_head(features)
            return logits
        
        def get_features(self, observations):
            """Extract features without policy head.
            
            Args:
                observations: Dictionary of observations
                
            Returns:
                Feature vector
            """
            return self.feature_extractor(observations)
    
    policy_network = PolicyNetwork(feature_extractor, policy_head, features_dim)
    
    logger.info(f"Created policy network with {features_dim}-dim features")
    logger.info(f"Policy architecture: {net_arch} -> {action_space.n} actions")
    
    return policy_network


def save_policy_checkpoint(
    policy_network: nn.Module,
    path: str,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    architecture_config: Optional[ArchitectureConfig] = None,
) -> None:
    """Save policy network checkpoint in format compatible with RL training.
    
    Saves checkpoint with 'policy_state_dict' key as expected by
    architecture_trainer.py for loading pretrained weights.
    
    Args:
        policy_network: Policy network to save
        path: Path to save checkpoint
        epoch: Training epoch (optional)
        metrics: Training metrics (optional)
        architecture_config: Architecture configuration (optional)
    """
    checkpoint = {
        'policy_state_dict': policy_network.state_dict(),
    }
    
    # Add optional metadata
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if architecture_config is not None:
        checkpoint['architecture'] = architecture_config.name
    
    # Save checkpoint
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)
    logger.info(f"Saved policy checkpoint to {path}")
    
    # Log checkpoint info
    if epoch is not None:
        logger.info(f"  Epoch: {epoch}")
    if metrics is not None:
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")


def load_policy_checkpoint(
    policy_network: nn.Module,
    path: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load policy network checkpoint.
    
    Args:
        policy_network: Policy network to load weights into
        path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint = torch.load(path, map_location=device)
    
    if 'policy_state_dict' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'policy_state_dict'. "
            f"Found keys: {list(checkpoint.keys())}"
        )
    
    # Load state dict
    policy_network.load_state_dict(checkpoint['policy_state_dict'])
    
    logger.info(f"Loaded policy checkpoint from {path}")
    
    # Extract and log metadata
    metadata = {}
    if 'epoch' in checkpoint:
        metadata['epoch'] = checkpoint['epoch']
        logger.info(f"  Epoch: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        metadata['metrics'] = checkpoint['metrics']
        for key, value in checkpoint['metrics'].items():
            logger.info(f"  {key}: {value:.4f}")
    
    if 'architecture' in checkpoint:
        metadata['architecture'] = checkpoint['architecture']
        logger.info(f"  Architecture: {checkpoint['architecture']}")
    
    return metadata


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_info(model: nn.Module, name: str = "Model") -> None:
    """Log information about a model.
    
    Args:
        model: PyTorch model
        name: Model name for logging
    """
    num_params = count_parameters(model)
    logger.info(f"{name} info:")
    logger.info(f"  Total trainable parameters: {num_params:,}")
    logger.info(f"  Model size: {num_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Log parameter breakdown by layer if verbose
    logger.debug(f"\n{name} architecture:")
    for name, module in model.named_children():
        module_params = count_parameters(module)
        logger.debug(f"  {name}: {module_params:,} parameters")
