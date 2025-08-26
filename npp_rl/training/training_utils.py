"""
Training utilities for NPP-RL.

This module contains shared utility functions for training,
including policy creation and configuration helpers.
"""

from typing import Union
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Discrete, Dict as SpacesDict

from npp_rl.models.feature_extractors import create_feature_extractor


def create_training_policy(
    observation_space: SpacesDict,
    action_space: Discrete,
    policy_class: str = 'npp',
    use_graph_obs: bool = False,
    features_dim: int = 512
) -> nn.Module:
    """
    Create a policy network for training.
    
    This function centralizes policy creation logic that can be shared
    across different training approaches (BC, RL, etc.).
    
    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        policy_class: Policy architecture ('npp' or 'simple')
        use_graph_obs: Whether to use graph observations
        features_dim: Feature extractor output dimension
        
    Returns:
        PyTorch neural network module representing the policy
    """
    if policy_class == 'npp':
        # Create custom policy with multimodal feature extractor
        features_extractor = create_feature_extractor(
            observation_space=observation_space,
            features_dim=features_dim,
            use_graph_obs=use_graph_obs
        )
        
        # Create policy network with appropriate architecture
        policy = nn.Sequential(
            features_extractor,
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_space.n)
        )
        
    elif policy_class == 'simple':
        # Simple MLP policy for testing and baselines
        # Calculate observation dimension, excluding graph-specific components
        obs_dim = sum(
            np.prod(space.shape) for space in observation_space.spaces.values()
            if not (space.dtype == np.int32 and len(space.shape) == 2)  # Skip edge_index
        )
        
        policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_space.n)
        )
        
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    
    return policy


def setup_device(device_str: str = 'auto') -> torch.device:
    """
    Set up the appropriate device for training.
    
    Args:
        device_str: Device specification ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        PyTorch device object
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    return device


def freeze_model_parameters(model: nn.Module, layer_names: list[str]):
    """
    Freeze specified layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name in layer_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            for param in layer.parameters():
                param.requires_grad = False


def unfreeze_model_parameters(model: nn.Module):
    """
    Unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
