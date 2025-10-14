"""
Vision-Free Feature Extractor for N++ RL Training.

This extractor uses only non-visual modalities for faster training and
debugging. It processes:
- Game state (physics state)
- Reachability features
- Graph observations
- Entity positions

This is suitable for CPU training and simplified architecture testing.
"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class VisionFreeExtractor(BaseFeaturesExtractor):
    """
    Vision-free feature extractor using only state-based observations.
    
    Processes:
    - game_state: 30-dim physics state vector
    - reachability_features: 8-dim reachability information
    - entity_positions: Variable-length entity state
    
    Ignores visual observations (player_frame, global_view) for speed.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        hidden_dim: int = 256,
    ):
        """
        Args:
            observation_space: Gymnasium Dict space with observation components
            features_dim: Output dimension for the feature vector
            hidden_dim: Hidden layer dimension for MLPs
        """
        super().__init__(observation_space, features_dim)
        
        self.hidden_dim = hidden_dim
        
        # Check which modalities are available
        self.has_state = "game_state" in observation_space.spaces
        self.has_reachability = "reachability_features" in observation_space.spaces
        self.has_entity_pos = "entity_positions" in observation_space.spaces
        
        feature_dims = []
        
        # 1. Game state MLP
        if self.has_state:
            state_dim = observation_space.spaces["game_state"].shape[0]
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            feature_dims.append(hidden_dim // 2)
        else:
            self.state_mlp = None
        
        # 2. Reachability features MLP
        if self.has_reachability:
            reach_dim = observation_space.spaces["reachability_features"].shape[0]
            self.reachability_mlp = nn.Sequential(
                nn.Linear(reach_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
            )
            feature_dims.append(hidden_dim // 4)
        else:
            self.reachability_mlp = None
        
        # 3. Entity positions MLP (with attention pooling)
        if self.has_entity_pos:
            entity_dim = observation_space.spaces["entity_positions"].shape[0]
            self.entity_mlp = nn.Sequential(
                nn.Linear(entity_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
            )
            feature_dims.append(hidden_dim // 4)
        else:
            self.entity_mlp = None
        
        # Fusion network
        total_dim = sum(feature_dims) if feature_dims else 1
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from non-visual observations.
        
        Args:
            observations: Dict of observation tensors
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        features = []
        
        # Process game state
        if self.state_mlp is not None and "game_state" in observations:
            state_features = self.state_mlp(observations["game_state"])
            features.append(state_features)
        
        # Process reachability features
        if self.reachability_mlp is not None and "reachability_features" in observations:
            reach_features = self.reachability_mlp(observations["reachability_features"])
            features.append(reach_features)
        
        # Process entity positions
        if self.entity_mlp is not None and "entity_positions" in observations:
            entity_features = self.entity_mlp(observations["entity_positions"])
            features.append(entity_features)
        
        # Concatenate and fuse features
        if features:
            combined = torch.cat(features, dim=1)
        else:
            # Fallback if no features available (shouldn't happen)
            batch_size = observations[list(observations.keys())[0]].shape[0]
            combined = torch.zeros((batch_size, 1), device=next(self.parameters()).device)
        
        output = self.fusion(combined)
        return output


class MinimalStateExtractor(BaseFeaturesExtractor):
    """
    Minimal state-only extractor for fastest training.
    
    Uses only game_state and reachability_features, ignoring everything else.
    Suitable for rapid prototyping and CPU-based training.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
    ):
        """
        Args:
            observation_space: Gymnasium Dict space with observation components
            features_dim: Output dimension for the feature vector
        """
        super().__init__(observation_space, features_dim)
        
        # Get dimensions
        state_dim = observation_space.spaces.get("game_state", gym.spaces.Box(0, 1, (30,))).shape[0]
        reach_dim = observation_space.spaces.get("reachability_features", gym.spaces.Box(0, 1, (8,))).shape[0]
        
        # Simple 2-layer MLP for state
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Simple 2-layer MLP for reachability
        self.reach_net = nn.Sequential(
            nn.Linear(reach_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from minimal state observations.
        
        Args:
            observations: Dict of observation tensors
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        # Process state
        state_features = self.state_net(observations["game_state"])
        
        # Process reachability if available, otherwise use zeros
        if "reachability_features" in observations:
            reach_features = self.reach_net(observations["reachability_features"])
        else:
            batch_size = state_features.shape[0]
            reach_features = torch.zeros((batch_size, 32), device=state_features.device)
        
        # Fuse
        combined = torch.cat([state_features, reach_features], dim=1)
        output = self.fusion(combined)
        return output
