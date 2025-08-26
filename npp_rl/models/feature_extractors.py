"""
Multimodal feature extractors that combine CNN, MLP, and GNN encoders.

This module extends the existing feature extractors to support graph-based
structural observations alongside visual and symbolic features.
"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

from npp_rl.models.gnn import create_graph_encoder


class NppMultimodalGraphExtractor(BaseFeaturesExtractor):
    """
    Multimodal feature extractor that combines CNN, MLP, and GNN encoders.
    
    This extractor processes:
    - Visual observations (player_frame, global_view) with CNNs
    - Symbolic observations (game_state) with MLPs  
    - Structural observations (graph_*) with GNNs
    
    The outputs are fused into a single feature representation.
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = 512,
        use_graph_obs: bool = False,
        use_3d_conv: bool = True,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_output_dim: int = 256,
        **kwargs
    ):
        """
        Initialize multimodal feature extractor.
        
        Args:
            observation_space: Gym observation space
            features_dim: Final output feature dimension
            use_graph_obs: Whether to process graph observations
            use_3d_conv: Whether to use 3D convolutions for temporal modeling
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_output_dim: Output dimension of GNN encoder
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, features_dim)
        
        self.use_graph_obs = use_graph_obs
        self.use_3d_conv = use_3d_conv
        
        # Extract observation space components
        self.has_player_frame = 'player_frame' in observation_space.spaces
        self.has_global_view = 'global_view' in observation_space.spaces
        self.has_game_state = 'game_state' in observation_space.spaces
        self.has_graph_obs = use_graph_obs and all(
            key in observation_space.spaces for key in [
                'graph_node_feats', 'graph_edge_index', 'graph_edge_feats',
                'graph_node_mask', 'graph_edge_mask'
            ]
        )
        
        # Initialize component encoders
        self._init_visual_encoders(observation_space)
        self._init_symbolic_encoder(observation_space)
        if self.has_graph_obs:
            self._init_graph_encoder(observation_space)
        
        # Calculate total feature dimension and create fusion network
        self._init_fusion_network()
    
    def _init_visual_encoders(self, observation_space: SpacesDict):
        """Initialize CNN encoders for visual observations."""
        self.visual_feature_dim = 0
        
        if self.has_player_frame:
            player_frame_shape = observation_space['player_frame'].shape
            
            if self.use_3d_conv and len(player_frame_shape) == 3 and player_frame_shape[2] > 1:
                # 3D convolutions for temporal modeling
                self.player_encoder = self._create_3d_cnn_encoder(player_frame_shape)
                self.player_output_dim = 512
            else:
                # 2D convolutions
                self.player_encoder = self._create_2d_cnn_encoder(player_frame_shape)
                self.player_output_dim = 512
                
            self.visual_feature_dim += self.player_output_dim
        
        if self.has_global_view:
            global_view_shape = observation_space['global_view'].shape
            self.global_encoder = self._create_2d_cnn_encoder(global_view_shape, prefix='global')
            self.global_output_dim = 256
            self.visual_feature_dim += self.global_output_dim
    
    def _init_symbolic_encoder(self, observation_space: SpacesDict):
        """Initialize MLP encoder for symbolic observations."""
        self.symbolic_feature_dim = 0
        
        if self.has_game_state:
            game_state_dim = observation_space['game_state'].shape[0]
            self.game_state_encoder = nn.Sequential(
                nn.Linear(game_state_dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.1),
                
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),
                
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.symbolic_feature_dim = 64
    
    def _init_graph_encoder(self, observation_space: SpacesDict):
        """Initialize GNN encoder for graph observations."""
        if not self.has_graph_obs:
            self.graph_feature_dim = 0
            return
            
        node_feat_dim = observation_space['graph_node_feats'].shape[1]
        edge_feat_dim = observation_space['graph_edge_feats'].shape[1]
        
        self.graph_encoder = create_graph_encoder(
            node_feature_dim=node_feat_dim,
            edge_feature_dim=edge_feat_dim,
            hidden_dim=128,
            num_layers=3,
            output_dim=256,
            aggregator='mean',
            global_pool='mean_max'
        )
        self.graph_feature_dim = 256
    
    def _init_fusion_network(self):
        """Initialize fusion network to combine all modalities."""
        total_dim = (
            self.visual_feature_dim + 
            self.symbolic_feature_dim + 
            (self.graph_feature_dim if self.has_graph_obs else 0)
        )
        
        if total_dim == 0:
            raise ValueError("No valid observation components found")
        
        # Multi-layer fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, self.features_dim),
            nn.ReLU(),
        )
    
    def _create_3d_cnn_encoder(self, input_shape: tuple) -> nn.Module:
        """Create 3D CNN encoder for temporal visual data."""
        height, width, temporal_frames = input_shape
        
        encoder = nn.Sequential(
            # First 3D conv layer
            nn.Conv3d(
                in_channels=1,
                out_channels=32,
                kernel_size=(4, 7, 7),
                stride=(2, 2, 2),
                padding=(1, 3, 3)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            
            # Second 3D conv layer
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            # Third 3D conv layer
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(2, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            # Adaptive pooling and flattening
            nn.AdaptiveAvgPool3d((1, 4, 4)),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(128 * 1 * 4 * 4, 512),
            nn.ReLU()
        )
        
        return encoder
    
    def _create_2d_cnn_encoder(self, input_shape: tuple, prefix: str = 'player') -> nn.Module:
        """Create 2D CNN encoder for visual data."""
        if len(input_shape) == 3:
            height, width, channels = input_shape
        else:
            height, width = input_shape
            channels = 1
        
        # Determine output dimension based on prefix
        output_dim = 512 if prefix == 'player' else 256
        
        encoder = nn.Sequential(
            # First conv layer
            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third conv layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Adaptive pooling and flattening
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(128 * 4 * 4, output_dim),
            nn.ReLU()
        )
        
        return encoder
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal feature extractor.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Fused feature representation
        """
        features = []
        
        # Process visual observations
        if self.has_player_frame:
            player_frame = observations['player_frame']
            
            if self.use_3d_conv and player_frame.dim() == 4 and player_frame.shape[3] > 1:
                # Reshape for 3D conv: (B, H, W, T) -> (B, 1, T, H, W)
                player_frame = player_frame.permute(0, 3, 1, 2).float() / 255.0
                player_frame = player_frame.unsqueeze(1)
            else:
                # Reshape for 2D conv: (B, H, W, C) -> (B, C, H, W)
                player_frame = player_frame.permute(0, 3, 1, 2).float() / 255.0
            
            player_features = self.player_encoder(player_frame)
            features.append(player_features)
        
        if self.has_global_view:
            global_view = observations['global_view']
            global_view = global_view.permute(0, 3, 1, 2).float() / 255.0
            global_features = self.global_encoder(global_view)
            features.append(global_features)
        
        # Process symbolic observations
        if self.has_game_state:
            game_state = observations['game_state'].float()
            game_state_features = self.game_state_encoder(game_state)
            features.append(game_state_features)
        
        # Process graph observations
        if self.has_graph_obs:
            graph_obs = {
                key: observations[key] for key in [
                    'graph_node_feats', 'graph_edge_index', 'graph_edge_feats',
                    'graph_node_mask', 'graph_edge_mask'
                ]
            }
            graph_features = self.graph_encoder(graph_obs)
            features.append(graph_features)
        
        # Fuse all features
        if len(features) == 0:
            raise ValueError("No valid observations to process")
        
        combined_features = torch.cat(features, dim=1)
        output = self.fusion_network(combined_features)
        
        return output


class NppMultimodalExtractor(BaseFeaturesExtractor):
    """
    Multimodal feature extractor without graph support (fallback).
    
    This is a simplified version that only processes visual and symbolic
    observations, for use when graph observations are not available.
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = 512,
        use_3d_conv: bool = True,
        **kwargs
    ):
        """Initialize multimodal feature extractor without graph support."""
        super().__init__(observation_space, features_dim)
        
        # Use the graph extractor but disable graph processing
        self.extractor = NppMultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            use_graph_obs=False,
            use_3d_conv=use_3d_conv,
            **kwargs
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multimodal extractor."""
        return self.extractor(observations)


def create_feature_extractor(
    observation_space: SpacesDict,
    features_dim: int = 512,
    use_graph_obs: bool = False,
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Factory function to create appropriate feature extractor.
    
    Args:
        observation_space: Gym observation space
        features_dim: Output feature dimension
        use_graph_obs: Whether to use graph observations
        **kwargs: Additional arguments
        
    Returns:
        Configured feature extractor
    """
    if use_graph_obs:
        return NppMultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            use_graph_obs=True,
            **kwargs
        )
    else:
        return NppMultimodalExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            **kwargs
        )