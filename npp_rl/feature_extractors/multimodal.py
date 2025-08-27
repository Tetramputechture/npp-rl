"""
Multimodal Feature Extractors for N++ RL Agent

This module implements advanced multimodal feature extractors that can handle
various types of observations including visual, symbolic, and graph-based data.

Key features:
- Multimodal fusion of visual, symbolic, and graph observations
- Graph Neural Network (GNN) support for structural data
- Flexible architecture that adapts to available observation types
- Factory functions for easy instantiation

The extractors in this module extend the basic temporal modeling with support
for graph-based structural observations and more sophisticated fusion techniques.
"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

from npp_rl.models.gnn import create_graph_encoder
from npp_rl.models.hgt_gnn import create_hgt_encoder


class MultimodalGraphExtractor(BaseFeaturesExtractor):
    """
    Advanced multimodal feature extractor with graph neural network support.
    
    This extractor processes multiple observation modalities:
    - Visual observations (player_frame, global_view) with CNNs
    - Symbolic observations (game_state) with MLPs  
    - Structural observations (graph_*) with GNNs
    
    The outputs are fused into a single feature representation using a
    multi-layer fusion network with batch normalization and dropout.
    
    This extractor is particularly useful for environments where structural
    relationships between entities are important for decision making.
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = 512,
        use_graph_obs: bool = False,
        use_hgt: bool = True,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        gnn_output_dim: int = 256,
        **kwargs
    ):
        """
        Initialize multimodal graph feature extractor.
        
        Args:
            observation_space: Gym observation space dictionary
            features_dim: Final output feature dimension
            use_graph_obs: Whether to process graph observations
            use_hgt: Whether to use HGT instead of basic GraphSAGE
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_output_dim: Output dimension of GNN encoder
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, features_dim)
        
        self.use_graph_obs = use_graph_obs
        self.use_hgt = use_hgt
        
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
            self._init_graph_encoder(observation_space, gnn_hidden_dim, gnn_num_layers, gnn_output_dim)
        
        # Calculate total feature dimension and create fusion network
        self._init_fusion_network()
    
    def _init_visual_encoders(self, observation_space: SpacesDict):
        """Initialize CNN encoders for visual observations."""
        self.visual_feature_dim = 0
        
        if self.has_player_frame:
            player_frame_shape = observation_space['player_frame'].shape
            
            self.player_encoder = self._create_3d_cnn_encoder(player_frame_shape)
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
    
    def _init_graph_encoder(self, observation_space: SpacesDict, hidden_dim: int, num_layers: int, output_dim: int):
        """Initialize GNN encoder for graph observations."""
        if not self.has_graph_obs:
            self.graph_feature_dim = 0
            return
            
        node_feat_dim = observation_space['graph_node_feats'].shape[1]
        edge_feat_dim = observation_space['graph_edge_feats'].shape[1]
        
        if self.use_hgt:
            # Use Heterogeneous Graph Transformer
            self.graph_encoder = create_hgt_encoder(
                node_feature_dim=node_feat_dim,
                edge_feature_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                num_heads=8,
                global_pool='mean_max'
            )
        else:
            # Use basic GraphSAGE
            self.graph_encoder = create_graph_encoder(
                node_feature_dim=node_feat_dim,
                edge_feature_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                aggregator='mean',
                global_pool='mean_max'
            )
        self.graph_feature_dim = output_dim
    
    def _init_fusion_network(self):
        """Initialize fusion network to combine all modalities."""
        total_dim = (
            self.visual_feature_dim + 
            self.symbolic_feature_dim + 
            (self.graph_feature_dim if self.has_graph_obs else 0)
        )
        
        if total_dim == 0:
            raise ValueError("No valid observation components found")
        
        # Multi-layer fusion network with residual connections
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
            # First 3D conv layer - optimized for temporal feature extraction
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
        """Create 2D CNN encoder for static visual data."""
        if len(input_shape) == 3:
            height, width, channels = input_shape
        else:
            height, width = input_shape
            channels = 1
        
        # Determine output dimension based on prefix
        output_dim = 512 if prefix == 'player' else 256
        
        encoder = nn.Sequential(
            # First conv layer - larger receptive field
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
            observations: Dictionary of observations with keys:
                - 'player_frame': Temporal visual observations (optional)
                - 'global_view': Static visual observations (optional)
                - 'game_state': Symbolic features (optional)
                - 'graph_*': Graph observations (optional)
            
        Returns:
            Fused feature representation of shape (batch_size, features_dim)
        """
        features = []
        
        # Process visual observations
        if self.has_player_frame:
            player_frame = observations['player_frame']
            
            # Reshape for 3D conv: (B, H, W, T) -> (B, 1, T, H, W)
            player_frame = player_frame.permute(0, 3, 1, 2).float() / 255.0
            player_frame = player_frame.unsqueeze(1)
            
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


class MultimodalExtractor(BaseFeaturesExtractor):
    """
    Simplified multimodal feature extractor without graph support.
    
    This is a fallback extractor that only processes visual and symbolic
    observations, for use when graph observations are not available or needed.
    
    It provides the same interface as MultimodalGraphExtractor but with
    reduced complexity for scenarios where structural information is not relevant.
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = 512,
        **kwargs
    ):
        """
        Initialize multimodal feature extractor without graph support.
        
        Args:
            observation_space: Gym observation space dictionary
            features_dim: Output feature dimension
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__(observation_space, features_dim)
        
        # Use the graph extractor but disable graph processing
        self.extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            use_graph_obs=False,
            **kwargs
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal extractor.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Extracted features of shape (batch_size, features_dim)
        """
        return self.extractor(observations)


def create_multimodal_extractor(
    observation_space: SpacesDict,
    features_dim: int = 512,
    use_graph_obs: bool = False,
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Factory function to create appropriate multimodal feature extractor.
    
    This function automatically selects between the graph-enabled and 
    standard multimodal extractors based on the use_graph_obs parameter
    and available observation space.
    
    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension
        use_graph_obs: Whether to use graph observations
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Configured multimodal feature extractor
    """
    if use_graph_obs:
        return MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            use_graph_obs=True,
            **kwargs
        )
    else:
        return MultimodalExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            **kwargs
        )


def create_hgt_multimodal_extractor(
    observation_space: SpacesDict,
    features_dim: int = 512,
    gnn_hidden_dim: int = 256,
    gnn_num_layers: int = 3,
    gnn_output_dim: int = 512,
    **kwargs
) -> MultimodalGraphExtractor:
    """
    Factory function to create HGT-enabled multimodal feature extractor.
    
    This function creates a multimodal extractor specifically configured
    to use Heterogeneous Graph Transformers for graph processing.
    
    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension
        gnn_hidden_dim: Hidden dimension for HGT layers
        gnn_num_layers: Number of HGT layers
        gnn_output_dim: Output dimension of HGT encoder
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Configured HGT multimodal feature extractor
    """
    return MultimodalGraphExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        use_graph_obs=True,
        use_hgt=True,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_output_dim=gnn_output_dim,
        **kwargs
    )


# Backward compatibility aliases
NppMultimodalGraphExtractor = MultimodalGraphExtractor
NppMultimodalExtractor = MultimodalExtractor
create_feature_extractor = create_multimodal_extractor
