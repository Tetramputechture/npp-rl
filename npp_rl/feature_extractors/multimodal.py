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
from npp_rl.models.spatial_attention import SpatialAttentionModule
from npp_rl.models.attention_constants import (
    DEFAULT_EMBED_DIM,
    DEFAULT_NUM_ATTENTION_HEADS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_SPATIAL_HEIGHT,
    DEFAULT_SPATIAL_WIDTH,
    FEATURE_EXPANSION_FACTOR
)


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
        use_cross_modal_attention: bool = True,
        use_spatial_attention: bool = True,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
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
            use_cross_modal_attention: Whether to use cross-modal attention
            use_spatial_attention: Whether to use graph-informed spatial attention
            num_attention_heads: Number of attention heads for transformers
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, features_dim)
        
        self.use_graph_obs = use_graph_obs
        self.use_hgt = use_hgt
        self.use_cross_modal_attention = use_cross_modal_attention
        self.use_spatial_attention = use_spatial_attention
        self.num_attention_heads = num_attention_heads
        
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
        self._init_fusion_network(observation_space)
    
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
    
    def _init_fusion_network(self, observation_space: SpacesDict):
        """Initialize fusion network with cross-modal attention and transformers."""
        total_dim = (
            self.visual_feature_dim + 
            self.symbolic_feature_dim + 
            (self.graph_feature_dim if self.has_graph_obs else 0)
        )
        
        if total_dim == 0:
            raise ValueError("No valid observation components found")
        
        # Common embedding dimension for cross-modal attention
        self.embed_dim = DEFAULT_EMBED_DIM
        
        # Project each modality to common embedding dimension
        if self.visual_feature_dim > 0:
            # Create separate projections for each visual component
            if self.has_player_frame:
                self.player_projection = nn.Linear(self.player_output_dim, self.embed_dim)
            if self.has_global_view:
                self.global_projection = nn.Linear(self.global_output_dim, self.embed_dim)
            
            # Combined visual projection for when both are present
            self.visual_projection = nn.Linear(self.visual_feature_dim, self.embed_dim)
            
        if self.symbolic_feature_dim > 0:
            self.symbolic_projection = nn.Linear(self.symbolic_feature_dim, self.embed_dim)
        if self.has_graph_obs:
            self.graph_projection = nn.Linear(self.graph_feature_dim, self.embed_dim)
        
        # Cross-modal attention mechanism
        if self.use_cross_modal_attention:
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_attention_heads,
                dropout=DEFAULT_DROPOUT_RATE,
                batch_first=True
            )
        
        # Graph-visual fusion transformer
        if self.use_cross_modal_attention and self.has_graph_obs:
            self.graph_visual_fusion = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_attention_heads,
                dim_feedforward=self.embed_dim * FEATURE_EXPANSION_FACTOR,
                dropout=DEFAULT_DROPOUT_RATE,
                batch_first=True
            )
        
        # Spatial attention for visual features
        if self.use_spatial_attention and self.has_graph_obs and self.visual_feature_dim > 0:
            # Get raw graph node feature dimension
            graph_node_dim = observation_space['graph_node_feats'].shape[1]
            self.spatial_attention = SpatialAttentionModule(
                graph_dim=graph_node_dim,
                visual_dim=self.visual_feature_dim,
                spatial_height=DEFAULT_SPATIAL_HEIGHT,
                spatial_width=DEFAULT_SPATIAL_WIDTH,
                num_attention_heads=self.num_attention_heads,
                dropout=DEFAULT_DROPOUT_RATE
            )
        
        # Calculate number of modalities for fusion
        num_modalities = sum([
            self.visual_feature_dim > 0,
            self.symbolic_feature_dim > 0,
            self.has_graph_obs
        ])
        
        # fusion network
        if self.use_cross_modal_attention:
            fusion_input_dim = self.embed_dim * num_modalities
        else:
            fusion_input_dim = total_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, self.features_dim),
            nn.ReLU(),
        )
        
        # Residual connection for features
        if self.use_cross_modal_attention:
            self.residual_projection = nn.Linear(fusion_input_dim, self.features_dim)
            self.residual_weight = nn.Parameter(torch.tensor(0.3))
    
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
        raw_features = []
        projected_features = []
        
        # Process visual observations
        visual_features = None
        visual_projected_features = []
        
        if self.has_player_frame:
            player_frame = observations['player_frame']
            
            # Reshape for 3D conv: (B, H, W, T) -> (B, 1, T, H, W)
            player_frame = player_frame.permute(0, 3, 1, 2).float() / 255.0
            player_frame = player_frame.unsqueeze(1)
            
            player_features = self.player_encoder(player_frame)
            raw_features.append(player_features)
            visual_features = player_features
            
            if self.use_cross_modal_attention:
                visual_projected_features.append(self.player_projection(player_features))
        
        if self.has_global_view:
            global_view = observations['global_view']
            global_view = global_view.permute(0, 3, 1, 2).float() / 255.0
            global_features = self.global_encoder(global_view)
            
            if visual_features is None:
                raw_features.append(global_features)
                visual_features = global_features
                if self.use_cross_modal_attention:
                    visual_projected_features.append(self.global_projection(global_features))
            else:
                # Combine with existing visual features
                combined_visual = torch.cat([visual_features, global_features], dim=1)
                visual_features = combined_visual
                raw_features[-1] = combined_visual  # Replace last visual features
                if self.use_cross_modal_attention:
                    visual_projected_features.append(self.global_projection(global_features))
        
        # Combine visual projected features if multiple visual modalities
        if self.use_cross_modal_attention and len(visual_projected_features) > 0:
            if len(visual_projected_features) == 1:
                projected_features.append(visual_projected_features[0])
            else:
                # Average or sum multiple visual projections
                combined_visual_proj = torch.stack(visual_projected_features, dim=0).mean(dim=0)
                projected_features.append(combined_visual_proj)
        
        # Process symbolic observations
        if self.has_game_state:
            game_state = observations['game_state'].float()
            game_state_features = self.game_state_encoder(game_state)
            raw_features.append(game_state_features)
            
            if self.use_cross_modal_attention:
                projected_features.append(self.symbolic_projection(game_state_features))
        
        # Process graph observations
        graph_features = None
        if self.has_graph_obs:
            graph_obs = {
                key: observations[key] for key in [
                    'graph_node_feats', 'graph_edge_index', 'graph_edge_feats',
                    'graph_node_mask', 'graph_edge_mask'
                ]
            }
            graph_features = self.graph_encoder(graph_obs)
            raw_features.append(graph_features)
            
            if self.use_cross_modal_attention:
                projected_features.append(self.graph_projection(graph_features))
        
        # Apply spatial attention if available
        if (self.use_spatial_attention and visual_features is not None and 
            graph_features is not None and hasattr(self, 'spatial_attention')):
            
            # Use raw graph node features for spatial attention (not aggregated features)
            graph_node_features = observations.get('graph_node_feats')
            if graph_node_features is not None:
                new_visual, attention_map = self.spatial_attention(
                    visual_features, graph_node_features
                )
            else:
                new_visual = visual_features
            
            # Update visual features with spatial attention
            visual_features = new_visual    
            
            # Update raw features
            for i, feat in enumerate(raw_features):
                if feat.shape == visual_features.shape:  # Find visual features by shape
                    raw_features[i] = new_visual
                    break
            
            # Update projected features if using cross-modal attention
            if self.use_cross_modal_attention and len(projected_features) > 0:
                # Re-project the visual features
                if self.has_player_frame and self.has_global_view:
                    # For combined visual features, use the combined projection
                    projected_features[0] = self.visual_projection(new_visual)
                elif self.has_player_frame:
                    projected_features[0] = self.player_projection(new_visual)
                elif self.has_global_view:
                    projected_features[0] = self.global_projection(new_visual)
        
        # fusion with cross-modal attention
        if self.use_cross_modal_attention and len(projected_features) > 1:
            # Stack features for attention
            stacked_features = torch.stack(projected_features, dim=1)  # [B, num_modalities, embed_dim]
            
            # Apply cross-modal attention
            attended_features, attention_weights = self.cross_modal_attention(
                query=stacked_features,
                key=stacked_features,
                value=stacked_features
            )
            
            # Apply graph-visual fusion transformer if available
            if (hasattr(self, 'graph_visual_fusion') and self.has_graph_obs and 
                visual_features is not None):
                attended_features = self.graph_visual_fusion(attended_features)
            
            # Flatten for fusion network
            combined_features = attended_features.view(attended_features.shape[0], -1)
            
        else:
            # Fallback to basic concatenation
            if len(raw_features) == 0:
                raise ValueError("No valid observations to process")
            combined_features = torch.cat(raw_features, dim=1)
        
        # Final fusion
        output = self.fusion_network(combined_features)
        
        # Apply residual connection if using cross-modal attention
        if self.use_cross_modal_attention and hasattr(self, 'residual_projection'):
            residual = self.residual_projection(combined_features)
            output = self.residual_weight * output + (1 - self.residual_weight) * residual
        
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
    use_cross_modal_attention: bool = True,
    use_spatial_attention: bool = True,
    num_attention_heads: int = 8,
    **kwargs
) -> MultimodalGraphExtractor:
    """
    Factory function to create HGT-enabled multimodal feature extractor with fusion.
    
    This function creates a multimodal extractor specifically configured
    to use Heterogeneous Graph Transformers for graph processing with
    cross-modal attention and spatial attention mechanisms.
    
    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension
        gnn_hidden_dim: Hidden dimension for HGT layers
        gnn_num_layers: Number of HGT layers
        gnn_output_dim: Output dimension of HGT encoder
        use_cross_modal_attention: Whether to use cross-modal attention
        use_spatial_attention: Whether to use graph-informed spatial attention
        num_attention_heads: Number of attention heads for transformers
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Configured HGT multimodal feature extractor with fusion
    """
    return MultimodalGraphExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        use_graph_obs=True,
        use_hgt=True,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_output_dim=gnn_output_dim,
        use_cross_modal_attention=use_cross_modal_attention,
        use_spatial_attention=use_spatial_attention,
        num_attention_heads=num_attention_heads,
        **kwargs
    )


# Backward compatibility aliases
NppMultimodalGraphExtractor = MultimodalGraphExtractor
NppMultimodalExtractor = MultimodalExtractor
create_feature_extractor = create_multimodal_extractor
