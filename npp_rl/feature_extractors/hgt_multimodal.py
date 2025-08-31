"""
HGT-based Multimodal Feature Extractor - Primary Architecture

This module implements the state-of-the-art Heterogeneous Graph Transformer (HGT)
based multimodal feature extractor for N++ RL agents. This is the PRIMARY and
RECOMMENDED architecture for the project.

Key advantages of HGT approach:
- Handles heterogeneous node types (grid cells, entities, hazards)
- Specialized attention mechanisms for different edge types (movement, functional)
- Superior performance on complex spatial reasoning tasks
- Type-aware processing with entity-specific embeddings
- Advanced multimodal fusion with cross-modal attention

Based on "Heterogeneous Graph Transformer" by Wang et al. (2020).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

from npp_rl.models.hgt_gnn import create_hgt_encoder
from npp_rl.models.spatial_attention import SpatialAttentionModule
from npp_rl.models.hgt_config import (
    CNN_CONFIG,
    POOLING_CONFIG,
    DEFAULT_CONFIG,
    FACTORY_CONFIG,
    MULTIPLIER_CONFIG
)


class HGTMultimodalExtractor(BaseFeaturesExtractor):
    """
    State-of-the-art HGT-based multimodal feature extractor.
    
    This is the PRIMARY and RECOMMENDED feature extractor for N++ RL agents.
    It combines Heterogeneous Graph Transformers with advanced multimodal fusion
    to achieve superior performance on complex spatial reasoning tasks.
    
    Architecture:
    1. Visual Processing: 3D CNN for temporal frames + 2D CNN for global view
    2. Graph Processing: HGT with type-specific attention for heterogeneous graphs
    3. State Processing: MLP for physics/game state features
    4. Multimodal Fusion: Cross-modal attention with spatial awareness
    
    The HGT component handles:
    - Heterogeneous node types: grid cells, entities, hazards, switches, etc.
    - Specialized edge types: movement (walk, jump, fall), functional relationships
    - Type-aware attention mechanisms for optimal feature learning
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = DEFAULT_CONFIG.embed_dim,
        # HGT parameters
        hgt_hidden_dim: int = DEFAULT_CONFIG.hidden_dim,
        hgt_num_layers: int = DEFAULT_CONFIG.num_layers,
        hgt_output_dim: int = DEFAULT_CONFIG.output_dim,
        hgt_num_heads: int = DEFAULT_CONFIG.num_attention_heads,
        # Visual processing parameters
        visual_hidden_dim: int = DEFAULT_CONFIG.visual_hidden_dim,
        global_hidden_dim: int = DEFAULT_CONFIG.global_hidden_dim,
        # State processing parameters
        state_hidden_dim: int = DEFAULT_CONFIG.state_hidden_dim,
        # Fusion parameters
        use_cross_modal_attention: bool = True,
        use_spatial_attention: bool = True,
        num_attention_heads: int = DEFAULT_CONFIG.num_attention_heads,
        dropout: float = DEFAULT_CONFIG.dropout_rate,
        **kwargs
    ):
        """
        Initialize HGT-based multimodal feature extractor.
        
        Args:
            observation_space: Gym observation space dictionary
            features_dim: Final output feature dimension
            hgt_hidden_dim: Hidden dimension for HGT layers
            hgt_num_layers: Number of HGT layers
            hgt_output_dim: Output dimension of HGT encoder
            hgt_num_heads: Number of attention heads in HGT
            visual_hidden_dim: Hidden dimension for visual processing
            global_hidden_dim: Hidden dimension for global view processing
            state_hidden_dim: Hidden dimension for state processing
            use_cross_modal_attention: Whether to use cross-modal attention
            use_spatial_attention: Whether to use spatial attention
            num_attention_heads: Number of attention heads for fusion
            dropout: Dropout probability
        """
        super().__init__(observation_space, features_dim)
        
        self.use_cross_modal_attention = use_cross_modal_attention
        self.use_spatial_attention = use_spatial_attention
        
        # Visual processing branch (temporal frames)
        if 'player_frame' in observation_space.spaces:
            visual_shape = observation_space['player_frame'].shape
            self.visual_cnn = nn.Sequential(
                # 3D CNN for temporal modeling
                nn.Conv3d(CNN_CONFIG.conv3d_layer1.in_channels, CNN_CONFIG.conv3d_layer1.out_channels, 
                         kernel_size=CNN_CONFIG.conv3d_layer1.kernel_size, 
                         stride=CNN_CONFIG.conv3d_layer1.stride, 
                         padding=CNN_CONFIG.conv3d_layer1.padding),
                nn.ReLU(),
                nn.Conv3d(CNN_CONFIG.conv3d_layer2.in_channels, CNN_CONFIG.conv3d_layer2.out_channels, 
                         kernel_size=CNN_CONFIG.conv3d_layer2.kernel_size, 
                         stride=CNN_CONFIG.conv3d_layer2.stride, 
                         padding=CNN_CONFIG.conv3d_layer2.padding),
                nn.ReLU(),
                nn.Conv3d(CNN_CONFIG.conv3d_layer3.in_channels, CNN_CONFIG.conv3d_layer3.out_channels, 
                         kernel_size=CNN_CONFIG.conv3d_layer3.kernel_size, 
                         stride=CNN_CONFIG.conv3d_layer3.stride, 
                         padding=CNN_CONFIG.conv3d_layer3.padding),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(POOLING_CONFIG.adaptive_pool3d_output_size),
                nn.Flatten(),
                nn.Linear(POOLING_CONFIG.cnn_flattened_size, visual_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.has_visual = True
        else:
            self.has_visual = False
            visual_hidden_dim = 0
        
        # Global view processing branch
        if 'global_view' in observation_space.spaces:
            global_shape = observation_space['global_view'].shape
            self.global_cnn = nn.Sequential(
                nn.Conv2d(CNN_CONFIG.conv2d_layer1.in_channels, CNN_CONFIG.conv2d_layer1.out_channels, 
                         kernel_size=CNN_CONFIG.conv2d_layer1.kernel_size[0], 
                         stride=CNN_CONFIG.conv2d_layer1.stride[0], 
                         padding=CNN_CONFIG.conv2d_layer1.padding[0]),
                nn.ReLU(),
                nn.Conv2d(CNN_CONFIG.conv2d_layer2.in_channels, CNN_CONFIG.conv2d_layer2.out_channels, 
                         kernel_size=CNN_CONFIG.conv2d_layer2.kernel_size[0], 
                         stride=CNN_CONFIG.conv2d_layer2.stride[0], 
                         padding=CNN_CONFIG.conv2d_layer2.padding[0]),
                nn.ReLU(),
                nn.Conv2d(CNN_CONFIG.conv2d_layer3.in_channels, CNN_CONFIG.conv2d_layer3.out_channels, 
                         kernel_size=CNN_CONFIG.conv2d_layer3.kernel_size[0], 
                         stride=CNN_CONFIG.conv2d_layer3.stride[0], 
                         padding=CNN_CONFIG.conv2d_layer3.padding[0]),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(POOLING_CONFIG.adaptive_pool2d_output_size),
                nn.Flatten(),
                nn.Linear(POOLING_CONFIG.cnn_flattened_size, global_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.has_global = True
        else:
            self.has_global = False
            global_hidden_dim = 0
        
        # State processing branch
        if 'game_state' in observation_space.spaces:
            state_dim = observation_space['game_state'].shape[0]
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, state_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_hidden_dim, state_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.has_state = True
        else:
            self.has_state = False
            state_hidden_dim = 0
        
        # HGT graph processing branch
        if 'graph_node_feats' in observation_space.spaces and 'graph_edge_feats' in observation_space.spaces:
            node_feat_dim = observation_space['graph_node_feats'].shape[1]
            edge_feat_dim = observation_space['graph_edge_feats'].shape[1]
            
            self.hgt_encoder = create_hgt_encoder(
                node_feature_dim=node_feat_dim,
                edge_feature_dim=edge_feat_dim,
                hidden_dim=hgt_hidden_dim,
                num_layers=hgt_num_layers,
                output_dim=hgt_output_dim,
                num_heads=hgt_num_heads,
                global_pool='mean_max'
            )
            self.has_graph = True
            # HGT with mean_max pooling outputs multiplier * hgt_output_dim
            graph_output_dim = MULTIPLIER_CONFIG.hgt_output_multiplier * hgt_output_dim
        else:
            self.has_graph = False
            graph_output_dim = 0
        
        # Spatial attention module (if enabled)
        if self.use_spatial_attention and self.has_graph and self.has_visual:
            self.spatial_attention = SpatialAttentionModule(
                graph_dim=graph_output_dim,
                visual_dim=visual_hidden_dim,
                spatial_height=DEFAULT_CONFIG.spatial_height,
                spatial_width=DEFAULT_CONFIG.spatial_width,
                num_heads=num_attention_heads
            )
        
        # Calculate total feature dimension
        total_dim = visual_hidden_dim + global_hidden_dim + state_hidden_dim + graph_output_dim
        
        # Cross-modal attention fusion (if enabled)
        if self.use_cross_modal_attention and total_dim > 0:
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=total_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(total_dim)
        
        # Final fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, features_dim * MULTIPLIER_CONFIG.fusion_expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim * MULTIPLIER_CONFIG.fusion_expansion_factor, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through HGT multimodal extractor.
        
        Args:
            observations: Dictionary of observation tensors
            
        Returns:
            Fused feature representation
        """
        features = []
        
        # Process visual observations
        if self.has_visual and 'player_frame' in observations:
            visual_obs = observations['player_frame']
            if visual_obs.dim() == 4:  # Add batch dimension if missing
                visual_obs = visual_obs.unsqueeze(1)
            visual_features = self.visual_cnn(visual_obs)
            features.append(visual_features)
        
        # Process global view
        if self.has_global and 'global_view' in observations:
            global_obs = observations['global_view']
            if global_obs.dim() == 3:  # Add channel dimension if missing
                global_obs = global_obs.unsqueeze(1)
            global_features = self.global_cnn(global_obs)
            features.append(global_features)
        
        # Process state observations
        if self.has_state and 'game_state' in observations:
            state_features = self.state_mlp(observations['game_state'])
            features.append(state_features)
        
        # Process graph observations with HGT
        if self.has_graph and all(key in observations for key in 
                                 ['graph_node_feats', 'graph_edge_feats', 'graph_edge_index',
                                  'graph_node_types', 'graph_edge_types', 'graph_node_mask', 'graph_edge_mask']):
            graph_features = self.hgt_encoder(
                node_features=observations['graph_node_feats'],
                edge_features=observations['graph_edge_feats'],
                edge_index=observations['graph_edge_index'],
                node_types=observations['graph_node_types'],
                edge_types=observations['graph_edge_types'],
                node_mask=observations['graph_node_mask'],
                edge_mask=observations['graph_edge_mask']
            )
            features.append(graph_features)
        
        # Concatenate all features
        if not features:
            raise ValueError("No valid observations found")
        
        combined_features = torch.cat(features, dim=1)
        
        # Apply spatial attention if enabled
        if (self.use_spatial_attention and self.has_graph and self.has_visual and 
            len(features) >= 2):
            # Apply spatial attention between graph and visual features
            combined_features = self.spatial_attention(
                graph_features=features[-1],  # Graph features (last added)
                visual_features=features[0],  # Visual features (first added)
                combined_features=combined_features
            )
        
        # Apply cross-modal attention if enabled
        if self.use_cross_modal_attention:
            # Reshape for attention (batch_size, seq_len=1, feature_dim)
            attn_input = combined_features.unsqueeze(1)
            attn_output, _ = self.cross_modal_attention(
                attn_input, attn_input, attn_input
            )
            combined_features = self.attention_norm(
                combined_features + attn_output.squeeze(1)
            )
        
        # Final fusion
        output_features = self.fusion_network(combined_features)
        
        return output_features


def create_hgt_multimodal_extractor(
    observation_space: SpacesDict,
    features_dim: int = FACTORY_CONFIG.features_dim,
    hgt_hidden_dim: int = FACTORY_CONFIG.hgt_hidden_dim,
    hgt_num_layers: int = FACTORY_CONFIG.hgt_num_layers,
    hgt_output_dim: int = FACTORY_CONFIG.hgt_output_dim,
    **kwargs
) -> HGTMultimodalExtractor:
    """
    Factory function to create the primary HGT-based multimodal feature extractor.
    
    This is the RECOMMENDED way to create feature extractors for N++ RL agents.
    
    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension
        hgt_hidden_dim: Hidden dimension for HGT layers
        hgt_num_layers: Number of HGT layers
        hgt_output_dim: Output dimension of HGT encoder
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Configured HGT multimodal feature extractor
    """
    return HGTMultimodalExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        hgt_hidden_dim=hgt_hidden_dim,
        hgt_num_layers=hgt_num_layers,
        hgt_output_dim=hgt_output_dim,
        use_cross_modal_attention=True,
        use_spatial_attention=True,
        **kwargs
    )