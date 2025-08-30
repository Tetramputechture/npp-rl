"""
Spatial Attention Module for Graph-Informed Visual Processing

This module implements spatial attention mechanisms that use graph structure
to guide CNN attention, enabling the visual encoder to focus on regions
that are structurally important according to the graph representation.

Key Components:
- SpatialAttentionModule: Graph-guided spatial attention for CNN features
- GraphSpatialGuidance: Converts graph structure to spatial attention maps
- MultiScaleSpatialAttention: Multi-resolution spatial attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

from .attention_constants import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_GUIDANCE_DIM,
    DEFAULT_SPATIAL_HEIGHT,
    DEFAULT_SPATIAL_WIDTH,
    DEFAULT_NUM_ATTENTION_HEADS,
    DEFAULT_RESIDUAL_WEIGHT,
    DEFAULT_SCALES,
    CONV_KERNEL_SIZE_3x3,
    CONV_KERNEL_SIZE_1x1,
    CONV_PADDING_SAME,
    FEATURE_EXPANSION_FACTOR,
    GUIDANCE_DIM_REDUCTION_FACTOR,
    ATTENTION_HEAD_REDUCTION_FACTOR,
    MIN_ATTENTION_HEADS,
    MIN_SPATIAL_DIM,
    MIN_FEATURE_DIM,
    BILINEAR_MODE,
    ALIGN_CORNERS,
    LAYER_NORM_EPS
)


class GraphSpatialGuidance(nn.Module):
    """
    Converts graph structure information into spatial attention guidance.
    
    This module takes graph node features and connectivity information
    and produces spatial attention maps that can guide CNN processing.
    """
    
    def __init__(
        self,
        graph_dim: int,
        spatial_height: int = DEFAULT_SPATIAL_HEIGHT,
        spatial_width: int = DEFAULT_SPATIAL_WIDTH,
        guidance_dim: int = DEFAULT_GUIDANCE_DIM,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS // ATTENTION_HEAD_REDUCTION_FACTOR,
        dropout: float = DEFAULT_DROPOUT_RATE
    ):
        """
        Initialize graph spatial guidance module.
        
        Args:
            graph_dim: Dimension of graph node features
            spatial_height: Height of spatial attention map
            spatial_width: Width of spatial attention map
            guidance_dim: Dimension for guidance processing
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Validate inputs
        if graph_dim < MIN_FEATURE_DIM:
            raise ValueError(f"graph_dim must be at least {MIN_FEATURE_DIM}, got {graph_dim}")
        if spatial_height < MIN_SPATIAL_DIM or spatial_width < MIN_SPATIAL_DIM:
            raise ValueError(f"Spatial dimensions must be at least {MIN_SPATIAL_DIM}")
        if num_attention_heads < MIN_ATTENTION_HEADS:
            raise ValueError(f"num_attention_heads must be at least {MIN_ATTENTION_HEADS}")
        if guidance_dim % num_attention_heads != 0:
            raise ValueError(f"guidance_dim ({guidance_dim}) must be divisible by num_attention_heads ({num_attention_heads})")
        
        self.graph_dim = graph_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.guidance_dim = guidance_dim
        self.num_heads = num_attention_heads
        self.dropout = dropout
        
        # Graph feature processor
        self.graph_processor = self._create_feature_processor(graph_dim, guidance_dim, dropout)
        
        # Spatial position encoder (2D coordinates -> guidance_dim)
        spatial_intermediate_dim = guidance_dim // GUIDANCE_DIM_REDUCTION_FACTOR
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, spatial_intermediate_dim),  # x, y coordinates
            nn.ReLU(),
            nn.Linear(spatial_intermediate_dim, guidance_dim),
            nn.ReLU()
        )
        
        # Cross-attention between graph and spatial positions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=guidance_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Spatial attention map generator
        attention_intermediate_dim = guidance_dim // GUIDANCE_DIM_REDUCTION_FACTOR
        self.attention_generator = nn.Sequential(
            nn.Linear(guidance_dim, attention_intermediate_dim),
            nn.ReLU(),
            nn.Linear(attention_intermediate_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize spatial position grid
        self.register_buffer('spatial_positions', self._create_spatial_positions())
    
    def _create_feature_processor(self, input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        """Create a standard feature processing block."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim, eps=LAYER_NORM_EPS),
            nn.Dropout(dropout)
        )
    
    def _create_spatial_positions(self) -> torch.Tensor:
        """Create normalized spatial position grid."""
        y_coords = torch.linspace(0, 1, self.spatial_height)
        x_coords = torch.linspace(0, 1, self.spatial_width)
        
        # Create meshgrid and flatten
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        return positions  # [H*W, 2]
    
    def forward(
        self,
        graph_features: torch.Tensor,
        graph_node_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate spatial attention map from graph structure.
        
        Args:
            graph_features: Graph node features [batch_size, num_nodes, graph_dim]
            graph_node_positions: Optional node positions [batch_size, num_nodes, 2]
            
        Returns:
            Spatial attention map [batch_size, spatial_height, spatial_width]
        """
        batch_size, num_nodes, feature_dim = graph_features.shape
        
        # Validate input dimensions
        if feature_dim != self.graph_dim:
            raise ValueError(f"Expected graph features with dim {self.graph_dim}, got {feature_dim}")
        if num_nodes == 0:
            raise ValueError("Graph features cannot be empty")
        
        # Process graph features
        processed_graph = self.graph_processor(graph_features)  # [B, N, guidance_dim]
        
        # Encode spatial positions
        spatial_positions = self.spatial_positions.unsqueeze(0).expand(batch_size, -1, -1)
        encoded_spatial = self.spatial_encoder(spatial_positions)  # [B, H*W, guidance_dim]
        
        # Cross-attention between spatial positions and graph nodes
        attended_spatial, _ = self.cross_attention(
            query=encoded_spatial,  # Spatial positions as queries
            key=processed_graph,    # Graph nodes as keys
            value=processed_graph   # Graph nodes as values
        )
        
        # Generate attention map
        attention_map = self.attention_generator(attended_spatial)  # [B, H*W, 1]
        attention_map = attention_map.squeeze(-1)  # [B, H*W]
        
        # Reshape to spatial dimensions
        attention_map = attention_map.view(batch_size, self.spatial_height, self.spatial_width)
        
        return attention_map


class SpatialAttentionModule(nn.Module):
    """
    Graph-informed spatial attention module for CNN features.
    
    This module applies spatial attention to CNN feature maps based on
    graph structure information, allowing the visual encoder to focus
    on regions that are important according to the graph representation.
    """
    
    def __init__(
        self,
        graph_dim: int,
        visual_dim: int,
        spatial_height: int = DEFAULT_SPATIAL_HEIGHT,
        spatial_width: int = DEFAULT_SPATIAL_WIDTH,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT_RATE
    ):
        """
        Initialize spatial attention module.
        
        Args:
            graph_dim: Dimension of graph features
            visual_dim: Dimension of visual features
            spatial_height: Height of spatial attention maps
            spatial_width: Width of spatial attention maps
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Validate inputs
        if graph_dim < MIN_FEATURE_DIM or visual_dim < MIN_FEATURE_DIM:
            raise ValueError(f"Feature dimensions must be at least {MIN_FEATURE_DIM}")
        if spatial_height < MIN_SPATIAL_DIM or spatial_width < MIN_SPATIAL_DIM:
            raise ValueError(f"Spatial dimensions must be at least {MIN_SPATIAL_DIM}")
        if num_attention_heads < MIN_ATTENTION_HEADS:
            raise ValueError(f"num_attention_heads must be at least {MIN_ATTENTION_HEADS}")
        
        self.graph_dim = graph_dim
        self.visual_dim = visual_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.num_heads = num_attention_heads
        self.dropout = dropout
        
        # Graph spatial guidance
        guidance_dim = max(visual_dim // GUIDANCE_DIM_REDUCTION_FACTOR, MIN_FEATURE_DIM)
        guidance_heads = max(num_attention_heads // ATTENTION_HEAD_REDUCTION_FACTOR, MIN_ATTENTION_HEADS)
        
        self.spatial_guidance = GraphSpatialGuidance(
            graph_dim=graph_dim,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            guidance_dim=guidance_dim,
            num_attention_heads=guidance_heads,
            dropout=dropout
        )
        
        # Visual feature processor
        self.visual_processor = self._create_feature_processor(visual_dim, visual_dim, dropout)
        
        # Spatial attention fusion (CNN for attention map refinement)
        conv_channels = [16, 8]  # Configurable channel progression
        self.attention_fusion = self._create_attention_fusion_cnn(conv_channels, dropout)
        
        # Feature enhancement
        enhanced_dim = visual_dim * FEATURE_EXPANSION_FACTOR
        self.feature_enhancer = nn.Sequential(
            nn.Linear(visual_dim, enhanced_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(enhanced_dim, visual_dim),
            nn.LayerNorm(visual_dim, eps=LAYER_NORM_EPS)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(DEFAULT_RESIDUAL_WEIGHT))
    
    def _create_feature_processor(self, input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        """Create a standard feature processing block."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim, eps=LAYER_NORM_EPS),
            nn.Dropout(dropout)
        )
    
    def _create_attention_fusion_cnn(self, channels: list, dropout: float) -> nn.Module:
        """Create CNN for attention map fusion."""
        layers = []
        in_channels = 1
        
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=CONV_KERNEL_SIZE_3x3, 
                         padding=CONV_PADDING_SAME),
                nn.ReLU(),
                nn.Dropout2d(dropout)
            ])
            in_channels = out_channels
        
        # Final output layer
        layers.extend([
            nn.Conv2d(in_channels, 1, kernel_size=CONV_KERNEL_SIZE_1x1),
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        graph_features: torch.Tensor,
        feature_map_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply graph-informed spatial attention to visual features.
        
        Args:
            visual_features: Visual features [batch_size, visual_dim]
            graph_features: Graph features [batch_size, num_nodes, graph_dim]
            feature_map_size: Optional size for spatial attention (H, W)
            
        Returns:
            Tuple of (enhanced_visual_features, attention_map)
        """
        batch_size, visual_dim = visual_features.shape
        
        # Validate inputs
        if visual_dim != self.visual_dim:
            raise ValueError(f"Expected visual features with dim {self.visual_dim}, got {visual_dim}")
        if len(graph_features.shape) != 3:
            raise ValueError(f"Expected 3D graph features, got {len(graph_features.shape)}D")
        if graph_features.shape[0] != batch_size:
            raise ValueError("Batch size mismatch between visual and graph features")
        
        # Use provided feature map size or default
        if feature_map_size is not None:
            spatial_h, spatial_w = feature_map_size
            if spatial_h < MIN_SPATIAL_DIM or spatial_w < MIN_SPATIAL_DIM:
                raise ValueError(f"Feature map size must be at least {MIN_SPATIAL_DIM}x{MIN_SPATIAL_DIM}")
        else:
            spatial_h, spatial_w = self.spatial_height, self.spatial_width
        
        # Generate spatial attention map from graph
        attention_map = self.spatial_guidance(graph_features)  # [B, H, W]
        
        # Resize attention map if needed
        if attention_map.shape[1] != spatial_h or attention_map.shape[2] != spatial_w:
            attention_map = F.interpolate(
                attention_map.unsqueeze(1),
                size=(spatial_h, spatial_w),
                mode=BILINEAR_MODE,
                align_corners=ALIGN_CORNERS
            ).squeeze(1)
        
        # Process visual features
        processed_visual = self.visual_processor(visual_features)
        
        # Apply spatial attention fusion (enhance attention map)
        enhanced_attention = self.attention_fusion(attention_map.unsqueeze(1))  # [B, 1, H, W]
        enhanced_attention = enhanced_attention.squeeze(1)  # [B, H, W]
        
        # Global average pooling of attention map for feature weighting
        # Add small epsilon to prevent division by zero
        attention_weight = torch.mean(enhanced_attention.view(batch_size, -1), dim=1, keepdim=True)
        attention_weight = torch.clamp(attention_weight, min=1e-8)  # Prevent zero weights
        
        # Apply attention to visual features
        attended_visual = processed_visual * attention_weight
        
        # Feature enhancement
        enhanced_visual = self.feature_enhancer(attended_visual)
        
        # Residual connection with clamped weight
        residual_weight = torch.clamp(self.residual_weight, 0.0, 1.0)
        output_visual = residual_weight * enhanced_visual + (1 - residual_weight) * visual_features
        
        return output_visual, enhanced_attention


class MultiScaleSpatialAttention(nn.Module):
    """
    Multi-scale spatial attention that operates at different resolution levels.
    
    This module applies spatial attention at multiple scales to capture
    both fine-grained and coarse-grained spatial relationships.
    """
    
    def __init__(
        self,
        graph_dim: int,
        visual_dim: int,
        scales: list = None,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT_RATE
    ):
        """
        Initialize multi-scale spatial attention.
        
        Args:
            graph_dim: Dimension of graph features
            visual_dim: Dimension of visual features
            scales: List of spatial scales (heights/widths)
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Use default scales if none provided
        if scales is None:
            scales = DEFAULT_SCALES.copy()
        
        # Validate inputs
        if graph_dim < MIN_FEATURE_DIM or visual_dim < MIN_FEATURE_DIM:
            raise ValueError(f"Feature dimensions must be at least {MIN_FEATURE_DIM}")
        if not scales or any(s < MIN_SPATIAL_DIM for s in scales):
            raise ValueError(f"All scales must be at least {MIN_SPATIAL_DIM}")
        if num_attention_heads < MIN_ATTENTION_HEADS:
            raise ValueError(f"num_attention_heads must be at least {MIN_ATTENTION_HEADS}")
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create spatial attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            SpatialAttentionModule(
                graph_dim=graph_dim,
                visual_dim=visual_dim,
                spatial_height=scale,
                spatial_width=scale,
                num_attention_heads=num_attention_heads,
                dropout=dropout
            )
            for scale in scales
        ])
        
        # Scale fusion network
        fusion_input_dim = visual_dim * self.num_scales
        fusion_hidden_dim = visual_dim * FEATURE_EXPANSION_FACTOR
        
        self.scale_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, visual_dim),
            nn.LayerNorm(visual_dim, eps=LAYER_NORM_EPS)
        )
        
        # Scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        graph_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply multi-scale spatial attention.
        
        Args:
            visual_features: Visual features [batch_size, visual_dim]
            graph_features: Graph features [batch_size, num_nodes, graph_dim]
            
        Returns:
            Tuple of (enhanced_features, attention_maps)
        """
        scale_features = []
        attention_maps = {}
        
        # Apply attention at each scale
        for i, (scale, attention_module) in enumerate(zip(self.scales, self.scale_attentions)):
            enhanced_features, attention_map = attention_module(
                visual_features, graph_features, feature_map_size=(scale, scale)
            )
            scale_features.append(enhanced_features)
            attention_maps[f'scale_{scale}'] = attention_map
        
        # Weighted combination of scale features
        weighted_features = []
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        for i, features in enumerate(scale_features):
            weighted_features.append(features * scale_weights[i])
        
        # Concatenate and fuse
        concatenated_features = torch.cat(scale_features, dim=-1)
        fused_features = self.scale_fusion(concatenated_features)
        
        return fused_features, attention_maps