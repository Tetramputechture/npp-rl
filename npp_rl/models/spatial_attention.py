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


class GraphSpatialGuidance(nn.Module):
    """
    Converts graph structure information into spatial attention guidance.
    
    This module takes graph node features and connectivity information
    and produces spatial attention maps that can guide CNN processing.
    """
    
    def __init__(
        self,
        graph_dim: int,
        spatial_height: int,
        spatial_width: int,
        guidance_dim: int = 64,
        num_attention_heads: int = 4
    ):
        """
        Initialize graph spatial guidance module.
        
        Args:
            graph_dim: Dimension of graph node features
            spatial_height: Height of spatial attention map
            spatial_width: Width of spatial attention map
            guidance_dim: Dimension for guidance processing
            num_attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.graph_dim = graph_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.guidance_dim = guidance_dim
        self.num_heads = num_attention_heads
        
        # Graph feature processor
        self.graph_processor = nn.Sequential(
            nn.Linear(graph_dim, guidance_dim),
            nn.ReLU(),
            nn.LayerNorm(guidance_dim),
            nn.Dropout(0.1)
        )
        
        # Spatial position encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, guidance_dim // 2),  # x, y coordinates
            nn.ReLU(),
            nn.Linear(guidance_dim // 2, guidance_dim),
            nn.ReLU()
        )
        
        # Cross-attention between graph and spatial positions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=guidance_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Spatial attention map generator
        self.attention_generator = nn.Sequential(
            nn.Linear(guidance_dim, guidance_dim // 2),
            nn.ReLU(),
            nn.Linear(guidance_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize spatial position grid
        self.register_buffer('spatial_positions', self._create_spatial_positions())
    
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
        batch_size = graph_features.shape[0]
        num_spatial_positions = self.spatial_positions.shape[0]
        
        # Process graph features
        processed_graph = self.graph_processor(graph_features)  # [B, N, guidance_dim]
        
        # Encode spatial positions
        spatial_positions = self.spatial_positions.unsqueeze(0).expand(batch_size, -1, -1)
        encoded_spatial = self.spatial_encoder(spatial_positions)  # [B, H*W, guidance_dim]
        
        # Cross-attention between spatial positions and graph nodes
        attended_spatial, attention_weights = self.cross_attention(
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
        spatial_height: int = 16,
        spatial_width: int = 16,
        num_attention_heads: int = 8,
        dropout: float = 0.1
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
        
        self.graph_dim = graph_dim
        self.visual_dim = visual_dim
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.num_heads = num_attention_heads
        
        # Graph spatial guidance
        self.spatial_guidance = GraphSpatialGuidance(
            graph_dim=graph_dim,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
            guidance_dim=visual_dim // 2,
            num_attention_heads=num_attention_heads // 2
        )
        
        # Visual feature processor
        self.visual_processor = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.LayerNorm(visual_dim),
            nn.Dropout(dropout)
        )
        
        # Spatial attention fusion
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature enhancement
        self.feature_enhancer = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim * 2, visual_dim),
            nn.LayerNorm(visual_dim)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.7))
    
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
        batch_size = visual_features.shape[0]
        
        # Use provided feature map size or default
        if feature_map_size is not None:
            spatial_h, spatial_w = feature_map_size
        else:
            spatial_h, spatial_w = self.spatial_height, self.spatial_width
        
        # Generate spatial attention map from graph
        attention_map = self.spatial_guidance(graph_features)  # [B, H, W]
        
        # Resize attention map if needed
        if attention_map.shape[1] != spatial_h or attention_map.shape[2] != spatial_w:
            attention_map = F.interpolate(
                attention_map.unsqueeze(1),
                size=(spatial_h, spatial_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
        
        # Process visual features
        processed_visual = self.visual_processor(visual_features)
        
        # Apply spatial attention fusion (enhance attention map)
        enhanced_attention = self.attention_fusion(attention_map.unsqueeze(1))  # [B, 1, H, W]
        enhanced_attention = enhanced_attention.squeeze(1)  # [B, H, W]
        
        # Global average pooling of attention map for feature weighting
        attention_weight = torch.mean(enhanced_attention.view(batch_size, -1), dim=1, keepdim=True)
        
        # Apply attention to visual features
        attended_visual = processed_visual * attention_weight
        
        # Feature enhancement
        enhanced_visual = self.feature_enhancer(attended_visual)
        
        # Residual connection
        output_visual = self.residual_weight * enhanced_visual + (1 - self.residual_weight) * visual_features
        
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
        scales: list = [8, 16, 32],
        num_attention_heads: int = 8,
        dropout: float = 0.1
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
        self.scale_fusion = nn.Sequential(
            nn.Linear(visual_dim * self.num_scales, visual_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim * 2, visual_dim),
            nn.LayerNorm(visual_dim)
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