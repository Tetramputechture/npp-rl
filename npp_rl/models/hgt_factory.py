"""
HGT Factory Functions for Production NPP-RL System.

This module provides factory functions and utilities for creating
production-ready HGT components with optimized configurations
for the NPP-RL system.

Includes integration with PyTorch Geometric for enhanced performance
and compatibility with modern graph learning frameworks.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from .hgt_layer import HGTLayer, create_hgt_layer
from .hgt_encoder import HGTEncoder, create_hgt_encoder
from .attention_mechanisms import create_cross_modal_attention
from nclone.graph.common import (
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)
from torch_geometric.nn import HGTConv


class HGTFactoryConfig:
    """Configuration class for HGT factory system."""

    # Architecture parameters (optimized for production)
    HIDDEN_DIM = 128  # Reduced for efficiency
    NUM_LAYERS = 3  # Good balance of depth and speed
    NUM_HEADS = 8  # Standard multi-head attention
    OUTPUT_DIM = 256  # Reduced for efficiency
    DROPOUT = 0.1  # Standard dropout rate

    # Performance parameters
    GLOBAL_POOL = "mean_max"  # Effective pooling strategy
    USE_NORM = True  # Layer normalization
    USE_RESIDUAL = True  # Residual connections


class HGTFactory:
    """
    Factory class for creating production-ready HGT components.

    Provides standardized creation of HGT layers, encoders, and complete
    systems with optimized configurations for the NPP-RL domain.
    """

    def __init__(self, config: Optional[HGTFactoryConfig] = None):
        """
        Initialize HGT factory.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or HGTFactoryConfig()
        self.logger = logging.getLogger(__name__)

    def create_hgt_layer(
        self, in_dim: Optional[int] = None, out_dim: Optional[int] = None, **kwargs
    ) -> HGTLayer:
        """
        Create production-ready HGT layer.

        Args:
            in_dim: Input dimension (uses config default if None)
            out_dim: Output dimension (uses config default if None)
            **kwargs: Additional parameters to override defaults

        Returns:
            Configured HGTLayer instance
        """
        params = {
            "in_dim": in_dim or self.config.HIDDEN_DIM,
            "out_dim": out_dim or self.config.HIDDEN_DIM,
            "num_heads": self.config.NUM_HEADS,
            "num_node_types": self.config.NUM_NODE_TYPES,
            "num_edge_types": self.config.NUM_EDGE_TYPES,
            "dropout": self.config.DROPOUT,
            "use_norm": self.config.USE_NORM,
        }

        # Override with provided kwargs
        params.update(kwargs)

        return create_hgt_layer(**params)

    def create_pytorch_geometric_hgt(
        self, metadata: Optional[tuple] = None, **kwargs
    ) -> Optional[nn.Module]:
        """
        Create PyTorch Geometric HGT implementation if available.

        Args:
            metadata: Graph metadata (node_types, edge_types)
            **kwargs: Additional parameters

        Returns:
            PyTorch Geometric HGT model or None if not available
        """
        # Default metadata for NPP-RL
        if metadata is None:
            node_types = ["tile", "ninja", "hazard", "collectible", "switch", "exit"]
            edge_types = [
                ("tile", "adjacent", "tile"),
                ("tile", "logical", "tile"),
                ("tile", "reachable", "tile"),
            ]
            metadata = (node_types, edge_types)

        # Create PyTorch Geometric HGT
        try:
            model = HGTConv(
                in_channels=NODE_FEATURE_DIM,
                out_channels=self.config.HIDDEN_DIM,
                metadata=metadata,
                heads=self.config.NUM_HEADS,
                dropout=self.config.DROPOUT,
                **kwargs,
            )

            self.logger.info("Created PyTorch Geometric HGT model")
            return model

        except Exception as e:
            self.logger.error(f"Failed to create PyTorch Geometric HGT: {e}")
            return None

    def create_multimodal_hgt_system(
        self, visual_dim: int = 512, state_dim: int = 64, **kwargs
    ) -> nn.Module:
        """
        Create complete multimodal HGT system for NPP-RL.

        Args:
            visual_dim: Visual feature dimension
            state_dim: State feature dimension
            **kwargs: Additional parameters

        Returns:
            Complete multimodal HGT system
        """
        return MultimodalHGTSystem(
            hgt_encoder=create_hgt_encoder(
                node_feature_dim=NODE_FEATURE_DIM,
                edge_feature_dim=EDGE_FEATURE_DIM,
                **kwargs,
            ),
            visual_dim=visual_dim,
            state_dim=state_dim,
            graph_dim=self.config.OUTPUT_DIM,
            config=self.config,
        )


class MultimodalHGTSystem(nn.Module):
    """
    Complete multimodal HGT system for NPP-RL production use.

    Integrates graph processing with visual and state information
    for comprehensive level understanding.
    """

    def __init__(
        self,
        hgt_encoder: HGTEncoder,
        visual_dim: int = 512,
        state_dim: int = 64,
        graph_dim: int = 256,
        config: Optional[HGTFactoryConfig] = None,
    ):
        """
        Initialize multimodal HGT system.

        Args:
            hgt_encoder: Pre-configured HGT encoder
            visual_dim: Visual feature dimension
            state_dim: State feature dimension
            graph_dim: Graph feature dimension
            config: System configuration
        """
        super().__init__()

        self.config = config or HGTFactoryConfig()
        self.hgt_encoder = hgt_encoder

        # Cross-modal attention mechanisms
        self.visual_graph_attention = create_cross_modal_attention(
            graph_dim=graph_dim,
            other_dim=visual_dim,
            num_heads=self.config.NUM_HEADS,
            dropout=self.config.DROPOUT,
        )

        self.state_graph_attention = create_cross_modal_attention(
            graph_dim=graph_dim,
            other_dim=state_dim,
            num_heads=self.config.NUM_HEADS,
            dropout=self.config.DROPOUT,
        )

        # Feature fusion
        fusion_dim = graph_dim + visual_dim + state_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(fusion_dim // 2, graph_dim),
        )

        # Final output projection
        self.output_projection = nn.Linear(graph_dim, self.config.OUTPUT_DIM)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize multimodal system parameters."""
        for module in [self.feature_fusion, self.output_projection]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        graph_obs: Dict[str, torch.Tensor],
        visual_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through multimodal HGT system.

        Args:
            graph_obs: Graph observation dictionary
            visual_features: Visual features [batch_size, visual_dim]
            state_features: State features [batch_size, state_dim]

        Returns:
            Multimodal embeddings [batch_size, output_dim]
        """
        # Use individual processing methods for consistency
        graph_features = self.process_graph(graph_obs)
        visual_processed = self.process_visual(visual_features)
        state_processed = self.process_state(state_features)

        return self.fuse_features(graph_features, visual_processed, state_processed)

    def process_graph(self, graph_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process graph observations through HGT encoder."""
        return self.hgt_encoder(graph_obs)

    def process_visual(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Process visual features (identity for now)."""
        return visual_features

    def process_state(self, state_features: torch.Tensor) -> torch.Tensor:
        """Process state features (identity for now)."""
        return state_features

    def fuse_features(
        self,
        graph_features: torch.Tensor,
        visual_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multimodal features."""
        # Apply cross-modal attention (simplified for batch processing)
        enhanced_graph = graph_features

        # Simple feature concatenation and fusion
        multimodal_features = torch.cat(
            [enhanced_graph, visual_features, state_features], dim=-1
        )

        # Feature fusion
        fused_features = self.feature_fusion(multimodal_features)

        # Final output projection
        output = self.output_projection(fused_features)

        return output
