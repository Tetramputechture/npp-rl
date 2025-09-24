"""
Heterogeneous Graph Transformer (HGT) Encoder Implementation.

This module implements the HGT encoder that stacks multiple HGT layers
for deep graph processing in the NPP-RL system. Optimized for production
use with the simplified feature set.

Based on "Heterogeneous Graph Transformer" by Wang et al. (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .hgt_layer import HGTLayer, EdgeType
from .entity_type_system import (
    EntitySpecializedEmbedding,
    HazardAwareAttention,
    create_entity_type_system,
)


class HGTEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer encoder for N++ level understanding.
    
    Processes heterogeneous graphs with specialized attention mechanisms
    for different entity types and functional relationships. Optimized
    for the simplified NPP-RL feature set (8 node features, 4 edge features).
    """
    
    def __init__(
        self,
        node_feature_dim: int = 8,  # Simplified node features
        edge_feature_dim: int = 4,  # Simplified edge features
        hidden_dim: int = 128,      # Reduced for efficiency
        num_layers: int = 3,
        num_heads: int = 8,
        output_dim: int = 256,      # Reduced for efficiency
        num_node_types: int = 6,    # From entity_type_system.py
        num_edge_types: int = 3,    # Simplified: ADJACENT, LOGICAL, REACHABLE
        dropout: float = 0.1,
        global_pool: str = "mean_max",
    ):
        """
        Initialize HGT encoder with production-optimized defaults.
        
        Args:
            node_feature_dim: Input node feature dimension (8 simplified)
            edge_feature_dim: Input edge feature dimension (4 simplified)
            hidden_dim: Hidden layer dimension (reduced for efficiency)
            num_layers: Number of HGT layers
            num_heads: Number of attention heads
            output_dim: Final output dimension (reduced for efficiency)
            num_node_types: Number of node types (6 from entity system)
            num_edge_types: Number of edge types (3 simplified)
            dropout: Dropout probability
            global_pool: Global pooling method ('mean', 'max', 'mean_max')
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.global_pool = global_pool
        
        # Entity type system for specialized processing
        self.entity_type_system = create_entity_type_system()
        
        # Specialized input embedding for simplified features
        self.input_embedding = EntitySpecializedEmbedding(
            input_dim=node_feature_dim,
            output_dim=hidden_dim,
            entity_type_system=self.entity_type_system,
            dropout=dropout,
        )
        
        # Edge feature embedding (simple linear projection)
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim // 4)
        
        # Stack of HGT layers
        self.hgt_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = HGTLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                num_node_types=num_node_types,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            self.hgt_layers.append(layer)
        
        # Hazard-aware attention for final processing
        self.hazard_attention = HazardAwareAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            entity_type_system=self.entity_type_system,
            dropout=dropout,
        )
        
        # Global pooling and output projection
        pool_dim = hidden_dim
        if global_pool == "mean_max":
            pool_dim = hidden_dim * 2
        
        self.output_projection = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize encoder parameters with proper scaling."""
        # Initialize edge embedding
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        if self.edge_embedding.bias is not None:
            nn.init.zeros_(self.edge_embedding.bias)
        
        # Initialize output projection
        for module in self.output_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, graph_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through HGT encoder.
        
        Args:
            graph_obs: Dictionary containing:
                - graph_node_feats: [batch_size, num_nodes, 8] - Simplified node features
                - graph_edge_index: [batch_size, 2, num_edges] - Edge connectivity
                - graph_edge_feats: [batch_size, num_edges, 4] - Simplified edge features
                - graph_node_mask: [batch_size, num_nodes] - Node validity mask
                - graph_edge_mask: [batch_size, num_edges] - Edge validity mask
                - graph_node_types: [batch_size, num_nodes] - Node type indices (optional)
                - graph_edge_types: [batch_size, num_edges] - Edge type indices (optional)
                
        Returns:
            Graph embeddings [batch_size, output_dim]
        """
        # Extract graph components
        node_feats = graph_obs["graph_node_feats"]
        edge_index = graph_obs["graph_edge_index"]
        edge_feats = graph_obs["graph_edge_feats"]
        node_mask = graph_obs["graph_node_mask"]
        edge_mask = graph_obs["graph_edge_mask"]
        
        # Get node and edge types (with defaults)
        node_types = graph_obs.get("graph_node_types", 
                                  torch.zeros(node_feats.shape[:2], dtype=torch.long, device=node_feats.device))
        edge_types = graph_obs.get("graph_edge_types",
                                  torch.zeros(edge_feats.shape[:2], dtype=torch.long, device=edge_feats.device))
        
        batch_size = node_feats.size(0)
        graph_embeddings = []
        
        # Process each graph in the batch
        for b in range(batch_size):
            # Extract single graph
            b_node_feats = node_feats[b]  # [num_nodes, 8]
            b_edge_index = edge_index[b]  # [2, num_edges]
            b_edge_feats = edge_feats[b]  # [num_edges, 4]
            b_node_mask = node_mask[b]    # [num_nodes]
            b_edge_mask = edge_mask[b]    # [num_edges]
            b_node_types = node_types[b]  # [num_nodes]
            b_edge_types = edge_types[b]  # [num_edges]
            
            # Filter valid nodes and edges
            valid_nodes = b_node_mask.bool()
            valid_edges = b_edge_mask.bool()
            
            if not valid_nodes.any():
                # Handle empty graph case
                graph_embeddings.append(torch.zeros(self.output_dim, device=node_feats.device))
                continue
            
            # Apply masks
            b_node_feats = b_node_feats[valid_nodes]
            b_node_types = b_node_types[valid_nodes]
            
            if valid_edges.any():
                b_edge_index = b_edge_index[:, valid_edges]
                b_edge_feats = b_edge_feats[valid_edges]
                b_edge_types = b_edge_types[valid_edges]
                
                # Remap edge indices to valid node indices
                b_edge_index, b_edge_feats, b_edge_types = self._remap_edge_indices_and_features(
                    b_edge_index, b_edge_feats, b_edge_types, valid_nodes
                )
            else:
                # No edges case
                b_edge_index = torch.empty((2, 0), dtype=torch.long, device=node_feats.device)
                b_edge_feats = torch.empty((0, self.edge_feature_dim), device=node_feats.device)
                b_edge_types = torch.empty((0,), dtype=torch.long, device=node_feats.device)
            
            # Process single graph
            graph_emb = self._process_single_graph(
                b_node_feats, b_edge_index, b_edge_feats,
                b_node_types, b_edge_types
            )
            graph_embeddings.append(graph_emb)
        
        # Stack batch results
        return torch.stack(graph_embeddings, dim=0)
    
    def _remap_edge_indices_and_features(
        self, 
        edge_index: torch.Tensor, 
        edge_feats: torch.Tensor,
        edge_types: torch.Tensor,
        valid_nodes: torch.Tensor
    ) -> tuple:
        """Remap edge indices and filter features to account for filtered nodes."""
        # Create mapping from original indices to new indices
        node_mapping = torch.full((valid_nodes.size(0),), -1, dtype=torch.long, device=edge_index.device)
        node_mapping[valid_nodes] = torch.arange(valid_nodes.sum(), device=edge_index.device)
        
        # Remap edge indices
        remapped_edge_index = node_mapping[edge_index]
        
        # Filter out edges with invalid nodes
        valid_edge_mask = (remapped_edge_index >= 0).all(dim=0)
        
        return (
            remapped_edge_index[:, valid_edge_mask],
            edge_feats[valid_edge_mask],
            edge_types[valid_edge_mask]
        )
    
    def _process_single_graph(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        node_types: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """Process a single graph through the HGT encoder."""
        # Input embedding with entity specialization (add batch dimension)
        x = self.input_embedding(node_feats.unsqueeze(0), node_types.unsqueeze(0))
        x = x.squeeze(0)  # Remove batch dimension
        
        # Embed edge features (simple projection)
        if edge_feats.size(0) > 0:
            edge_emb = self.edge_embedding(edge_feats)
        else:
            edge_emb = torch.empty((0, self.hidden_dim // 4), device=node_feats.device)
        
        # Pass through HGT layers
        for layer in self.hgt_layers:
            x = layer(
                node_features=x,
                edge_index=edge_index,
                edge_features=edge_emb,
                node_types=node_types,
                edge_types=edge_types
            )
        
        # Apply hazard-aware attention (add batch dimension)
        x_batched = x.unsqueeze(0)  # [1, num_nodes, embed_dim]
        node_types_batched = node_types.unsqueeze(0)  # [1, num_nodes]
        
        x_out, _ = self.hazard_attention(
            query=x_batched,
            key=x_batched,
            value=x_batched,
            entity_types=node_types_batched
        )
        x = x_out.squeeze(0)  # Remove batch dimension
        
        # Global pooling
        graph_emb = self._global_pool(x)
        
        # Output projection
        graph_emb = self.output_projection(graph_emb)
        
        return graph_emb
    
    def _global_pool(self, node_features: torch.Tensor) -> torch.Tensor:
        """Apply global pooling to node features."""
        if self.global_pool == "mean":
            return torch.mean(node_features, dim=0)
        elif self.global_pool == "max":
            return torch.max(node_features, dim=0)[0]
        elif self.global_pool == "mean_max":
            mean_pool = torch.mean(node_features, dim=0)
            max_pool = torch.max(node_features, dim=0)[0]
            return torch.cat([mean_pool, max_pool], dim=0)
        else:
            raise ValueError(f"Unknown global pooling method: {self.global_pool}")


def create_hgt_encoder(
    node_feature_dim: int = 8,
    edge_feature_dim: int = 4,
    **kwargs
) -> HGTEncoder:
    """
    Factory function to create HGT encoder with production defaults.
    
    Args:
        node_feature_dim: Input node feature dimension (8 simplified)
        edge_feature_dim: Input edge feature dimension (4 simplified)
        **kwargs: Additional parameters for HGTEncoder
        
    Returns:
        Configured HGTEncoder instance optimized for production
    """
    # Production-optimized defaults
    default_params = {
        "hidden_dim": 128,      # Reduced for efficiency
        "num_layers": 3,
        "num_heads": 8,
        "output_dim": 256,      # Reduced for efficiency
        "num_node_types": 6,    # From entity_type_system.py
        "num_edge_types": 3,    # Simplified: ADJACENT, LOGICAL, REACHABLE
        "dropout": 0.1,
        "global_pool": "mean_max",
    }
    
    # Override defaults with provided kwargs
    params = {**default_params, **kwargs}
    
    return HGTEncoder(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        **params
    )