"""
Graph Convolutional Network (GCN) implementation.

Implements GCN layers and encoder based on Kipf & Welling (2017)
"Semi-Supervised Classification with Graph Convolutional Networks".

This is a simplified baseline for Task 3.1 architecture comparison.

PERFORMANCE NOTES:
- Uses vectorized scatter_add for efficient aggregation (optimized for CPU)
- Avoids nested Python loops over edges - uses torch operations instead
- Suitable for N++ level graphs with up to ~130K edges
- Much faster than naive loop-based implementations on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.
    
    Simple message passing where each node aggregates features from its neighbors
    using normalized adjacency matrix. No attention mechanism.
    
    Based on Kipf & Welling (2017) "Semi-Supervised Classification with GCNs"
    
    Note: This implementation uses iterative aggregation for simplicity.
    For large-scale graphs, use PyTorch Geometric's GCNConv for better performance.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, max_nodes, in_dim]
            edge_index: [batch_size, 2, num_edges] - source and target node indices
            node_mask: [batch_size, max_nodes] - mask for valid nodes
            
        Returns:
            Updated node features [batch_size, max_nodes, out_dim]
        """
        batch_size, max_nodes, in_dim = node_features.shape
        
        # Transform features
        h = self.linear(node_features)  # [batch, max_nodes, out_dim]
        out_dim = h.shape[-1]
        
        # Vectorized aggregation using scatter_add
        # This is MUCH faster than nested Python loops
        aggregated = torch.zeros_like(h)
        
        for b in range(batch_size):
            edges = edge_index[b]  # [2, num_edges]
            if edges.shape[1] == 0:
                continue
                
            src_nodes = edges[0].long()  # Source nodes [num_edges]
            tgt_nodes = edges[1].long()  # Target nodes [num_edges]
            
            # Get source node features to aggregate
            src_features = h[b, src_nodes]  # [num_edges, out_dim]
            
            # Aggregate features from source to target using scatter_add
            # This replaces the slow nested loop with a single vectorized operation
            aggregated[b].scatter_add_(
                0,  # dimension to scatter along
                tgt_nodes.unsqueeze(1).expand(-1, out_dim),  # [num_edges, out_dim]
                src_features  # [num_edges, out_dim]
            )
        
        # Normalize by degree (approximate - using sqrt of max_nodes as in original)
        h = h + aggregated / (max_nodes ** 0.5)
        
        # Apply dropout and normalization
        h = self.dropout(h)
        h = self.norm(h)
        h = F.relu(h)
        
        # Apply mask if provided
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)
        
        return h


class GCNEncoder(nn.Module):
    """
    Multi-layer GCN encoder for graph representation learning.
    
    Simplest graph baseline without attention or heterogeneous types.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GCN layers
        self.layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [batch_size, max_nodes, node_feature_dim]
            edge_index: [batch_size, 2, num_edges]
            node_mask: [batch_size, max_nodes]
            
        Returns:
            node_embeddings: [batch_size, max_nodes, output_dim]
            graph_embedding: [batch_size, output_dim] - pooled graph representation
        """
        # Project input features
        h = self.input_proj(node_features)
        
        # Apply GCN layers
        for layer in self.layers:
            h = layer(h, edge_index, node_mask)
        
        # Output projection
        node_embeddings = self.output_proj(h)
        
        # Global pooling for graph-level representation
        if node_mask is not None:
            # Masked mean pooling
            masked_h = node_embeddings * node_mask.unsqueeze(-1)
            graph_embedding = masked_h.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            graph_embedding = node_embeddings.mean(dim=1)
        
        return node_embeddings, graph_embedding
