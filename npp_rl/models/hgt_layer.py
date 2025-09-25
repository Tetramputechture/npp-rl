"""
Heterogeneous Graph Transformer (HGT) Layer Implementation.

This module implements the core HGT layer with type-specific attention
mechanisms for processing heterogeneous graphs in the NPP-RL system.

Based on "Heterogeneous Graph Transformer" by Wang et al. (2020).
Optimized for production use with clean, maintainable code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from enum import IntEnum


class EdgeType(IntEnum):
    """Types of edges in the heterogeneous graph."""
    
    # Movement edges (simplified from complex physics)
    ADJACENT = 0    # Basic 4-connectivity between traversable tiles
    LOGICAL = 1     # Switch-door relationships  
    REACHABLE = 2   # Simple flood-fill connectivity


class HGTLayer(nn.Module):
    """
    Heterogeneous Graph Transformer layer with type-specific attention.
    
    Implements multi-head attention with separate parameters for different
    node and edge types, enabling specialized processing of heterogeneous
    graph structures. Optimized for the simplified NPP-RL feature set.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_node_types: int = 6,  # From entity_type_system.py
        num_edge_types: int = 3,  # Simplified: ADJACENT, LOGICAL, REACHABLE
        dropout: float = 0.1,
        use_norm: bool = True,
    ):
        """
        Initialize HGT layer with simplified configuration.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension  
            num_heads: Number of attention heads
            num_node_types: Number of node types (6 from entity system)
            num_edge_types: Number of edge types (3 simplified)
            dropout: Dropout probability
            use_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.d_k = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Type-specific linear transformations for K, Q, V projections
        self.k_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_node_types)
        ])
        self.q_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_node_types)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_node_types)
        ])
        
        # Edge-type specific attention parameters
        self.relation_pri = nn.Parameter(torch.ones(num_edge_types, num_heads))
        self.relation_att = nn.ModuleList([
            nn.Linear(self.d_k, self.d_k, bias=False) for _ in range(num_edge_types)
        ])
        self.relation_msg = nn.ModuleList([
            nn.Linear(self.d_k, self.d_k, bias=False) for _ in range(num_edge_types)
        ])
        
        # Skip connection and normalization
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize layer parameters with proper scaling."""
        # Xavier initialization for linear layers
        for module in [*self.k_linears, *self.q_linears, *self.v_linears]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Initialize relation parameters
        nn.init.ones_(self.relation_pri)
        for module in [*self.relation_att, *self.relation_msg]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_types: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of HGT layer.
        
        Args:
            node_features: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge features [num_edges, edge_feat_dim]
            node_types: Node type indices [num_nodes]
            edge_types: Edge type indices [num_edges]
            node_mask: Optional node mask [num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            # Handle empty graph case
            output = self.skip(node_features)
            return self.norm(output)
        
        # Compute K, Q, V for each node based on its type
        K = self._compute_type_specific_projections(node_features, node_types, self.k_linears)
        Q = self._compute_type_specific_projections(node_features, node_types, self.q_linears)  
        V = self._compute_type_specific_projections(node_features, node_types, self.v_linears)
        
        # Reshape for multi-head attention
        K = K.view(num_nodes, self.num_heads, self.d_k)
        Q = Q.view(num_nodes, self.num_heads, self.d_k)
        V = V.view(num_nodes, self.num_heads, self.d_k)
        
        # Compute attention and messages
        output = self._compute_attention_and_aggregate(
            K, Q, V, edge_index, edge_types, node_features
        )
        
        # Skip connection and normalization
        output = output + self.skip(node_features)
        output = self.norm(output)
        output = self.dropout(output)
        
        # Apply node mask if provided
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
        
        return output
    
    def _compute_type_specific_projections(
        self, 
        node_features: torch.Tensor, 
        node_types: torch.Tensor, 
        linears: nn.ModuleList
    ) -> torch.Tensor:
        """Compute type-specific linear projections for nodes."""
        num_nodes = node_features.size(0)
        output = torch.zeros(num_nodes, self.out_dim, device=node_features.device)
        
        # Process each node type separately
        for node_type in range(self.num_node_types):
            mask = (node_types == node_type)
            if mask.any():
                output[mask] = linears[node_type](node_features[mask])
        
        return output
    
    def _compute_attention_and_aggregate(
        self,
        K: torch.Tensor,
        Q: torch.Tensor, 
        V: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention weights and aggregate messages."""
        src_nodes, tgt_nodes = edge_index[0], edge_index[1]
        num_nodes = node_features.size(0)
        
        # Initialize output
        output = torch.zeros(num_nodes, self.out_dim, device=node_features.device)
        
        # Process each edge type separately
        for edge_type in range(self.num_edge_types):
            edge_mask = (edge_types == edge_type)
            if not edge_mask.any():
                continue
            
            # Get edges of this type
            type_src = src_nodes[edge_mask]
            type_tgt = tgt_nodes[edge_mask]
            
            # Compute attention for this edge type
            type_output = self._compute_edge_type_attention(
                K, Q, V, type_src, type_tgt, edge_type
            )
            
            # Aggregate to target nodes
            output.index_add_(0, type_tgt, type_output)
        
        return output
    
    def _compute_edge_type_attention(
        self,
        K: torch.Tensor,
        Q: torch.Tensor,
        V: torch.Tensor, 
        src_nodes: torch.Tensor,
        tgt_nodes: torch.Tensor,
        edge_type: int
    ) -> torch.Tensor:
        """Compute attention for specific edge type."""
        # Get source and target features
        K_src = K[src_nodes]  # [num_edges_type, num_heads, d_k]
        Q_tgt = Q[tgt_nodes]  # [num_edges_type, num_heads, d_k]
        V_src = V[src_nodes]  # [num_edges_type, num_heads, d_k]
        
        # Apply edge-type specific transformations
        K_src = self.relation_att[edge_type](K_src)
        V_src = self.relation_msg[edge_type](V_src)
        
        # Compute attention scores
        att_scores = torch.sum(K_src * Q_tgt, dim=-1)  # [num_edges_type, num_heads]
        att_scores = att_scores / math.sqrt(self.d_k)
        
        # Apply edge-type priority
        att_scores = att_scores * self.relation_pri[edge_type]
        
        # Softmax normalization (per target node)
        att_weights = F.softmax(att_scores, dim=0)  # [num_edges_type, num_heads]
        
        # Apply attention to values
        messages = att_weights.unsqueeze(-1) * V_src  # [num_edges_type, num_heads, d_k]
        
        # Reshape back to [num_edges_type, out_dim]
        messages = messages.view(messages.size(0), -1)
        
        return messages


def create_hgt_layer(
    in_dim: int,
    out_dim: int,
    num_heads: int = 8,
    num_node_types: int = 6,
    num_edge_types: int = 3,
    dropout: float = 0.1,
    use_norm: bool = True
) -> HGTLayer:
    """
    Factory function to create HGT layer with production defaults.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        num_heads: Number of attention heads
        num_node_types: Number of node types
        num_edge_types: Number of edge types  
        dropout: Dropout probability
        use_norm: Whether to use layer normalization
        
    Returns:
        Configured HGTLayer instance
    """
    return HGTLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        dropout=dropout,
        use_norm=use_norm
    )