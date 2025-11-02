"""
Attention Mechanisms for Heterogeneous Graph Transformer (HGT).

This module implements specialized attention mechanisms used in the HGT
architecture, including type-specific attention and hazard-aware processing
for the NPP-RL system.

Optimized for production use with clean, maintainable implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TypeSpecificAttention(nn.Module):
    """
    Type-specific attention mechanism for heterogeneous graphs.
    
    Computes attention weights based on node and edge types, allowing
    the model to learn specialized attention patterns for different
    types of game elements (tiles, entities, etc.).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_node_types: int = 6,
        num_edge_types: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize type-specific attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_node_types: Number of node types
            num_edge_types: Number of edge types
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Type-specific query, key, value projections
        self.q_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_node_types)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_node_types)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_node_types)
        ])
        
        # Edge-type specific attention bias
        self.edge_bias = nn.Parameter(torch.zeros(num_edge_types, num_heads))
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize attention parameters."""
        # Initialize projections
        for proj_list in [self.q_proj, self.k_proj, self.v_proj]:
            for proj in proj_list:
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        
        # Initialize edge bias
        nn.init.zeros_(self.edge_bias)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        node_types: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of type-specific attention.
        
        Args:
            query: Query tensor [num_nodes, embed_dim]
            key: Key tensor [num_nodes, embed_dim]
            value: Value tensor [num_nodes, embed_dim]
            node_types: Node type indices [num_nodes]
            edge_index: Edge connectivity [2, num_edges] (optional)
            edge_types: Edge type indices [num_edges] (optional)
            attn_mask: Attention mask [num_nodes, num_nodes] (optional)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        num_nodes = query.size(0)
        
        # Compute type-specific Q, K, V
        Q = self._compute_type_specific_projection(query, node_types, self.q_proj)
        K = self._compute_type_specific_projection(key, node_types, self.k_proj)
        V = self._compute_type_specific_projection(value, node_types, self.v_proj)
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('ihd,jhd->hij', Q, K) / math.sqrt(self.head_dim)
        
        # Apply edge-type specific bias if available
        if edge_index is not None and edge_types is not None:
            attn_scores = self._apply_edge_bias(attn_scores, edge_index, edge_types)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.einsum('hij,jhd->ihd', attn_weights, V)
        output = output.contiguous().view(num_nodes, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def _compute_type_specific_projection(
        self,
        x: torch.Tensor,
        node_types: torch.Tensor,
        proj_list: nn.ModuleList
    ) -> torch.Tensor:
        """Compute type-specific projections."""
        num_nodes = x.size(0)
        output = torch.zeros_like(x)
        
        for node_type in range(self.num_node_types):
            mask = (node_types == node_type)
            if mask.any():
                output[mask] = proj_list[node_type](x[mask])
        
        return output
    
    def _apply_edge_bias(
        self,
        attn_scores: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """Apply edge-type specific bias to attention scores."""
        src_nodes, tgt_nodes = edge_index[0], edge_index[1]
        
        # Apply bias for each edge
        for i, (src, tgt) in enumerate(zip(src_nodes, tgt_nodes)):
            edge_type = edge_types[i]
            attn_scores[tgt, src] += self.edge_bias[edge_type]
        
        return attn_scores


# HazardAwareAttention is imported from entity_type_system.py to avoid duplication


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for integrating graph features with other modalities.
    
    Enables the HGT to attend to visual features, state information, or
    other modalities when processing graph structures.
    """
    
    def __init__(
        self,
        graph_dim: int,
        other_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            graph_dim: Graph feature dimension
            other_dim: Other modality feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.graph_dim = graph_dim
        self.other_dim = other_dim
        self.num_heads = num_heads
        self.head_dim = graph_dim // num_heads
        
        assert graph_dim % num_heads == 0, "graph_dim must be divisible by num_heads"
        
        # Cross-attention projections
        self.q_proj = nn.Linear(graph_dim, graph_dim)  # Query from graph
        self.k_proj = nn.Linear(other_dim, graph_dim)  # Key from other modality
        self.v_proj = nn.Linear(other_dim, graph_dim)  # Value from other modality
        
        # Output projection
        self.out_proj = nn.Linear(graph_dim, graph_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize cross-modal attention parameters."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def forward(
        self,
        graph_features: torch.Tensor,
        other_features: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of cross-modal attention.
        
        Args:
            graph_features: Graph features [num_nodes, graph_dim]
            other_features: Other modality features [seq_len, other_dim]
            attn_mask: Attention mask [num_nodes, seq_len] (optional)
            
        Returns:
            Enhanced graph features [num_nodes, graph_dim]
        """
        num_nodes = graph_features.size(0)
        seq_len = other_features.size(0)
        
        # Compute Q, K, V
        Q = self.q_proj(graph_features)  # [num_nodes, graph_dim]
        K = self.k_proj(other_features)  # [seq_len, graph_dim]
        V = self.v_proj(other_features)  # [seq_len, graph_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(seq_len, self.num_heads, self.head_dim)
        V = V.view(seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('ihd,jhd->hij', Q, K) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.einsum('hij,jhd->ihd', attn_weights, V)
        output = output.contiguous().view(num_nodes, self.graph_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


# Factory functions for easy creation
def create_type_specific_attention(
    embed_dim: int,
    num_heads: int = 8,
    num_node_types: int = 6,
    num_edge_types: int = 3,
    dropout: float = 0.1
) -> TypeSpecificAttention:
    """Create type-specific attention with production defaults."""
    return TypeSpecificAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        dropout=dropout
    )


# create_hazard_aware_attention is available from entity_type_system.py


def create_cross_modal_attention(
    graph_dim: int,
    other_dim: int,
    num_heads: int = 8,
    dropout: float = 0.1
) -> CrossModalAttention:
    """Create cross-modal attention with production defaults."""
    return CrossModalAttention(
        graph_dim=graph_dim,
        other_dim=other_dim,
        num_heads=num_heads,
        dropout=dropout
    )