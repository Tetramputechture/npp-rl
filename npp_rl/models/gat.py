"""
Graph Attention Network (GAT) implementation.

Implements GAT layers and encoder based on Veličković et al. (2018)
"Graph Attention Networks".

Uses attention mechanism to weight neighbor contributions for Task 3.1 comparison.

PERFORMANCE NOTES:
- Uses sparse edge-based attention (optimized for CPU)
- Avoids dense 15K x 15K attention matrices - only computes attention on edges
- Suitable for N++ level graphs with up to ~130K edges
- Much more memory and compute efficient than dense attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    
    Uses attention mechanism to weight neighbor contributions, but without
    the heterogeneous type-specific processing of HGT.
    
    Based on Veličković et al. (2018) "Graph Attention Networks"
    
    Note: This simplified implementation uses dense attention over all nodes.
    For sparse edge-based attention, use PyTorch Geometric's GATConv.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        concat_heads: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat_heads = concat_heads
        
        # Per-head dimension
        self.head_dim = out_dim // num_heads if concat_heads else out_dim
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(in_dim, self.head_dim * num_heads)
        self.key = nn.Linear(in_dim, self.head_dim * num_heads)
        self.value = nn.Linear(in_dim, self.head_dim * num_heads)
        
        # Attention scoring
        self.attention = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        
        if concat_heads:
            self.out_proj = nn.Linear(self.head_dim * num_heads, out_dim)
        else:
            self.out_proj = nn.Linear(out_dim, out_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, max_nodes, in_dim]
            edge_index: [batch_size, 2, num_edges]
            node_mask: [batch_size, max_nodes]
            
        Returns:
            Updated node features [batch_size, max_nodes, out_dim]
        """
        batch_size, max_nodes, _ = node_features.shape
        
        # Project to Q, K, V
        Q = self.query(node_features)  # [batch, max_nodes, head_dim * num_heads]
        K = self.key(node_features)
        V = self.value(node_features)
        
        # Reshape for multi-head attention: [batch, max_nodes, num_heads, head_dim]
        Q = Q.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        K = K.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        V = V.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        
        # Use sparse edge-based attention instead of dense attention
        # This is MUCH more efficient for large graphs
        out = torch.zeros(batch_size, max_nodes, self.num_heads, self.head_dim, device=node_features.device)
        
        for b in range(batch_size):
            edges = edge_index[b]  # [2, num_edges]
            if edges.shape[1] == 0:
                # No edges, just use self-attention
                out[b] = V[b]
                continue
                
            src_nodes = edges[0].long()  # [num_edges]
            tgt_nodes = edges[1].long()  # [num_edges]
            
            # Get Q, K, V for edges
            q_tgt = Q[b, tgt_nodes]  # [num_edges, num_heads, head_dim]
            k_src = K[b, src_nodes]  # [num_edges, num_heads, head_dim]
            v_src = V[b, src_nodes]  # [num_edges, num_heads, head_dim]
            
            # Compute attention scores for edges only
            # scores: [num_edges, num_heads]
            scores = (q_tgt * k_src).sum(dim=-1) / (self.head_dim ** 0.5)
            
            # For each target node, compute softmax over its incoming edges
            # Group edges by target node
            unique_tgts = torch.unique(tgt_nodes)
            
            for tgt in unique_tgts:
                # Find all edges incoming to this target
                edge_mask = (tgt_nodes == tgt)
                tgt_scores = scores[edge_mask]  # [num_incoming_edges, num_heads]
                tgt_values = v_src[edge_mask]  # [num_incoming_edges, num_heads, head_dim]
                
                # Softmax over incoming edges
                attn_weights = F.softmax(tgt_scores, dim=0)  # [num_incoming_edges, num_heads]
                attn_weights = self.dropout(attn_weights)
                
                # Weighted sum of values
                # attn_weights: [num_incoming_edges, num_heads] -> [num_incoming_edges, num_heads, 1]
                # tgt_values: [num_incoming_edges, num_heads, head_dim]
                aggregated = (attn_weights.unsqueeze(-1) * tgt_values).sum(dim=0)  # [num_heads, head_dim]
                out[b, tgt] = aggregated
        
        # Transpose to [batch, max_nodes, num_heads, head_dim]
        # Concatenate or average heads
        if self.concat_heads:
            out = out.reshape(batch_size, max_nodes, -1)  # [batch, max_nodes, head_dim * num_heads]
        else:
            out = out.mean(dim=2)  # [batch, max_nodes, head_dim]
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Residual connection if dimensions match
        if node_features.shape[-1] == self.out_dim:
            out = self.norm(out + node_features)
        else:
            out = self.norm(out)
        
        # Apply mask
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        
        return out


class GATEncoder(nn.Module):
    """
    Multi-layer GAT encoder for graph representation learning.
    
    Uses attention mechanism but without heterogeneous type processing.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GAT layers
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout, concat_heads=True)
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
            graph_embedding: [batch_size, output_dim]
        """
        # Project input features
        h = self.input_proj(node_features)
        
        # Apply GAT layers
        for layer in self.layers:
            h = layer(h, edge_index, node_mask)
        
        # Output projection
        node_embeddings = self.output_proj(h)
        
        # Global pooling
        if node_mask is not None:
            masked_h = node_embeddings * node_mask.unsqueeze(-1)
            graph_embedding = masked_h.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            graph_embedding = node_embeddings.mean(dim=1)
        
        return node_embeddings, graph_embedding
