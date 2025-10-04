"""
Simplified Graph Neural Network architectures for Task 3.1 comparison.

This module implements GAT (Graph Attention Network) and GCN (Graph Convolutional
Network) as simpler alternatives to the full HGT architecture.

These serve as baselines to evaluate whether the complexity of HGT is necessary
for the N++ completion task.
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
        batch_size, max_nodes, _ = node_features.shape
        
        # Transform features
        h = self.linear(node_features)  # [batch, max_nodes, out_dim]
        
        # Simple aggregation: mean of neighbor features
        # For each node, collect features from all nodes that connect to it
        aggregated = torch.zeros_like(h)
        
        for b in range(batch_size):
            edges = edge_index[b]  # [2, num_edges]
            if edges.shape[1] == 0:
                continue
                
            src_nodes = edges[0]  # Source nodes
            tgt_nodes = edges[1]  # Target nodes
            
            # Aggregate features from source to target
            for i in range(edges.shape[1]):
                src, tgt = src_nodes[i].item(), tgt_nodes[i].item()
                if node_mask is None or (node_mask[b, src] and node_mask[b, tgt]):
                    aggregated[b, tgt] += h[b, src]
        
        # Normalize by degree (approximate)
        h = h + aggregated / (max_nodes ** 0.5)
        
        # Apply dropout and normalization
        h = self.dropout(h)
        h = self.norm(h)
        h = F.relu(h)
        
        # Apply mask if provided
        if node_mask is not None:
            h = h * node_mask.unsqueeze(-1)
        
        return h


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    
    Uses attention mechanism to weight neighbor contributions, but without
    the heterogeneous type-specific processing of HGT.
    
    Based on Veličković et al. (2018) "Graph Attention Networks"
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
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        K = K.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        V = V.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        
        # Simplified attention: use dense attention over all nodes
        # In practice, should use edge_index to compute sparse attention
        # For simplicity, computing dense attention here
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask to attention scores
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_nodes]
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax over nodes
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [batch, max_nodes, num_heads, head_dim]
        
        # Concatenate or average heads
        if self.concat_heads:
            out = out.reshape(batch_size, max_nodes, -1)  # [batch, max_nodes, head_dim * num_heads]
        else:
            out = out.mean(dim=2)  # [batch, max_nodes, head_dim]
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + node_features[:, :, :self.out_dim])  # Residual if dims match
        
        # Apply mask
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1)
        
        return out


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


class SimplifiedHGTEncoder(nn.Module):
    """
    Simplified HGT encoder with reduced complexity.
    
    Maintains heterogeneous type processing but with:
    - Fewer layers
    - Reduced dimensions
    - Simplified attention mechanism
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_node_types: int = 6,
        num_edge_types: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        
        # Type embeddings (simplified)
        self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Simplified attention layers (using GAT-like mechanism)
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [batch_size, max_nodes, node_feature_dim]
            edge_index: [batch_size, 2, num_edges]
            node_types: [batch_size, max_nodes] - integer type indices
            node_mask: [batch_size, max_nodes]
            
        Returns:
            node_embeddings: [batch_size, max_nodes, output_dim]
            graph_embedding: [batch_size, output_dim]
        """
        # Project features and add type embeddings
        h = self.input_proj(node_features)
        type_emb = self.node_type_embed(node_types)
        h = h + type_emb
        
        # Apply attention layers
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
