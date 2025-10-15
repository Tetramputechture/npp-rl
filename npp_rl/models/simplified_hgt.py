"""
Simplified Heterogeneous Graph Transformer (HGT) implementation.

Maintains heterogeneous type processing but with reduced complexity for
Task 3.1 architecture comparison:
- Fewer layers
- Reduced dimensions
- Simplified attention mechanism

This serves as a middle ground between simple GNN variants (GCN/GAT) and
the full HGT architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .gat import GATLayer


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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Type embeddings (simplified)
        self.node_type_embed = nn.Embedding(num_node_types, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Simplified attention layers (using GAT-like mechanism)
        self.layers = nn.ModuleList(
            [
                GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
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
        # Ensure node_types is long for embedding lookup
        type_emb = self.node_type_embed(node_types.long())
        h = h + type_emb

        # Apply attention layers
        for layer in self.layers:
            h = layer(h, edge_index, node_mask)

        # Output projection
        node_embeddings = self.output_proj(h)

        # Global pooling
        if node_mask is not None:
            masked_h = node_embeddings * node_mask.unsqueeze(-1)
            graph_embedding = masked_h.sum(dim=1) / node_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        else:
            graph_embedding = node_embeddings.mean(dim=1)

        return node_embeddings, graph_embedding
