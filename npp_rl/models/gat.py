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

# Try to import torch_scatter for efficient edge aggregation
try:
    from torch_scatter import scatter_softmax, scatter

    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False
    import warnings

    warnings.warn("torch_scatter not available, GAT will use slower fallback")


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
        concat_heads: bool = True,
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
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        GAT forward with batched sparse edge attention.

        Args:
            node_features: [batch_size, max_nodes, in_dim]
            edge_index: [batch_size, 2, num_edges]
            node_mask: [batch_size, max_nodes] (optional)
            edge_mask: [batch_size, num_edges] (optional) - 1 for valid edges, 0 for padding

        Returns:
            Updated node features [batch_size, max_nodes, out_dim]
        """
        batch_size, max_nodes, _ = node_features.shape
        device = node_features.device

        # Project to Q, K, V
        Q = self.query(node_features)  # [batch, max_nodes, head_dim * num_heads]
        K = self.key(node_features)
        V = self.value(node_features)

        # Reshape for multi-head: [batch, max_nodes, num_heads, head_dim]
        Q = Q.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        K = K.view(batch_size, max_nodes, self.num_heads, self.head_dim)
        V = V.view(batch_size, max_nodes, self.num_heads, self.head_dim)

        # Process all batches together
        outputs = []
        for b in range(batch_size):
            edges = edge_index[b]  # [2, num_edges]

            # CRITICAL: Filter edges using mask before processing
            if edge_mask is not None:
                valid_edge_indices = torch.where(edge_mask[b] == 1)[0]
                if len(valid_edge_indices) == 0:
                    # No valid edges: return zero output
                    outputs.append(torch.zeros_like(V[b]))
                    continue
                edges = edges[:, valid_edge_indices]  # Only process valid edges!

            # Early exit if no edges after filtering
            if edges.shape[1] == 0:
                outputs.append(torch.zeros_like(V[b]))
                continue

            src_nodes = edges[0].long()  # [num_edges] - now only valid edges
            tgt_nodes = edges[1].long()  # [num_edges] - now only valid edges

            # CRITICAL: Validate edge indices to prevent out-of-bounds access
            # This prevents hangs when scatter operations receive invalid indices
            if (src_nodes >= max_nodes).any() or (src_nodes < 0).any():
                invalid_src = torch.where((src_nodes >= max_nodes) | (src_nodes < 0))[0]
                raise ValueError(
                    f"[GAT] Invalid source node indices in batch {b}: "
                    f"max_nodes={max_nodes}, invalid_indices={src_nodes[invalid_src[:10]].tolist()}, "
                    f"invalid_count={len(invalid_src)}"
                )
            if (tgt_nodes >= max_nodes).any() or (tgt_nodes < 0).any():
                invalid_tgt = torch.where((tgt_nodes >= max_nodes) | (tgt_nodes < 0))[0]
                raise ValueError(
                    f"[GAT] Invalid target node indices in batch {b}: "
                    f"max_nodes={max_nodes}, invalid_indices={tgt_nodes[invalid_tgt[:10]].tolist()}, "
                    f"invalid_count={len(invalid_tgt)}"
                )

            # Get source and target features for edges
            q_tgt = Q[b, tgt_nodes]  # [num_edges, num_heads, head_dim]
            k_src = K[b, src_nodes]  # [num_edges, num_heads, head_dim]
            v_src = V[b, src_nodes]  # [num_edges, num_heads, head_dim]

            # GAT attention formula: e_ij = LeakyReLU(a^T [W q_i || W k_j])
            # Concatenate Q and K along feature dim
            qk_concat = torch.cat(
                [q_tgt, k_src], dim=-1
            )  # [num_edges, num_heads, 2*head_dim]

            # Apply learnable attention parameter
            # self.attention: [num_heads, 2*head_dim]
            # Expand for broadcasting: [1, num_heads, 2*head_dim]
            attn_param = self.attention.unsqueeze(0)  # [1, num_heads, 2*head_dim]

            # Compute attention scores: sum over feature dim
            # [num_edges, num_heads, 2*head_dim] * [1, num_heads, 2*head_dim] → [num_edges, num_heads]
            scores = (qk_concat * attn_param).sum(dim=-1)  # [num_edges, num_heads]
            scores = F.leaky_relu(scores, negative_slope=0.2)

            # Normalize with softmax per target node
            if TORCH_SCATTER_AVAILABLE:
                # Use scatter_softmax for efficient per-node normalization
                # Applies softmax over all edges with same target
                # CRITICAL: Ensure tgt_nodes are within valid range for scatter operations
                # scatter_softmax can hang or produce incorrect results with out-of-bounds indices
                try:
                    attn_weights = scatter_softmax(
                        scores,  # [num_edges, num_heads]
                        tgt_nodes,  # [num_edges] - group by target
                        dim=0,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"[GAT] scatter_softmax failed in batch {b}: {e}. "
                        f"tgt_nodes range: [{tgt_nodes.min().item()}, {tgt_nodes.max().item()}], "
                        f"max_nodes: {max_nodes}, num_edges: {len(tgt_nodes)}"
                    ) from e
            else:
                # Fallback: manual softmax per target (slower)
                attn_weights = torch.zeros_like(scores)
                unique_tgts = torch.unique(tgt_nodes)
                for tgt in unique_tgts:
                    mask = tgt_nodes == tgt
                    tgt_scores = scores[mask]
                    tgt_weights = F.softmax(tgt_scores, dim=0)
                    attn_weights[mask] = tgt_weights

            # Apply dropout
            attn_weights = self.dropout(attn_weights)  # [num_edges, num_heads]

            # Aggregate messages with attention weights
            # [num_edges, num_heads, 1] * [num_edges, num_heads, head_dim] → [num_edges, num_heads, head_dim]
            messages = attn_weights.unsqueeze(-1) * v_src

            # Sum messages per target node
            if TORCH_SCATTER_AVAILABLE:
                # Efficient scatter sum
                # CRITICAL: dim_size must match max_nodes to prevent hangs
                try:
                    aggregated = scatter(
                        messages,  # [num_edges, num_heads, head_dim]
                        tgt_nodes,  # [num_edges]
                        dim=0,
                        dim_size=max_nodes,
                        reduce="sum",
                    )
                    # Validate output shape
                    if aggregated.shape[0] != max_nodes:
                        raise RuntimeError(
                            f"[GAT] scatter output shape mismatch: expected {max_nodes}, "
                            f"got {aggregated.shape[0]}"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"[GAT] scatter failed in batch {b}: {e}. "
                        f"tgt_nodes range: [{tgt_nodes.min().item()}, {tgt_nodes.max().item()}], "
                        f"max_nodes: {max_nodes}, num_edges: {len(tgt_nodes)}, "
                        f"messages shape: {messages.shape}"
                    ) from e
            else:
                # Fallback: manual sum per target
                aggregated = torch.zeros(
                    max_nodes, self.num_heads, self.head_dim, device=device
                )
                for i, tgt in enumerate(tgt_nodes):
                    # Additional safety check in fallback path
                    if tgt >= max_nodes or tgt < 0:
                        raise ValueError(
                            f"[GAT] Invalid target index {tgt.item()} in fallback aggregation "
                            f"(batch {b}, edge {i}, max_nodes={max_nodes})"
                        )
                    aggregated[tgt] += messages[i]

            outputs.append(aggregated)  # [max_nodes, num_heads, head_dim]

        # Stack batch dimension
        out = torch.stack(outputs, dim=0)  # [batch, max_nodes, num_heads, head_dim]

        # Concatenate or average heads
        if self.concat_heads:
            out = out.reshape(
                batch_size, max_nodes, -1
            )  # [batch, max_nodes, head_dim * num_heads]
        else:
            out = out.mean(dim=2)  # [batch, max_nodes, out_dim]

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection if dimensions match
        if node_features.shape[-1] == self.out_dim:
            out = self.norm(out + node_features)
        else:
            out = self.norm(out)

        # Apply node mask
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
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GAT layers
        self.layers = nn.ModuleList(
            [
                GATLayer(hidden_dim, hidden_dim, num_heads, dropout, concat_heads=True)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [batch_size, max_nodes, node_feature_dim]
            edge_index: [batch_size, 2, num_edges]
            node_mask: [batch_size, max_nodes]
            edge_mask: [batch_size, num_edges] - 1 for valid edges, 0 for padding

        Returns:
            node_embeddings: [batch_size, max_nodes, output_dim]
            graph_embedding: [batch_size, output_dim]
        """
        # Project input features
        h = self.input_proj(node_features)

        # Apply GAT layers
        for layer in self.layers:
            h = layer(h, edge_index, node_mask, edge_mask)

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
