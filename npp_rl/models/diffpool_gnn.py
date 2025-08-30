"""
DiffPool GNN implementation for hierarchical graph processing.

This module implements differentiable graph pooling (DiffPool) for learning
hierarchical representations of N++ levels. DiffPool enables end-to-end
training of multi-resolution graph neural networks through soft cluster
assignments and learnable pooling operations.

Based on:
- Ying et al. (2018) "Hierarchical Graph Representation Learning with Differentiable Pooling"
- Adapted for N++ level structure and physics-informed features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List

from .gnn import GraphSAGELayer


class DiffPoolLayer(nn.Module):
    """
    Differentiable pooling layer for hierarchical graph coarsening.
    
    Learns soft cluster assignments to pool nodes into coarser representations
    while preserving important structural and feature information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_clusters: int,
        gnn_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize DiffPool layer.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output node feature dimension
            num_clusters: Number of clusters to pool into
            gnn_layers: Number of GNN layers for embedding and assignment
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_clusters = num_clusters
        
        # GNN for node embeddings
        self.embedding_gnn = nn.ModuleList()
        for i in range(gnn_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < gnn_layers - 1 else output_dim
            self.embedding_gnn.append(
                GraphSAGELayer(in_dim, out_dim, dropout=dropout)
            )
        
        # GNN for cluster assignments
        self.assignment_gnn = nn.ModuleList()
        for i in range(gnn_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < gnn_layers - 1 else num_clusters
            self.assignment_gnn.append(
                GraphSAGELayer(in_dim, out_dim, dropout=dropout)
            )
        
        # Link prediction for auxiliary loss
        self.link_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        ninja_physics_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through DiffPool layer.
        
        Args:
            node_features: Node features [batch_size, num_nodes, input_dim]
            edge_index: Edge indices [batch_size, 2, num_edges]
            node_mask: Node mask [batch_size, num_nodes]
            edge_mask: Edge mask [batch_size, num_edges]
            ninja_physics_state: Optional ninja physics state
            
        Returns:
            Tuple of:
            - pooled_node_features: [batch_size, num_clusters, output_dim]
            - pooled_edge_index: [batch_size, 2, pooled_num_edges]
            - pooled_node_mask: [batch_size, num_clusters]
            - pooled_edge_mask: [batch_size, pooled_num_edges]
            - auxiliary_losses: Dict with link prediction and entropy losses
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Generate node embeddings
        embeddings = node_features
        for gnn_layer in self.embedding_gnn:
            embeddings = gnn_layer(
                embeddings, edge_index, node_mask, edge_mask, ninja_physics_state
            )
            embeddings = self.dropout(embeddings)
        
        # Generate cluster assignments
        assignments = node_features
        for gnn_layer in self.assignment_gnn:
            assignments = gnn_layer(
                assignments, edge_index, node_mask, edge_mask, ninja_physics_state
            )
            assignments = self.dropout(assignments)
        
        # Apply softmax to get soft cluster assignments with numerical stability
        assignments = F.softmax(assignments, dim=-1)  # [batch_size, num_nodes, num_clusters]
        
        # Apply node mask to assignments and ensure numerical stability
        assignments = assignments * node_mask.unsqueeze(-1)
        
        # Normalize assignments to ensure they sum to 1 for valid nodes
        assignment_sums = torch.sum(assignments, dim=-1, keepdim=True)
        assignment_sums = torch.clamp(assignment_sums, min=1e-8)  # Prevent division by zero
        assignments = assignments / assignment_sums
        
        # Pool node features: X_pooled = S^T * Z
        pooled_node_features = torch.bmm(
            assignments.transpose(1, 2),  # [batch_size, num_clusters, num_nodes]
            embeddings  # [batch_size, num_nodes, output_dim]
        )  # [batch_size, num_clusters, output_dim]
        
        # Create pooled node mask based on cluster assignments
        pooled_node_mask = torch.sum(assignments, dim=1)  # [batch_size, num_clusters]
        pooled_node_mask = (pooled_node_mask > 1e-6).float()  # Threshold for active clusters
        
        # Pool adjacency matrix: A_pooled = S^T * A * S
        pooled_adj, pooled_edge_index, pooled_edge_mask = self._pool_adjacency(
            edge_index, edge_mask, assignments, batch_size, num_nodes
        )
        
        # Compute auxiliary losses
        auxiliary_losses = self._compute_auxiliary_losses(
            embeddings, edge_index, edge_mask, assignments, pooled_adj, node_mask
        )
        
        return (
            pooled_node_features,
            pooled_edge_index,
            pooled_node_mask,
            pooled_edge_mask,
            auxiliary_losses
        )
    
    def _pool_adjacency(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        assignments: torch.Tensor,
        batch_size: int,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool adjacency matrix using cluster assignments.
        
        Args:
            edge_index: Edge indices [batch_size, 2, num_edges]
            edge_mask: Edge mask [batch_size, num_edges]
            assignments: Cluster assignments [batch_size, num_nodes, num_clusters]
            batch_size: Batch size
            num_nodes: Number of nodes
            
        Returns:
            Tuple of (pooled_adjacency, pooled_edge_index, pooled_edge_mask)
        """
        device = edge_index.device
        
        # Convert edge_index to dense adjacency matrices
        adj_matrices = []
        
        for b in range(batch_size):
            # Create dense adjacency matrix for this batch
            adj = torch.zeros(num_nodes, num_nodes, device=device)
            
            # Get valid edges for this batch
            valid_edges = edge_mask[b].bool()
            if valid_edges.any():
                src_nodes = edge_index[b, 0, valid_edges]
                tgt_nodes = edge_index[b, 1, valid_edges]
                adj[src_nodes, tgt_nodes] = 1.0
            
            adj_matrices.append(adj)
        
        adj_matrices = torch.stack(adj_matrices)  # [batch_size, num_nodes, num_nodes]
        
        # Pool adjacency: A_pooled = S^T * A * S
        pooled_adj = torch.bmm(
            torch.bmm(assignments.transpose(1, 2), adj_matrices),  # S^T * A
            assignments  # * S
        )  # [batch_size, num_clusters, num_clusters]
        
        # Convert back to edge_index format
        pooled_edge_indices = []
        pooled_edge_masks = []
        max_pooled_edges = self.num_clusters * self.num_clusters
        
        for b in range(batch_size):
            # Find non-zero entries in pooled adjacency
            src_nodes, tgt_nodes = torch.nonzero(pooled_adj[b] > 1e-6, as_tuple=True)
            
            # Create edge index for this batch
            num_edges = len(src_nodes)
            edge_idx = torch.zeros(2, max_pooled_edges, dtype=torch.long, device=device)
            edge_msk = torch.zeros(max_pooled_edges, device=device)
            
            if num_edges > 0:
                edge_idx[0, :num_edges] = src_nodes
                edge_idx[1, :num_edges] = tgt_nodes
                edge_msk[:num_edges] = 1.0
            
            pooled_edge_indices.append(edge_idx)
            pooled_edge_masks.append(edge_msk)
        
        pooled_edge_index = torch.stack(pooled_edge_indices)
        pooled_edge_mask = torch.stack(pooled_edge_masks)
        
        return pooled_adj, pooled_edge_index, pooled_edge_mask
    
    def _compute_auxiliary_losses(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        assignments: torch.Tensor,
        pooled_adj: torch.Tensor,
        node_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for DiffPool training.
        
        Args:
            embeddings: Node embeddings [batch_size, num_nodes, output_dim]
            edge_index: Original edge indices
            edge_mask: Original edge mask
            assignments: Cluster assignments
            pooled_adj: Pooled adjacency matrix
            node_mask: Node mask
            
        Returns:
            Dictionary with auxiliary losses
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Link prediction loss
        link_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            valid_edges = edge_mask[b].bool()
            if not valid_edges.any():
                continue
            
            # Get edge embeddings
            src_nodes = edge_index[b, 0, valid_edges]
            tgt_nodes = edge_index[b, 1, valid_edges]
            
            src_embeddings = embeddings[b, src_nodes]  # [num_valid_edges, output_dim]
            tgt_embeddings = embeddings[b, tgt_nodes]  # [num_valid_edges, output_dim]
            
            # Concatenate source and target embeddings
            edge_embeddings = torch.cat([src_embeddings, tgt_embeddings], dim=-1)
            
            # Predict link existence
            link_probs = self.link_predictor(edge_embeddings).squeeze(-1)
            
            # Target is 1 for existing edges
            targets = torch.ones_like(link_probs)
            
            # Binary cross entropy loss
            link_loss += F.binary_cross_entropy(link_probs, targets)
        
        link_loss = link_loss / batch_size
        
        # Entropy regularization loss to encourage diverse cluster assignments
        entropy_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            # Get valid assignments for this batch
            valid_assignments = assignments[b] * node_mask[b].unsqueeze(-1)
            
            # Compute entropy of cluster assignments
            cluster_probs = torch.sum(valid_assignments, dim=0)  # [num_clusters]
            cluster_probs = cluster_probs / (torch.sum(cluster_probs) + 1e-8)
            
            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8))
            entropy_loss += entropy
        
        entropy_loss = entropy_loss / batch_size
        
        # Orthogonality loss to encourage diverse cluster assignments
        ortho_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            # Get valid assignments
            valid_assignments = assignments[b] * node_mask[b].unsqueeze(-1)
            
            # Compute S^T * S
            gram_matrix = torch.mm(valid_assignments.t(), valid_assignments)
            
            # Encourage orthogonality (identity matrix)
            identity = torch.eye(self.num_clusters, device=device)
            ortho_loss += F.mse_loss(gram_matrix, identity)
        
        ortho_loss = ortho_loss / batch_size
        
        return {
            'link_prediction_loss': link_loss,
            'entropy_loss': entropy_loss,
            'orthogonality_loss': ortho_loss
        }


class HierarchicalDiffPoolGNN(nn.Module):
    """
    Hierarchical GNN using DiffPool for multi-resolution processing.
    
    Processes graphs at multiple resolutions using differentiable pooling
    to learn hierarchical representations suitable for both local movement
    decisions and global pathfinding strategies.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_levels: int = 3,
        pooling_ratios: List[float] = [0.25, 0.25, 0.5],
        gnn_layers_per_level: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical DiffPool GNN.
        
        Args:
            input_dims: Input dimensions for each level {'sub_cell': dim, 'tile': dim, 'region': dim}
            hidden_dim: Hidden dimension for processing
            output_dim: Final output dimension
            num_levels: Number of hierarchical levels
            pooling_ratios: Pooling ratios for each level
            gnn_layers_per_level: Number of GNN layers per pooling level
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.pooling_ratios = pooling_ratios
        
        # Input projections for each resolution level
        self.input_projections = nn.ModuleDict()
        for level_name, input_dim in input_dims.items():
            self.input_projections[level_name] = nn.Linear(input_dim, hidden_dim)
        
        # DiffPool layers for hierarchical processing
        self.diffpool_layers = nn.ModuleList()
        
        current_dim = hidden_dim
        for i in range(num_levels - 1):  # n-1 pooling layers for n levels
            # Calculate number of clusters based on pooling ratio
            # This is a simplified calculation - in practice, you'd determine this
            # based on the actual graph structure
            num_clusters = max(1, int(1000 * pooling_ratios[i]))  # Placeholder calculation
            
            diffpool_layer = DiffPoolLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_clusters=num_clusters,
                gnn_layers=gnn_layers_per_level,
                dropout=dropout
            )
            self.diffpool_layers.append(diffpool_layer)
        
        # Final processing layers
        self.final_gnn = nn.ModuleList([
            GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(2)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Global pooling for graph-level representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hierarchical_graph_data: Dict[str, torch.Tensor],
        ninja_physics_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical DiffPool GNN.
        
        Args:
            hierarchical_graph_data: Dictionary containing graph data for each level
            ninja_physics_state: Optional ninja physics state
            
        Returns:
            Tuple of (graph_embedding, auxiliary_losses)
        """
        # Start with the finest resolution (sub-cell level)
        level_name = 'sub_cell'
        node_features = hierarchical_graph_data[f'{level_name}_node_features']
        edge_index = hierarchical_graph_data[f'{level_name}_edge_index']
        node_mask = hierarchical_graph_data[f'{level_name}_node_mask']
        edge_mask = hierarchical_graph_data[f'{level_name}_edge_mask']
        
        # Project input features
        node_features = self.input_projections[level_name](node_features)
        node_features = F.relu(node_features)
        node_features = self.dropout(node_features)
        
        # Collect auxiliary losses
        total_auxiliary_losses = {
            'link_prediction_loss': torch.tensor(0.0, device=node_features.device),
            'entropy_loss': torch.tensor(0.0, device=node_features.device),
            'orthogonality_loss': torch.tensor(0.0, device=node_features.device)
        }
        
        # Apply DiffPool layers for hierarchical coarsening
        for i, diffpool_layer in enumerate(self.diffpool_layers):
            node_features, edge_index, node_mask, edge_mask, aux_losses = diffpool_layer(
                node_features, edge_index, node_mask, edge_mask, ninja_physics_state
            )
            
            # Accumulate auxiliary losses
            for loss_name, loss_value in aux_losses.items():
                total_auxiliary_losses[loss_name] += loss_value
        
        # Final GNN processing at coarsest level
        for gnn_layer in self.final_gnn:
            node_features = gnn_layer(
                node_features, edge_index, node_mask, edge_mask, ninja_physics_state
            )
            node_features = self.dropout(node_features)
        
        # Global pooling to get graph-level representation
        # Apply node mask before pooling
        masked_features = node_features * node_mask.unsqueeze(-1)
        
        # Sum pooling with normalization
        graph_embedding = torch.sum(masked_features, dim=1)  # [batch_size, hidden_dim]
        
        # Normalize by number of active nodes
        num_active_nodes = torch.sum(node_mask, dim=1, keepdim=True)
        graph_embedding = graph_embedding / (num_active_nodes + 1e-8)
        
        # Final output projection
        graph_embedding = self.output_projection(graph_embedding)
        
        return graph_embedding, total_auxiliary_losses
    
    def compute_total_loss(
        self,
        main_loss: torch.Tensor,
        auxiliary_losses: Dict[str, torch.Tensor],
        aux_loss_weights: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Compute total loss including auxiliary losses.
        
        Args:
            main_loss: Main task loss (e.g., RL policy loss)
            auxiliary_losses: Dictionary of auxiliary losses
            aux_loss_weights: Weights for auxiliary losses
            
        Returns:
            Total weighted loss
        """
        if aux_loss_weights is None:
            aux_loss_weights = {
                'link_prediction_loss': 0.1,
                'entropy_loss': 0.01,
                'orthogonality_loss': 0.1
            }
        
        total_loss = main_loss
        
        for loss_name, loss_value in auxiliary_losses.items():
            weight = aux_loss_weights.get(loss_name, 0.0)
            total_loss += weight * loss_value
        
        return total_loss


class MultiScaleGraphAttention(nn.Module):
    """
    Multi-scale attention mechanism for hierarchical graph features.
    
    Learns to attend to different resolution levels based on the current
    context and ninja state, enabling adaptive focus on local vs global features.
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize multi-scale attention.
        
        Args:
            feature_dims: Feature dimensions for each scale
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Project features from different scales to common dimension
        self.scale_projections = nn.ModuleDict()
        for scale_name, feat_dim in feature_dims.items():
            self.scale_projections[scale_name] = nn.Linear(feat_dim, hidden_dim)
        
        # Multi-head attention components
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Context-aware attention weights
        self.context_attention = nn.Sequential(
            nn.Linear(hidden_dim + 18, hidden_dim),  # +18 for ninja physics state
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        scale_features: Dict[str, torch.Tensor],
        ninja_physics_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale attention.
        
        Args:
            scale_features: Features from different scales
            ninja_physics_state: Ninja physics state for context
            
        Returns:
            Attended multi-scale features
        """
        batch_size = next(iter(scale_features.values())).shape[0]
        device = next(iter(scale_features.values())).device
        
        # Project all scales to common dimension
        projected_features = {}
        for scale_name, features in scale_features.items():
            projected_features[scale_name] = self.scale_projections[scale_name](features)
        
        # Stack features for attention computation
        scale_names = list(projected_features.keys())
        stacked_features = torch.stack([projected_features[name] for name in scale_names], dim=1)
        # [batch_size, num_scales, hidden_dim]
        
        # Compute context-aware scale attention weights
        if ninja_physics_state is not None:
            # Use mean of scale features as context
            context_features = torch.mean(stacked_features, dim=1)  # [batch_size, hidden_dim]
            
            # Expand ninja physics state to match batch size if needed
            if ninja_physics_state.dim() == 1:
                ninja_physics_state = ninja_physics_state.unsqueeze(0).expand(batch_size, -1)
            
            # Combine context and physics state
            context_input = torch.cat([context_features, ninja_physics_state], dim=-1)
            scale_weights = self.context_attention(context_input)  # [batch_size, num_scales]
        else:
            # Uniform attention if no physics state
            scale_weights = torch.ones(batch_size, len(scale_names), device=device) / len(scale_names)
        
        # Apply scale attention weights
        weighted_features = stacked_features * scale_weights.unsqueeze(-1)
        attended_features = torch.sum(weighted_features, dim=1)  # [batch_size, hidden_dim]
        
        # Multi-head self-attention on attended features
        queries = self.query_projection(attended_features)
        keys = self.key_projection(attended_features)
        values = self.value_projection(attended_features)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, self.num_heads, self.head_dim)
        values = values.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.sum(queries * keys, dim=-1) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = values * attention_weights.unsqueeze(-1)
        attended_values = attended_values.view(batch_size, self.hidden_dim)
        
        # Output projection and residual connection
        output = self.output_projection(attended_values)
        output = self.dropout(output)
        output = self.layer_norm(output + attended_features)
        
        return output