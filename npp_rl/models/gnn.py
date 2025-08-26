"""
Graph Neural Network (GNN) implementation for structural level understanding.

This module implements GraphSAGE-style message passing for processing
graph-based observations of N++ levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with masked message passing.
    
    Implements the GraphSAGE aggregation scheme with support for
    padded graphs using node and edge masks.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregator: str = 'mean',
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize GraphSAGE layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            aggregator: Aggregation method ('mean', 'max', 'sum')
            activation: Activation function ('relu', 'tanh', 'none')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        
        # Linear transformations
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)
        
        # Activation and regularization
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GraphSAGE layer.
        
        Args:
            node_features: Node features [batch_size, num_nodes, in_dim]
            edge_index: Edge indices [batch_size, 2, num_edges]
            node_mask: Node mask [batch_size, num_nodes]
            edge_mask: Edge mask [batch_size, num_edges]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Self transformation
        self_features = self.self_linear(node_features)
        
        # Neighbor aggregation
        neighbor_features = self._aggregate_neighbors(
            node_features, edge_index, node_mask, edge_mask
        )
        neighbor_features = self.neighbor_linear(neighbor_features)
        
        # Combine self and neighbor features
        output = self_features + neighbor_features
        output = self.activation(output)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        # Apply node mask
        output = output * node_mask.unsqueeze(-1)
        
        return output
    
    def _aggregate_neighbors(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate neighbor features using specified aggregation method.
        
        Args:
            node_features: Node features [batch_size, num_nodes, in_dim]
            edge_index: Edge indices [batch_size, 2, num_edges]
            node_mask: Node mask [batch_size, num_nodes]
            edge_mask: Edge mask [batch_size, num_edges]
            
        Returns:
            Aggregated neighbor features [batch_size, num_nodes, in_dim]
        """
        batch_size, num_nodes, in_dim = node_features.shape
        _, _, num_edges = edge_index.shape
        
        # Initialize aggregated features
        aggregated = torch.zeros_like(node_features)
        
        for b in range(batch_size):
            # Get valid edges for this batch
            valid_edges = edge_mask[b].bool()
            if not valid_edges.any():
                continue
                
            src_nodes = edge_index[b, 0, valid_edges]
            tgt_nodes = edge_index[b, 1, valid_edges]
            
            # Get source node features
            src_features = node_features[b, src_nodes]  # [num_valid_edges, in_dim]
            
            # Aggregate by target node
            for i in range(num_nodes):
                # Find edges targeting node i
                target_mask = (tgt_nodes == i)
                if not target_mask.any():
                    continue
                
                # Get features of nodes that connect to node i
                incoming_features = src_features[target_mask]  # [num_incoming, in_dim]
                
                # Aggregate
                if self.aggregator == 'mean':
                    aggregated[b, i] = incoming_features.mean(dim=0)
                elif self.aggregator == 'max':
                    aggregated[b, i] = incoming_features.max(dim=0)[0]
                elif self.aggregator == 'sum':
                    aggregated[b, i] = incoming_features.sum(dim=0)
        
        return aggregated


class GraphEncoder(nn.Module):
    """
    Graph encoder using multiple GraphSAGE layers with global pooling.
    
    Processes graph observations and produces fixed-size embeddings
    suitable for use in RL policies.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 256,
        aggregator: str = 'mean',
        global_pool: str = 'mean_max',
        dropout: float = 0.1
    ):
        """
        Initialize graph encoder.
        
        Args:
            node_feature_dim: Input node feature dimension
            edge_feature_dim: Input edge feature dimension (currently unused)
            hidden_dim: Hidden layer dimension
            num_layers: Number of GraphSAGE layers
            output_dim: Final output dimension
            aggregator: Node aggregation method
            global_pool: Global pooling method ('mean', 'max', 'mean_max')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.global_pool = global_pool
        
        # Input projection
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # GraphSAGE layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GraphSAGELayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                aggregator=aggregator,
                dropout=dropout
            )
            self.gnn_layers.append(layer)
        
        # Global pooling and output projection
        pool_dim = hidden_dim
        if global_pool == 'mean_max':
            pool_dim = hidden_dim * 2
            
        self.output_projection = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, graph_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through graph encoder.
        
        Args:
            graph_obs: Dictionary containing:
                - graph_node_feats: [batch_size, num_nodes, node_feat_dim]
                - graph_edge_index: [batch_size, 2, num_edges]
                - graph_edge_feats: [batch_size, num_edges, edge_feat_dim]
                - graph_node_mask: [batch_size, num_nodes]
                - graph_edge_mask: [batch_size, num_edges]
                
        Returns:
            Graph embedding [batch_size, output_dim]
        """
        node_features = graph_obs['graph_node_feats']
        edge_index = graph_obs['graph_edge_index']
        node_mask = graph_obs['graph_node_mask']
        edge_mask = graph_obs['graph_edge_mask']
                
        # Input projection
        x = self.input_projection(node_features)
        x = F.relu(x)
        
        # Apply GraphSAGE layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, node_mask, edge_mask)
        
        # Global pooling
        graph_embedding = self._global_pool(x, node_mask)
        
        # Output projection
        output = self.output_projection(graph_embedding)
        
        return output
    
    def _global_pool(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply global pooling to get graph-level representation.
        
        Args:
            node_features: Node features [batch_size, num_nodes, hidden_dim]
            node_mask: Node mask [batch_size, num_nodes]
            
        Returns:
            Graph embedding [batch_size, pool_dim]
        """
        # Mask out invalid nodes
        masked_features = node_features * node_mask.unsqueeze(-1)
        
        if self.global_pool == 'mean':
            # Mean pooling
            num_valid_nodes = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            graph_emb = masked_features.sum(dim=1) / num_valid_nodes
            
        elif self.global_pool == 'max':
            # Max pooling
            masked_features = masked_features.masked_fill(
                ~node_mask.unsqueeze(-1).bool(), float('-inf')
            )
            graph_emb = masked_features.max(dim=1)[0]
            
        elif self.global_pool == 'mean_max':
            # Concatenate mean and max pooling
            num_valid_nodes = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pool = masked_features.sum(dim=1) / num_valid_nodes
            
            masked_for_max = masked_features.masked_fill(
                ~node_mask.unsqueeze(-1).bool(), float('-inf')
            )
            max_pool = masked_for_max.max(dim=1)[0]
            
            graph_emb = torch.cat([mean_pool, max_pool], dim=1)
            
        else:
            raise ValueError(f"Unknown global pooling method: {self.global_pool}")
        
        return graph_emb


def create_graph_encoder(
    node_feature_dim: int,
    edge_feature_dim: int,
    **kwargs
) -> GraphEncoder:
    """
    Create a graph encoder with default parameters.
    
    Args:
        node_feature_dim: Input node feature dimension
        edge_feature_dim: Input edge feature dimension
        **kwargs: Additional parameters for GraphEncoder
        
    Returns:
        Configured GraphEncoder instance
    """
    default_params = {
        'hidden_dim': 128,
        'num_layers': 3,
        'output_dim': 256,
        'aggregator': 'mean',
        'global_pool': 'mean_max',
        'dropout': 0.1
    }
    
    # Override defaults with provided kwargs
    params = {**default_params, **kwargs}
    
    return GraphEncoder(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        **params
    )