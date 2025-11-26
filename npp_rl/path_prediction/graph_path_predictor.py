"""Graph-based Path Predictor using GNN + Pointer Networks.

This module implements path prediction as discrete node selection using:
1. Existing GNN encoders (GCNEncoder or GATEncoder) for graph representation
2. Pointer Network decoders for selecting sequences of graph nodes
3. Multiple decoder heads for diverse path candidates

Predictions are guaranteed to be on valid graph nodes (geometric validity).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
import logging
import importlib.util
from pathlib import Path

# Import pointer decoder, adapter, and fusion module
from .pointer_decoder import MultiHeadPointerDecoder
from .graph_data_adapter import GraphDataAdapter
from .multimodal_fusion import PathPredictionFusion


# Bypass circular import by loading GCN/GAT directly
def _import_gcn_gat():
    """Import GCN and GAT encoders directly to avoid circular imports."""
    # Get the path to the models directory
    models_dir = Path(__file__).parent.parent / "models"

    # Import GCNEncoder
    gcn_spec = importlib.util.spec_from_file_location(
        "gcn_module", models_dir / "gcn.py"
    )
    gcn_module = importlib.util.module_from_spec(gcn_spec)
    gcn_spec.loader.exec_module(gcn_module)

    # Import GATEncoder
    gat_spec = importlib.util.spec_from_file_location(
        "gat_module", models_dir / "gat.py"
    )
    gat_module = importlib.util.module_from_spec(gat_spec)
    gat_spec.loader.exec_module(gat_module)

    return gcn_module.GCNEncoder, gat_module.GATEncoder


GCNEncoder, GATEncoder = _import_gcn_gat()

logger = logging.getLogger(__name__)


class GraphPathPredictor(nn.Module):
    """Graph-based path predictor using GNN + Pointer Networks.

    Architecture:
    1. GNN Encoder: Learns node embeddings from graph structure
    2. Multi-Head Pointer Decoder: Selects sequences of nodes as paths
    3. Multiple heads generate diverse path candidates

    Output: Discrete node IDs (not continuous coordinates)
    """

    def __init__(
        self,
        node_feature_dim: int = 16,  # Updated from 8 to 16 for enhanced features
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_gnn_layers: int = 3,
        num_path_candidates: int = 4,
        max_waypoints: int = 20,
        gnn_type: str = "gcn",  # "gcn" or "gat"
        num_gat_heads: int = 8,
        dropout: float = 0.1,
        max_nodes: int = 5000,
        max_edges: int = 40000,
        context_dim: int = 256,  # Fusion output dimension
        fusion_hidden_dim: int = 128,  # Physics encoder hidden dimension
        use_fusion: bool = True,  # Enable multimodal fusion
    ):
        """Initialize graph-based path predictor with multimodal fusion.

        Args:
            node_feature_dim: Dimension of input node features (16 for enhanced features)
            hidden_dim: Hidden dimension for GNN
            output_dim: Output dimension for GNN
            num_gnn_layers: Number of GNN layers
            num_path_candidates: Number of diverse paths to generate
            max_waypoints: Maximum waypoints per path
            gnn_type: Type of GNN ("gcn" or "gat")
            num_gat_heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            max_nodes: Maximum nodes for padding
            max_edges: Maximum edges for padding
            context_dim: Dimension of fused context for pointer decoder
            fusion_hidden_dim: Hidden dimension for physics encoder in fusion
            use_fusion: Whether to use multimodal fusion (True for new architecture)
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_path_candidates = num_path_candidates
        self.max_waypoints = max_waypoints
        self.gnn_type = gnn_type
        self.use_fusion = use_fusion
        self.context_dim = context_dim

        # Graph data adapter
        self.graph_adapter = GraphDataAdapter(max_nodes=max_nodes, max_edges=max_edges)

        # GNN Encoder
        if gnn_type == "gcn":
            self.gnn_encoder = GCNEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
            )
            logger.info(f"Using GCNEncoder with {num_gnn_layers} layers")
        elif gnn_type == "gat":
            self.gnn_encoder = GATEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_gnn_layers,
                num_heads=num_gat_heads,
                dropout=dropout,
            )
            logger.info(
                f"Using GATEncoder with {num_gnn_layers} layers, {num_gat_heads} heads"
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Choose 'gcn' or 'gat'")

        # Multimodal fusion module (if enabled)
        if use_fusion:
            self.fusion = PathPredictionFusion(
                graph_output_dim=output_dim,
                node_feature_dim=node_feature_dim,
                hidden_dim=fusion_hidden_dim,
                uniform_dim=256,  # Standard uniform dimension for fusion
                context_dim=context_dim,
                num_heads=8,  # Multi-head attention heads
                dropout=dropout,
            )
            logger.info(
                f"Enabled multimodal fusion: 4 modalities → {context_dim}D context"
            )
        else:
            self.fusion = None
            # Backward compatibility: use graph embedding directly as context
            context_dim = output_dim

        # Multi-head pointer decoder with proper context_dim
        self.pointer_decoder = MultiHeadPointerDecoder(
            num_heads=num_path_candidates,
            hidden_dim=output_dim,
            max_waypoints=max_waypoints,
            dropout=dropout,
            context_dim=context_dim,
        )

        # Update head projections to use context_dim (for diversity across heads)
        self.pointer_decoder.head_projections = nn.ModuleList(
            [nn.Linear(context_dim, context_dim) for _ in range(num_path_candidates)]
        )

        fusion_str = (
            "WITH multimodal fusion" if use_fusion else "WITHOUT fusion (legacy)"
        )
        logger.info(
            f"Initialized GraphPathPredictor: {gnn_type.upper()}-{num_gnn_layers}L "
            f"→ {num_path_candidates} paths × {max_waypoints} waypoints ({fusion_str})"
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        start_node_ids: Optional[torch.Tensor] = None,
        goal_node_ids: Optional[torch.Tensor] = None,
        ninja_states: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        use_sampling: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode graph, fuse modalities, and generate path candidates.

        Args:
            node_features: [batch, max_nodes, node_feature_dim]
            edge_index: [batch, 2, max_edges]
            node_mask: [batch, max_nodes] - 1 for valid nodes
            edge_mask: [batch, max_edges] - 1 for valid edges
            start_node_ids: [batch] - start node ID for each sample (for fusion)
            goal_node_ids: [batch, num_goals] - goal node IDs (for fusion)
            ninja_states: [batch, 40] - full ninja physics state (for fusion)
            temperature: Softmax temperature for pointer network
            use_sampling: If True, sample nodes; if False, use argmax

        Returns:
            Dictionary containing:
            - node_indices: [batch, num_heads, max_waypoints] - predicted node IDs
            - logits: [batch, num_heads, max_waypoints, max_nodes] - pointer logits
            - confidences: [batch, num_heads] - path confidence scores
            - node_embeddings: [batch, max_nodes, output_dim] - node representations
            - graph_embedding: [batch, output_dim] - global graph representation
            - fused_context: [batch, context_dim] - fused multimodal context (if fusion enabled)
        """
        # Encode graph with GNN
        node_embeddings, graph_embedding = self.gnn_encoder(
            node_features, edge_index, node_mask, edge_mask
        )

        # Fuse modalities if fusion is enabled and all inputs are provided
        if (
            self.use_fusion
            and start_node_ids is not None
            and goal_node_ids is not None
            and ninja_states is not None
        ):
            batch_size = node_embeddings.size(0)

            # Extract start node embeddings
            # start_node_ids: [batch] -> [batch, 1, 1] for gathering
            start_indices = (
                start_node_ids.unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, 1, self.hidden_dim)
            )
            start_node_embedding = torch.gather(
                node_embeddings, 1, start_indices
            ).squeeze(1)
            # [batch, hidden_dim]

            # Extract goal node embeddings
            # goal_node_ids: [batch, num_goals] -> [batch, num_goals, hidden_dim]
            num_goals = goal_node_ids.size(1)
            goal_indices = goal_node_ids.unsqueeze(2).expand(
                batch_size, num_goals, self.hidden_dim
            )
            goal_node_embeddings = torch.gather(node_embeddings, 1, goal_indices)
            # [batch, num_goals, hidden_dim]

            # Fuse modalities
            fused_context = self.fusion(
                graph_embedding=graph_embedding,
                start_node_embedding=start_node_embedding,
                goal_node_embeddings=goal_node_embeddings,
                ninja_state=ninja_states,
            )
            # [batch, context_dim]
        else:
            # Backward compatibility: use graph embedding directly
            fused_context = graph_embedding

        # Generate paths with pointer network
        if use_sampling:
            node_indices, logits, confidences = (
                self.pointer_decoder.forward_with_sampling(
                    node_embeddings, fused_context, node_mask, temperature
                )
            )
        else:
            node_indices, logits, confidences = self.pointer_decoder(
                node_embeddings, fused_context, node_mask, temperature
            )

        return {
            "node_indices": node_indices,  # [batch, heads, max_waypoints]
            "logits": logits,  # [batch, heads, max_waypoints, max_nodes]
            "confidences": confidences,  # [batch, heads]
            "node_embeddings": node_embeddings,  # [batch, max_nodes, hidden_dim]
            "graph_embedding": graph_embedding,  # [batch, hidden_dim]
            "fused_context": fused_context,  # [batch, context_dim]
        }

    def forward_from_adjacency(
        self,
        adjacency: Dict,
        start_pos: Optional[Tuple[float, float]] = None,
        goal_positions: Optional[List[Tuple[float, float]]] = None,
        ninja_state: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], Dict]:
        """Convenience method to predict from adjacency dict with multimodal inputs.

        Args:
            adjacency: Graph adjacency dictionary
            start_pos: Starting position (x, y) for fusion
            goal_positions: List of goal positions [(x1, y1), (x2, y2), ...] for fusion
            ninja_state: Full ninja physics state [40] for fusion
            temperature: Softmax temperature
            device: Device for tensors

        Returns:
            Tuple of:
            - node_indices: [num_heads, max_waypoints] - predicted node IDs
            - positions: List of predicted waypoint positions
            - metadata: Dict with position mappings and graph info
        """
        # Convert adjacency to tensors
        node_features, edge_index, node_mask, edge_mask, metadata = (
            self.graph_adapter.adjacency_to_tensors(
                adjacency,
                self.node_feature_dim,
                start_pos=start_pos,
                goal_positions=goal_positions,
            )
        )

        # Add batch dimension and move to device
        node_features = node_features.unsqueeze(0).to(device)
        edge_index = edge_index.unsqueeze(0).to(device)
        node_mask = node_mask.unsqueeze(0).to(device)
        edge_mask = edge_mask.unsqueeze(0).to(device)

        # Process fusion inputs if provided
        start_node_ids = None
        goal_node_ids = None
        ninja_states = None

        if (
            self.use_fusion
            and start_pos is not None
            and goal_positions is not None
            and ninja_state is not None
        ):
            # Import pathfinding utilities
            from nclone.graph.reachability.pathfinding_utils import (
                find_closest_node_to_position,
            )

            # Convert start position to node ID
            start_node = find_closest_node_to_position(
                start_pos, adjacency, threshold=50.0
            )
            if start_node is not None:
                start_node_id = metadata["position_to_id"].get(start_node, 0)
                start_node_ids = torch.tensor(
                    [start_node_id], dtype=torch.long, device=device
                )

            # Convert goal positions to node IDs
            goal_ids = []
            for goal_pos in goal_positions:
                goal_node = find_closest_node_to_position(
                    goal_pos, adjacency, threshold=50.0
                )
                if goal_node is not None:
                    goal_node_id = metadata["position_to_id"].get(goal_node, 0)
                    goal_ids.append(goal_node_id)

            if goal_ids:
                goal_node_ids = torch.tensor(
                    [goal_ids], dtype=torch.long, device=device
                )

            # Add batch dimension to ninja state
            if isinstance(ninja_state, torch.Tensor):
                ninja_states = ninja_state.unsqueeze(0).to(device)
            else:
                ninja_states = torch.tensor(
                    [ninja_state], dtype=torch.float32, device=device
                )

        # Forward pass
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                node_features,
                edge_index,
                node_mask,
                edge_mask,
                start_node_ids=start_node_ids,
                goal_node_ids=goal_node_ids,
                ninja_states=ninja_states,
                temperature=temperature,
            )

        # Extract predictions (remove batch dimension)
        node_indices = outputs["node_indices"][0]  # [num_heads, max_waypoints]

        # Convert node IDs to positions
        positions_all_heads = []
        for head_idx in range(self.num_path_candidates):
            head_indices = node_indices[head_idx]
            positions = self.graph_adapter.node_ids_to_positions(
                head_indices, metadata["id_to_position"]
            )
            positions_all_heads.append(positions)

        return node_indices, positions_all_heads, metadata

    def get_statistics(self) -> Dict[str, any]:
        """Get model statistics.

        Returns:
            Dictionary with model configuration and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "gnn_type": self.gnn_type,
            "hidden_dim": self.hidden_dim,
            "num_path_candidates": self.num_path_candidates,
            "max_waypoints": self.max_waypoints,
        }


def create_graph_path_predictor(config: Dict) -> GraphPathPredictor:
    """Factory function to create GraphPathPredictor from config.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Configured GraphPathPredictor instance
    """
    return GraphPathPredictor(
        node_feature_dim=config.get("node_feature_dim", 16),  # Updated default
        hidden_dim=config.get("hidden_dim", 256),
        output_dim=config.get("output_dim", 256),
        num_gnn_layers=config.get("num_gnn_layers", 3),
        num_path_candidates=config.get("num_path_candidates", 4),
        max_waypoints=config.get("max_waypoints", 20),
        gnn_type=config.get("gnn_type", "gcn"),
        num_gat_heads=config.get("num_gat_heads", 8),
        dropout=config.get("dropout", 0.1),
        max_nodes=config.get("max_nodes", 5000),
        max_edges=config.get("max_edges", 40000),
        context_dim=config.get("context_dim", 256),
        fusion_hidden_dim=config.get("fusion_hidden_dim", 128),
        use_fusion=config.get("use_fusion", True),
    )
