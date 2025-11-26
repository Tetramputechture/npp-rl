"""Utilities for extracting graph observations consistently across training and inference.

This module ensures that graph observations are created identically whether we're:
1. Training the path predictor
2. Visualizing predictions
3. Running evaluation
4. Using the predictor in RL training

Consistency is critical for model performance and debugging.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List


def extract_graph_observation(
    graph_data: Optional[Dict[str, Any]],
    target_dim: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract and normalize graph observation features to fixed dimension.

    This function provides a consistent way to convert graph adjacency data
    into a fixed-size feature vector suitable for neural network input.

    The extraction strategy:
    1. If graph has node features: pool them via mean and pad/truncate to target_dim
    2. If graph has only adjacency: encode graph statistics (size, density, etc.)
    3. If no graph data: return zeros

    Args:
        graph_data: Dictionary containing graph information with keys:
                   - 'node_features': Optional[torch.Tensor] of shape [num_nodes, feature_dim]
                   - 'adjacency': Optional[Dict] mapping nodes to neighbors
                   - 'num_nodes': Optional[int] node count
                   - 'num_edges': Optional[int] edge count
        target_dim: Target dimension for output feature vector
        device: Device to place tensor on

    Returns:
        Feature tensor of shape [target_dim]
    """
    import logging

    logger = logging.getLogger(__name__)

    if graph_data is None:
        return torch.zeros(target_dim, device=device)

    # Strategy 1: Use node features if available
    if "node_features" in graph_data and graph_data["node_features"] is not None:
        node_feats = graph_data["node_features"]

        # Convert to tensor if needed
        if not isinstance(node_feats, torch.Tensor):
            node_feats = torch.tensor(node_feats, dtype=torch.float32)

        node_feats = node_feats.to(device)

        # DEFENSIVE: Check for NaN/Inf in node features before processing
        if torch.isnan(node_feats).any() or torch.isinf(node_feats).any():
            logger.warning("NaN/Inf detected in node_features from graph_data")
            logger.warning(f"  Node features shape: {node_feats.shape}")
            logger.warning(f"  NaN count: {torch.isnan(node_feats).sum().item()}")
            logger.warning(f"  Inf count: {torch.isinf(node_feats).sum().item()}")
            # Replace NaN/Inf with zeros to prevent propagation
            node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
            logger.warning("  Replaced NaN/Inf with zeros")

        # Handle different tensor shapes
        if node_feats.dim() == 0 or node_feats.numel() == 0:
            # Scalar or empty tensor - create zeros
            logger.warning("Graph has empty node features - returning zeros")
            return torch.zeros(target_dim, device=device)
        elif node_feats.dim() == 1:
            # Already a feature vector
            pooled = node_feats
        elif node_feats.dim() == 2:
            # Node feature matrix [num_nodes, feature_dim] - pool via mean
            # Check if empty first dimension
            if node_feats.shape[0] == 0:
                logger.warning("Graph has zero nodes - returning zeros")
                return torch.zeros(target_dim, device=device)
            pooled = torch.mean(node_feats, dim=0)
        else:
            # Higher dimensional - flatten and pool
            pooled = torch.mean(node_feats.reshape(-1, node_feats.shape[-1]), dim=0)

        # Pad or truncate to target dimension
        if len(pooled) < target_dim:
            padded = torch.cat(
                [pooled, torch.zeros(target_dim - len(pooled), device=device)]
            )
            return padded
        else:
            return pooled[:target_dim]

    # Strategy 2: Encode graph statistics if adjacency is available
    if "adjacency" in graph_data and graph_data["adjacency"] is not None:
        adjacency = graph_data["adjacency"]
        features = torch.zeros(target_dim, device=device)

        # Compute graph statistics
        num_nodes = (
            len(adjacency)
            if isinstance(adjacency, dict)
            else graph_data.get("num_nodes", 0)
        )

        if num_nodes > 0:
            # Feature 0: Normalized node count (log scale)
            features[0] = min(np.log1p(num_nodes) / 10.0, 1.0)

            if isinstance(adjacency, dict):
                # Compute edge count and degree statistics
                degrees = [len(neighbors) for neighbors in adjacency.values()]
                num_edges = sum(degrees) // 2  # Undirected graph

                # Feature 1: Normalized edge count (log scale)
                features[1] = min(np.log1p(num_edges) / 12.0, 1.0)

                # Feature 2: Graph density
                max_edges = num_nodes * (num_nodes - 1) / 2
                if max_edges > 0:
                    features[2] = num_edges / max_edges

                # Feature 3-7: Degree distribution statistics
                if degrees:
                    features[3] = (
                        np.mean(degrees) / 8.0
                    )  # Avg degree (normalized by 8-connectivity)
                    features[4] = np.max(degrees) / 8.0  # Max degree
                    features[5] = np.min(degrees) / 8.0  # Min degree
                    features[6] = np.std(degrees) / 4.0  # Degree std
                    features[7] = np.median(degrees) / 8.0  # Median degree

        return features

    # Strategy 3: Use explicit node/edge counts if provided
    if "num_nodes" in graph_data or "num_edges" in graph_data:
        features = torch.zeros(target_dim, device=device)

        num_nodes = graph_data.get("num_nodes", 0)
        num_edges = graph_data.get("num_edges", 0)

        # Feature 0: Normalized node count
        features[0] = min(np.log1p(num_nodes) / 10.0, 1.0)

        # Feature 1: Normalized edge count
        features[1] = min(np.log1p(num_edges) / 12.0, 1.0)

        # Feature 2: Graph density estimate
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1) / 2
            features[2] = min(num_edges / max_edges, 1.0)

        return features

    # Fallback: return zeros
    return torch.zeros(target_dim, device=device)


def extract_graph_observation_batch(
    graph_data_list: List[Optional[Dict[str, Any]]],
    target_dim: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract graph observations for a batch of samples.

    Args:
        graph_data_list: List of graph data dictionaries
        target_dim: Target dimension for output feature vectors
        device: Device to place tensors on

    Returns:
        Feature tensor of shape [batch_size, target_dim]
    """
    import logging

    logger = logging.getLogger(__name__)

    batch_features = []

    for idx, graph_data in enumerate(graph_data_list):
        features = extract_graph_observation(graph_data, target_dim, device)

        # DEFENSIVE: Check for NaN in extracted features
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.warning(f"NaN/Inf detected in graph features for batch item {idx}")
            logger.warning(
                f"  Graph data keys: {list(graph_data.keys()) if graph_data else 'None'}"
            )
            if graph_data and "node_features" in graph_data:
                node_feats = graph_data["node_features"]
                if isinstance(node_feats, torch.Tensor):
                    logger.warning(f"  Node features shape: {node_feats.shape}")
                    if node_feats.numel() > 0:
                        logger.warning(
                            f"  Node features stats: min={node_feats.min()}, max={node_feats.max()}"
                        )
                        logger.warning(
                            f"  Node features has NaN: {torch.isnan(node_feats).any()}"
                        )
                    else:
                        logger.warning("  Node features tensor is empty!")

        batch_features.append(features)

    return torch.stack(batch_features)


def build_graph_observation_from_level(
    level_data: Dict[str, Any],
    graph_builder: Optional[Any] = None,
    target_dim: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Build graph observation directly from level data.

    This is useful when we don't have pre-computed graph observations
    (e.g., during online evaluation or visualization).

    Args:
        level_data: Level data containing tiles, entities, etc.
        graph_builder: Optional GraphBuilder instance to build graph
        target_dim: Target dimension for output feature vector
        device: Device to place tensor on

    Returns:
        Feature tensor of shape [target_dim]
    """
    # If no graph builder provided, create minimal observation
    if graph_builder is None:
        return torch.zeros(target_dim, device=device)

    # Build graph
    graph_result = graph_builder.build_graph(level_data)

    # Convert adjacency to graph_data format
    adjacency = graph_result["adjacency"]

    graph_data = {
        "adjacency": adjacency,
        "num_nodes": len(adjacency),
        "num_edges": sum(len(neighbors) for neighbors in adjacency.values()) // 2,
    }

    return extract_graph_observation(graph_data, target_dim, device)


def debug_print_graph_observation(
    graph_obs: torch.Tensor,
    graph_data: Optional[Dict[str, Any]] = None,
    name: str = "Graph Observation",
):
    """Print debug information about graph observation.

    Useful for debugging and ensuring consistency.

    Args:
        graph_obs: Graph observation tensor
        graph_data: Original graph data (optional)
        name: Name for this observation in output
    """
    print(f"\n=== {name} ===")
    print(f"Shape: {graph_obs.shape}")
    print(f"Device: {graph_obs.device}")
    print(f"Dtype: {graph_obs.dtype}")
    print(f"Non-zero elements: {torch.count_nonzero(graph_obs).item()}")
    print(f"Min: {graph_obs.min().item():.6f}, Max: {graph_obs.max().item():.6f}")
    print(f"Mean: {graph_obs.mean().item():.6f}, Std: {graph_obs.std().item():.6f}")

    # Print first few features
    print(f"First 10 features: {graph_obs[:10].tolist()}")

    if graph_data is not None:
        print(f"\nOriginal graph data keys: {list(graph_data.keys())}")
        if "adjacency" in graph_data and graph_data["adjacency"]:
            adj = graph_data["adjacency"]
            if isinstance(adj, dict):
                print(f"  Adjacency nodes: {len(adj)}")
                print(f"  Adjacency edges: {sum(len(n) for n in adj.values()) // 2}")
        if "node_features" in graph_data and graph_data["node_features"] is not None:
            nf = graph_data["node_features"]
            if isinstance(nf, torch.Tensor):
                print(f"  Node features shape: {nf.shape}")
