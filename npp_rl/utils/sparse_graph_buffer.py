"""Sparse graph observation storage for memory-efficient rollout buffers.

This module provides utilities to compress graph observations by storing only
valid nodes and edges instead of full padded arrays. This reduces memory usage
by 80-90% for typical levels.

Memory Comparison (typical level with ~400 nodes, ~1600 edges):
- Dense storage: 4500 nodes + 18500 edges = ~1.7 MB per observation
- Sparse storage: 400 nodes + 1600 edges = ~0.15 MB per observation
- Reduction: ~91% memory saved

The compression is lossless - no information is lost during packing/unpacking.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import torch


@dataclass
class SparseGraphObservation:
    """Sparse representation of graph observations.

    Instead of storing full padded arrays (4500 nodes, 18500 edges),
    stores only the valid nodes/edges based on masks.

    Attributes:
        node_feats: [num_valid_nodes, NODE_FEATURE_DIM] - only valid nodes
        edge_index: [2, num_valid_edges] - only valid edges
        edge_feats: [num_valid_edges, EDGE_FEATURE_DIM] - only valid edges
        node_types: [num_valid_nodes] - node types for valid nodes
        edge_types: [num_valid_edges] - edge types for valid edges
        num_nodes: int - number of valid nodes
        num_edges: int - number of valid edges
        max_nodes: int - original max size for unpacking
        max_edges: int - original max size for unpacking
    """

    node_feats: np.ndarray
    edge_index: np.ndarray
    edge_feats: np.ndarray
    node_types: np.ndarray
    edge_types: np.ndarray
    num_nodes: int
    num_edges: int
    max_nodes: int
    max_edges: int


def pack_graph_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Pack graph observations to sparse format to save memory.

    This function extracts only the valid nodes and edges based on masks,
    significantly reducing memory usage in the rollout buffer.

    Handles both single observations and batched observations from vectorized environments.

    Args:
        obs: Observation dictionary containing graph observations with keys:
            - graph_node_feats: [max_nodes, node_feat_dim] or [batch, max_nodes, node_feat_dim]
            - graph_edge_index: [2, max_edges] or [batch, 2, max_edges]
            - graph_edge_feats: [max_edges, edge_feat_dim] or [batch, max_edges, edge_feat_dim]
            - graph_node_mask: [max_nodes] or [batch, max_nodes] - 1 for valid, 0 for padding
            - graph_edge_mask: [max_edges] or [batch, max_edges] - 1 for valid, 0 for padding
            - graph_node_types: [max_nodes] or [batch, max_nodes]
            - graph_edge_types: [max_edges] or [batch, max_edges]

    Returns:
        Modified observation dictionary with sparse graph stored in '_sparse_graph'
        and dense graph keys removed.
    """
    # Check if graph observations exist
    if "graph_node_feats" not in obs:
        return obs

    # Check if this is a batched observation (from vectorized env)
    node_feats = obs["graph_node_feats"]
    if isinstance(node_feats, torch.Tensor):
        is_batched = node_feats.ndim == 3
    else:
        is_batched = node_feats.ndim == 3

    # If batched, pack each observation separately
    if is_batched:
        batch_size = node_feats.shape[0]
        sparse_graphs = []

        for i in range(batch_size):
            # Extract single observation from batch
            single_obs = {
                key: value[i] for key, value in obs.items() if key.startswith("graph_")
            }
            # Pack it (recursive call with single obs)
            packed = pack_graph_observation(single_obs)
            sparse_graphs.append(packed["_sparse_graph"])

        # Create new observation dict without graph data
        packed_obs = {k: v for k, v in obs.items() if not k.startswith("graph_")}
        packed_obs["_sparse_graph"] = np.array(sparse_graphs, dtype=object)

        return packed_obs

    # Single observation case
    # Extract masks to determine valid nodes/edges
    node_mask = obs["graph_node_mask"]
    edge_mask = obs["graph_edge_mask"]

    # Handle both numpy arrays and torch tensors
    if isinstance(node_mask, torch.Tensor):
        node_mask = node_mask.cpu().numpy()
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.cpu().numpy()

    # Convert to boolean masks
    node_mask_bool = node_mask.astype(bool)
    edge_mask_bool = edge_mask.astype(bool)

    # Count valid elements
    num_valid_nodes = np.sum(node_mask_bool)
    num_valid_edges = np.sum(edge_mask_bool)

    # Get original dimensions
    max_nodes = obs["graph_node_feats"].shape[0]
    max_edges = obs["graph_edge_feats"].shape[0]

    # Extract only valid elements (slice arrays)
    # Convert to numpy if needed
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    sparse_graph = SparseGraphObservation(
        node_feats=to_numpy(obs["graph_node_feats"])[node_mask_bool].copy(),
        edge_index=to_numpy(obs["graph_edge_index"])[:, edge_mask_bool].copy(),
        edge_feats=to_numpy(obs["graph_edge_feats"])[edge_mask_bool].copy(),
        node_types=to_numpy(obs["graph_node_types"])[node_mask_bool].copy(),
        edge_types=to_numpy(obs["graph_edge_types"])[edge_mask_bool].copy(),
        num_nodes=int(num_valid_nodes),
        num_edges=int(num_valid_edges),
        max_nodes=int(max_nodes),
        max_edges=int(max_edges),
    )

    # Create new observation dict without dense graph data
    packed_obs = {k: v for k, v in obs.items() if not k.startswith("graph_")}
    packed_obs["_sparse_graph"] = sparse_graph

    return packed_obs


def unpack_graph_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Unpack sparse graph observations back to dense format for training.

    This reconstructs the full padded arrays from sparse storage, allowing
    the policy network to process them normally.

    Args:
        obs: Observation dictionary containing '_sparse_graph' key

    Returns:
        Modified observation dictionary with dense graph observations restored
    """
    # Check if sparse graph exists
    if "_sparse_graph" not in obs:
        return obs

    sparse_graph = obs["_sparse_graph"]

    # Reconstruct full padded arrays
    node_feats = np.zeros(
        (sparse_graph.max_nodes, sparse_graph.node_feats.shape[1]),
        dtype=sparse_graph.node_feats.dtype,
    )
    edge_index = np.zeros(
        (2, sparse_graph.max_edges), dtype=sparse_graph.edge_index.dtype
    )
    edge_feats = np.zeros(
        (sparse_graph.max_edges, sparse_graph.edge_feats.shape[1]),
        dtype=sparse_graph.edge_feats.dtype,
    )
    node_mask = np.zeros(sparse_graph.max_nodes, dtype=np.int32)
    edge_mask = np.zeros(sparse_graph.max_edges, dtype=np.int32)
    node_types = np.zeros(sparse_graph.max_nodes, dtype=sparse_graph.node_types.dtype)
    edge_types = np.zeros(sparse_graph.max_edges, dtype=sparse_graph.edge_types.dtype)

    # Fill in valid elements
    num_nodes = sparse_graph.num_nodes
    num_edges = sparse_graph.num_edges

    node_feats[:num_nodes] = sparse_graph.node_feats
    edge_index[:, :num_edges] = sparse_graph.edge_index
    edge_feats[:num_edges] = sparse_graph.edge_feats
    node_types[:num_nodes] = sparse_graph.node_types
    edge_types[:num_edges] = sparse_graph.edge_types
    node_mask[:num_nodes] = 1
    edge_mask[:num_edges] = 1

    # Create new observation dict with dense graph data
    unpacked_obs = {k: v for k, v in obs.items() if k != "_sparse_graph"}
    unpacked_obs.update(
        {
            "graph_node_feats": node_feats,
            "graph_edge_index": edge_index,
            "graph_edge_feats": edge_feats,
            "graph_node_mask": node_mask,
            "graph_edge_mask": edge_mask,
            "graph_node_types": node_types,
            "graph_edge_types": edge_types,
        }
    )

    return unpacked_obs


def pack_batch_observations(obs_batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Pack a batch of observations for efficient storage.

    Args:
        obs_batch: Dictionary of batched observations where each value has
                  shape [batch_size, ...observation_dims...]

    Returns:
        Modified batch with sparse graph storage
    """
    # Check if this is a dictionary observation with graphs
    if not isinstance(obs_batch, dict) or "graph_node_feats" not in obs_batch:
        return obs_batch

    batch_size = obs_batch["graph_node_feats"].shape[0]

    # Pack each observation in the batch
    sparse_graphs = []
    for i in range(batch_size):
        # Extract single observation
        single_obs = {
            key: value[i]
            for key, value in obs_batch.items()
            if key.startswith("graph_")
        }
        # Pack it
        packed = pack_graph_observation(single_obs)
        sparse_graphs.append(packed["_sparse_graph"])

    # Create new batch without graph observations
    packed_batch = {k: v for k, v in obs_batch.items() if not k.startswith("graph_")}
    packed_batch["_sparse_graph_batch"] = sparse_graphs

    return packed_batch


def unpack_batch_observations(obs_batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Unpack a batch of sparse observations back to dense format.

    Args:
        obs_batch: Dictionary containing '_sparse_graph_batch' key

    Returns:
        Modified batch with dense graph observations
    """
    # Check if sparse batch exists
    if "_sparse_graph_batch" not in obs_batch:
        return obs_batch

    sparse_graphs = obs_batch["_sparse_graph_batch"]
    batch_size = len(sparse_graphs)

    # Get dimensions from first sparse graph
    first_graph = sparse_graphs[0]
    max_nodes = first_graph.max_nodes
    max_edges = first_graph.max_edges
    node_feat_dim = first_graph.node_feats.shape[1]
    edge_feat_dim = first_graph.edge_feats.shape[1]

    # Pre-allocate dense arrays for entire batch
    node_feats_batch = np.zeros(
        (batch_size, max_nodes, node_feat_dim), dtype=np.float32
    )
    edge_index_batch = np.zeros((batch_size, 2, max_edges), dtype=np.int32)
    edge_feats_batch = np.zeros(
        (batch_size, max_edges, edge_feat_dim), dtype=np.float32
    )
    node_mask_batch = np.zeros((batch_size, max_nodes), dtype=np.int32)
    edge_mask_batch = np.zeros((batch_size, max_edges), dtype=np.int32)
    node_types_batch = np.zeros((batch_size, max_nodes), dtype=np.int32)
    edge_types_batch = np.zeros((batch_size, max_edges), dtype=np.int32)

    # Fill in each observation
    for i, sparse_graph in enumerate(sparse_graphs):
        num_nodes = sparse_graph.num_nodes
        num_edges = sparse_graph.num_edges

        node_feats_batch[i, :num_nodes] = sparse_graph.node_feats
        edge_index_batch[i, :, :num_edges] = sparse_graph.edge_index
        edge_feats_batch[i, :num_edges] = sparse_graph.edge_feats
        node_types_batch[i, :num_nodes] = sparse_graph.node_types
        edge_types_batch[i, :num_edges] = sparse_graph.edge_types
        node_mask_batch[i, :num_nodes] = 1
        edge_mask_batch[i, :num_edges] = 1

    # Create new batch with dense graph data
    unpacked_batch = {k: v for k, v in obs_batch.items() if k != "_sparse_graph_batch"}
    unpacked_batch.update(
        {
            "graph_node_feats": node_feats_batch,
            "graph_edge_index": edge_index_batch,
            "graph_edge_feats": edge_feats_batch,
            "graph_node_mask": node_mask_batch,
            "graph_edge_mask": edge_mask_batch,
            "graph_node_types": node_types_batch,
            "graph_edge_types": edge_types_batch,
        }
    )

    return unpacked_batch


def get_memory_savings(
    num_valid_nodes: int,
    num_valid_edges: int,
    max_nodes: int = 4500,
    max_edges: int = 18500,
    node_feat_dim: int = 21,
    edge_feat_dim: int = 14,
) -> Tuple[float, float, float]:
    """Calculate memory savings from sparse storage.

    Args:
        num_valid_nodes: Number of actual valid nodes in graph
        num_valid_edges: Number of actual valid edges in graph
        max_nodes: Maximum node capacity (default: 4500)
        max_edges: Maximum edge capacity (default: 18500)
        node_feat_dim: Node feature dimension (default: 21)
        edge_feat_dim: Edge feature dimension (default: 14)

    Returns:
        Tuple of (dense_mb, sparse_mb, reduction_percent)
    """
    # Calculate dense storage (in MB)
    dense_bytes = (
        max_nodes * node_feat_dim * 4  # node_feats (float32)
        + 2 * max_edges * 4  # edge_index (int32)
        + max_edges * edge_feat_dim * 4  # edge_feats (float32)
        + max_nodes * 4  # node_mask (int32)
        + max_edges * 4  # edge_mask (int32)
        + max_nodes * 4  # node_types (int32)
        + max_edges * 4  # edge_types (int32)
    )
    dense_mb = dense_bytes / (1024**2)

    # Calculate sparse storage (in MB)
    sparse_bytes = (
        num_valid_nodes * node_feat_dim * 4  # node_feats
        + 2 * num_valid_edges * 4  # edge_index
        + num_valid_edges * edge_feat_dim * 4  # edge_feats
        + num_valid_nodes * 4  # node_types
        + num_valid_edges * 4  # edge_types
        + 8 * 4  # metadata (8 ints)
    )
    sparse_mb = sparse_bytes / (1024**2)

    # Calculate reduction
    reduction_percent = 100 * (1 - sparse_mb / dense_mb)

    return dense_mb, sparse_mb, reduction_percent
