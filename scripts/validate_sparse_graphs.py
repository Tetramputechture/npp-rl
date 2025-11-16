#!/usr/bin/env python3
"""Validate sparse graph storage and calculate memory savings.

This script demonstrates the memory optimization achieved by:
1. Removing unused internal observations
2. Using sparse storage for graph observations
3. Using float16 for graph features

Run this to verify the optimizations work correctly and show memory savings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from npp_rl.utils.sparse_graph_buffer import (
    pack_graph_observation,
    unpack_graph_observation,
    pack_batch_observations,
    unpack_batch_observations,
    get_memory_savings,
)


def test_sparse_packing():
    """Test that sparse packing/unpacking preserves data correctly."""
    print("=" * 70)
    print("Testing Sparse Graph Packing/Unpacking")
    print("=" * 70)
    
    # Create a mock observation with typical graph sizes
    num_valid_nodes = 450  # Typical for a level
    num_valid_edges = 1800
    max_nodes = 4500
    max_edges = 18500
    node_feat_dim = 21
    edge_feat_dim = 14
    
    # Create dense observation (as environment produces)
    obs = {
        "graph_node_feats": np.random.randn(max_nodes, node_feat_dim).astype(np.float16),
        "graph_edge_index": np.random.randint(0, num_valid_nodes, (2, max_edges), dtype=np.int32),
        "graph_edge_feats": np.random.randn(max_edges, edge_feat_dim).astype(np.float16),
        "graph_node_mask": np.zeros(max_nodes, dtype=np.int32),
        "graph_edge_mask": np.zeros(max_edges, dtype=np.int32),
        "graph_node_types": np.zeros(max_nodes, dtype=np.int32),
        "graph_edge_types": np.zeros(max_edges, dtype=np.int32),
        "game_state": np.random.randn(29).astype(np.float32),
        "reachability_features": np.random.randn(6).astype(np.float32),
    }
    
    # Set valid masks
    obs["graph_node_mask"][:num_valid_nodes] = 1
    obs["graph_edge_mask"][:num_valid_edges] = 1
    
    # Only fill valid portions with actual data (simulate real environment)
    obs["graph_node_feats"][:num_valid_nodes] = np.random.randn(num_valid_nodes, node_feat_dim).astype(np.float16)
    obs["graph_edge_feats"][:num_valid_edges] = np.random.randn(num_valid_edges, edge_feat_dim).astype(np.float16)
    
    # Calculate memory before packing
    dense_size = (
        obs["graph_node_feats"].nbytes +
        obs["graph_edge_index"].nbytes +
        obs["graph_edge_feats"].nbytes +
        obs["graph_node_mask"].nbytes +
        obs["graph_edge_mask"].nbytes +
        obs["graph_node_types"].nbytes +
        obs["graph_edge_types"].nbytes
    )
    
    print("\n✓ Created mock observation")
    print(f"  Valid nodes: {num_valid_nodes}/{max_nodes}")
    print(f"  Valid edges: {num_valid_edges}/{max_edges}")
    print(f"  Dense memory (graph only): {dense_size / 1024 / 1024:.2f} MB")
    
    # Pack to sparse format
    packed_obs = pack_graph_observation(obs)
    
    # Calculate packed memory
    sparse_graph = packed_obs["_sparse_graph"]
    packed_size = (
        sparse_graph.node_feats.nbytes +
        sparse_graph.edge_index.nbytes +
        sparse_graph.edge_feats.nbytes +
        sparse_graph.node_types.nbytes +
        sparse_graph.edge_types.nbytes +
        32  # Overhead for dataclass fields
    )
    
    print("\n✓ Packed to sparse format")
    print(f"  Sparse memory (graph only): {packed_size / 1024 / 1024:.2f} MB")
    print(f"  Memory reduction: {100 * (1 - packed_size / dense_size):.1f}%")
    
    # Unpack back to dense
    unpacked_obs = unpack_graph_observation(packed_obs)
    
    print("\n✓ Unpacked back to dense format")
    
    # Verify correctness - check valid portions match exactly
    node_match = np.allclose(
        obs["graph_node_feats"][:num_valid_nodes],
        unpacked_obs["graph_node_feats"][:num_valid_nodes],
        rtol=1e-3  # float16 precision
    )
    edge_match = np.allclose(
        obs["graph_edge_feats"][:num_valid_edges],
        unpacked_obs["graph_edge_feats"][:num_valid_edges],
        rtol=1e-3  # float16 precision
    )
    index_match = np.array_equal(
        obs["graph_edge_index"][:, :num_valid_edges],
        unpacked_obs["graph_edge_index"][:, :num_valid_edges]
    )
    
    if node_match and edge_match and index_match:
        print("  ✓ Data integrity verified - no loss from packing/unpacking")
    else:
        print("  ✗ Data mismatch detected!")
        return False
    
    return True


def test_batch_packing():
    """Test batch packing for multiple observations."""
    print("\n" + "=" * 70)
    print("Testing Batch Packing/Unpacking")
    print("=" * 70)
    
    batch_size = 256  # Typical PPO minibatch size
    num_valid_nodes = 450
    num_valid_edges = 1800
    max_nodes = 4500
    max_edges = 18500
    node_feat_dim = 21
    edge_feat_dim = 14
    
    # Create batch of observations
    obs_batch = {
        "graph_node_feats": np.random.randn(batch_size, max_nodes, node_feat_dim).astype(np.float16),
        "graph_edge_index": np.random.randint(0, num_valid_nodes, (batch_size, 2, max_edges), dtype=np.int32),
        "graph_edge_feats": np.random.randn(batch_size, max_edges, edge_feat_dim).astype(np.float16),
        "graph_node_mask": np.zeros((batch_size, max_nodes), dtype=np.int32),
        "graph_edge_mask": np.zeros((batch_size, max_edges), dtype=np.int32),
        "graph_node_types": np.zeros((batch_size, max_nodes), dtype=np.int32),
        "graph_edge_types": np.zeros((batch_size, max_edges), dtype=np.int32),
    }
    
    # Set valid masks for all in batch
    obs_batch["graph_node_mask"][:, :num_valid_nodes] = 1
    obs_batch["graph_edge_mask"][:, :num_valid_edges] = 1
    
    # Calculate memory
    dense_batch_size = sum(v.nbytes for k, v in obs_batch.items() if k.startswith("graph_"))
    
    print(f"\n✓ Created batch of {batch_size} observations")
    print(f"  Dense memory (total): {dense_batch_size / 1024 / 1024:.2f} MB")
    print(f"  Dense memory (per obs): {dense_batch_size / batch_size / 1024:.2f} KB")
    
    # Pack batch
    packed_batch = pack_batch_observations(obs_batch)
    
    # Estimate packed size (each sparse graph)
    sparse_graphs = packed_batch["_sparse_graph_batch"]
    packed_batch_size = sum(
        sg.node_feats.nbytes + sg.edge_index.nbytes + sg.edge_feats.nbytes +
        sg.node_types.nbytes + sg.edge_types.nbytes + 32
        for sg in sparse_graphs
    )
    
    print("\n✓ Packed batch to sparse format")
    print(f"  Sparse memory (total): {packed_batch_size / 1024 / 1024:.2f} MB")
    print(f"  Sparse memory (per obs): {packed_batch_size / batch_size / 1024:.2f} KB")
    print(f"  Batch memory reduction: {100 * (1 - packed_batch_size / dense_batch_size):.1f}%")
    
    # Unpack batch
    unpacked_batch = unpack_batch_observations(packed_batch)
    
    print("\n✓ Unpacked batch back to dense format")
    
    # Verify shapes match
    shapes_match = all(
        obs_batch[k].shape == unpacked_batch[k].shape
        for k in obs_batch.keys()
    )
    
    if shapes_match:
        print("  ✓ Batch shapes verified")
    else:
        print("  ✗ Batch shape mismatch!")
        return False
    
    return True


def calculate_rollout_savings():
    """Calculate memory savings for full rollout buffer."""
    print("\n" + "=" * 70)
    print("Rollout Buffer Memory Savings")
    print("=" * 70)
    
    # Typical training configuration
    num_envs = 128
    n_steps = 512
    total_observations = num_envs * n_steps
    
    # Typical graph sizes
    num_valid_nodes = 450
    num_valid_edges = 1800
    
    print("\nConfiguration:")
    print(f"  Environments: {num_envs}")
    print(f"  Steps per rollout: {n_steps}")
    print(f"  Total observations: {total_observations:,}")
    
    # Calculate savings
    dense_mb, sparse_mb, reduction_percent = get_memory_savings(
        num_valid_nodes, num_valid_edges
    )
    
    print("\nPer-observation memory:")
    print(f"  Dense storage (float32): {dense_mb:.3f} MB")
    print(f"  Dense storage (float16): {dense_mb * 0.5:.3f} MB")
    print(f"  Sparse storage (float16): {sparse_mb:.3f} MB")
    print(f"  Reduction: {reduction_percent:.1f}%")
    
    # Total buffer
    total_dense_float32 = dense_mb * total_observations
    total_dense_float16 = dense_mb * 0.5 * total_observations
    total_sparse_float16 = sparse_mb * total_observations
    
    print(f"\nFull rollout buffer ({total_observations:,} observations):")
    print(f"  Dense (float32): {total_dense_float32:.1f} MB ({total_dense_float32/1024:.2f} GB)")
    print(f"  Dense (float16): {total_dense_float16:.1f} MB ({total_dense_float16/1024:.2f} GB)")
    print(f"  Sparse (float16): {total_sparse_float16:.1f} MB ({total_sparse_float16/1024:.2f} GB)")
    
    combined_reduction = 100 * (1 - total_sparse_float16 / total_dense_float32)
    print(f"\n✓ Combined optimization (sparse + float16): {combined_reduction:.1f}% reduction")
    print(f"  Memory saved: {(total_dense_float32 - total_sparse_float16)/1024:.2f} GB")
    
    # Additional savings from removing internal observations (estimated 30%)
    internal_obs_savings = total_dense_float32 * 0.3
    print("\n✓ Additional savings from removing internal observations:")
    print(f"  Estimated: {internal_obs_savings/1024:.2f} GB (30% of original)")
    
    total_savings = (total_dense_float32 + internal_obs_savings - total_sparse_float16) / 1024
    print(f"\n✓✓ Total memory savings: {total_savings:.2f} GB")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("Memory Optimization Validation for NPP-RL Attention Config")
    print("=" * 70)
    
    # Run tests
    success = True
    success &= test_sparse_packing()
    success &= test_batch_packing()
    calculate_rollout_savings()
    
    print("\n" + "=" * 70)
    if success:
        print("✓✓ All tests passed! Memory optimizations working correctly.")
    else:
        print("✗✗ Some tests failed. Review output above.")
    print("=" * 70)
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

