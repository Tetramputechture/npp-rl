#!/usr/bin/env python3
"""
Validate that all architectures can perform forward passes successfully.

This script tests each architecture's feature extractor with realistic
graph observations to ensure they work correctly before training.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.architecture_configs import list_available_architectures, get_architecture_config
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from gymnasium.spaces import Dict as SpacesDict, Box
import numpy as np


def create_mock_observation_space():
    """Create mock observation space matching N++ environment."""
    return SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(176, 100, 1), dtype=np.uint8),
        'game_state': Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32),
        'reachability_features': Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        'graph_node_feats': Box(low=-np.inf, high=np.inf, shape=(15856, 55), dtype=np.float32),
        'graph_edge_index': Box(low=0, high=15856, shape=(2, 126848), dtype=np.int64),
        'graph_edge_feats': Box(low=-np.inf, high=np.inf, shape=(126848, 6), dtype=np.float32),
        'graph_node_mask': Box(low=0, high=1, shape=(15856,), dtype=np.float32),
        'graph_edge_mask': Box(low=0, high=1, shape=(126848,), dtype=np.float32),
        'graph_node_types': Box(low=0, high=6, shape=(15856,), dtype=np.int64),
        'graph_edge_types': Box(low=0, high=4, shape=(126848,), dtype=np.int64),
    })


def create_mock_observation(batch_size=1, num_actual_edges=1000):
    """Create mock observation tensors with realistic graph data."""
    obs = {
        'player_frame': torch.randint(0, 256, (batch_size, 84, 84, 1), dtype=torch.uint8),
        'global_view': torch.randint(0, 256, (batch_size, 176, 100, 1), dtype=torch.uint8),
        'game_state': torch.randn(batch_size, 30),
        'reachability_features': torch.randn(batch_size, 8),
        'graph_node_feats': torch.randn(batch_size, 15856, 55),
        'graph_edge_index': torch.randint(0, 15856, (batch_size, 2, 126848)),
        'graph_edge_feats': torch.randn(batch_size, 126848, 6),
        'graph_node_mask': torch.zeros(batch_size, 15856),
        'graph_edge_mask': torch.zeros(batch_size, 126848),
        'graph_node_types': torch.randint(0, 6, (batch_size, 15856)),
        'graph_edge_types': torch.randint(0, 4, (batch_size, 126848)),
    }
    
    # Set realistic masks - only first num_actual_edges are valid
    obs['graph_edge_mask'][:, :num_actual_edges] = 1.0
    # Estimate nodes from edges (rough approximation)
    num_actual_nodes = min(15856, num_actual_edges // 4)
    obs['graph_node_mask'][:, :num_actual_nodes] = 1.0
    
    return obs


def validate_architecture(arch_name, num_actual_edges=1000, batch_size=2):
    """Validate a single architecture."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {arch_name}")
    print(f"{'=' * 60}")
    
    try:
        # Get architecture config
        arch_config = get_architecture_config(arch_name)
        print(f"Description: {arch_config.description}")
        print(f"Modalities: {', '.join(arch_config.modalities.get_enabled_modalities())}")
        print(f"Features dim: {arch_config.features_dim}")
        
        # Create observation space
        obs_space = create_mock_observation_space()
        
        # Create feature extractor
        print("\nCreating feature extractor...")
        extractor = ConfigurableMultimodalExtractor(obs_space, arch_config)
        print("✓ Feature extractor created")
        
        # Create mock observations
        print(f"\nCreating mock observations (batch_size={batch_size}, edges={num_actual_edges})...")
        obs = create_mock_observation(batch_size, num_actual_edges)
        
        # Test forward pass
        print("Testing forward pass...")
        start_time = time.time()
        with torch.no_grad():
            features = extractor(obs)
        elapsed = time.time() - start_time
        
        print(f"✓ Forward pass completed in {elapsed:.3f}s")
        print(f"  Output shape: {features.shape}")
        print(f"  Expected shape: ({batch_size}, {arch_config.features_dim})")
        
        # Validate output shape
        expected_shape = (batch_size, arch_config.features_dim)
        if features.shape != expected_shape:
            raise ValueError(f"Shape mismatch: got {features.shape}, expected {expected_shape}")
        
        # Validate no NaN/Inf values
        if torch.isnan(features).any():
            raise ValueError("Output contains NaN values")
        if torch.isinf(features).any():
            raise ValueError("Output contains Inf values")
        
        print("✓ Output validation passed")
        print(f"\n{'=' * 60}")
        print(f"✓ {arch_name} PASSED")
        print(f"{'=' * 60}")
        
        return True, elapsed
        
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"✗ {arch_name} FAILED: {e}")
        print(f"{'=' * 60}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Validate all architectures."""
    print("=" * 60)
    print("NPP-RL Architecture Validation")
    print("=" * 60)
    print("\nThis script validates that all architectures can perform")
    print("forward passes with realistic N++ graph observations.")
    print()
    
    # Get all architectures
    architectures = list_available_architectures()
    print(f"Found {len(architectures)} architectures to validate")
    print()
    
    # Test with different edge counts
    edge_counts = [100, 1000]  # Start small, then realistic
    
    results = {}
    
    for num_edges in edge_counts:
        print(f"\n{'#' * 60}")
        print(f"# Testing with {num_edges} edges")
        print(f"{'#' * 60}")
        
        for arch_name in architectures:
            passed, elapsed = validate_architecture(arch_name, num_edges, batch_size=2)
            if arch_name not in results:
                results[arch_name] = []
            results[arch_name].append((num_edges, passed, elapsed))
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for arch in results if all(r[1] for r in results[arch]))
    failed_count = len(results) - passed_count
    
    print(f"\nTotal architectures: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print()
    
    # Detailed results
    print("Detailed Results:")
    print("-" * 60)
    for arch_name, arch_results in results.items():
        all_passed = all(r[1] for r in arch_results)
        status = "✓ PASS" if all_passed else "✗ FAIL"
        print(f"{status:8} {arch_name:25}", end="")
        
        if all_passed:
            # Show timing for largest edge count
            largest_test = max(arch_results, key=lambda x: x[0])
            print(f"  {largest_test[2]:.3f}s ({largest_test[0]} edges)")
        else:
            print()
    
    print("=" * 60)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
