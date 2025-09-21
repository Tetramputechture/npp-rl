#!/usr/bin/env python3
"""
Test script for reachability-aware ICM implementation.

This script validates the basic functionality of the enhanced ICM module
with reachability awareness, ensuring it meets performance requirements.
"""

import time
import numpy as np
import torch
from typing import Dict, Any

# Import the enhanced ICM components
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.intrinsic.utils import (
    extract_reachability_info_from_observations,
    create_reachability_aware_icm_config,
)


def create_mock_observations(batch_size: int = 4) -> Dict[str, Any]:
    """Create mock observations for testing."""
    return {
        "player_x": np.random.uniform(0, 1000, batch_size),
        "player_y": np.random.uniform(0, 600, batch_size),
        "reachability_features": np.random.rand(batch_size, 64),
        "player_frames": np.random.rand(batch_size, 3, 64, 64),
    }


def create_mock_features(batch_size: int = 4, feature_dim: int = 512) -> torch.Tensor:
    """Create mock feature tensors for testing."""
    return torch.randn(batch_size, feature_dim)


def test_basic_icm_functionality():
    """Test basic ICM functionality without reachability awareness."""
    print("Testing basic ICM functionality...")
    
    # Create ICM network
    icm = ICMNetwork(
        feature_dim=512,
        action_dim=6,
        enable_reachability_awareness=False,
    )
    
    # Create mock data
    batch_size = 4
    features_current = create_mock_features(batch_size)
    features_next = create_mock_features(batch_size)
    actions = torch.randint(0, 6, (batch_size,))
    
    # Test forward pass
    output = icm(features_current, features_next, actions)
    
    # Validate output
    assert "intrinsic_reward" in output
    assert output["intrinsic_reward"].shape == (batch_size,)
    assert "inverse_loss" in output
    assert "forward_loss" in output
    
    print("✓ Basic ICM functionality works")


def test_reachability_aware_icm():
    """Test reachability-aware ICM functionality."""
    print("Testing reachability-aware ICM functionality...")
    
    # Create reachability-aware ICM network
    icm = ICMNetwork(
        feature_dim=512,
        action_dim=6,
        enable_reachability_awareness=True,
        reachability_dim=64,
    )
    
    # Create mock data
    batch_size = 4
    features_current = create_mock_features(batch_size)
    features_next = create_mock_features(batch_size)
    actions = torch.randint(0, 6, (batch_size,))
    
    # Create mock reachability info
    reachability_info = {
        "current_positions": [(100.0, 200.0), (300.0, 400.0), (500.0, 100.0), (200.0, 300.0)],
        "target_positions": [(120.0, 220.0), (320.0, 420.0), (520.0, 120.0), (220.0, 320.0)],
        "reachable_positions": [
            {(4, 8), (5, 8), (6, 8), (4, 9), (5, 9)},
            {(12, 16), (13, 16), (14, 16), (12, 17), (13, 17)},
            {(20, 4), (21, 4), (22, 4), (20, 5), (21, 5)},
            {(8, 12), (9, 12), (10, 12), (8, 13), (9, 13)},
        ],
        "door_positions": [(200.0, 300.0), (400.0, 500.0)],
        "switch_positions": [(150.0, 250.0)],
        "exit_position": (800.0, 100.0),
    }
    
    # Test forward pass with reachability info
    output = icm(features_current, features_next, actions, reachability_info)
    
    # Validate output
    assert "intrinsic_reward" in output
    assert output["intrinsic_reward"].shape == (batch_size,)
    assert "reachability_modulation" in output
    assert output["reachability_modulation"].shape == (batch_size,)
    
    # Test that modulation factors are reasonable
    modulation = output["reachability_modulation"]
    assert torch.all(modulation > 0), "Modulation factors should be positive"
    assert torch.all(modulation < 10), "Modulation factors should be reasonable"
    
    print("✓ Reachability-aware ICM functionality works")


def test_reachability_info_extraction():
    """Test reachability information extraction from observations."""
    print("Testing reachability info extraction...")
    
    # Create mock observations
    observations = create_mock_observations(batch_size=2)
    
    # Extract reachability info
    reachability_info = extract_reachability_info_from_observations(observations)
    
    # Validate extraction
    assert reachability_info is not None
    assert "current_positions" in reachability_info
    assert "target_positions" in reachability_info
    assert "reachable_positions" in reachability_info
    assert len(reachability_info["current_positions"]) == 2
    assert len(reachability_info["target_positions"]) == 2
    assert len(reachability_info["reachable_positions"]) == 2
    
    print("✓ Reachability info extraction works")


def test_performance_requirements():
    """Test that performance requirements are met."""
    print("Testing performance requirements...")
    
    # Create ICM network
    icm = ICMNetwork(
        feature_dim=512,
        action_dim=6,
        enable_reachability_awareness=True,
        reachability_dim=64,
    )
    
    # Create mock data
    batch_size = 32  # Larger batch for performance testing
    features_current = create_mock_features(batch_size)
    features_next = create_mock_features(batch_size)
    actions = torch.randint(0, 6, (batch_size,))
    
    # Create mock reachability info
    reachability_info = {
        "current_positions": [(100.0 + i * 10, 200.0 + i * 10) for i in range(batch_size)],
        "target_positions": [(120.0 + i * 10, 220.0 + i * 10) for i in range(batch_size)],
        "reachable_positions": [{(4 + i, 8), (5 + i, 8), (6 + i, 8)} for i in range(batch_size)],
        "door_positions": [(200.0, 300.0)],
        "switch_positions": [(150.0, 250.0)],
        "exit_position": (800.0, 100.0),
    }
    
    # Measure computation time
    num_trials = 100
    times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        
        with torch.no_grad():
            intrinsic_reward = icm.compute_intrinsic_reward(
                features_current, features_next, actions, reachability_info
            )
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Check performance targets
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    
    print(f"Average computation time: {avg_time:.3f}ms")
    print(f"95th percentile time: {p95_time:.3f}ms")
    
    # Performance requirements from task specification
    assert avg_time < 1.0, f"Average time too high: {avg_time:.3f}ms (target: <1ms)"
    assert p95_time < 2.0, f"95th percentile time too high: {p95_time:.3f}ms (target: <2ms)"
    
    print("✓ Performance requirements met")


def test_memory_usage():
    """Test memory usage requirements."""
    print("Testing memory usage...")
    
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create ICM network
    icm = ICMNetwork(
        feature_dim=512,
        action_dim=6,
        enable_reachability_awareness=True,
        reachability_dim=64,
    )
    
    # Run for extended period
    batch_size = 16
    for _ in range(1000):
        features_current = create_mock_features(batch_size)
        features_next = create_mock_features(batch_size)
        actions = torch.randint(0, 6, (batch_size,))
        
        reachability_info = {
            "current_positions": [(100.0, 200.0)] * batch_size,
            "target_positions": [(120.0, 220.0)] * batch_size,
            "reachable_positions": [{(4, 8), (5, 8)}] * batch_size,
            "door_positions": [(200.0, 300.0)],
            "switch_positions": [(150.0, 250.0)],
            "exit_position": (800.0, 100.0),
        }
        
        with torch.no_grad():
            reward = icm.compute_intrinsic_reward(
                features_current, features_next, actions, reachability_info
            )
    
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"Memory increase: {memory_increase:.1f}MB")
    
    # Memory requirement from task specification
    assert memory_increase < 50, f"Memory usage too high: {memory_increase:.1f}MB (target: <50MB)"
    
    print("✓ Memory usage requirements met")


def main():
    """Run all tests."""
    print("Running reachability-aware ICM tests...\n")
    
    try:
        test_basic_icm_functionality()
        test_reachability_aware_icm()
        test_reachability_info_extraction()
        test_performance_requirements()
        test_memory_usage()
        
        print("\n✅ All tests passed! Reachability-aware ICM implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()