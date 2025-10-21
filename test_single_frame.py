#!/usr/bin/env python3
"""
Test script to validate single-frame grayscale support in npp-rl.

This script tests that the ConfigurableMultimodalExtractor can correctly
handle the new single-frame observation format (84x84x1) instead of the
old 12-frame temporal stacking (84x84x12).
"""

import torch
from npp_rl.feature_extractors.configurable_extractor import (
    ConfigurableMultimodalExtractor,
    ArchitectureConfig,
    ModalityConfig,
    VisualConfig,
    GraphConfig,
    FusionConfig,
    GraphArchitectureType,
    FusionType,
)
from gymnasium import spaces


def test_single_frame_architecture():
    """Test that architecture works with single-frame observations."""
    print("=" * 70)
    print("Testing Single-Frame Grayscale Architecture")
    print("=" * 70)

    # Create mock observation space matching new single-frame format
    observation_space = spaces.Dict(
        {
            "player_frame": spaces.Box(0, 255, shape=(84, 84, 1), dtype="uint8"),
            "global_view": spaces.Box(0, 255, shape=(176, 100, 1), dtype="uint8"),
            "game_state": spaces.Box(
                -float("inf"), float("inf"), shape=(30,), dtype="float32"
            ),
            "reachability_features": spaces.Box(
                -float("inf"), float("inf"), shape=(8,), dtype="float32"
            ),
        }
    )

    # Create architecture config
    config = ArchitectureConfig(
        name="test_single_frame",
        description="Test config for single frame",
        detailed_description="Test config for single frame",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=False,
            use_game_state=True,
            use_reachability=True,
        ),
        graph=GraphConfig(architecture=GraphArchitectureType.NONE),
        visual=VisualConfig(),
        state=None,
        fusion=FusionConfig(fusion_type=FusionType.CONCAT),
        features_dim=512,
    )

    # Instantiate extractor
    print("\n1. Creating feature extractor...")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)
    print("   ✓ Feature extractor created successfully")
    print(f"   - Features dim: {extractor.features_dim}")
    print(f"   - Has player frame CNN: {extractor.temporal_cnn is not None}")
    print(f"   - Has global view CNN: {extractor.global_cnn is not None}")

    # Test forward pass with different input formats
    print("\n2. Testing forward pass with different input formats...")
    batch_size = 4

    # Format 1: [batch, H, W, 1] (what nclone provides)
    print("\n   Testing format 1: [batch, H, W, 1]")
    obs1 = {
        "player_frame": torch.randint(0, 256, (batch_size, 84, 84, 1), dtype=torch.uint8),
        "global_view": torch.randint(0, 256, (batch_size, 176, 100, 1), dtype=torch.uint8),
        "game_state": torch.randn(batch_size, 30),
        "reachability_features": torch.randn(batch_size, 8),
    }
    features1 = extractor(obs1)
    print(f"   ✓ Format 1 successful")
    print(f"     Input: {obs1['player_frame'].shape}")
    print(f"     Output: {features1.shape}")

    # Format 2: [batch, 1, H, W] (PyTorch standard)
    print("\n   Testing format 2: [batch, 1, H, W]")
    obs2 = {
        "player_frame": torch.randint(0, 256, (batch_size, 1, 84, 84), dtype=torch.uint8),
        "global_view": torch.randint(0, 256, (batch_size, 1, 176, 100), dtype=torch.uint8),
        "game_state": torch.randn(batch_size, 30),
        "reachability_features": torch.randn(batch_size, 8),
    }
    features2 = extractor(obs2)
    print(f"   ✓ Format 2 successful")
    print(f"     Input: {obs2['player_frame'].shape}")
    print(f"     Output: {features2.shape}")

    # Format 3: [batch, H, W] (squeezed)
    print("\n   Testing format 3: [batch, H, W]")
    obs3 = {
        "player_frame": torch.randint(0, 256, (batch_size, 84, 84), dtype=torch.uint8),
        "global_view": torch.randint(0, 256, (batch_size, 176, 100), dtype=torch.uint8),
        "game_state": torch.randn(batch_size, 30),
        "reachability_features": torch.randn(batch_size, 8),
    }
    features3 = extractor(obs3)
    print(f"   ✓ Format 3 successful")
    print(f"     Input: {obs3['player_frame'].shape}")
    print(f"     Output: {features3.shape}")

    # Validate shapes
    print("\n3. Validating output shapes...")
    expected_shape = (batch_size, config.features_dim)
    assert features1.shape == expected_shape, f"Shape mismatch: {features1.shape} vs {expected_shape}"
    assert features2.shape == expected_shape, f"Shape mismatch: {features2.shape} vs {expected_shape}"
    assert features3.shape == expected_shape, f"Shape mismatch: {features3.shape} vs {expected_shape}"
    print(f"   ✓ All outputs match expected shape: {expected_shape}")

    # Count parameters
    print("\n4. Model statistics...")
    num_params = sum(p.numel() for p in extractor.parameters())
    print(f"   - Total parameters: {num_params:,}")
    print(f"   - Model size: ~{num_params * 4 / 1024 / 1024:.1f} MB (float32)")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print("- Single-frame architecture instantiates correctly")
    print("- Forward pass works with multiple input formats")
    print("- Output shapes are correct")
    print("- Model is ready for training with nclone's grayscale observations")
    print("\nNext steps:")
    print("1. Install/update nclone with grayscale support")
    print("2. Run training with updated architectures")
    print("3. Compare performance against old 12-frame system")


if __name__ == "__main__":
    test_single_frame_architecture()
