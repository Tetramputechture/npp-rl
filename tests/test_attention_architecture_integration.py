"""Integration tests for full attention architecture."""

import torch
import pytest
import gymnasium as gym

# Use normal imports - they work fine
from npp_rl.training.architecture_configs import get_architecture_config
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor


def create_mock_observation_space():
    """Create mock observation space for testing."""
    return gym.spaces.Dict(
        {
            "player_frame": gym.spaces.Box(0, 255, (84, 84, 1), dtype="uint8"),
            "global_view": gym.spaces.Box(0, 255, (176, 100, 1), dtype="uint8"),
            "graph_node_feats": gym.spaces.Box(-10, 10, (1000, 55), dtype="float32"),
            "graph_edge_index": gym.spaces.Box(0, 1000, (2, 5000), dtype="int32"),
            "graph_node_mask": gym.spaces.Box(0, 1, (1000,), dtype="float32"),
            "graph_edge_mask": gym.spaces.Box(0, 1, (5000,), dtype="float32"),
            "game_state": gym.spaces.Box(-10, 10, (58,), dtype="float32"),
            "reachability_features": gym.spaces.Box(0, 1, (8,), dtype="float32"),
        }
    )


def create_mock_observation(batch_size=4):
    """Create mock observation batch."""
    return {
        "player_frame": torch.randint(
            0, 255, (batch_size, 84, 84, 1), dtype=torch.uint8
        ),
        "global_view": torch.randint(
            0, 255, (batch_size, 176, 100, 1), dtype=torch.uint8
        ),
        "graph_node_feats": torch.randn(batch_size, 1000, 55),
        "graph_edge_index": torch.randint(0, 1000, (batch_size, 2, 500)),
        "graph_node_mask": torch.ones(batch_size, 1000),
        "graph_edge_mask": torch.ones(batch_size, 500),
        "game_state": torch.randn(batch_size, 58),
        "reachability_features": torch.randn(batch_size, 8),
    }


def test_attention_architecture_end_to_end():
    """Test full forward pass through attention architecture."""
    config = get_architecture_config("attention")
    obs_space = create_mock_observation_space()

    extractor = ConfigurableMultimodalExtractor(
        observation_space=obs_space,
        config=config,
        frame_stack_config={
            "enable_visual_frame_stacking": False,
            "enable_state_stacking": False,
        },
    )

    batch_size = 4
    obs = create_mock_observation(batch_size)

    # Forward pass
    features = extractor(obs)

    # Check output shape
    assert features.shape == (batch_size, config.features_dim), (
        f"Expected shape ({batch_size}, {config.features_dim}), got {features.shape}"
    )


def test_attention_architecture_with_frame_stacking():
    """Test with frame stacking enabled."""
    config = get_architecture_config("attention")
    obs_space = create_mock_observation_space()

    frame_stack_config = {
        "enable_visual_frame_stacking": True,
        "visual_stack_size": 4,
        "enable_state_stacking": True,
        "state_stack_size": 4,
    }

    extractor = ConfigurableMultimodalExtractor(
        observation_space=obs_space,
        config=config,
        frame_stack_config=frame_stack_config,
    )

    batch_size = 4
    obs = {
        "player_frame": torch.randint(
            0, 255, (batch_size, 4, 84, 84, 1), dtype=torch.uint8
        ),  # Stacked
        "global_view": torch.randint(
            0, 255, (batch_size, 176, 100, 1), dtype=torch.uint8
        ),  # Not stacked
        "graph_node_feats": torch.randn(batch_size, 1000, 55),
        "graph_edge_index": torch.randint(0, 1000, (batch_size, 2, 500)),
        "graph_node_mask": torch.ones(batch_size, 1000),
        "graph_edge_mask": torch.ones(batch_size, 500),
        "game_state": torch.randn(batch_size, 4, 58),  # Stacked
        "reachability_features": torch.randn(batch_size, 8),
    }

    features = extractor(obs)
    assert features.shape == (batch_size, config.features_dim)


def test_attention_architecture_gradient_flow():
    """Test gradients flow through all components."""
    config = get_architecture_config("attention")
    obs_space = create_mock_observation_space()

    extractor = ConfigurableMultimodalExtractor(
        observation_space=obs_space,
        config=config,
    )

    batch_size = 2
    obs = create_mock_observation(batch_size)

    features = extractor(obs)
    loss = features.sum()
    loss.backward()

    # Check major components have gradients
    has_grad_count = 0
    for name, param in extractor.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad_count += 1

    assert has_grad_count > 0, "No parameters received gradients"
    print(f"✓ {has_grad_count} parameters received gradients")


def test_attention_architecture_memory_usage():
    """Profile memory usage."""
    config = get_architecture_config("attention")
    obs_space = create_mock_observation_space()

    extractor = ConfigurableMultimodalExtractor(
        observation_space=obs_space,
        config=config,
    )

    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Should be reasonable (< 10M parameters)
    assert total_params < 10_000_000, f"Too many parameters: {total_params:,}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
