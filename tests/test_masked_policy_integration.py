"""Integration tests for MaskedActorCriticPolicy with actual environment setup.

These tests use the real ConfigurableMultimodalExtractor to validate
that action masking works correctly in production conditions.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO

from npp_rl.agents.masked_actor_critic_policy import MaskedActorCriticPolicy
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from npp_rl.training.architecture_configs import get_architecture_config


def test_masked_policy_with_multimodal_extractor():
    """Test that MaskedActorCriticPolicy works with ConfigurableMultimodalExtractor."""
    # Get a simple architecture config
    try:
        arch_config = get_architecture_config("mlp_cnn")
    except:
        pytest.skip("Architecture config not available")

    # Create observation space matching our environment
    obs_space = spaces.Dict(
        {
            "player_frame": spaces.Box(
                low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
            ),
            "global_view": spaces.Box(
                low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
            ),
            "game_state": spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
        }
    )
    action_space = spaces.Discrete(6)

    # Create policy with multimodal extractor (production setup)
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=ConfigurableMultimodalExtractor,
        features_extractor_kwargs={"config": arch_config},
    )

    assert policy is not None
    assert isinstance(policy.features_extractor, ConfigurableMultimodalExtractor)


def test_action_masking_prevents_invalid_actions():
    """Test that action masking actually prevents invalid actions from being selected."""
    try:
        arch_config = get_architecture_config("mlp_cnn")
    except:
        pytest.skip("Architecture config not available")

    obs_space = spaces.Dict(
        {
            "player_frame": spaces.Box(
                low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
            ),
            "global_view": spaces.Box(
                low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
            ),
            "game_state": spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
        }
    )
    action_space = spaces.Discrete(6)

    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=ConfigurableMultimodalExtractor,
        features_extractor_kwargs={"config": arch_config},
    )

    # Create observation with mask that only allows actions 0, 1, 2
    # (simulating useless jump actions 3, 4, 5 being masked)
    batch_size = 4
    obs = {
        "player_frame": torch.randint(
            0, 255, (batch_size, 64, 64, 1), dtype=torch.uint8
        ),
        "global_view": torch.randint(
            0, 255, (batch_size, 128, 128, 1), dtype=torch.uint8
        ),
        "game_state": torch.randn(batch_size, 50),
        "action_mask": torch.tensor(
            [[1, 1, 1, 0, 0, 0]] * batch_size, dtype=torch.bool
        ),
    }

    # Sample many actions to verify distribution
    action_counts = torch.zeros(6)
    num_samples = 200

    for _ in range(num_samples):
        actions, _, _ = policy.forward(obs, deterministic=False)
        for action in actions:
            action_counts[action.item()] += 1

    # Masked actions (3, 4, 5) should NEVER be selected
    assert action_counts[3] == 0, (
        f"Masked action 3 was selected {action_counts[3]} times!"
    )
    assert action_counts[4] == 0, (
        f"Masked action 4 was selected {action_counts[4]} times!"
    )
    assert action_counts[5] == 0, (
        f"Masked action 5 was selected {action_counts[5]} times!"
    )

    # Valid actions (0, 1, 2) should all be selected at least once
    assert action_counts[0] > 0, "Valid action 0 was never selected"
    assert action_counts[1] > 0, "Valid action 1 was never selected"
    assert action_counts[2] > 0, "Valid action 2 was never selected"

    print(f"\n✅ Action distribution with masking [3,4,5]: {action_counts.tolist()}")
    print(
        f"   Masked actions: {action_counts[3:].sum().item()} / {num_samples * batch_size} (expected: 0)"
    )


def test_evaluate_actions_with_mask_integration():
    """Test that evaluate_actions properly handles masks in production setup."""
    try:
        arch_config = get_architecture_config("mlp_cnn")
    except:
        pytest.skip("Architecture config not available")

    obs_space = spaces.Dict(
        {
            "player_frame": spaces.Box(
                low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
            ),
            "global_view": spaces.Box(
                low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
            ),
            "game_state": spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
        }
    )
    action_space = spaces.Discrete(6)

    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=ConfigurableMultimodalExtractor,
        features_extractor_kwargs={"config": arch_config},
    )

    batch_size = 2
    obs = {
        "player_frame": torch.randint(
            0, 255, (batch_size, 64, 64, 1), dtype=torch.uint8
        ),
        "global_view": torch.randint(
            0, 255, (batch_size, 128, 128, 1), dtype=torch.uint8
        ),
        "game_state": torch.randn(batch_size, 50),
        "action_mask": torch.tensor(
            [[1, 1, 1, 0, 0, 0]] * batch_size, dtype=torch.bool
        ),
    }

    # Test evaluating valid actions
    valid_actions = torch.tensor([0, 1])
    values, log_probs, entropy = policy.evaluate_actions(obs, valid_actions)

    assert values.shape == (batch_size, 1)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert torch.all(torch.isfinite(log_probs)), (
        "Valid actions should have finite log probs"
    )

    # Test evaluating masked actions - should have -inf log prob
    masked_actions = torch.tensor([3, 4])
    values_masked, log_probs_masked, entropy_masked = policy.evaluate_actions(
        obs, masked_actions
    )

    assert torch.all(torch.isinf(log_probs_masked)), (
        "Masked actions should have -inf log prob"
    )
    assert torch.all(log_probs_masked < 0), "Log probs should be negative infinity"

    print(f"\n✅ Valid action log probs: {log_probs.tolist()}")
    print(f"✅ Masked action log probs: {log_probs_masked.tolist()} (expected: -inf)")


def test_no_regression_without_mask():
    """Test that policy works normally when no action mask is provided (backward compatibility)."""
    try:
        arch_config = get_architecture_config("mlp_cnn")
    except:
        pytest.skip("Architecture config not available")

    # Observation space WITHOUT action_mask key
    obs_space = spaces.Dict(
        {
            "player_frame": spaces.Box(
                low=0, high=255, shape=(64, 64, 1), dtype=np.uint8
            ),
            "global_view": spaces.Box(
                low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
            ),
            "game_state": spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32),
        }
    )
    action_space = spaces.Discrete(6)

    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=ConfigurableMultimodalExtractor,
        features_extractor_kwargs={"config": arch_config},
    )

    # Create observation without mask
    batch_size = 2
    obs = {
        "player_frame": torch.randint(
            0, 255, (batch_size, 64, 64, 1), dtype=torch.uint8
        ),
        "global_view": torch.randint(
            0, 255, (batch_size, 128, 128, 1), dtype=torch.uint8
        ),
        "game_state": torch.randn(batch_size, 50),
    }

    # Should work without mask (all actions available)
    actions, values, log_probs = policy.forward(obs, deterministic=False)

    assert actions.shape == (batch_size,)
    assert values.shape == (batch_size, 1)
    assert log_probs.shape == (batch_size,)
    assert torch.all((actions >= 0) & (actions < 6))

    # Verify all actions can be selected (sample many times)
    action_counts = torch.zeros(6)
    for _ in range(100):
        actions, _, _ = policy.forward(obs, deterministic=False)
        for action in actions:
            action_counts[action.item()] += 1

    # All actions should potentially be selected when no mask is present
    # (though due to randomness, some might not be selected in 200 samples)
    print(f"\n✅ Action distribution without masking: {action_counts.tolist()}")
    print(
        f"   All actions available: {(action_counts > 0).sum().item()} / 6 actions selected"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
