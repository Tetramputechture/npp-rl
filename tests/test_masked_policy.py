"""Tests for MaskedActorCriticPolicy to ensure no regressions."""

import pytest
import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from npp_rl.agents.masked_actor_critic_policy import MaskedActorCriticPolicy


@pytest.fixture
def simple_env_spaces():
    """Create simple observation and action spaces for testing."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)
    return obs_space, action_space


def test_policy_instantiation(simple_env_spaces):
    """Test that the masked policy can be instantiated correctly."""
    obs_space, action_space = simple_env_spaces
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    assert policy is not None
    assert hasattr(policy, 'forward')
    assert hasattr(policy, 'evaluate_actions')
    assert hasattr(policy, '_apply_action_mask')


def test_forward_without_mask():
    """Test that forward pass works without action mask (backward compatibility)."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
    })
    action_space = spaces.Discrete(6)
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    # Create observation without mask
    obs = {
        "vector": torch.randn(2, 10),  # Batch of 2
    }
    
    # Forward should work without mask
    actions, values, log_probs = policy.forward(obs, deterministic=False)
    
    assert actions.shape == (2,)
    assert values.shape == (2, 1)
    assert log_probs.shape == (2,)
    assert torch.all((actions >= 0) & (actions < 6))


def test_forward_with_mask():
    """Test that forward pass properly applies action mask."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    # Create observation with mask that only allows actions 0, 1, 2
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    obs = {
        "vector": torch.randn(2, 10),
        "action_mask": mask,
    }
    
    # Sample many times to check distribution
    action_counts = torch.zeros(6)
    for _ in range(100):
        actions, values, log_probs = policy.forward(obs, deterministic=False)
        for action in actions:
            action_counts[action.item()] += 1
    
    # Masked actions (3, 4, 5) should never be selected
    assert action_counts[3] == 0, "Action 3 should be masked"
    assert action_counts[4] == 0, "Action 4 should be masked"
    assert action_counts[5] == 0, "Action 5 should be masked"
    
    # Valid actions (0, 1, 2) should be selected
    assert action_counts[0] > 0, "Action 0 should be available"
    assert action_counts[1] > 0, "Action 1 should be available"
    assert action_counts[2] > 0, "Action 2 should be available"


def test_evaluate_actions_with_mask():
    """Test that evaluate_actions properly handles masks."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    # Create observation with mask
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    obs = {
        "vector": torch.randn(2, 10),
        "action_mask": mask,
    }
    
    # Valid actions
    valid_actions = torch.tensor([0, 1])
    
    values, log_probs, entropy = policy.evaluate_actions(obs, valid_actions)
    
    assert values.shape == (2, 1)
    assert log_probs.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.all(torch.isfinite(log_probs)), "Log probs should be finite for valid actions"


def test_masked_actions_have_zero_probability():
    """Test that masked actions have effectively zero probability."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    # Create observation with mask that masks actions 3, 4, 5
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    obs = {
        "vector": torch.randn(1, 10),
        "action_mask": mask,
    }
    
    # Try to evaluate a masked action
    masked_action = torch.tensor([3])
    
    values, log_probs, entropy = policy.evaluate_actions(obs, masked_action)
    
    # Log prob should be -inf for masked action
    assert torch.isinf(log_probs), "Masked actions should have -inf log probability"
    assert log_probs < 0, "Log probability should be negative infinity"


def test_apply_action_mask_helper():
    """Test the _apply_action_mask helper method directly."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    # Test with batch of logits
    logits = torch.randn(2, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1]], dtype=torch.bool)
    
    masked_logits = policy._apply_action_mask(logits, mask)
    
    # Masked actions should be -inf
    assert torch.isinf(masked_logits[0, 3]) and masked_logits[0, 3] < 0
    assert torch.isinf(masked_logits[0, 4]) and masked_logits[0, 4] < 0
    assert torch.isinf(masked_logits[0, 5]) and masked_logits[0, 5] < 0
    assert torch.isinf(masked_logits[1, 2]) and masked_logits[1, 2] < 0
    assert torch.isinf(masked_logits[1, 3]) and masked_logits[1, 3] < 0
    assert torch.isinf(masked_logits[1, 4]) and masked_logits[1, 4] < 0
    
    # Valid actions should be unchanged
    assert torch.isfinite(masked_logits[0, 0])
    assert torch.isfinite(masked_logits[0, 1])
    assert torch.isfinite(masked_logits[0, 2])
    assert torch.isfinite(masked_logits[1, 5])


def test_single_action_mask_broadcasting():
    """Test that a single action mask can be broadcast to a batch."""
    obs_space = spaces.Dict({
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)
    
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )
    
    # Test with single mask for entire batch
    logits = torch.randn(4, 6)  # Batch of 4
    mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)  # Single mask
    
    masked_logits = policy._apply_action_mask(logits, mask)
    
    # All batch elements should have the same mask applied
    for i in range(4):
        assert torch.isinf(masked_logits[i, 3]) and masked_logits[i, 3] < 0
        assert torch.isinf(masked_logits[i, 4]) and masked_logits[i, 4] < 0
        assert torch.isinf(masked_logits[i, 5]) and masked_logits[i, 5] < 0
        assert torch.isfinite(masked_logits[i, 0])
        assert torch.isfinite(masked_logits[i, 1])
        assert torch.isfinite(masked_logits[i, 2])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

