"""Integration tests for evaluation with frame stacking and action masking.

These tests verify that:
1. Frame-stacked models can be evaluated without warnings
2. Action masking works correctly during evaluation
3. Observation shapes match between training and evaluation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from npp_rl.agents.masked_actor_critic_policy import MaskedActorCriticPolicy
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from npp_rl.training.architecture_configs import get_architecture_config
from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator


def test_masked_policy_predict_with_action_mask():
    """Test that MaskedActorCriticPolicy._predict() properly handles action masks."""
    try:
        arch_config = get_architecture_config("mlp_baseline")
    except:
        pytest.skip("Architecture config not available")

    # Create observation space with action_mask
    obs_space = spaces.Dict({
        "player_frame": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
        "global_view": spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
        "game_state": spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)

    # Create policy
    policy = MaskedActorCriticPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=ConfigurableMultimodalExtractor,
        features_extractor_kwargs={"config": arch_config},
    )

    # Create observation with mask that only allows actions 0, 1, 2
    obs = {
        "player_frame": np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8),
        "global_view": np.random.randint(0, 255, (128, 128, 1), dtype=np.uint8),
        "game_state": np.random.randn(50).astype(np.float32),
        "action_mask": np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
    }

    # Sample many actions to verify distribution
    action_counts = np.zeros(6)
    num_samples = 200

    for _ in range(num_samples):
        # Use _predict() directly (called by model.predict())
        action = policy._predict(obs, deterministic=False)
        action_counts[action.item()] += 1

    # Masked actions (3, 4, 5) should NEVER be selected
    assert action_counts[3] == 0, f"Masked action 3 was selected {action_counts[3]} times!"
    assert action_counts[4] == 0, f"Masked action 4 was selected {action_counts[4]} times!"
    assert action_counts[5] == 0, f"Masked action 5 was selected {action_counts[5]} times!"

    # Valid actions (0, 1, 2) should be selected
    assert action_counts[0] > 0, "Action 0 should be available"
    assert action_counts[1] > 0, "Action 1 should be available"
    assert action_counts[2] > 0, "Action 2 should be available"


def test_frame_stacked_observation_space_detection():
    """Test that frame-stacked observation spaces are correctly detected."""
    try:
        arch_config = get_architecture_config("mlp_baseline")
    except:
        pytest.skip("Architecture config not available")

    # Create observation space with frame stacking (4 frames)
    obs_space_stacked = spaces.Dict({
        "player_frame": spaces.Box(low=0, high=255, shape=(4, 64, 64, 1), dtype=np.uint8),
        "global_view": spaces.Box(low=0, high=255, shape=(4, 128, 128, 1), dtype=np.uint8),
        "game_state": spaces.Box(low=-1, high=1, shape=(4, 50), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })
    action_space = spaces.Discrete(6)

    # Create a mock model with frame-stacked observation space
    class MockModel:
        def __init__(self, obs_space):
            self.observation_space = obs_space

    model = MockModel(obs_space_stacked)

    # Test detection
    evaluator = ComprehensiveEvaluator(
        test_dataset_path=str(Path(__file__).parent.parent / "data" / "test_dataset"),
        device="cpu",
    )

    detected_config = evaluator.detect_frame_stack_config(model)
    
    # Should detect visual stacking
    assert detected_config is not None
    assert detected_config["enable_visual_frame_stacking"] is True
    assert detected_config["visual_stack_size"] == 4


def test_frame_stacked_observation_space_validation():
    """Test that validation correctly handles frame-stacked observation spaces."""
    try:
        arch_config = get_architecture_config("mlp_baseline")
    except:
        pytest.skip("Architecture config not available")

    # Create observation space with frame stacking (4 frames)
    obs_space_stacked = spaces.Dict({
        "player_frame": spaces.Box(low=0, high=255, shape=(4, 64, 64, 1), dtype=np.uint8),
        "global_view": spaces.Box(low=0, high=255, shape=(4, 128, 128, 1), dtype=np.uint8),
        "game_state": spaces.Box(low=-1, high=1, shape=(4, 50), dtype=np.float32),
        "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
    })

    # Create a mock model with frame-stacked observation space
    class MockModel:
        def __init__(self, obs_space):
            self.observation_space = obs_space

    model = MockModel(obs_space_stacked)

    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        test_dataset_path=str(Path(__file__).parent.parent / "data" / "test_dataset"),
        device="cpu",
    )

    # Test validation with correct config
    frame_stack_config = {
        "enable_visual_frame_stacking": True,
        "visual_stack_size": 4,
        "enable_state_stacking": True,
        "state_stack_size": 4,
        "padding_type": "zero",
    }

    # Should not raise an error
    try:
        evaluator._validate_frame_stack_config(model, frame_stack_config)
    except ValueError as e:
        pytest.fail(f"Validation should pass but raised: {e}")

    # Test validation with incorrect stack size
    frame_stack_config_wrong = {
        "enable_visual_frame_stacking": True,
        "visual_stack_size": 2,  # Wrong size
        "enable_state_stacking": False,
        "state_stack_size": 4,
        "padding_type": "zero",
    }

    # Should raise ValueError
    with pytest.raises(ValueError, match="Frame stacking size mismatch"):
        evaluator._validate_frame_stack_config(model, frame_stack_config_wrong)


def test_action_masking_with_model_predict():
    """Test that action masking works when using model.predict() (integration test)."""
    try:
        arch_config = get_architecture_config("mlp_baseline")
    except:
        pytest.skip("Architecture config not available")

    # Create a simple mock environment that returns observations with action masks
    class MockEnv:
        def __init__(self):
            self.observation_space = spaces.Dict({
                "player_frame": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
                "global_view": spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
                "game_state": spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
            })
            self.action_space = spaces.Discrete(6)

        def reset(self):
            obs = {
                "player_frame": np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8),
                "global_view": np.random.randint(0, 255, (128, 128, 1), dtype=np.uint8),
                "game_state": np.random.randn(50).astype(np.float32),
                "action_mask": np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),  # Only actions 0,1,2 allowed
            }
            return obs, {}

        def step(self, action):
            obs = {
                "player_frame": np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8),
                "global_view": np.random.randint(0, 255, (128, 128, 1), dtype=np.uint8),
                "game_state": np.random.randn(50).astype(np.float32),
                "action_mask": np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
            }
            return obs, 0.0, False, False, {}

    # Create vectorized environment
    env = DummyVecEnv([lambda: MockEnv()])

    # Create model with MaskedActorCriticPolicy
    model = PPO(
        policy=MaskedActorCriticPolicy,
        env=env,
        policy_kwargs={
            "features_extractor_class": ConfigurableMultimodalExtractor,
            "features_extractor_kwargs": {"config": arch_config},
        },
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
    )

    # Reset environment
    obs, _ = env.reset()

    # Sample many actions using model.predict()
    action_counts = np.zeros(6)
    num_samples = 200

    for _ in range(num_samples):
        action, _ = model.predict(obs, deterministic=False)
        # DummyVecEnv returns array, extract first element
        if isinstance(action, np.ndarray):
            action = action[0] if len(action) > 0 else action.item()
        action_counts[action] += 1

    # Masked actions (3, 4, 5) should NEVER be selected
    assert action_counts[3] == 0, f"Masked action 3 was selected {action_counts[3]} times!"
    assert action_counts[4] == 0, f"Masked action 4 was selected {action_counts[4]} times!"
    assert action_counts[5] == 0, f"Masked action 5 was selected {action_counts[5]} times!"

    # Valid actions (0, 1, 2) should be selected
    assert action_counts[0] > 0, "Action 0 should be available"
    assert action_counts[1] > 0, "Action 1 should be available"
    assert action_counts[2] > 0, "Action 2 should be available"

