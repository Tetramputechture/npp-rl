"""Test curriculum progression with multiple environments (n_envs > 1).

This test verifies that curriculum stage progression is tracked globally
across all environments and that stage changes are synchronized properly.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from npp_rl.training.curriculum_manager import CurriculumManager
from npp_rl.wrappers.curriculum_env import (
    CurriculumEnv,
    CurriculumVecEnvWrapper,
)


class MockNppEnvironment(gym.Env):
    """Mock N++ environment for testing."""

    def __init__(self, success_pattern=None):
        """Initialize mock environment.

        Args:
            success_pattern: List of booleans indicating success/failure for episodes
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(6)
        self.success_pattern = success_pattern or []
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = 10

        # Mock nplay_headless for map loading
        self.nplay_headless = MagicMock()
        self.nplay_headless.load_map_from_map_data = MagicMock()

    def reset(self, **kwargs):
        """Reset environment."""
        self.step_count = 0
        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        info = {}
        return obs, info

    def step(self, action):
        """Execute step."""
        self.step_count += 1
        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        reward = 0.0

        # Episode ends after max_steps
        terminated = self.step_count >= self.max_steps
        truncated = False

        # Determine success based on pattern
        if terminated:
            if self.episode_count < len(self.success_pattern):
                success = self.success_pattern[self.episode_count]
            else:
                success = False
            self.episode_count += 1
        else:
            success = False

        info = {"is_success": success}

        return obs, reward, terminated, truncated, info


def create_mock_curriculum_manager():
    """Create a mock curriculum manager with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create minimal test suite structure
        test_suite_dir = tmpdir_path / "test_suite"
        test_suite_dir.mkdir()

        # Create test levels for each stage
        for stage in ["simplest", "simple", "medium"]:
            stage_dir = test_suite_dir / stage
            stage_dir.mkdir()

            # Create a simple level file
            level_data = {
                "level_id": f"{stage}_level_1",
                "category": stage,
                "map_data": "mock_map_data",
            }

            import json

            level_file = stage_dir / "level_1.json"
            with open(level_file, "w") as f:
                json.dump(level_data, f)

        # Create curriculum manager
        manager = CurriculumManager(
            dataset_path=str(test_suite_dir),
            starting_stage="simplest",
            performance_window=10,
            allow_stage_mixing=False,
        )

        yield manager


def test_curriculum_env_local_tracking_disabled():
    """Test that local tracking can be disabled in CurriculumEnv."""
    mock_env = MockNppEnvironment(success_pattern=[True, True, False])

    with patch("npp_rl.wrappers.curriculum_env.TestSuiteLoader"):
        curriculum_manager = MagicMock()
        curriculum_manager.get_current_stage.return_value = "simplest"
        curriculum_manager.sample_level.return_value = {
            "level_id": "test_level",
            "category": "simplest",
            "map_data": "test_data",
        }
        curriculum_manager.CURRICULUM_ORDER = ["simplest", "simple", "medium"]

        # Create curriculum env with local tracking disabled
        curriculum_env = CurriculumEnv(
            mock_env,
            curriculum_manager,
            check_advancement_freq=5,
            enable_local_tracking=False,
        )

        # Run some episodes
        for _ in range(3):
            obs, info = curriculum_env.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = curriculum_env.step(0)
                done = terminated or truncated

        # Verify that record_episode was NOT called (local tracking disabled)
        curriculum_manager.record_episode.assert_not_called()
        curriculum_manager.check_advancement.assert_not_called()


def test_curriculum_vec_env_global_tracking():
    """Test that CurriculumVecEnvWrapper tracks episodes globally across all envs."""
    n_envs = 4

    # Create success patterns for each env
    # Env 0: alternating success/failure
    # Env 1-3: mostly success
    success_patterns = [
        [True, False, True, False, True, False],  # Env 0
        [True, True, True, True, True, True],  # Env 1
        [True, True, True, True, False, True],  # Env 2
        [True, True, True, True, True, True],  # Env 3
    ]

    with patch("npp_rl.wrappers.curriculum_env.TestSuiteLoader"):
        # Create mock curriculum manager
        curriculum_manager = MagicMock()
        curriculum_manager.get_current_stage.return_value = "simplest"
        curriculum_manager.check_advancement.return_value = False
        curriculum_manager.get_stage_performance.return_value = {
            "success_rate": 0.5,
            "episodes": 10,
            "can_advance": False,
        }
        curriculum_manager.CURRICULUM_ORDER = ["simplest", "simple", "medium"]

        def mock_sample_level():
            return {
                "level_id": "test_level",
                "category": "simplest",
                "map_data": "test_data",
            }

        curriculum_manager.sample_level = mock_sample_level

        # Create environment factory functions
        def make_env(idx):
            def _init():
                mock_env = MockNppEnvironment(success_pattern=success_patterns[idx])
                curriculum_env = CurriculumEnv(
                    mock_env,
                    curriculum_manager,
                    check_advancement_freq=10,
                    enable_local_tracking=False,  # Disabled for VecEnv
                )
                return curriculum_env

            return _init

        # Create vectorized environment
        env_fns = [make_env(i) for i in range(n_envs)]
        vec_env = DummyVecEnv(env_fns)

        # Wrap with curriculum tracking
        curriculum_vec_env = CurriculumVecEnvWrapper(
            vec_env,
            curriculum_manager,
            check_advancement_freq=10,
        )

        # Reset all environments
        obs = curriculum_vec_env.reset()
        assert obs.shape[0] == n_envs

        # Run steps until we complete at least 15 episodes
        total_completed = 0
        max_steps = 200
        step_count = 0

        while total_completed < 15 and step_count < max_steps:
            actions = np.array([0] * n_envs)
            obs, rewards, dones, infos = curriculum_vec_env.step(actions)
            total_completed += np.sum(dones)
            step_count += 1

        # Verify that record_episode was called for each completed episode
        # Each env completed multiple episodes
        assert curriculum_manager.record_episode.call_count >= 12

        # Verify check_advancement was called at least once
        # (should be called every 10 episodes)
        assert curriculum_manager.check_advancement.call_count >= 1


def test_curriculum_stage_advancement_and_sync():
    """Test that stage advancement is detected and synced to all environments."""
    n_envs = 3

    # All envs will succeed most of the time to trigger advancement
    success_patterns = [
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True],
    ]

    with patch("npp_rl.wrappers.curriculum_env.TestSuiteLoader"):
        # Create mock curriculum manager
        curriculum_manager = MagicMock()
        initial_stage = "simplest"
        next_stage = "simple"

        # Track current stage
        current_stage = [initial_stage]

        def get_current_stage():
            return current_stage[0]

        def check_advancement():
            # Advance after 5 successful episodes
            if curriculum_manager.record_episode.call_count >= 5:
                current_stage[0] = next_stage
                return True
            return False

        def get_stage_performance(stage):
            if curriculum_manager.record_episode.call_count < 5:
                return {
                    "success_rate": 0.5,
                    "episodes": curriculum_manager.record_episode.call_count,
                    "can_advance": False,
                }
            else:
                return {
                    "success_rate": 0.9,
                    "episodes": curriculum_manager.record_episode.call_count,
                    "can_advance": True,
                }

        curriculum_manager.get_current_stage = get_current_stage
        curriculum_manager.check_advancement = check_advancement
        curriculum_manager.get_stage_performance = get_stage_performance
        curriculum_manager.CURRICULUM_ORDER = ["simplest", "simple", "medium"]

        def mock_sample_level():
            return {
                "level_id": f"{current_stage[0]}_level",
                "category": current_stage[0],
                "map_data": "test_data",
            }

        curriculum_manager.sample_level = mock_sample_level

        # Create environment factory functions
        def make_env(idx):
            def _init():
                mock_env = MockNppEnvironment(success_pattern=success_patterns[idx])
                curriculum_env = CurriculumEnv(
                    mock_env,
                    curriculum_manager,
                    check_advancement_freq=10,
                    enable_local_tracking=False,
                )
                return curriculum_env

            return _init

        # Create vectorized environment
        env_fns = [make_env(i) for i in range(n_envs)]
        vec_env = DummyVecEnv(env_fns)

        # Wrap with curriculum tracking
        curriculum_vec_env = CurriculumVecEnvWrapper(
            vec_env,
            curriculum_manager,
            check_advancement_freq=5,  # Check frequently
        )

        # Reset all environments
        obs = curriculum_vec_env.reset()

        # Run until we've completed enough episodes to trigger advancement
        total_completed = 0
        max_steps = 200
        step_count = 0

        while total_completed < 10 and step_count < max_steps:
            actions = np.array([0] * n_envs)
            obs, rewards, dones, infos = curriculum_vec_env.step(actions)
            total_completed += np.sum(dones)
            step_count += 1

        # Verify that stage advanced
        assert curriculum_manager.check_advancement.call_count >= 2
        assert current_stage[0] == next_stage

        # Verify env_method was called to sync stage to all environments
        assert vec_env.env_method.call_count >= 2  # Initial sync + advancement sync


def test_curriculum_manager_global_performance_tracking():
    """Test that curriculum manager tracks performance globally, not per-env."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create minimal test suite structure
        test_suite_dir = tmpdir_path / "test_suite"
        test_suite_dir.mkdir()

        # Create test level
        stage = "simplest"
        stage_dir = test_suite_dir / stage
        stage_dir.mkdir()

        import json

        level_data = {
            "level_id": f"{stage}_level_1",
            "category": stage,
            "map_data": "mock_map_data",
        }
        level_file = stage_dir / "level_1.json"
        with open(level_file, "w") as f:
            json.dump(level_data, f)

        # Create curriculum manager
        manager = CurriculumManager(
            dataset_path=str(test_suite_dir),
            starting_stage="simplest",
            advancement_threshold=0.7,
            min_episodes_per_stage=5,
            performance_window=10,
        )

        # Record episodes from "different environments" (but tracked globally)
        for i in range(10):
            # Simulate episodes from different envs (70% success rate)
            success = i % 10 < 7
            manager.record_episode("simplest", success)

        # Check performance
        perf = manager.get_stage_performance("simplest")
        assert perf["success_rate"] == 0.7
        assert perf["episodes"] == 10
        assert perf["can_advance"]  # Should be able to advance

        # Verify advancement works
        advanced = manager.check_advancement()
        assert advanced
        assert manager.get_current_stage() == "simpler"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
