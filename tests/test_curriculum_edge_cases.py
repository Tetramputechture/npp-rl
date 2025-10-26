"""Test edge cases and error handling in curriculum learning system."""

import unittest
from collections import deque
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path


class TestCurriculumManagerEdgeCases(unittest.TestCase):
    """Test edge cases in CurriculumManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the TestSuiteLoader to avoid file dependencies
        self.patcher = patch('npp_rl.training.curriculum_manager.TestSuiteLoader')
        self.mock_loader_class = self.patcher.start()
        
        # Create a mock loader instance
        mock_loader = MagicMock()
        mock_loader.load_all_levels.return_value = {
            "simplest": [{"level_id": "test1", "map_data": "data", "category": "simplest"}],
            "simpler": [{"level_id": "test2", "map_data": "data", "category": "simpler"}],
            "simple": [{"level_id": "test3", "map_data": "data", "category": "simple"}],
        }
        self.mock_loader_class.return_value = mock_loader

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()

    def test_get_stage_performance_returns_all_keys_when_empty(self):
        """Test that get_stage_performance returns all keys even with no data."""
        from npp_rl.training.curriculum_manager import CurriculumManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CurriculumManager(
                dataset_path=tmpdir,
                starting_stage="simplest",
                advancement_threshold=0.7,
            )
            
            # Get performance for stage with no episodes
            perf = manager.get_stage_performance("simplest")
            
            # Verify all required keys exist
            self.assertIn("success_rate", perf)
            self.assertIn("episodes", perf)
            self.assertIn("can_advance", perf)
            self.assertIn("advancement_threshold", perf)
            
            # Verify values
            self.assertEqual(perf["success_rate"], 0.0)
            self.assertEqual(perf["episodes"], 0)
            self.assertEqual(perf["can_advance"], False)
            self.assertEqual(perf["advancement_threshold"], 0.7)

    def test_get_stage_performance_returns_all_keys_with_data(self):
        """Test that get_stage_performance returns all keys with data."""
        from npp_rl.training.curriculum_manager import CurriculumManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CurriculumManager(
                dataset_path=tmpdir,
                starting_stage="simplest",
                advancement_threshold=0.7,
            )
            
            # Record some episodes
            manager.record_episode("simplest", True)
            manager.record_episode("simplest", True)
            manager.record_episode("simplest", False)
            
            # Get performance
            perf = manager.get_stage_performance("simplest")
            
            # Verify all required keys exist
            self.assertIn("success_rate", perf)
            self.assertIn("episodes", perf)
            self.assertIn("can_advance", perf)
            self.assertIn("advancement_threshold", perf)
            
            # Verify values
            self.assertAlmostEqual(perf["success_rate"], 2/3, places=2)
            self.assertEqual(perf["episodes"], 3)
            self.assertEqual(perf["advancement_threshold"], 0.7)

    def test_record_episode_handles_unknown_stage(self):
        """Test that record_episode handles unknown stages gracefully."""
        from npp_rl.training.curriculum_manager import CurriculumManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CurriculumManager(
                dataset_path=tmpdir,
                starting_stage="simplest",
            )
            
            # Try to record for unknown stage (should not crash)
            manager.record_episode("nonexistent_stage", True)
            
            # Verify no data was recorded
            perf = manager.get_stage_performance("simplest")
            self.assertEqual(perf["episodes"], 0)

    def test_record_episode_initializes_missing_episode_count(self):
        """Test that record_episode handles missing episode count."""
        from npp_rl.training.curriculum_manager import CurriculumManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CurriculumManager(
                dataset_path=tmpdir,
                starting_stage="simplest",
            )
            
            # Manually delete episode count to simulate edge case
            if "simplest" in manager.stage_episode_counts:
                del manager.stage_episode_counts["simplest"]
            
            # Record episode (should not crash)
            manager.record_episode("simplest", True)
            
            # Verify episode was recorded
            self.assertEqual(manager.stage_episode_counts["simplest"], 1)

    def test_get_stage_performance_handles_missing_episode_count(self):
        """Test that get_stage_performance handles missing episode count."""
        from npp_rl.training.curriculum_manager import CurriculumManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CurriculumManager(
                dataset_path=tmpdir,
                starting_stage="simplest",
            )
            
            # Add performance data but remove episode count
            manager.stage_performance["simplest"].append(1)
            if "simplest" in manager.stage_episode_counts:
                del manager.stage_episode_counts["simplest"]
            
            # Get performance (should not crash, should default to 0)
            perf = manager.get_stage_performance("simplest")
            
            self.assertEqual(perf["episodes"], 0)


class TestCurriculumEnvEdgeCases(unittest.TestCase):
    """Test edge cases in CurriculumEnv wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock environment
        self.mock_env = MagicMock()
        self.mock_env.unwrapped = self.mock_env
        
        # Create mock curriculum manager
        self.mock_curriculum = MagicMock()
        self.mock_curriculum.sample_level.return_value = {
            "level_id": "test1",
            "map_data": "test_data",
            "category": "simplest",
        }
        self.mock_curriculum.CURRICULUM_ORDER = [
            "simplest", "simpler", "simple", "medium", "complex"
        ]

    def test_reset_handles_none_level_data(self):
        """Test that reset handles None from sample_level."""
        from npp_rl.wrappers.curriculum_env import CurriculumEnv
        
        # Configure curriculum to return None
        self.mock_curriculum.sample_level.return_value = None
        self.mock_env.reset.return_value = (MagicMock(), {})
        
        env = CurriculumEnv(self.mock_env, self.mock_curriculum)
        
        # Should not crash, should fall back to default reset
        obs, info = env.reset()
        
        # Verify default reset was called
        self.mock_env.reset.assert_called()

    def test_reset_handles_missing_category(self):
        """Test that reset handles level data without category."""
        from npp_rl.wrappers.curriculum_env import CurriculumEnv
        
        # Configure curriculum to return level without category
        self.mock_curriculum.sample_level.return_value = {
            "level_id": "test1",
            "map_data": "test_data",
            # No category field
        }
        self.mock_env.reset.return_value = (MagicMock(), {})
        
        env = CurriculumEnv(self.mock_env, self.mock_curriculum)
        
        # Should not crash, should use "unknown"
        obs, info = env.reset()
        
        self.assertEqual(env.current_level_stage, "unknown")

    def test_step_handles_missing_current_level_stage(self):
        """Test that step handles missing current_level_stage attribute."""
        from npp_rl.wrappers.curriculum_env import CurriculumEnv
        
        self.mock_env.step.return_value = (MagicMock(), 0, False, False, {})
        
        env = CurriculumEnv(self.mock_env, self.mock_curriculum)
        
        # Delete the attribute to simulate edge case
        if hasattr(env, 'current_level_stage'):
            delattr(env, 'current_level_stage')
        
        # Should not crash, should add "unknown" to info
        obs, reward, term, trunc, info = env.step(0)
        
        self.assertEqual(info["curriculum_stage"], "unknown")

    def test_on_episode_end_handles_no_stage(self):
        """Test that _on_episode_end handles missing stage."""
        from npp_rl.wrappers.curriculum_env import CurriculumEnv
        
        env = CurriculumEnv(self.mock_env, self.mock_curriculum)
        
        # Set stage to None
        env.current_level_stage = None
        
        # Should not crash, should not record
        env._on_episode_end({"is_success": True})
        
        # Verify record_episode was not called
        self.mock_curriculum.record_episode.assert_not_called()

    def test_set_curriculum_stage_handles_invalid_stage(self):
        """Test that set_curriculum_stage handles invalid stage names."""
        from npp_rl.wrappers.curriculum_env import CurriculumEnv
        
        env = CurriculumEnv(self.mock_env, self.mock_curriculum)
        
        # Should not crash with invalid stage
        env.set_curriculum_stage("invalid_stage_name")
        
        # Verify last known stage was updated
        self.assertEqual(env._last_known_stage, "invalid_stage_name")


class TestCurriculumVecEnvWrapperEdgeCases(unittest.TestCase):
    """Test edge cases in CurriculumVecEnvWrapper."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock vectorized environment
        self.mock_venv = MagicMock()
        self.mock_venv.num_envs = 4
        self.mock_venv.env_method = MagicMock()
        
        # Create mock curriculum manager
        self.mock_curriculum = MagicMock()
        self.mock_curriculum.get_current_stage.return_value = "simplest"
        self.mock_curriculum.get_stage_performance.return_value = {
            "success_rate": 0.0,
            "episodes": 0,
            "can_advance": False,
            "advancement_threshold": 0.7,
        }
        self.mock_curriculum.check_advancement.return_value = False

    def test_step_wait_handles_missing_curriculum_stage(self):
        """Test that step_wait handles info dicts without curriculum_stage."""
        from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper
        
        # Configure venv to return info without curriculum_stage
        self.mock_venv.step_wait.return_value = (
            [MagicMock()] * 4,  # obs
            [0] * 4,  # rewards
            [True, False, False, False],  # dones
            [{"is_success": True}, {}, {}, {}],  # infos (first missing stage)
        )
        
        wrapper = CurriculumVecEnvWrapper(
            self.mock_venv, self.mock_curriculum, check_advancement_freq=10
        )
        
        # Should not crash
        obs, rewards, dones, infos = wrapper.step_wait()
        
        # Verify warning was logged (curriculum manager should not be called)
        self.mock_curriculum.record_episode.assert_not_called()

    def test_step_wait_handles_record_episode_error(self):
        """Test that step_wait handles errors in record_episode."""
        from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper
        
        # Configure curriculum to raise error
        self.mock_curriculum.record_episode.side_effect = Exception("Test error")
        
        # Configure venv
        self.mock_venv.step_wait.return_value = (
            [MagicMock()] * 4,
            [0] * 4,
            [True, False, False, False],
            [{"is_success": True, "curriculum_stage": "simplest"}, {}, {}, {}],
        )
        
        wrapper = CurriculumVecEnvWrapper(
            self.mock_venv, self.mock_curriculum, check_advancement_freq=10
        )
        
        # Should not crash, should catch and log error
        obs, rewards, dones, infos = wrapper.step_wait()

    def test_step_wait_handles_advancement_check_error(self):
        """Test that step_wait handles errors during advancement check."""
        from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper
        
        # Configure curriculum to raise error on get_stage_performance
        self.mock_curriculum.get_stage_performance.side_effect = Exception("Test error")
        
        # Configure venv to complete enough episodes to trigger check
        self.mock_venv.step_wait.return_value = (
            [MagicMock()] * 4,
            [0] * 4,
            [True, False, False, False],
            [{"is_success": True, "curriculum_stage": "simplest"}, {}, {}, {}],
        )
        
        wrapper = CurriculumVecEnvWrapper(
            self.mock_venv, self.mock_curriculum, check_advancement_freq=1
        )
        
        # Should not crash, should catch and log error
        obs, rewards, dones, infos = wrapper.step_wait()

    def test_sync_handles_missing_env_method(self):
        """Test that _sync_curriculum_stage handles missing env_method."""
        from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper
        
        # Remove env_method
        del self.mock_venv.env_method
        
        wrapper = CurriculumVecEnvWrapper(
            self.mock_venv, self.mock_curriculum, check_advancement_freq=10
        )
        
        # Should not crash, should log warning
        wrapper._sync_curriculum_stage("simpler")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
