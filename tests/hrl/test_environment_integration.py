"""
Unit tests for hierarchical environment integration.

This module tests the integration between hierarchical components and the environment:
- HierarchicalNppWrapper functionality
- Environment factory functions
- Reward shaping and logging
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from gymnasium import spaces

from npp_rl.environments.environment_factory import (
    create_hierarchical_env,
    HierarchicalNppWrapper
)
from npp_rl.hrl.completion_controller import CompletionController, Subtask


class MockNppEnvironment:
    """Mock NPP environment for testing."""
    
    def __init__(self):
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'reachability_features': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(6)
        self.metadata = {'render_modes': ['rgb_array', 'human']}
        
        self.reset_count = 0
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.reset_count += 1
        obs = {
            'observation': np.zeros((64, 64, 3), dtype=np.uint8),
            'reachability_features': np.zeros(8, dtype=np.float32)
        }
        info = {
            'ninja_pos': (10, 10),
            'level_data': {},
            'switch_states': {},
        }
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        obs = {
            'observation': np.zeros((64, 64, 3), dtype=np.uint8),
            'reachability_features': np.zeros(8, dtype=np.float32)
        }
        reward = 1.0
        terminated = False
        truncated = False
        info = {
            'ninja_pos': (10 + self.step_count, 10),
            'level_data': {},
            'switch_states': {},
        }
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        pass


class TestHierarchicalNppWrapper(unittest.TestCase):
    """Test cases for HierarchicalNppWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockNppEnvironment()
        self.mock_controller = Mock(spec=CompletionController)
        self.mock_controller.get_current_subtask.return_value = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.mock_controller.get_subtask_features.return_value = np.array([1, 0, 0, 0])
        self.mock_controller.get_subtask_metrics.return_value = {
            'current_subtask': 'NAVIGATE_TO_EXIT_SWITCH',
            'subtask_step_count': 5,
            'subtask_duration': 1.0,
            'total_transitions': 1,
            'recent_transitions': []
        }
        
        self.wrapper = HierarchicalNppWrapper(
            env=self.mock_env,
            completion_controller=self.mock_controller,
            enable_subtask_rewards=True,
            subtask_reward_scale=0.1,
        )
    
    def test_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.env, self.mock_env)
        self.assertEqual(self.wrapper.completion_controller, self.mock_controller)
        self.assertTrue(self.wrapper.enable_subtask_rewards)
        self.assertEqual(self.wrapper.subtask_reward_scale, 0.1)
        
        # Check environment attributes are exposed
        self.assertEqual(self.wrapper.observation_space, self.mock_env.observation_space)
        self.assertEqual(self.wrapper.action_space, self.mock_env.action_space)
        self.assertEqual(self.wrapper.metadata, self.mock_env.metadata)
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.wrapper.reset()
        
        # Check base environment was reset
        self.assertEqual(self.mock_env.reset_count, 1)
        
        # Check controller was reset
        self.mock_controller.reset.assert_called_once()
        
        # Check hierarchical info was added
        self.assertIn('hierarchical', info)
        hierarchical_info = info['hierarchical']
        self.assertIn('current_subtask', hierarchical_info)
        self.assertIn('subtask_features', hierarchical_info)
        self.assertIn('subtask_metrics', hierarchical_info)
    
    def test_step_without_subtask_rewards(self):
        """Test step without subtask reward shaping."""
        wrapper = HierarchicalNppWrapper(
            env=self.mock_env,
            completion_controller=self.mock_controller,
            enable_subtask_rewards=False,
        )
        
        obs, reward, terminated, truncated, info = wrapper.step(0)
        
        # Reward should be unchanged
        self.assertEqual(reward, 1.0)
        
        # Check hierarchical info was added
        self.assertIn('hierarchical', info)
    
    def test_step_with_subtask_rewards(self):
        """Test step with subtask reward shaping."""
        obs, reward, terminated, truncated, info = self.wrapper.step(0)
        
        # Reward should be modified (base reward + subtask reward * scale)
        self.assertNotEqual(reward, 1.0)
        
        # Check controller was updated
        self.mock_controller.step.assert_called_once()
        
        # Check hierarchical info was added
        self.assertIn('hierarchical', info)
    
    def test_subtask_transition_logging(self):
        """Test subtask transition logging."""
        # Mock subtask change
        self.wrapper.last_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.mock_controller.get_current_subtask.return_value = Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
        
        obs, reward, terminated, truncated, info = self.wrapper.step(0)
        
        # Check transition was logged
        self.assertIn('subtask_transition', info)
        transition = info['subtask_transition']
        self.assertEqual(transition['from'], 'NAVIGATE_TO_EXIT_SWITCH')
        self.assertEqual(transition['to'], 'NAVIGATE_TO_LOCKED_DOOR_SWITCH')
    
    def test_calculate_subtask_reward(self):
        """Test subtask-specific reward calculation."""
        obs_dict = {'observation': np.zeros((64, 64, 3))}
        info = {'switch_distance': 10.0}
        
        # Test exit switch navigation reward
        reward = self.wrapper._calculate_subtask_reward(
            Subtask.NAVIGATE_TO_EXIT_SWITCH, obs_dict, info, False
        )
        self.assertLess(reward, 0)  # Should be negative distance reward
        
        # Test subtask transition bonus
        info['subtask_transition'] = {'from': 'A', 'to': 'B'}
        reward = self.wrapper._calculate_subtask_reward(
            Subtask.NAVIGATE_TO_EXIT_SWITCH, obs_dict, info, False
        )
        self.assertGreater(reward, -0.1)  # Should include transition bonus
        
        # Test long subtask penalty
        self.wrapper.subtask_step_count = 600  # Over penalty threshold
        reward = self.wrapper._calculate_subtask_reward(
            Subtask.NAVIGATE_TO_EXIT_SWITCH, obs_dict, info, False
        )
        self.assertLess(reward, 0.4)  # Should include penalty
    
    def test_obs_to_dict_conversion(self):
        """Test observation to dictionary conversion."""
        obs = np.zeros((64, 64, 3))
        obs_dict = self.wrapper._obs_to_dict(obs)
        
        self.assertIn('observation', obs_dict)
        np.testing.assert_array_equal(obs_dict['observation'], obs)
    
    def test_attribute_delegation(self):
        """Test attribute delegation to base environment."""
        # Test accessing base environment attribute
        self.assertEqual(self.wrapper.reset_count, self.mock_env.reset_count)
        
        # Test calling base environment method
        result = self.wrapper.render()
        np.testing.assert_array_equal(result, np.zeros((64, 64, 3), dtype=np.uint8))


class TestEnvironmentFactory(unittest.TestCase):
    """Test cases for environment factory functions."""
    
    @patch('npp_rl.environments.environment_factory.create_reachability_aware_env')
    @patch('npp_rl.environments.environment_factory.HierarchicalNppWrapper')
    def test_create_hierarchical_env(self, mock_wrapper_class, mock_create_base):
        """Test hierarchical environment creation."""
        mock_base_env = Mock()
        mock_create_base.return_value = mock_base_env
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper
        
        mock_controller = Mock()
        
        result = create_hierarchical_env(
            render_mode="rgb_array",
            level_set="intro",
            max_episode_steps=2000,
            completion_controller=mock_controller,
            enable_subtask_rewards=True,
            subtask_reward_scale=0.2,
        )
        
        # Check base environment was created
        mock_create_base.assert_called_once_with(
            render_mode="rgb_array",
            level_set="intro",
            max_episode_steps=2000,
        )
        
        # Check wrapper was created with correct arguments
        mock_wrapper_class.assert_called_once_with(
            mock_base_env,
            completion_controller=mock_controller,
            enable_subtask_rewards=True,
            subtask_reward_scale=0.2,
        )
        
        self.assertEqual(result, mock_wrapper)
    
    @patch('npp_rl.environments.environment_factory.create_reachability_aware_env')
    @patch('npp_rl.environments.environment_factory.HierarchicalNppWrapper')
    def test_create_hierarchical_env_default_controller(self, mock_wrapper_class, mock_create_base):
        """Test hierarchical environment creation with default controller."""
        mock_base_env = Mock()
        mock_create_base.return_value = mock_base_env
        mock_wrapper = Mock()
        mock_wrapper_class.return_value = mock_wrapper
        
        result = create_hierarchical_env()
        
        # Check wrapper was created with a completion controller
        mock_wrapper_class.assert_called_once()
        call_args = mock_wrapper_class.call_args
        self.assertIsNotNone(call_args[1]['completion_controller'])


if __name__ == '__main__':
    unittest.main()