"""
Test hierarchical RL environment integration.

Tests:
- Hierarchical environment factory function
- Integration with completion controller
- Basic hierarchical functionality
"""

import unittest
from unittest.mock import Mock, MagicMock
import numpy as np

from nclone.gym_environment import create_hierarchical_env
from npp_rl.hrl.completion_controller import CompletionController


class TestHierarchicalIntegration(unittest.TestCase):
    """Test cases for hierarchical RL integration."""
    
    def test_create_hierarchical_env(self):
        """Test hierarchical environment creation."""
        # Create mock completion planner
        mock_planner = Mock()
        mock_planner.get_current_subtask.return_value = "NAVIGATE_TO_SWITCH"
        mock_planner.get_subtask_features.return_value = np.array([1.0, 0.0, 0.0, 0.5])
        
        # Create hierarchical environment
        env = create_hierarchical_env(
            completion_planner=mock_planner,
            enable_subtask_rewards=True,
            subtask_reward_scale=0.1,
            max_subtask_steps=100,
            debug=True
        )
        
        # Check environment was created with hierarchical features
        self.assertTrue(hasattr(env, 'enable_hierarchical'))
        self.assertTrue(env.enable_hierarchical)
        self.assertEqual(env.subtask_reward_scale, 0.1)
        self.assertEqual(env.max_subtask_steps, 100)
        
        # Check observation space includes hierarchical features
        self.assertIn('subtask_features', env.observation_space.spaces)
        
        # Clean up
        env.close()
    
    def test_hierarchical_env_without_planner(self):
        """Test hierarchical environment creation without completion planner."""
        env = create_hierarchical_env(
            completion_planner=None,
            enable_subtask_rewards=False
        )
        
        # Check environment was created
        self.assertTrue(hasattr(env, 'enable_hierarchical'))
        self.assertTrue(env.enable_hierarchical)
        self.assertFalse(env.enable_subtask_rewards)
        
        # Clean up
        env.close()
    
    def test_hierarchical_env_default_params(self):
        """Test hierarchical environment creation with default parameters."""
        env = create_hierarchical_env()
        
        # Check default parameters
        self.assertTrue(env.enable_hierarchical)
        self.assertTrue(env.enable_subtask_rewards)
        self.assertEqual(env.subtask_reward_scale, 0.1)
        self.assertEqual(env.max_subtask_steps, 1000)
        
        # Clean up
        env.close()


if __name__ == '__main__':
    unittest.main()