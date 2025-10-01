"""
Unit tests for the hierarchical PPO implementation.

This module tests the hierarchical policy network and PPO integration including:
- Policy network architecture
- Action selection logic
- Value function computation
- Integration with completion controller
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

from npp_rl.agents.hierarchical_ppo import (
    HierarchicalPolicyNetwork,
    HierarchicalActorCriticPolicy,
    HierarchicalPPO
)
from npp_rl.hrl.completion_controller import CompletionController


class MockFeaturesExtractor(nn.Module):
    """Mock features extractor for testing."""
    
    def __init__(self, features_dim=256):
        super().__init__()
        self.features_dim = features_dim
        
    def forward(self, obs):
        batch_size = obs.shape[0]
        return torch.randn(batch_size, self.features_dim)


class TestHierarchicalPolicyNetwork(unittest.TestCase):
    """Test cases for HierarchicalPolicyNetwork."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.features_extractor = MockFeaturesExtractor(features_dim=256)
        self.network = HierarchicalPolicyNetwork(
            features_extractor=self.features_extractor,
            features_dim=256,
            high_level_actions=4,
            low_level_actions=6,
        )
        
        # Mock inputs
        self.batch_size = 8
        self.obs = torch.randn(self.batch_size, 64, 64, 3)
        self.current_subtask = torch.zeros(self.batch_size, 4)
        self.current_subtask[:, 0] = 1.0  # First subtask active
    
    def test_network_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.network.features_dim, 256)
        self.assertIsInstance(self.network.high_level_policy, nn.Sequential)
        self.assertIsInstance(self.network.low_level_policy, nn.Sequential)
        self.assertIsInstance(self.network.value_net, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        high_level_action, low_level_action, high_level_log_prob, low_level_log_prob, value = \
            self.network(self.obs, self.current_subtask)
        
        # Check output shapes
        self.assertEqual(high_level_action.shape, (self.batch_size,))
        self.assertEqual(low_level_action.shape, (self.batch_size,))
        self.assertEqual(high_level_log_prob.shape, (self.batch_size,))
        self.assertEqual(low_level_log_prob.shape, (self.batch_size,))
        self.assertEqual(value.shape, (self.batch_size, 1))
        
        # Check action ranges
        self.assertTrue(torch.all(high_level_action >= 0))
        self.assertTrue(torch.all(high_level_action < 4))
        self.assertTrue(torch.all(low_level_action >= 0))
        self.assertTrue(torch.all(low_level_action < 6))
    
    def test_deterministic_forward(self):
        """Test deterministic forward pass."""
        # Run multiple times to check consistency
        results1 = self.network(self.obs, self.current_subtask, deterministic=True)
        results2 = self.network(self.obs, self.current_subtask, deterministic=True)
        
        # Actions should be identical in deterministic mode
        torch.testing.assert_close(results1[0], results2[0])  # high_level_action
        torch.testing.assert_close(results1[1], results2[1])  # low_level_action
    
    def test_get_value(self):
        """Test value function computation."""
        value = self.network.get_value(self.obs, self.current_subtask)
        
        self.assertEqual(value.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(value)))
    
    def test_subtask_cooldown(self):
        """Test subtask switching cooldown mechanism."""
        # Test with cooldown active
        step_count = 5  # Within cooldown period
        high_level_action1, _, _, _, _ = self.network(
            self.obs, self.current_subtask, step_count=step_count
        )
        
        # Test with cooldown expired
        step_count = 15  # Beyond cooldown period
        high_level_action2, _, _, _, _ = self.network(
            self.obs, self.current_subtask, step_count=step_count
        )
        
        # Both should be valid actions
        self.assertTrue(torch.all(high_level_action1 >= 0))
        self.assertTrue(torch.all(high_level_action1 < 4))
        self.assertTrue(torch.all(high_level_action2 >= 0))
        self.assertTrue(torch.all(high_level_action2 < 4))


class TestHierarchicalActorCriticPolicy(unittest.TestCase):
    """Test cases for HierarchicalActorCriticPolicy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'reachability_features': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(6)
        
        # Mock completion controller
        self.mock_controller = Mock(spec=CompletionController)
        self.mock_controller.get_subtask_features.return_value = np.array([1, 0, 0, 0])
        self.mock_controller.get_subtask_metrics.return_value = {
            'current_subtask': 'NAVIGATE_TO_EXIT_SWITCH',
            'subtask_step_count': 10,
            'subtask_duration': 1.5,
            'total_transitions': 2,
            'recent_transitions': []
        }
        
        # Mock lr_schedule
        self.lr_schedule = lambda x: 3e-4
    
    @patch('npp_rl.agents.hierarchical_ppo.HierarchicalPolicyNetwork')
    def test_policy_initialization(self, mock_network_class):
        """Test policy initialization."""
        mock_network = Mock()
        mock_network_class.return_value = mock_network
        
        policy = HierarchicalActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            completion_controller=self.mock_controller,
        )
        
        self.assertIsNotNone(policy.completion_controller)
        self.assertEqual(policy.step_count, 0)
    
    def test_reset_episode(self):
        """Test episode reset functionality."""
        with patch('npp_rl.agents.hierarchical_ppo.HierarchicalPolicyNetwork'):
            policy = HierarchicalActorCriticPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
                completion_controller=self.mock_controller,
            )
            
            # Set some state
            policy.step_count = 100
            policy.current_subtask_tensor = torch.ones(1, 4)
            
            # Reset
            policy.reset_episode()
            
            # Check state is reset
            self.mock_controller.reset.assert_called_once()
            self.assertIsNone(policy.current_subtask_tensor)
            self.assertEqual(policy.step_count, 0)
    
    def test_get_subtask_metrics(self):
        """Test subtask metrics retrieval."""
        with patch('npp_rl.agents.hierarchical_ppo.HierarchicalPolicyNetwork'):
            policy = HierarchicalActorCriticPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=self.lr_schedule,
                completion_controller=self.mock_controller,
            )
            
            metrics = policy.get_subtask_metrics()
            
            self.mock_controller.get_subtask_metrics.assert_called_once()
            self.assertIn('current_subtask', metrics)
            self.assertIn('subtask_step_count', metrics)


class TestHierarchicalPPO(unittest.TestCase):
    """Test cases for HierarchicalPPO wrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_controller = Mock(spec=CompletionController)
        self.hierarchical_ppo = HierarchicalPPO(
            completion_controller=self.mock_controller,
            learning_rate=3e-4,
            n_steps=2048,
        )
    
    def test_initialization(self):
        """Test HierarchicalPPO initialization."""
        self.assertIsNotNone(self.hierarchical_ppo.completion_controller)
        self.assertIn('completion_controller', self.hierarchical_ppo.ppo_kwargs['policy_kwargs'])
        self.assertIsNone(self.hierarchical_ppo.ppo_model)
    
    @patch('npp_rl.agents.hierarchical_ppo.PPO')
    def test_create_model(self, mock_ppo_class):
        """Test model creation."""
        mock_env = Mock()
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        result = self.hierarchical_ppo.create_model(mock_env)
        
        # Check PPO was called with correct arguments
        mock_ppo_class.assert_called_once()
        call_kwargs = mock_ppo_class.call_args[1]
        self.assertEqual(call_kwargs['env'], mock_env)
        self.assertEqual(call_kwargs['policy'], self.hierarchical_ppo.policy_class)
        
        # Check model is stored
        self.assertEqual(self.hierarchical_ppo.ppo_model, mock_model)
        self.assertEqual(result, mock_model)
    
    def test_methods_require_model(self):
        """Test that methods require model to be created first."""
        with self.assertRaises(ValueError):
            self.hierarchical_ppo.learn()
        
        with self.assertRaises(ValueError):
            self.hierarchical_ppo.predict(np.zeros((1, 64, 64, 3)))
        
        with self.assertRaises(ValueError):
            self.hierarchical_ppo.save("test_model")
    
    @patch('npp_rl.agents.hierarchical_ppo.PPO')
    def test_methods_with_model(self, mock_ppo_class):
        """Test methods work after model creation."""
        mock_env = Mock()
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        # Create model
        self.hierarchical_ppo.create_model(mock_env)
        
        # Test learn
        self.hierarchical_ppo.learn(total_timesteps=1000)
        mock_model.learn.assert_called_once_with(total_timesteps=1000)
        
        # Test predict
        obs = np.zeros((1, 64, 64, 3))
        self.hierarchical_ppo.predict(obs)
        mock_model.predict.assert_called_once_with(obs)
        
        # Test save
        self.hierarchical_ppo.save("test_model")
        mock_model.save.assert_called_once_with("test_model")


if __name__ == '__main__':
    unittest.main()