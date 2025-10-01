"""
Unit tests for the CompletionController class.

This module tests the hierarchical controller functionality including:
- Subtask selection logic
- Completion planner integration
- State transitions and metrics
"""

import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import time

from npp_rl.hrl.completion_controller import CompletionController, Subtask


class TestCompletionController(unittest.TestCase):
    """Test cases for CompletionController."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_planner = Mock()
        self.controller = CompletionController(completion_planner=self.mock_planner)
        
        # Mock observation and info
        self.mock_obs = {
            'observation': np.zeros((64, 64, 3)),
            'reachability_features': np.zeros(8)
        }
        self.mock_info = {
            'ninja_pos': (10, 10),
            'level_data': {'switches': [], 'doors': []},
            'switch_states': {},
        }
    
    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.current_subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)
        self.assertEqual(self.controller.subtask_step_count, 0)
        self.assertEqual(len(self.controller.subtask_history), 0)
    
    def test_get_subtask_features(self):
        """Test subtask feature encoding."""
        features = self.controller.get_subtask_features()
        
        # Should be 4-dimensional one-hot vector
        self.assertEqual(features.shape, (4,))
        self.assertEqual(np.sum(features), 1.0)
        
        # First subtask should be active
        self.assertEqual(features[0], 1.0)
    
    def test_subtask_transition(self):
        """Test subtask transition logic."""
        initial_subtask = self.controller.current_subtask
        
        # Force transition to different subtask
        self.controller._transition_to_subtask(Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH)
        
        # Check transition occurred
        self.assertNotEqual(self.controller.current_subtask, initial_subtask)
        self.assertEqual(self.controller.current_subtask, Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH)
        self.assertEqual(len(self.controller.subtask_history), 1)
        
        # Check transition record
        transition = self.controller.subtask_history[0]
        self.assertEqual(transition['from_subtask'], initial_subtask.name)
        self.assertEqual(transition['to_subtask'], Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH.name)
    
    def test_should_switch_subtask_step_limit(self):
        """Test subtask switching due to step limit."""
        # Set step count to exceed limit
        self.controller.subtask_step_count = self.controller.max_subtask_steps + 1
        
        should_switch = self.controller.should_switch_subtask(self.mock_obs, self.mock_info)
        self.assertTrue(should_switch)
    
    def test_should_switch_subtask_switch_activation(self):
        """Test subtask switching due to switch activation."""
        # Set up switch state change
        self.controller.last_switch_states = {'switch_1': False}
        self.mock_info['switch_states'] = {'switch_1': True}
        
        should_switch = self.controller.should_switch_subtask(self.mock_obs, self.mock_info)
        self.assertTrue(should_switch)
    
    def test_fallback_subtask_selection(self):
        """Test fallback subtask selection when planner fails."""
        # Test with no switches activated
        switch_states = {}
        reachability_features = np.zeros(8)
        
        subtask = self.controller._fallback_subtask_selection(switch_states, reachability_features)
        self.assertEqual(subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)
        
        # Test with switches activated
        switch_states = {'switch_1': True}
        subtask = self.controller._fallback_subtask_selection(switch_states, reachability_features)
        self.assertEqual(subtask, Subtask.NAVIGATE_TO_EXIT_DOOR)
    
    def test_step_updates_state(self):
        """Test that step() updates controller state."""
        initial_step_count = self.controller.subtask_step_count
        
        self.controller.step(self.mock_obs, self.mock_info)
        
        self.assertEqual(self.controller.subtask_step_count, initial_step_count + 1)
        self.assertEqual(self.controller.last_switch_states, self.mock_info['switch_states'])
        self.assertEqual(self.controller.last_ninja_pos, self.mock_info['ninja_pos'])
    
    def test_reset_clears_state(self):
        """Test that reset() clears controller state."""
        # Set some state
        self.controller.subtask_step_count = 100
        self.controller.last_switch_states = {'switch_1': True}
        self.controller.last_ninja_pos = (20, 20)
        self.controller.current_subtask = Subtask.NAVIGATE_TO_EXIT_DOOR
        
        # Reset
        self.controller.reset()
        
        # Check state is cleared
        self.assertEqual(self.controller.current_subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)
        self.assertEqual(self.controller.subtask_step_count, 0)
        self.assertEqual(self.controller.last_switch_states, {})
        self.assertIsNone(self.controller.last_ninja_pos)
    
    def test_get_subtask_metrics(self):
        """Test subtask metrics collection."""
        # Add some history
        self.controller._transition_to_subtask(Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH)
        self.controller._transition_to_subtask(Subtask.NAVIGATE_TO_EXIT_DOOR)
        
        metrics = self.controller.get_subtask_metrics()
        
        # Check required fields
        self.assertIn('current_subtask', metrics)
        self.assertIn('subtask_step_count', metrics)
        self.assertIn('subtask_duration', metrics)
        self.assertIn('total_transitions', metrics)
        self.assertIn('recent_transitions', metrics)
        
        # Check values
        self.assertEqual(metrics['current_subtask'], Subtask.NAVIGATE_TO_EXIT_DOOR.name)
        self.assertEqual(metrics['total_transitions'], 2)
        self.assertEqual(len(metrics['recent_transitions']), 2)
    
    def test_completion_planner_integration(self):
        """Test integration with completion planner."""
        # Mock completion planner response
        mock_strategy = Mock()
        mock_strategy.steps = [Mock()]
        mock_strategy.steps[0].action_type = "navigate_and_activate"
        mock_strategy.steps[0].description = "Navigate to exit switch"
        
        self.mock_planner.plan_completion.return_value = mock_strategy
        
        # Test subtask determination
        ninja_pos = (10, 10)
        level_data = {}
        switch_states = {}
        reachability_features = np.zeros(8)
        
        subtask = self.controller._determine_next_subtask(
            ninja_pos, level_data, switch_states, reachability_features
        )
        
        # Should call planner and return mapped subtask
        self.mock_planner.plan_completion.assert_called_once()
        self.assertEqual(subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)
    
    def test_completion_planner_failure_fallback(self):
        """Test fallback when completion planner fails."""
        # Mock planner to raise exception
        self.mock_planner.plan_completion.side_effect = Exception("Planner failed")
        
        ninja_pos = (10, 10)
        level_data = {}
        switch_states = {}
        reachability_features = np.zeros(8)
        
        subtask = self.controller._determine_next_subtask(
            ninja_pos, level_data, switch_states, reachability_features
        )
        
        # Should fall back to default logic
        self.assertEqual(subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)


class TestSubtaskEnum(unittest.TestCase):
    """Test cases for Subtask enum."""
    
    def test_subtask_values(self):
        """Test subtask enum values."""
        self.assertEqual(Subtask.NAVIGATE_TO_EXIT_SWITCH.value, 0)
        self.assertEqual(Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH.value, 1)
        self.assertEqual(Subtask.NAVIGATE_TO_EXIT_DOOR.value, 2)
        self.assertEqual(Subtask.AVOID_MINE.value, 3)
    
    def test_subtask_names(self):
        """Test subtask enum names."""
        self.assertEqual(Subtask.NAVIGATE_TO_EXIT_SWITCH.name, "NAVIGATE_TO_EXIT_SWITCH")
        self.assertEqual(Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH.name, "NAVIGATE_TO_LOCKED_DOOR_SWITCH")
        self.assertEqual(Subtask.NAVIGATE_TO_EXIT_DOOR.name, "NAVIGATE_TO_EXIT_DOOR")
        self.assertEqual(Subtask.AVOID_MINE.name, "AVOID_MINE")


if __name__ == '__main__':
    unittest.main()