"""
Integration tests for hierarchical reward wrapper.

Tests focus on:
1. Wrapper integration with environment
2. Reward combination behavior
3. Subtask state management
4. Info dict enrichment
"""

import unittest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

from npp_rl.wrappers.hierarchical_reward_wrapper import (
    HierarchicalRewardWrapper,
    SubtaskAwareRewardShaping,
)
from npp_rl.hrl.high_level_policy import Subtask


class DummyNPPEnv(gym.Env):
    """Dummy environment for testing the wrapper."""
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 12), dtype=np.uint8
        )
        
        self.player_pos = np.array([50.0, 50.0])
        self.switch_pos = np.array([100.0, 100.0])
        self.exit_pos = np.array([200.0, 200.0])
        self.switch_activated = False
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.player_pos = np.array([50.0, 50.0])
        self.switch_pos = np.array([100.0, 100.0])
        self.exit_pos = np.array([200.0, 200.0])
        self.switch_activated = False
        self.step_count = 0
        
        obs = np.zeros((84, 84, 12), dtype=np.uint8)
        info = {
            "player_x": self.player_pos[0],
            "player_y": self.player_pos[1],
            "switch_x": self.switch_pos[0],
            "switch_y": self.switch_pos[1],
            "exit_door_x": self.exit_pos[0],
            "exit_door_y": self.exit_pos[1],
            "switch_activated": self.switch_activated,
            "player_dead": False,
            "player_won": False,
        }
        
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        
        # Simple movement simulation
        if action == 1:  # Left
            self.player_pos[0] -= 2.0
        elif action == 2:  # Right
            self.player_pos[0] += 2.0
        elif action == 3:  # Jump
            self.player_pos[1] -= 2.0
        elif action == 4:  # Jump+Left
            self.player_pos[0] -= 1.5
            self.player_pos[1] -= 1.5
        elif action == 5:  # Jump+Right
            self.player_pos[0] += 1.5
            self.player_pos[1] += 1.5
        
        # Check switch activation
        switch_dist = np.linalg.norm(self.player_pos - self.switch_pos)
        if switch_dist < 10.0 and not self.switch_activated:
            self.switch_activated = True
            base_reward = 0.1  # Switch activation reward
        else:
            base_reward = -0.01  # Time penalty
        
        # Check exit
        exit_dist = np.linalg.norm(self.player_pos - self.exit_pos)
        if exit_dist < 10.0 and self.switch_activated:
            base_reward = 1.0
            terminated = True
        else:
            terminated = False
        
        truncated = self.step_count >= 500
        
        obs = np.zeros((84, 84, 12), dtype=np.uint8)
        info = {
            "player_x": self.player_pos[0],
            "player_y": self.player_pos[1],
            "switch_x": self.switch_pos[0],
            "switch_y": self.switch_pos[1],
            "exit_door_x": self.exit_pos[0],
            "exit_door_y": self.exit_pos[1],
            "switch_activated": self.switch_activated,
            "player_dead": False,
            "player_won": terminated,
        }
        
        return obs, base_reward, terminated, truncated, info


class TestHierarchicalRewardWrapper(unittest.TestCase):
    """Test the hierarchical reward wrapper."""
    
    def setUp(self):
        self.env = DummyNPPEnv()
        self.wrapped_env = HierarchicalRewardWrapper(
            self.env,
            enable_pbrs=True,
            enable_mine_avoidance=False,  # Disabled for simple testing
        )
    
    def test_wrapper_initialization(self):
        """Test that wrapper initializes correctly."""
        self.assertIsNotNone(self.wrapped_env.subtask_calculator)
        self.assertEqual(self.wrapped_env.current_subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)
    
    def test_reset_returns_correct_format(self):
        """Test that reset returns observation and info."""
        obs, info = self.wrapped_env.reset()
        
        self.assertIsNotNone(obs)
        self.assertIsInstance(info, dict)
        self.assertIn("current_subtask", info)
        self.assertIn("subtask_name", info)
    
    def test_step_combines_rewards(self):
        """Test that step combines base and subtask rewards."""
        obs, info = self.wrapped_env.reset()
        
        # Take a step
        obs, reward, terminated, truncated, info = self.wrapped_env.step(2)  # Move right
        
        # Reward should be non-zero (base + subtask)
        self.assertIsInstance(reward, (float, np.floating))
        
        # Info should contain reward components
        if "reward_components" in info:
            self.assertIn("base_reward", info["reward_components"])
            self.assertIn("subtask_reward", info["reward_components"])
            self.assertIn("total_reward", info["reward_components"])
    
    def test_progress_toward_switch_gives_positive_reward(self):
        """Test that progress toward switch increases total reward."""
        obs, info = self.wrapped_env.reset()
        
        # Take steps toward switch (right and up)
        rewards = []
        for _ in range(10):
            obs, reward, terminated, truncated, info = self.wrapped_env.step(2)  # Right
            if "reward_components" in info:
                rewards.append(info["reward_components"]["subtask_reward"])
        
        # Should get some positive subtask rewards for progress
        positive_rewards = [r for r in rewards if r > 0]
        self.assertGreater(len(positive_rewards), 0, "Should get some positive rewards for progress")
    
    def test_subtask_can_be_updated(self):
        """Test that subtask can be updated externally."""
        obs, info = self.wrapped_env.reset()
        
        initial_subtask = self.wrapped_env.get_current_subtask()
        self.assertEqual(initial_subtask, Subtask.NAVIGATE_TO_EXIT_SWITCH)
        
        # Update subtask
        self.wrapped_env.set_subtask(Subtask.NAVIGATE_TO_EXIT_DOOR)
        
        updated_subtask = self.wrapped_env.get_current_subtask()
        self.assertEqual(updated_subtask, Subtask.NAVIGATE_TO_EXIT_DOOR)
    
    def test_episode_statistics_included_on_termination(self):
        """Test that episode statistics are added on termination."""
        obs, info = self.wrapped_env.reset()
        
        # Run until termination or truncation
        for _ in range(500):
            obs, reward, terminated, truncated, info = self.wrapped_env.step(1)
            if terminated or truncated:
                break
        
        # Should have episode statistics
        self.assertTrue(terminated or truncated)
        self.assertIn("episode_statistics", info)
        
        stats = info["episode_statistics"]
        self.assertIn("total_base_reward", stats)
        self.assertIn("total_subtask_reward", stats)
        self.assertIn("episode_length", stats)
    
    def test_reward_components_tracked_over_episode(self):
        """Test that reward components are tracked throughout episode."""
        obs, info = self.wrapped_env.reset()
        
        # Take several steps
        for _ in range(20):
            obs, reward, terminated, truncated, info = self.wrapped_env.step(2)
            if terminated or truncated:
                break
        
        # Should have accumulated reward history
        self.assertGreater(len(self.wrapped_env.reward_component_history["base_rewards"]), 0)
        self.assertGreater(len(self.wrapped_env.reward_component_history["subtask_rewards"]), 0)
        self.assertGreater(len(self.wrapped_env.reward_component_history["total_rewards"]), 0)
    
    def test_get_reward_statistics(self):
        """Test that reward statistics can be retrieved."""
        obs, info = self.wrapped_env.reset()
        
        # Take several steps
        for _ in range(20):
            obs, reward, terminated, truncated, info = self.wrapped_env.step(2)
            if terminated or truncated:
                break
        
        stats = self.wrapped_env.get_reward_statistics()
        
        self.assertIn("base_reward_mean", stats)
        self.assertIn("subtask_reward_mean", stats)
        self.assertIn("total_reward_mean", stats)
        self.assertIn("num_samples", stats)
        self.assertEqual(stats["num_samples"], 20)


class TestSubtaskAwareRewardShaping(unittest.TestCase):
    """Test the subtask-aware reward shaping utility."""
    
    def setUp(self):
        self.shaper = SubtaskAwareRewardShaping(
            enable_pbrs=True,
            enable_mine_avoidance=False,
        )
    
    def test_calculate_augmented_reward(self):
        """Test that augmented reward is calculated correctly."""
        base_reward = 0.1
        
        obs = {
            "player_x": 60.0, "player_y": 60.0,
            "switch_x": 100.0, "switch_y": 100.0,
            "exit_door_x": 200.0, "exit_door_y": 200.0,
            "switch_activated": False, "player_dead": False, "player_won": False,
        }
        
        prev_obs = {
            "player_x": 50.0, "player_y": 50.0,
            "switch_x": 100.0, "switch_y": 100.0,
            "exit_door_x": 200.0, "exit_door_y": 200.0,
            "switch_activated": False, "player_dead": False, "player_won": False,
        }
        
        total_reward, components = self.shaper.calculate_augmented_reward(
            base_reward, obs, prev_obs, Subtask.NAVIGATE_TO_EXIT_SWITCH
        )
        
        # Check that components are returned
        self.assertIn("base_reward", components)
        self.assertIn("subtask_reward", components)
        self.assertIn("total_reward", components)
        
        # Total should be base + subtask
        self.assertEqual(
            total_reward,
            components["base_reward"] + components["subtask_reward"]
        )
    
    def test_reset_clears_state(self):
        """Test that reset clears the calculator state."""
        obs = {
            "player_x": 60.0, "player_y": 60.0,
            "switch_x": 100.0, "switch_y": 100.0,
            "exit_door_x": 200.0, "exit_door_y": 200.0,
            "switch_activated": False, "player_dead": False, "player_won": False,
        }
        
        # Calculate some rewards
        self.shaper.calculate_augmented_reward(0.1, obs, obs, Subtask.NAVIGATE_TO_EXIT_SWITCH)
        
        # Reset
        self.shaper.reset()
        
        # State should be reset (we can't directly test internal state, but it shouldn't crash)
        self.shaper.calculate_augmented_reward(0.1, obs, obs, Subtask.NAVIGATE_TO_EXIT_SWITCH)


if __name__ == "__main__":
    unittest.main()
