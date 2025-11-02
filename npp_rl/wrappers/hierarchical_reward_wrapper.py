"""
Hierarchical Reward Wrapper

This wrapper integrates subtask-specific rewards with the base reward system
for hierarchical reinforcement learning. It combines base completion rewards
with dense subtask-aligned feedback to enable efficient learning of hierarchical
behaviors.

The wrapper:
1. Maintains subtask state (current active subtask)
2. Calculates subtask-specific dense rewards
3. Combines with base environment rewards
4. Provides logging for reward components
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple
from collections import deque

from npp_rl.hrl.subtask_rewards import SubtaskRewardCalculator
from npp_rl.hrl.high_level_policy import Subtask


class HierarchicalRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds subtask-specific rewards to base environment rewards.

    This wrapper augments the base reward signal with dense, subtask-aligned
    rewards that encourage efficient completion of hierarchical objectives.

    The wrapper expects the environment to provide subtask information either
    through the info dict or through an internal hierarchical controller.
    """

    def __init__(
        self,
        env: gym.Env,
        enable_mine_avoidance: bool = True,
        log_reward_components: bool = True,
        default_subtask: Subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH,
    ):
        """
        Initialize hierarchical reward wrapper.
        
        NOTE: PBRS is handled by base environment (nclone). This wrapper
        provides ONLY subtask-specific milestones and progress rewards.

        Args:
            env: Base environment
            enable_mine_avoidance: Whether to include mine avoidance rewards
            log_reward_components: Whether to log individual reward components
            default_subtask: Default subtask if none is provided
        """
        super().__init__(env)

        self.subtask_calculator = SubtaskRewardCalculator(
            enable_mine_avoidance=enable_mine_avoidance,
        )

        self.log_reward_components = log_reward_components
        self.default_subtask = default_subtask

        # Track previous observation for reward calculation
        self.prev_obs = None

        # Current subtask (updated from info or externally)
        self.current_subtask = default_subtask

        # Reward tracking for logging
        self.episode_base_reward = 0.0
        self.episode_subtask_reward = 0.0
        self.episode_length = 0

        # Track reward components over episode
        self.reward_component_history = {
            "base_rewards": deque(maxlen=1000),
            "subtask_rewards": deque(maxlen=1000),
            "total_rewards": deque(maxlen=1000),
        }

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment and reward calculator.

        Returns:
            Tuple of (observation, info dict)
        """
        obs, info = self.env.reset(**kwargs)

        # Reset subtask calculator
        self.subtask_calculator.reset()

        # Reset tracking
        self.prev_obs = self._extract_obs_dict(obs, info)
        self.current_subtask = self.default_subtask
        self.episode_base_reward = 0.0
        self.episode_subtask_reward = 0.0
        self.episode_length = 0

        # Add initial subtask to info
        info["current_subtask"] = self.current_subtask.value
        info["subtask_name"] = self.current_subtask.name

        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Take environment step and augment reward with subtask rewards.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Take step in base environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Extract observation dict for reward calculation
        obs_dict = self._extract_obs_dict(obs, info)

        # Update current subtask if provided in info
        if "current_subtask" in info:
            if isinstance(info["current_subtask"], Subtask):
                self.current_subtask = info["current_subtask"]
            elif isinstance(info["current_subtask"], int):
                self.current_subtask = Subtask(info["current_subtask"])

        # Calculate subtask reward
        subtask_reward = 0.0
        if self.prev_obs is not None:
            subtask_reward = self.subtask_calculator.calculate_subtask_reward(
                obs_dict,
                self.prev_obs,
                self.current_subtask,
            )

        # Combine rewards
        total_reward = base_reward + subtask_reward

        # Update tracking
        self.episode_base_reward += base_reward
        self.episode_subtask_reward += subtask_reward
        self.episode_length += 1

        # Store reward components
        self.reward_component_history["base_rewards"].append(base_reward)
        self.reward_component_history["subtask_rewards"].append(subtask_reward)
        self.reward_component_history["total_rewards"].append(total_reward)

        # Add reward breakdown to info
        if self.log_reward_components:
            info["reward_components"] = {
                "base_reward": base_reward,
                "subtask_reward": subtask_reward,
                "total_reward": total_reward,
                "current_subtask": self.current_subtask.value,
                "subtask_name": self.current_subtask.name,
            }

            # Add subtask-specific components
            subtask_components = self.subtask_calculator.get_subtask_components(
                self.current_subtask
            )
            info["subtask_components"] = subtask_components

        # Add episode statistics on termination
        if terminated or truncated:
            info["episode_statistics"] = {
                "total_base_reward": self.episode_base_reward,
                "total_subtask_reward": self.episode_subtask_reward,
                "total_combined_reward": self.episode_base_reward
                + self.episode_subtask_reward,
                "episode_length": self.episode_length,
                "avg_base_reward": self.episode_base_reward
                / max(1, self.episode_length),
                "avg_subtask_reward": self.episode_subtask_reward
                / max(1, self.episode_length),
            }
            # Add key that EnhancedTensorBoardCallback expects
            info["hierarchical_reward_episode"] = self.episode_subtask_reward

            # Calculate reward statistics
            if len(self.reward_component_history["base_rewards"]) > 0:
                info["reward_statistics"] = {
                    "base_reward_mean": np.mean(
                        list(self.reward_component_history["base_rewards"])
                    ),
                    "base_reward_std": np.std(
                        list(self.reward_component_history["base_rewards"])
                    ),
                    "subtask_reward_mean": np.mean(
                        list(self.reward_component_history["subtask_rewards"])
                    ),
                    "subtask_reward_std": np.std(
                        list(self.reward_component_history["subtask_rewards"])
                    ),
                    "total_reward_mean": np.mean(
                        list(self.reward_component_history["total_rewards"])
                    ),
                    "total_reward_std": np.std(
                        list(self.reward_component_history["total_rewards"])
                    ),
                }

        # Update previous observation
        self.prev_obs = obs_dict

        return obs, total_reward, terminated, truncated, info

    def _extract_obs_dict(self, obs: Any, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract observation dictionary for reward calculation.

        This method handles different observation formats and extracts the
        necessary information for reward calculation.

        Args:
            obs: Observation from environment
            info: Info dict from environment

        Returns:
            Dictionary with required observation fields
        """
        # Check if obs is a dict with the required raw fields
        # The observation processor transforms raw obs, removing player_x, player_y, etc.
        # We need to check for these fields before using obs directly
        if isinstance(obs, dict) and "player_x" in obs and "player_y" in obs:
            return obs

        # Try to extract from info (raw observation fields)
        if "game_state" in info:
            return info["game_state"]

        # Fallback: try to extract from info with common keys
        obs_dict = {}

        # Player position
        obs_dict["player_x"] = info.get("player_x", 0.0)
        obs_dict["player_y"] = info.get("player_y", 0.0)

        # Switch position and state
        obs_dict["switch_x"] = info.get("switch_x", 0.0)
        obs_dict["switch_y"] = info.get("switch_y", 0.0)
        obs_dict["switch_activated"] = info.get("switch_activated", False)

        # Exit position
        obs_dict["exit_door_x"] = info.get("exit_door_x", 0.0)
        obs_dict["exit_door_y"] = info.get("exit_door_y", 0.0)

        # Player state
        obs_dict["player_dead"] = info.get("player_dead", False)
        obs_dict["player_won"] = info.get("player_won", False)

        # Reachability features if available
        if "reachability_features" in info:
            obs_dict["reachability_features"] = info["reachability_features"]

        return obs_dict

    def set_subtask(self, subtask: Subtask):
        """
        Manually set the current subtask.

        This can be called externally to update the subtask when using
        a hierarchical controller.

        Args:
            subtask: New subtask to set
        """
        self.current_subtask = subtask

    def get_current_subtask(self) -> Subtask:
        """Get the current active subtask."""
        return self.current_subtask

    def get_reward_statistics(self) -> Dict[str, float]:
        """
        Get statistics about reward components.

        Returns:
            Dictionary of reward statistics
        """
        if len(self.reward_component_history["base_rewards"]) == 0:
            return {}

        return {
            "base_reward_mean": np.mean(
                list(self.reward_component_history["base_rewards"])
            ),
            "base_reward_std": np.std(
                list(self.reward_component_history["base_rewards"])
            ),
            "base_reward_min": np.min(
                list(self.reward_component_history["base_rewards"])
            ),
            "base_reward_max": np.max(
                list(self.reward_component_history["base_rewards"])
            ),
            "subtask_reward_mean": np.mean(
                list(self.reward_component_history["subtask_rewards"])
            ),
            "subtask_reward_std": np.std(
                list(self.reward_component_history["subtask_rewards"])
            ),
            "subtask_reward_min": np.min(
                list(self.reward_component_history["subtask_rewards"])
            ),
            "subtask_reward_max": np.max(
                list(self.reward_component_history["subtask_rewards"])
            ),
            "total_reward_mean": np.mean(
                list(self.reward_component_history["total_rewards"])
            ),
            "total_reward_std": np.std(
                list(self.reward_component_history["total_rewards"])
            ),
            "total_reward_min": np.min(
                list(self.reward_component_history["total_rewards"])
            ),
            "total_reward_max": np.max(
                list(self.reward_component_history["total_rewards"])
            ),
            "num_samples": len(self.reward_component_history["base_rewards"]),
        }


class SubtaskAwareRewardShaping:
    """
    Utility class for subtask-aware reward shaping.

    This provides helper methods for integrating subtask rewards into
    existing training pipelines without using a wrapper.
    """

    def __init__(
        self,
        enable_mine_avoidance: bool = True,
    ):
        """
        Initialize subtask-aware reward shaping.
        
        NOTE: PBRS is handled by base environment (nclone). This utility
        provides ONLY subtask-specific milestones and progress rewards.

        Args:
            enable_mine_avoidance: Whether to include mine avoidance rewards
        """
        self.calculator = SubtaskRewardCalculator(
            enable_mine_avoidance=enable_mine_avoidance,
        )

    def calculate_augmented_reward(
        self,
        base_reward: float,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        current_subtask: Subtask,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate augmented reward with subtask shaping.

        Args:
            base_reward: Base environment reward
            obs: Current observation
            prev_obs: Previous observation
            current_subtask: Current active subtask

        Returns:
            Tuple of (total_reward, reward_components_dict)
        """
        subtask_reward = self.calculator.calculate_subtask_reward(
            obs, prev_obs, current_subtask
        )

        total_reward = base_reward + subtask_reward

        components = {
            "base_reward": base_reward,
            "subtask_reward": subtask_reward,
            "total_reward": total_reward,
        }

        return total_reward, components

    def reset(self):
        """Reset the reward calculator."""
        self.calculator.reset()
