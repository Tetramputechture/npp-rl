"""
Exploration metrics for evaluating agent exploration behavior.

This module provides metrics to quantify how well agents explore
their environment, particularly useful for evaluating intrinsic
motivation methods like ICM.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict, deque
import math

from nclone.constants.physics_constants import (
    TILE_PIXEL_SIZE,
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
)


class ExplorationMetrics:
    """
    Computes and tracks exploration metrics for RL agents.

    Metrics include:
    - Unique tile coverage
    - Visitation entropy
    - Intrinsic reward statistics
    - Success rates on complex levels
    """

    def __init__(
        self,
        window_size: int = 100,
    ):
        """
        Initialize exploration metrics tracker.

        Args:
            window_size: Window size for rolling metrics
        """
        self.window_size = window_size

        # Episode-level metrics
        self.episode_metrics = []

        # Rolling metrics
        self.rolling_metrics = {
            "coverage": deque(maxlen=window_size),
            "entropy": deque(maxlen=window_size),
            "intrinsic_reward": deque(maxlen=window_size),
            "success_rate": deque(maxlen=window_size),
        }

        # Current episode tracking
        self.reset_episode()

    def reset_episode(self):
        """Reset tracking for a new episode."""
        self.visited_cells = set()
        self.position_history = []
        self.intrinsic_rewards = []
        self.episode_length = 0
        self.episode_success = False

    def update_step(
        self,
        position: Tuple[float, float],
        intrinsic_reward: float = 0.0,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Update metrics for a single step.

        Args:
            position: Agent position (x, y)
            intrinsic_reward: Intrinsic reward for this step
            info: Additional info from environment
        """
        # Convert position to grid cell
        cell_x = int(position[0] // TILE_PIXEL_SIZE)
        cell_y = int(position[1] // TILE_PIXEL_SIZE)

        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, MAP_TILE_WIDTH - 1))
        cell_y = max(0, min(cell_y, MAP_TILE_HEIGHT - 1))

        # Track visited cells
        self.visited_cells.add((cell_x, cell_y))

        # Track position history
        self.position_history.append((cell_x, cell_y))

        # Track intrinsic rewards
        self.intrinsic_rewards.append(intrinsic_reward)

        self.episode_length += 1

    def end_episode(self, success: bool = False) -> Dict[str, float]:
        """
        End episode and compute metrics.

        Args:
            success: Whether the episode was successful

        Returns:
            Dictionary of episode metrics
        """
        self.episode_success = success

        # Compute episode metrics
        metrics = self._compute_episode_metrics()

        # Store episode metrics
        self.episode_metrics.append(metrics)

        # Update rolling metrics
        self.rolling_metrics["coverage"].append(metrics["coverage"])
        self.rolling_metrics["entropy"].append(metrics["visitation_entropy"])
        self.rolling_metrics["intrinsic_reward"].append(
            metrics["mean_intrinsic_reward"]
        )
        self.rolling_metrics["success_rate"].append(float(success))

        return metrics

    def _compute_episode_metrics(self) -> Dict[str, float]:
        """Compute metrics for the current episode."""
        metrics = {}

        # Coverage metrics
        total_traversable_cells = MAP_TILE_WIDTH * MAP_TILE_HEIGHT  # Simplified
        unique_cells_visited = len(self.visited_cells)
        metrics["coverage"] = unique_cells_visited / total_traversable_cells
        metrics["unique_cells_visited"] = unique_cells_visited

        # Visitation entropy
        metrics["visitation_entropy"] = self._compute_visitation_entropy()

        # Intrinsic reward metrics
        if self.intrinsic_rewards:
            metrics["mean_intrinsic_reward"] = np.mean(self.intrinsic_rewards)
            metrics["total_intrinsic_reward"] = np.sum(self.intrinsic_rewards)
            metrics["max_intrinsic_reward"] = np.max(self.intrinsic_rewards)
        else:
            metrics["mean_intrinsic_reward"] = 0.0
            metrics["total_intrinsic_reward"] = 0.0
            metrics["max_intrinsic_reward"] = 0.0

        # Episode length
        metrics["episode_length"] = self.episode_length

        # Success
        metrics["success"] = float(self.episode_success)

        return metrics

    def _compute_visitation_entropy(self) -> float:
        """Compute entropy of position visitations."""
        if not self.position_history:
            return 0.0

        # Count visitations per cell
        cell_counts = defaultdict(int)
        for cell in self.position_history:
            cell_counts[cell] += 1

        # Compute probabilities
        total_visits = len(self.position_history)
        if total_visits == 0:
            return 0.0
        
        probabilities = [count / total_visits for count in cell_counts.values()]

        # Compute entropy
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        return entropy

    def get_rolling_metrics(self) -> Dict[str, float]:
        """Get rolling average metrics."""
        metrics = {}

        for key, values in self.rolling_metrics.items():
            if values:
                metrics[f"rolling_{key}"] = np.mean(values)
                metrics[f"rolling_{key}_std"] = np.std(values)
            else:
                metrics[f"rolling_{key}"] = 0.0
                metrics[f"rolling_{key}_std"] = 0.0

        return metrics

    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics across all episodes."""
        if not self.episode_metrics:
            return {}

        stats = {}

        # Aggregate metrics across episodes
        for key in self.episode_metrics[0].keys():
            values = [ep[key] for ep in self.episode_metrics]
            stats[f"{key}_mean"] = np.mean(values)
            stats[f"{key}_std"] = np.std(values)
            stats[f"{key}_min"] = np.min(values)
            stats[f"{key}_max"] = np.max(values)

        # Additional statistics
        stats["total_episodes"] = len(self.episode_metrics)
        stats["success_rate"] = np.mean([ep["success"] for ep in self.episode_metrics])

        return stats

    def get_tensorboard_scalars(self) -> Dict[str, float]:
        """Get metrics formatted for TensorBoard logging."""
        scalars = {}

        # Latest episode metrics
        if self.episode_metrics:
            latest = self.episode_metrics[-1]
            for key, value in latest.items():
                scalars[f"exploration/{key}"] = value

        # Rolling metrics
        rolling = self.get_rolling_metrics()
        for key, value in rolling.items():
            scalars[f"exploration/{key}"] = value

        return scalars


class LevelComplexityMetrics:
    """
    Metrics for evaluating performance on levels of different complexity.

    Tracks success rates and exploration metrics separately for:
    - Simple levels (direct path to goal)
    - Complex levels (requiring backtracking, switches, etc.)
    - Maze-like levels (complex navigation)
    """

    def __init__(self):
        """Initialize level complexity metrics."""
        self.level_metrics = {
            "simple": ExplorationMetrics(),
            "complex": ExplorationMetrics(),
            "maze": ExplorationMetrics(),
        }

        self.level_counts = defaultdict(int)
        self.level_successes = defaultdict(int)

    def update_step(
        self,
        level_type: str,
        position: Tuple[float, float],
        intrinsic_reward: float = 0.0,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Update metrics for a step on a specific level type."""
        if level_type in self.level_metrics:
            self.level_metrics[level_type].update_step(position, intrinsic_reward, info)

    def end_episode(self, level_type: str, success: bool = False) -> Dict[str, float]:
        """End episode for a specific level type."""
        self.level_counts[level_type] += 1
        if success:
            self.level_successes[level_type] += 1

        if level_type in self.level_metrics:
            return self.level_metrics[level_type].end_episode(success)

        return {}

    def reset_episode(self, level_type: str):
        """Reset episode tracking for a level type."""
        if level_type in self.level_metrics:
            self.level_metrics[level_type].reset_episode()

    def get_level_statistics(self) -> Dict[str, Any]:
        """Get statistics by level type."""
        stats = {}

        for level_type, metrics in self.level_metrics.items():
            level_stats = metrics.get_episode_statistics()

            # Add level-specific success rate
            if self.level_counts[level_type] > 0:
                level_stats["overall_success_rate"] = (
                    self.level_successes[level_type] / self.level_counts[level_type]
                )
            else:
                level_stats["overall_success_rate"] = 0.0

            level_stats["total_attempts"] = self.level_counts[level_type]
            level_stats["total_successes"] = self.level_successes[level_type]

            stats[level_type] = level_stats

        return stats

    def get_tensorboard_scalars(self) -> Dict[str, float]:
        """Get metrics formatted for TensorBoard logging."""
        scalars = {}

        for level_type, metrics in self.level_metrics.items():
            level_scalars = metrics.get_tensorboard_scalars()

            # Rename keys to include level type
            for key, value in level_scalars.items():
                new_key = key.replace("exploration/", f"exploration/{level_type}_")
                scalars[new_key] = value

            # Add overall success rate
            if self.level_counts[level_type] > 0:
                success_rate = (
                    self.level_successes[level_type] / self.level_counts[level_type]
                )
                scalars[f"success_rate/{level_type}"] = success_rate

        return scalars


def create_exploration_callback(metrics: ExplorationMetrics, log_frequency: int = 1000):
    """
    Create a callback for logging exploration metrics during training.

    Args:
        metrics: ExplorationMetrics instance
        log_frequency: How often to log metrics (in steps)

    Returns:
        Callback function compatible with SB3
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class ExplorationCallback(BaseCallback):
        def __init__(self, exploration_metrics, log_freq):
            super().__init__()
            self.exploration_metrics = exploration_metrics
            self.log_freq = log_freq
            self.episode_positions = []
            self.episode_intrinsic_rewards = []

        def _on_step(self) -> bool:
            # Extract position and intrinsic reward from info
            info = self.locals.get("infos", [{}])[0]

            # Get position (would need to be added to env info)
            if "player_x" in info and "player_y" in info:
                position = (info["player_x"], info["player_y"])
                self.episode_positions.append(position)

                # Update metrics
                intrinsic_reward = info.get("r_int", 0.0)
                self.exploration_metrics.update_step(position, intrinsic_reward, info)
                self.episode_intrinsic_rewards.append(intrinsic_reward)

            # Check for episode end
            if self.locals.get("dones", [False])[0]:
                success = info.get("success", False)
                self.exploration_metrics.end_episode(success)

                # Log episode metrics
                if self.num_timesteps % self.log_freq == 0:
                    scalars = self.exploration_metrics.get_tensorboard_scalars()
                    for key, value in scalars.items():
                        self.logger.record(key, value)

                # Reset for next episode
                self.exploration_metrics.reset_episode()
                self.episode_positions = []
                self.episode_intrinsic_rewards = []

            return True

    return ExplorationCallback(metrics, log_frequency)
