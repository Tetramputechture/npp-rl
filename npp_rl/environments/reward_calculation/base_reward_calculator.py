"""Base reward calculator with constants and common utilities."""
from typing import Dict, Any, List, Tuple
import numpy as np
from npp_rl.util.util import calculate_distance


class BaseRewardCalculator:
    """Base class containing reward constants and common utility methods."""

    # Base reward/penalty constants
    BASE_TIME_PENALTY = -0.005
    GOLD_COLLECTION_REWARD = 0.5
    SWITCH_ACTIVATION_REWARD = 10.0
    TERMINAL_REWARD = 25.0
    DEATH_PENALTY = -5.0
    TIMEOUT_PENALTY = -7.5

    # Movement assessment constants
    FINE_DISTANCE_THRESHOLD = 5.0

    # Distance-based reward scales
    DISTANCE_SCALE = 0.05
    APPROACH_REWARD_SCALE = 2.5
    RETREAT_PENALTY_SCALE = 0.1

    # Mine avoidance constants
    MINE_DANGER_RADIUS = 50.0  # Distance at which mines start affecting rewards
    MINE_PENALTY_SCALE = 0.15  # Scale factor for mine proximity penalties
    MINE_MIN_DISTANCE = 15.0  # Minimum safe distance from mines

    def __init__(self):
        """Initialize base reward calculator."""
        self.gamma = 0.99  # Discount factor for potential-based shaping

    def calculate_distance_to_objective(self,
                                        player_x: float,
                                        player_y: float,
                                        objective_x: float,
                                        objective_y: float) -> float:
        """Calculate distance between player and an objective."""
        return calculate_distance(player_x, player_y, objective_x, objective_y)

    def calculate_movement_vector(self,
                                  curr_state: Dict[str, Any],
                                  prev_state: Dict[str, Any]) -> np.ndarray:
        """Calculate movement vector between two states."""
        return np.array([
            curr_state['player_x'] - prev_state['player_x'],
            curr_state['player_y'] - prev_state['player_y']
        ])

    def calculate_time_reward(self,
                              curr_state: Dict[str, Any],
                              prev_state: Dict[str, Any]) -> float:
        """Calculate time-based rewards/penalties."""
        time_diff = curr_state['time_remaining'] - prev_state['time_remaining']
        if time_diff > 0:  # Collected gold
            return self.GOLD_COLLECTION_REWARD * time_diff
        else:  # Normal time decrease
            time_penalty_scale = 1.0 / max(curr_state['time_remaining'], 1.0)
            return self.BASE_TIME_PENALTY * min(time_penalty_scale, 5.0)

    def calculate_mine_proximity_penalty(self,
                                         player_x: float,
                                         player_y: float,
                                         mine_coords: List[Tuple[float, float]]) -> float:
        """Calculate penalty based on proximity to nearest mine.

        Args:
            player_x: Player's x coordinate
            player_y: Player's y coordinate
            mine_coords: List of (x, y) coordinates of mines

        Returns:
            float: Penalty value (negative or zero)
        """
        if not mine_coords:
            return 0.0

        # Find distance to nearest mine
        min_distance = min(
            calculate_distance(player_x, player_y, mine_x, mine_y)
            for mine_x, mine_y in mine_coords
        )

        # No penalty if outside danger radius
        if min_distance > self.MINE_DANGER_RADIUS:
            return 0.0

        # Calculate penalty based on proximity
        danger_factor = 1.0 - (min_distance / self.MINE_DANGER_RADIUS)
        penalty = -self.MINE_PENALTY_SCALE * danger_factor

        # Increase penalty exponentially when very close to mines
        if min_distance < self.MINE_MIN_DISTANCE:
            close_factor = 1.0 - (min_distance / self.MINE_MIN_DISTANCE)
            penalty *= (1.0 + close_factor * 2.0)

        return penalty
