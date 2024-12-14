"""Base reward calculator with constants and common utilities."""
from typing import Dict, Any
import numpy as np
from npp_rl.util.util import calculate_distance


class BaseRewardCalculator:
    """Base class containing reward constants and common utility methods."""

    # Base reward/penalty constants
    BASE_TIME_PENALTY = -0.01
    GOLD_COLLECTION_REWARD = 0.5
    SWITCH_ACTIVATION_REWARD = 10.0
    TERMINAL_REWARD = 25.0
    DEATH_PENALTY = -10.0
    TIMEOUT_PENALTY = -7.5

    # Movement assessment constants
    FINE_DISTANCE_THRESHOLD = 5.0

    # Distance-based reward scales
    DISTANCE_SCALE = 0.15
    APPROACH_REWARD_SCALE = 5.0
    RETREAT_PENALTY_SCALE = 0.3

    # Mine avoidance constants
    MINE_DANGER_RADIUS = 15.0
    MINE_PENALTY_SCALE = 0.5
    MAX_VELOCITY = 20000.0     # Maximum velocity magnitude for normalization

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

        # Normal time decrease
        time_penalty_scale = 1.0 / max(curr_state['time_remaining'], 1.0)
        return self.BASE_TIME_PENALTY * min(time_penalty_scale, 5.0)
