from typing import Dict, Any, Tuple, List
import numpy as np
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator
from npp_rl.environments.planning.waypoint_manager import WaypointManager, WaypointMetrics


class PlanningRewardCalculator(BaseRewardCalculator):
    """Calculates rewards based on planning and path following performance.

    This calculator evaluates the agent's ability to:
    1. Follow planned paths efficiently
    2. Reach waypoints in sequence
    3. Maintain progress towards objectives
    4. Handle dynamic replanning
    """

    # Planning reward constants
    WAYPOINT_REACHED_REWARD = 0.5  # Increased for more emphasis on reaching waypoints
    PATH_PROGRESS_SCALE = 0.3  # Scaled for balanced progress rewards
    # Adjusted for more reasonable path deviation penalties
    DEVIATION_PENALTY_SCALE = -0.4
    BACKTRACK_PENALTY = -0.3  # Reduced slightly to balance with other penalties
    DISTANCE_PENALTY_SCALE = -0.2  # New constant for distance-based penalties
    PROGRESS_REWARD_SCALE = 0.25  # New constant for progress-based rewards
    # Minimum progress (in pixels) to be considered meaningful
    MIN_PROGRESS_THRESHOLD = 1.0

    def __init__(self, waypoint_manager: WaypointManager):
        """Initialize planning reward calculator."""
        super().__init__()
        self.waypoint_manager = waypoint_manager
        self.last_metrics: WaypointMetrics = None
        self.replan_count = 0
        self.consecutive_waypoints = 0

    def calculate_planning_reward(self,
                                  curr_state: Dict[str, Any],
                                  prev_state: Dict[str, Any]) -> float:
        """Calculate comprehensive planning-based reward."""
        # Get current position
        current_pos = (curr_state['player_x'], curr_state['player_y'])
        prev_pos = (prev_state['player_x'], prev_state['player_y'])

        # Calculate waypoint metrics
        metrics = self.waypoint_manager.calculate_metrics(
            current_pos, prev_pos)

        # Get level dimensions for normalization
        level_width = curr_state.get('level_width', 1032.0)
        level_height = curr_state.get('level_height', 576.0)
        level_diagonal = np.sqrt(level_width**2 + level_height**2)

        # Initialize reward components
        waypoint_reward = 0.0
        progress_reward = 0.0
        deviation_penalty = 0.0
        distance_penalty = 0.0

        # Update waypoint first if reached
        if metrics.waypoint_reached:
            current_waypoint = self.waypoint_manager.get_current_waypoint()
            if current_waypoint and not self.waypoint_manager.is_backtracking(current_waypoint):
                self.consecutive_waypoints += 1
                capped_consecutive = min(self.consecutive_waypoints, 10)
                waypoint_reward = self.WAYPOINT_REACHED_REWARD * \
                    min(1.1 ** capped_consecutive, 10.0)

                # Update waypoint immediately when reached
                self.waypoint_manager.update_waypoint(
                    current_pos, prev_pos, metrics)
            else:
                waypoint_reward = self.BACKTRACK_PENALTY
                self.consecutive_waypoints = 0
        else:
            self.consecutive_waypoints = max(0, self.consecutive_waypoints - 1)

        # Calculate normalized distance for distance penalty
        normalized_distance = metrics.distance_to_waypoint / level_diagonal
        distance_penalty = self.DISTANCE_PENALTY_SCALE * normalized_distance

        # Calculate progress reward based on metrics
        if metrics.progress_to_waypoint > 0:
            normalized_progress = metrics.progress_to_waypoint / level_diagonal
            progress_reward = self.PROGRESS_REWARD_SCALE * normalized_progress

        # Calculate path deviation penalty
        if metrics.path_deviation > 0:
            normalized_deviation = metrics.path_deviation / level_diagonal
            deviation_penalty = self.DEVIATION_PENALTY_SCALE * normalized_deviation

        # Combine all reward components
        total_reward = (
            waypoint_reward +
            progress_reward +
            distance_penalty +
            deviation_penalty
        )

        # Store metrics for next iteration
        self.last_metrics = metrics

        # Ensure final reward is bounded
        final_reward = np.clip(total_reward, -5.0, 5.0)

        return final_reward

    def update_path(self, new_path: List[Tuple[float, float]]):
        """Update current path and track replanning."""
        self.waypoint_manager.update_path(new_path)
        self.replan_count += 1
        self.consecutive_waypoints = 0

    def get_planning_features(self) -> np.ndarray:
        """Get planning-related features for observation space."""
        if not self.last_metrics:
            return np.zeros(4, dtype=np.float32)

        # Get level dimensions for normalization (use typical values)
        level_diagonal = np.sqrt(1032.0**2 + 576.0**2)

        # Normalize values by level diagonal
        distance = np.clip(
            self.last_metrics.distance_to_waypoint / level_diagonal, 0.0, 1.0)
        progress = np.clip(
            self.last_metrics.progress_to_waypoint / level_diagonal, -1.0, 1.0)
        deviation = np.clip(
            self.last_metrics.path_deviation / level_diagonal, 0.0, 1.0)
        consecutive = np.clip(
            float(self.consecutive_waypoints) / 10.0, 0.0, 1.0)

        # Create array with normalized values
        features = np.array([
            distance,
            progress,
            deviation,
            consecutive
        ], dtype=np.float32)

        return features

    def reset(self, waypoint_manager: WaypointManager):
        """Reset internal state for new episode."""
        self.waypoint_manager = waypoint_manager
        self.last_metrics = None
        self.replan_count = 0
        self.consecutive_waypoints = 0
