"""Exploration reward calculator for evaluating area exploration and discovery."""
from typing import Dict, Any, Set, Tuple
from collections import deque
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator


class ExplorationRewardCalculator(BaseRewardCalculator):
    """Handles calculation of exploration and area-based rewards."""

    # Exploration-based rewards
    EXPLORATION_REWARD = 0.75
    STUCK_PENALTY = -0.5
    AREA_EXPLORATION_REWARD = 1.5
    NEW_TRANSITION_REWARD = 2.0
    BACKTRACK_PENALTY = -1.2
    LOCAL_MINIMA_PENALTY = -0.75
    AREA_REVISIT_DECAY = 0.4
    PROGRESSIVE_BACKTRACK_PENALTY = -0.3
    OBJECTIVE_DISTANCE_WEIGHT = 0.3

    # Add maximum penalty caps
    MAX_BACKTRACK_PENALTY = -3.0  # Maximum penalty for backtracking
    MAX_LOCAL_MINIMA_PENALTY = -2.0  # Maximum penalty for being stuck
    MAX_TOTAL_PENALTY = -5.0  # Maximum total negative reward per step

    def __init__(self):
        """Initialize exploration reward calculator."""
        super().__init__()
        # Grid sizes for position tracking
        self.position_grid_size = 10
        self.area_grid_size = 50

        # Exploration tracking
        self.exploration_memory = 1000
        self.stuck_threshold = 30
        self.local_minima_counter = 0
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.visited_areas: Set[Tuple[int, int]] = set()
        self.area_visit_counts = {}
        self.area_transition_points: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set(
        )

        # Path analysis
        self.prev_area = None
        self.path_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=100)
        self.distance_history = deque(maxlen=100)
        self.last_objective_distance = None

    def _get_grid_id(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous position to discrete grid position."""
        return (int(x / self.position_grid_size),
                int(y / self.position_grid_size))

    def _get_area_id(self, x: float, y: float) -> Tuple[int, int]:
        """Convert position to larger area grid position."""
        return (int(x / self.area_grid_size),
                int(y / self.area_grid_size))

    def _check_stuck_in_local_minima(self,
                                     curr_pos: Tuple[int, int],
                                     current_distance: float) -> bool:
        """Detect if agent is stuck in a local minimum.

        Args:
            curr_pos: Current grid position
            current_distance: Current distance to objective

        Returns:
            bool: Whether agent is stuck in local minimum
        """
        # Add current position and distance to history
        self.position_history.append(curr_pos)
        self.distance_history.append(current_distance)

        if len(self.position_history) < self.stuck_threshold:
            return False

        # Check position variety
        recent_positions = set(list(self.position_history)
                               [-self.stuck_threshold:])
        position_variety = len(recent_positions)

        # Check distance progress
        recent_distances = list(self.distance_history)[-self.stuck_threshold:]
        min_distance = min(recent_distances)
        max_distance = max(recent_distances)
        distance_progress = max_distance - min_distance

        # Determine if stuck
        is_stuck = (position_variety < 5 and
                    abs(distance_progress) < 50 and
                    len(recent_positions) >= self.stuck_threshold)

        if is_stuck:
            self.local_minima_counter += 1
        else:
            self.local_minima_counter = max(0, self.local_minima_counter - 1)

        return is_stuck

    def calculate_exploration_reward(self,
                                     curr_state: Dict[str, Any],
                                     prev_state: Dict[str, Any]) -> float:
        """Calculate comprehensive exploration reward.

        Args:
            curr_state: Current game state
            prev_state: Previous game state

        Returns:
            float: Total exploration reward
        """
        reward = 0.0
        current_pos = (curr_state['player_x'], curr_state['player_y'])
        grid_pos = self._get_grid_id(*current_pos)
        current_area = self._get_area_id(*current_pos)

        # Calculate current objective distance
        if not curr_state['switch_activated']:
            current_distance = self.calculate_distance_to_objective(
                curr_state['player_x'], curr_state['player_y'],
                curr_state['switch_x'], curr_state['switch_y']
            )
        else:
            current_distance = self.calculate_distance_to_objective(
                curr_state['player_x'], curr_state['player_y'],
                curr_state['exit_door_x'], curr_state['exit_door_y']
            )

        # Distance-weighted exploration reward for new positions
        if grid_pos not in self.visited_positions:
            # Reduced influence of distance to objective
            distance_factor = min(current_distance / 500.0,
                                  1.0) * self.OBJECTIVE_DISTANCE_WEIGHT
            # Base exploration reward is now more significant compared to distance factor
            exploration_reward = self.EXPLORATION_REWARD * \
                (1.0 + distance_factor)
            reward += exploration_reward
            self.visited_positions.add(grid_pos)

            if len(self.visited_positions) > self.exploration_memory:
                self.visited_positions.remove(
                    next(iter(self.visited_positions)))

        # Progressive backtracking penalty
        visit_count = self.area_visit_counts.get(current_area, 0)
        if visit_count > 0:
            # Increased penalty for revisits
            backtrack_penalty = self.PROGRESSIVE_BACKTRACK_PENALTY * \
                (visit_count ** 1.5)  # Exponential scaling
            reward += backtrack_penalty

        # Handle area transitions and exploration
        if self.prev_area != current_area:
            if current_area not in self.visited_areas:
                # Reduced influence of distance for new area rewards
                distance_factor = min(
                    current_distance / 500.0, 1.0) * self.OBJECTIVE_DISTANCE_WEIGHT
                area_reward = self.AREA_EXPLORATION_REWARD * \
                    (1.0 + distance_factor * 0.5)  # Further reduced distance influence
                reward += area_reward
                self.visited_areas.add(current_area)
            else:
                # Harsher revisit penalty
                visit_count = self.area_visit_counts.get(current_area, 0)
                revisit_penalty = self.AREA_EXPLORATION_REWARD * \
                    (self.AREA_REVISIT_DECAY **
                     (visit_count + 2))  # Increased decay power
                reward += revisit_penalty

            # Record transition point with reduced reward
            if self.prev_area is not None:
                transition_point = (self.prev_area, current_area)
                if transition_point not in self.area_transition_points:
                    # Only reward new transitions if they're not backtracking
                    if current_area not in list(self.path_history)[-20:]:
                        reward += self.NEW_TRANSITION_REWARD
                    self.area_transition_points.add(transition_point)

            self.area_visit_counts[current_area] = visit_count + 1

        # Enhanced backtracking detection with capped penalties
        if len(self.path_history) > 50:
            recent_areas = list(self.path_history)[-50:]
            area_count = len(set(recent_areas))
            if area_count < 3:
                if current_area in recent_areas[:-10]:
                    print(f"Backtracking detected in area {current_area}")
                    backtrack_count = recent_areas.count(current_area)
                    # Cap the backtrack penalty
                    backtrack_penalty = max(
                        # Reduced power from 1.5
                        self.BACKTRACK_PENALTY * (backtrack_count ** 1.2),
                        self.MAX_BACKTRACK_PENALTY
                    )
                    reward += backtrack_penalty

        # Track progress towards objective with reduced influence
        if self.last_objective_distance is not None:
            progress = self.last_objective_distance - current_distance
            if progress < -50:  # Significant movement away from objective
                # Reduced and capped penalty for moving away from objective
                reward += max(self.BACKTRACK_PENALTY * 0.3,
                              self.MAX_BACKTRACK_PENALTY)
            elif progress > 50:  # Significant movement towards objective
                reward += abs(self.BACKTRACK_PENALTY) * 0.1

        # Update history
        self.path_history.append(current_area)
        self.prev_area = current_area

        # Enhanced local minima detection with capped penalties
        if self._check_stuck_in_local_minima(grid_pos, current_distance):
            print(
                f"Stuck in local minima {grid_pos}. Counter: {self.local_minima_counter}")
            # Cap the local minima penalty
            penalty_scale = min(self.local_minima_counter **
                                1.2, 3.0)  # Reduced power and scale
            local_minima_penalty = max(
                self.LOCAL_MINIMA_PENALTY * penalty_scale,
                self.MAX_LOCAL_MINIMA_PENALTY
            )
            reward += local_minima_penalty

        # Final cap on total negative reward
        reward = max(reward, self.MAX_TOTAL_PENALTY)

        return reward

    def reset(self):
        """Reset internal state for new episode."""
        self.visited_positions.clear()
        self.visited_areas.clear()
        self.area_visit_counts.clear()
        self.area_transition_points.clear()
        self.prev_area = None
        self.path_history.clear()
        self.position_history.clear()
        self.distance_history.clear()
        self.local_minima_counter = 0
