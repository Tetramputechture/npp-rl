"""Navigation reward calculator for evaluating objective-based movement and progress."""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator
from npp_rl.util.util import calculate_velocity


class NavigationRewardCalculator(BaseRewardCalculator):
    """Handles calculation of navigation and objective-based rewards."""

    # Navigation constants
    DISTANCE_IMPROVEMENT_SCALE = 5.0  # Increased scale for distance improvements
    # Increased bonus for consecutive improvements
    CONSECUTIVE_IMPROVEMENT_BONUS = 0.4
    MOMENTUM_BONUS = 3.0  # Increased momentum bonus
    MOMENTUM_PENALTY = -0.2  # Reduced momentum penalty
    MIN_DISTANCE_THRESHOLD = 75.0  # Increased threshold for close proximity rewards
    PROXIMITY_BONUS_SCALE = 5.0  # Increased scale for proximity bonuses

    def __init__(self):
        """Initialize navigation reward calculator."""
        super().__init__()
        self.prev_improvement = None
        self.min_distance_to_switch = float('inf')
        self.min_distance_to_exit = float('inf')
        self.prev_distance_to_switch = float('inf')
        self.prev_distance_to_exit = float('inf')
        self.first_switch_distance_update = True
        self.first_exit_distance_update = True
        self.prev_potential = None
        self.consecutive_improvements = 0
        self.mine_coords: List[Tuple[float, float]] = []

        # Training progress tracking
        self.total_steps = 0
        self.early_training_threshold = 50000

        # Progress tracking for shaping
        self.best_switch_distance = float('inf')
        self.best_exit_distance = float('inf')
        self.episode_start_switch_distance = None
        self.episode_start_exit_distance = None

    def _get_penalty_scale(self) -> float:
        """Calculate penalty scaling based on training progress."""
        if self.total_steps < self.early_training_threshold:
            return 0.3 + (0.7 * (self.total_steps / self.early_training_threshold))
        return 1.0

    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """Calculate enhanced state potential for reward shaping."""
        # Dynamic scaling based on level size and progress
        level_diagonal = np.sqrt(
            state['level_width']**2 + state['level_height']**2)
        # Reduced scale for sharper potential gradients
        distance_scale = level_diagonal / 4

        # Calculate progress-based potential scaling
        if not state['switch_activated']:
            distance_to_switch = self.calculate_distance_to_objective(
                state['player_x'], state['player_y'],
                state['switch_x'], state['switch_y']
            )
            # Progress-based scaling with enhanced bonuses
            if self.episode_start_switch_distance is not None:
                progress = 1.0 - (distance_to_switch /
                                  self.episode_start_switch_distance)
                # Increased progress bonus
                progress_bonus = 3.0 * max(0, progress)  # Increased from 2.0
            else:
                progress_bonus = 0.0

            # Enhanced exponential potential with progress bonus
            switch_potential = (8.0 * np.exp(-distance_to_switch / distance_scale) +
                                progress_bonus)  # Increased base potential

            # Larger bonus for new best distances
            if distance_to_switch < self.best_switch_distance:
                switch_potential += 8.0  # Increased from 5.0
                self.best_switch_distance = distance_to_switch

            # Add proximity bonus for being close to switch
            if distance_to_switch < self.MIN_DISTANCE_THRESHOLD:
                proximity_bonus = self.PROXIMITY_BONUS_SCALE * \
                    (1.0 - distance_to_switch / self.MIN_DISTANCE_THRESHOLD)
                switch_potential += proximity_bonus

            return switch_potential
        else:
            distance_to_exit = self.calculate_distance_to_objective(
                state['player_x'], state['player_y'],
                state['exit_door_x'], state['exit_door_y']
            )
            # Progress-based scaling with enhanced bonuses
            if self.episode_start_exit_distance is not None:
                progress = 1.0 - (distance_to_exit /
                                  self.episode_start_exit_distance)
                # Larger bonus for exit progress
                progress_bonus = 3.0 * max(0, progress)  # Increased from 2.0
            else:
                progress_bonus = 0.0

            # Enhanced exponential potential with progress bonus
            exit_potential = (8.0 * np.exp(-distance_to_exit / distance_scale) +
                              8.0 + progress_bonus)  # Increased base potentials

            # Larger bonus for new best distances
            if distance_to_exit < self.best_exit_distance:
                exit_potential += 8.0  # Increased from 5.0
                self.best_exit_distance = distance_to_exit

            # Add proximity bonus for being close to exit
            if distance_to_exit < self.MIN_DISTANCE_THRESHOLD:
                proximity_bonus = self.PROXIMITY_BONUS_SCALE * \
                    (1.0 - distance_to_exit / self.MIN_DISTANCE_THRESHOLD)
                exit_potential += proximity_bonus

            return exit_potential

    def calculate_navigation_reward(self,
                                    curr_state: Dict[str, Any],
                                    prev_state: Dict[str, Any],
                                    navigation_scale: float) -> Tuple[float, bool]:
        """Calculate comprehensive navigation reward with enhanced shaping."""
        self.total_steps += 1
        reward = 0.0
        switch_activated = False

        # Calculate current distances
        curr_distance_to_switch = self.calculate_distance_to_objective(
            curr_state['player_x'], curr_state['player_y'],
            curr_state['switch_x'], curr_state['switch_y']
        )
        curr_distance_to_exit = self.calculate_distance_to_objective(
            curr_state['player_x'], curr_state['player_y'],
            curr_state['exit_door_x'], curr_state['exit_door_y']
        )

        # Initialize episode start distances
        if self.episode_start_switch_distance is None:
            self.episode_start_switch_distance = curr_distance_to_switch
        if curr_state['switch_activated'] and self.episode_start_exit_distance is None:
            self.episode_start_exit_distance = curr_distance_to_exit

        if not curr_state['switch_activated']:
            # Enhanced navigation to switch with progress-based scaling
            progress = 1.0 - (curr_distance_to_switch /
                              self.episode_start_switch_distance)
            # Scale rewards based on overall progress
            progress_scale = 1.0 + max(0, progress)

            navigation_reward = self.evaluate_navigation_quality(
                curr_distance_to_switch,
                self.prev_distance_to_switch,
                navigation_scale * progress_scale,
                curr_state, prev_state
            )
            reward += navigation_reward

            # Progressive rewards for new minimum distances to switch
            if curr_distance_to_switch < self.min_distance_to_switch:
                improvement = self.min_distance_to_switch - curr_distance_to_switch
                if not self.first_switch_distance_update:
                    reward += improvement * navigation_scale * progress_scale
                self.min_distance_to_switch = curr_distance_to_switch
                self.first_switch_distance_update = False

        else:
            # Enhanced switch activation reward
            if not prev_state['switch_activated']:
                reward += self.SWITCH_ACTIVATION_REWARD
                switch_activated = True
                self.min_distance_to_exit = curr_distance_to_exit
                self.first_exit_distance_update = True
                self.consecutive_improvements = 0
                self.episode_start_exit_distance = curr_distance_to_exit

            # Enhanced navigation to exit with progress-based scaling
            if self.episode_start_exit_distance is not None:
                progress = 1.0 - (curr_distance_to_exit /
                                  self.episode_start_exit_distance)
                progress_scale = 1.0 + max(0, progress)
            else:
                progress_scale = 1.0

            navigation_reward = self.evaluate_navigation_quality(
                curr_distance_to_exit,
                self.prev_distance_to_exit,
                navigation_scale * progress_scale,
                curr_state, prev_state
            )
            reward += navigation_reward

            # Progressive rewards for new minimum distances to exit
            if curr_distance_to_exit < self.min_distance_to_exit:
                improvement = self.min_distance_to_exit - curr_distance_to_exit
                if not self.first_exit_distance_update:
                    reward += improvement * navigation_scale * progress_scale
                self.min_distance_to_exit = curr_distance_to_exit
                self.first_exit_distance_update = False

        # Calculate potential-based shaping reward
        current_potential = self.calculate_potential(curr_state)
        if self.prev_potential is not None:
            shaping_reward = self.gamma * current_potential - self.prev_potential
            reward += shaping_reward
        self.prev_potential = current_potential

        # Add small survival reward that scales with progress
        if not curr_state['switch_activated']:
            progress = 1.0 - (curr_distance_to_switch /
                              self.episode_start_switch_distance)
        else:
            progress = 1.0 - (curr_distance_to_exit /
                              self.episode_start_exit_distance)
        survival_reward = 0.05 * (1.0 + max(0, progress))
        reward += survival_reward

        # Update previous distances
        self.prev_distance_to_switch = curr_distance_to_switch
        self.prev_distance_to_exit = curr_distance_to_exit

        return reward, switch_activated

    def reset(self):
        """Reset internal state for new episode."""
        self.prev_improvement = None
        self.min_distance_to_switch = float('inf')
        self.min_distance_to_exit = float('inf')
        self.prev_distance_to_switch = float('inf')
        self.prev_distance_to_exit = float('inf')
        self.first_switch_distance_update = True
        self.first_exit_distance_update = True
        self.prev_potential = None
        self.consecutive_improvements = 0
        self.episode_start_switch_distance = None
        self.episode_start_exit_distance = None

    def set_mine_coordinates(self, mine_coords: List[Tuple[float, float]]):
        """Set the mine coordinates for the current level.

        Args:
            mine_coords: List of (x, y) coordinates of mines
        """
        self.mine_coords = mine_coords

    def calculate_mine_proximity_penalty(self,
                                         curr_state: Dict[str, Any],
                                         prev_state: Dict[str, Any]) -> float:
        """Calculate penalty based on velocity towards nearest mine."""
        # # Get mine information from state
        # mine_vector = curr_state['closest_mine_vector']
        # mine_dist = curr_state['closest_mine_distance']

        # # If no mine vector or distance is too far, return 0
        # if mine_vector == (0.0, 0.0) or mine_dist > self.MINE_DANGER_RADIUS:
        #     return 0.0

        # # Create mine vector tuple in format expected by velocity calculation
        # mine_vector_tuple = (0, 0, mine_vector[0], mine_vector[1])

        # # Calculate velocity component towards mine
        # velocity_towards_mine = self.calculate_velocity_towards_mine(
        #     curr_state, prev_state, mine_vector_tuple)

        # # Only penalize positive velocity towards mine
        # if velocity_towards_mine <= 0:
        #     return 0.0

        # # Calculate penalty based on velocity and proximity with penalty scaling
        # proximity_factor = 1.0 - (mine_dist / self.MINE_DANGER_RADIUS)
        # penalty = -self.MINE_PENALTY_SCALE * velocity_towards_mine * \
        #     proximity_factor * self._get_penalty_scale()

        # return penalty
        return 0.0

    def evaluate_navigation_quality(self,
                                    curr_distance: float,
                                    prev_distance: float,
                                    navigation_scale: float,
                                    curr_state: Dict[str, Any],
                                    prev_state: Dict[str, Any]) -> float:
        """Evaluate navigation quality using temporal difference learning."""
        if prev_distance is None or prev_distance == float('inf'):
            return 0.0

        # Calculate absolute and relative improvement with enhanced scaling
        absolute_improvement = prev_distance - curr_distance
        relative_improvement = absolute_improvement / (prev_distance + 1e-6)

        # Enhanced progressive scaling based on consecutive improvements
        if absolute_improvement > 0:
            self.consecutive_improvements += 1
            progress_multiplier = min(
                4.0,  # Increased max multiplier
                1.0 + (self.consecutive_improvements * \
                       self.CONSECUTIVE_IMPROVEMENT_BONUS)
            )
        else:
            self.consecutive_improvements = max(
                0, self.consecutive_improvements - 1)
            progress_multiplier = max(
                0.7,  # Increased minimum multiplier
                # Reduced penalty
                1.0 - (0.05 * (self.consecutive_improvements == 0))
            )

        # Calculate base reward with stronger emphasis on absolute improvement
        base_reward = (
            0.95 * np.sign(absolute_improvement) * np.sqrt(abs(absolute_improvement)) * self.DISTANCE_IMPROVEMENT_SCALE +
            0.05 * np.sign(relative_improvement) *
            np.sqrt(abs(relative_improvement))
        )

        # Enhanced momentum bonus for consistent progress
        momentum_bonus = 0.0
        if self.prev_improvement is not None:
            if np.sign(relative_improvement) == np.sign(self.prev_improvement):
                momentum_bonus = self.MOMENTUM_BONUS * progress_multiplier
            else:
                momentum_bonus = self.MOMENTUM_PENALTY

        # Add proximity bonus for being close to objective
        proximity_bonus = 0.0
        if curr_distance < self.MIN_DISTANCE_THRESHOLD:
            proximity_bonus = self.PROXIMITY_BONUS_SCALE * \
                (1.0 - curr_distance / self.MIN_DISTANCE_THRESHOLD)

        # Store current improvement
        self.prev_improvement = relative_improvement

        # Calculate mine proximity penalty
        mine_penalty = self.calculate_mine_proximity_penalty(
            curr_state, prev_state)

        return (base_reward + momentum_bonus + proximity_bonus) * navigation_scale + mine_penalty

    def calculate_velocity_towards_mine(self,
                                        curr_state: Dict[str, Any],
                                        prev_state: Dict[str, Any],
                                        mine_vector: Optional[Tuple[int, int, int, int]]) -> float:
        """Calculate velocity component towards nearest mine."""
        # if mine_vector is None:
        #     return 0.0

        # velocity_x, velocity_y = calculate_velocity(
        #     curr_state['player_x'],
        #     curr_state['player_y'],
        #     prev_state['player_x'],
        #     prev_state['player_y'],
        #     TIMESTEP
        # )

        # mine_dx = mine_vector[2] - mine_vector[0]
        # mine_dy = mine_vector[3] - mine_vector[1]

        # mine_dist = np.sqrt(mine_dx**2 + mine_dy**2)
        # if mine_dist == 0:
        #     return 0.0

        # mine_dx /= mine_dist
        # mine_dy /= mine_dist

        # velocity_towards_mine = (velocity_x * mine_dx + velocity_y * mine_dy)
        # return velocity_towards_mine / self.MAX_VELOCITY
        return 0.0
