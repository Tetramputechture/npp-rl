"""Navigation reward calculator for evaluating objective-based movement and progress."""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator
from npp_rl.util.util import calculate_velocity
from npp_rl.environments.constants import TIMESTEP


class NavigationRewardCalculator(BaseRewardCalculator):
    """Handles calculation of navigation and objective-based rewards."""

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

    def set_mine_coordinates(self, mine_coords: List[Tuple[float, float]]):
        """Set the mine coordinates for the current level.

        Args:
            mine_coords: List of (x, y) coordinates of mines
        """
        self.mine_coords = mine_coords

    def calculate_mine_proximity_penalty(self,
                                         curr_state: Dict[str, Any],
                                         prev_state: Dict[str, Any]) -> float:
        """Calculate penalty based on velocity towards nearest mine.

        Args:
            curr_state: Current game state
            prev_state: Previous game state

        Returns:
            float: Penalty value (negative or zero)
        """
        # Get mine information from state
        mine_vector = curr_state['closest_mine_vector']
        mine_dist = curr_state['closest_mine_distance']

        # If no mine vector or distance is too far, return 0
        if mine_vector == (0.0, 0.0) or mine_dist > self.MINE_DANGER_RADIUS:
            return 0.0

        # Create mine vector tuple in format expected by velocity calculation
        mine_vector_tuple = (0, 0, mine_vector[0], mine_vector[1])

        # Calculate velocity component towards mine
        velocity_towards_mine = self.calculate_velocity_towards_mine(
            curr_state, prev_state, mine_vector_tuple)

        # Only penalize positive velocity towards mine
        if velocity_towards_mine <= 0:
            return 0.0

        # Calculate penalty based on velocity and proximity
        proximity_factor = 1.0 - (mine_dist / self.MINE_DANGER_RADIUS)
        penalty = -self.MINE_PENALTY_SCALE * velocity_towards_mine * proximity_factor

        return penalty

    def evaluate_navigation_quality(self,
                                    curr_distance: float,
                                    prev_distance: float,
                                    navigation_scale: float,
                                    curr_state: Dict[str, Any],
                                    prev_state: Dict[str, Any]) -> float:
        """Evaluate navigation quality using temporal difference learning.

        Args:
            curr_distance: Current distance to objective
            prev_distance: Previous distance to objective 
            navigation_scale: Current navigation reward scale
            curr_state: Current game state
            prev_state: Previous game state

        Returns:
            float: Navigation quality reward
        """
        if prev_distance is None or prev_distance == float('inf'):
            return 0.0

        # Calculate absolute and relative improvement
        absolute_improvement = prev_distance - curr_distance
        relative_improvement = absolute_improvement / (prev_distance + 1e-6)

        # Apply progressive scaling based on consecutive improvements
        if absolute_improvement > 0:
            self.consecutive_improvements += 1
            progress_multiplier = min(
                1.5, 1.0 + (self.consecutive_improvements * 0.1))
        else:
            self.consecutive_improvements = 0
            progress_multiplier = 1.0

        # Calculate base reward using both absolute and relative improvements
        base_reward = (
            0.7 * np.sign(absolute_improvement) * np.sqrt(abs(absolute_improvement)) +
            0.3 * np.sign(relative_improvement) *
            np.sqrt(abs(relative_improvement))
        )

        # Enhanced momentum bonus for consistent progress
        momentum_bonus = 0.0
        if self.prev_improvement is not None:
            if np.sign(relative_improvement) == np.sign(self.prev_improvement):
                momentum_bonus = 1.0 * progress_multiplier
            else:
                momentum_bonus = -0.5  # Increased penalty for inconsistent progress

        # Store current improvement
        self.prev_improvement = relative_improvement

        # Calculate mine proximity penalty based on velocity
        mine_penalty = self.calculate_mine_proximity_penalty(
            curr_state, prev_state)

        return (base_reward + momentum_bonus) * navigation_scale + mine_penalty

    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """Calculate state potential for reward shaping.

        Args:
            state: Current game state

        Returns:
            float: State potential value
        """
        # Dynamic scaling based on level size
        level_diagonal = np.sqrt(
            state['level_width']**2 + state['level_height']**2)
        distance_scale = level_diagonal / 4  # Adaptive scaling

        # Calculate mine avoidance potential using current state only
        mine_penalty = self.calculate_mine_proximity_penalty(state, state)

        if not state['switch_activated']:
            # Switch-focused potential with adaptive scaling
            distance_to_switch = self.calculate_distance_to_objective(
                state['player_x'], state['player_y'],
                state['switch_x'], state['switch_y']
            )
            switch_potential = 25.0 * \
                np.exp(-distance_to_switch / distance_scale)
            return switch_potential + mine_penalty
        else:
            # Exit-focused potential with adaptive scaling
            distance_to_exit = self.calculate_distance_to_objective(
                state['player_x'], state['player_y'],
                state['exit_door_x'], state['exit_door_y']
            )
            exit_potential = 30.0 * \
                np.exp(-distance_to_exit / distance_scale) + 15.0
            return exit_potential + mine_penalty

    def calculate_navigation_reward(self,
                                    curr_state: Dict[str, Any],
                                    prev_state: Dict[str, Any],
                                    navigation_scale: float) -> Tuple[float, bool]:
        """Calculate comprehensive navigation reward.

        Args:
            curr_state: Current game state
            prev_state: Previous game state
            navigation_scale: Current navigation reward scale

        Returns:
            tuple[float, bool]: Total navigation reward and whether switch was activated
        """
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

        if not curr_state['switch_activated']:
            # Enhanced navigation to switch
            navigation_reward = self.evaluate_navigation_quality(
                curr_distance_to_switch,
                self.prev_distance_to_switch,
                navigation_scale * 1.2,  # Increased focus on switch navigation
                curr_state, prev_state
            )
            reward += navigation_reward

            # Progressive rewards for new minimum distances to switch
            if curr_distance_to_switch < self.min_distance_to_switch:
                improvement = self.min_distance_to_switch - curr_distance_to_switch
                if not self.first_switch_distance_update:
                    reward += improvement * navigation_scale
                self.min_distance_to_switch = curr_distance_to_switch
                self.first_switch_distance_update = False

        else:
            # Enhanced switch activation reward
            if not prev_state['switch_activated']:
                reward += self.SWITCH_ACTIVATION_REWARD
                switch_activated = True
                # Reset exit distance tracking
                self.min_distance_to_exit = curr_distance_to_exit
                self.first_exit_distance_update = True
                self.consecutive_improvements = 0  # Reset for exit navigation

            # Enhanced navigation to exit
            navigation_reward = self.evaluate_navigation_quality(
                curr_distance_to_exit,
                self.prev_distance_to_exit,
                navigation_scale * 1.5,  # Further increased focus on exit navigation
                curr_state, prev_state
            )
            reward += navigation_reward

            # Progressive rewards for new minimum distances to exit
            if curr_distance_to_exit < self.min_distance_to_exit:
                improvement = self.min_distance_to_exit - curr_distance_to_exit
                if not self.first_exit_distance_update:
                    reward += improvement * navigation_scale * 1.2
                self.min_distance_to_exit = curr_distance_to_exit
                self.first_exit_distance_update = False

        # Calculate potential-based shaping reward
        current_potential = self.calculate_potential(curr_state)
        if self.prev_potential is not None:
            shaping_reward = self.gamma * current_potential - self.prev_potential
            reward += shaping_reward
        self.prev_potential = current_potential

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
        # Don't reset mine_coords as they stay constant for the level

    def calculate_velocity_towards_mine(self,
                                        curr_state: Dict[str, Any],
                                        prev_state: Dict[str, Any],
                                        mine_vector: Optional[Tuple[int, int, int, int]]) -> float:
        """Calculate velocity component towards nearest mine."""
        if mine_vector is None:
            return 0.0

        velocity_x, velocity_y = calculate_velocity(
            curr_state['player_x'],
            curr_state['player_y'],
            prev_state['player_x'],
            prev_state['player_y'],
            TIMESTEP
        )

        mine_dx = mine_vector[2] - mine_vector[0]
        mine_dy = mine_vector[3] - mine_vector[1]

        mine_dist = np.sqrt(mine_dx**2 + mine_dy**2)
        if mine_dist == 0:
            return 0.0

        mine_dx /= mine_dist
        mine_dy /= mine_dist

        velocity_towards_mine = (velocity_x * mine_dx + velocity_y * mine_dy)
        return velocity_towards_mine / self.MAX_VELOCITY
