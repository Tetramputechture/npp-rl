"""Navigation reward calculator for evaluating objective-based movement and progress."""
from typing import Dict, Any, Tuple
import numpy as np
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator


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

    def evaluate_navigation_quality(self,
                                    curr_distance: float,
                                    prev_distance: float,
                                    navigation_scale: float) -> float:
        """Evaluate navigation quality using temporal difference learning.

        Args:
            curr_distance: Current distance to objective
            prev_distance: Previous distance to objective
            navigation_scale: Current navigation reward scale

        Returns:
            float: Navigation quality reward
        """
        if prev_distance is None or prev_distance == float('inf'):
            return 0.0

        # Calculate relative improvement
        relative_improvement = (
            prev_distance - curr_distance) / (prev_distance + 1e-6)

        # Apply non-linear scaling
        scaled_improvement = np.sign(
            relative_improvement) * np.sqrt(abs(relative_improvement))

        # Add momentum bonus for consistent progress
        momentum_bonus = 0.0
        if self.prev_improvement is not None:
            momentum_bonus = 0.5 * (
                1.0 if np.sign(relative_improvement) == np.sign(self.prev_improvement)
                else -0.2
            )

        # Store current improvement
        self.prev_improvement = relative_improvement

        return (scaled_improvement + momentum_bonus) * navigation_scale

    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """Calculate state potential for reward shaping.

        Args:
            state: Current game state

        Returns:
            float: State potential value
        """
        # Base time potential
        potential = (state['time_remaining'] / 100)

        if not state['switch_activated']:
            # Switch-focused potential
            distance_to_switch = self.calculate_distance_to_objective(
                state['player_x'], state['player_y'],
                state['switch_x'], state['switch_y']
            )
            switch_potential = 20.0 * np.exp(-distance_to_switch / 250.0)
            potential += switch_potential
        else:
            # Exit-focused potential
            distance_to_exit = self.calculate_distance_to_objective(
                state['player_x'], state['player_y'],
                state['exit_door_x'], state['exit_door_y']
            )
            exit_potential = 25.0 * np.exp(-distance_to_exit / 250.0) + 10.0
            potential += exit_potential

        return potential

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
            # Navigation to switch
            navigation_reward = self.evaluate_navigation_quality(
                curr_distance_to_switch,
                self.prev_distance_to_switch,
                navigation_scale
            )
            reward += navigation_reward

            # Track minimum distance to switch
            if self.min_distance_to_switch is None:
                self.min_distance_to_switch = curr_distance_to_switch
            elif curr_distance_to_switch < self.min_distance_to_switch:
                if not self.first_switch_distance_update:
                    reward += (self.min_distance_to_switch -
                               curr_distance_to_switch) * 0.5
                self.min_distance_to_switch = curr_distance_to_switch
                self.first_switch_distance_update = False

        else:
            # Handle switch activation
            if not prev_state['switch_activated']:
                reward += self.SWITCH_ACTIVATION_REWARD
                switch_activated = True
                # Reset exit distance tracking
                self.min_distance_to_exit = None
                self.first_exit_distance_update = True

            # Navigation to exit
            navigation_reward = self.evaluate_navigation_quality(
                curr_distance_to_exit,
                self.prev_distance_to_exit,
                navigation_scale
            )
            reward += navigation_reward

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
