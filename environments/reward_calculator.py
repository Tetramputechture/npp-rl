import numpy as np
from typing import Dict, Any, List, Deque
from collections import deque
from environments.movement_evaluator import MovementEvaluator


class RewardCalculator:
    """
    A curriculum-based reward calculator for the N++ environment that progressively
    adapts rewards based on the agent's demonstrated capabilities and learning stage.

    The calculator implements three main learning stages:
    1. Movement Mastery: Basic control and precision
    2. Navigation: Efficient path-finding and objective targeting
    3. Optimization: Speed and perfect execution

    Each stage builds upon the skills learned in previous stages, with rewards
    automatically adjusting based on the agent's demonstrated competence.
    """

    def __init__(self, timestep: float):
        # Base reward/penalty constants
        self.BASE_TIME_PENALTY = -0.01
        self.GOLD_COLLECTION_REWARD = 1.0
        self.SWITCH_ACTIVATION_REWARD = 10.0
        self.TERMINAL_REWARD = 20.0
        self.DEATH_PENALTY = -15.0
        self.TIMEOUT_PENALTY = -10.0

        # Movement assessment constants
        self.FINE_DISTANCE_THRESHOLD = 5.0
        self.MIN_MOVEMENT_THRESHOLD = 0.1
        self.MOVEMENT_PENALTY = -0.01
        self.MAX_MOVEMENT_REWARD = 0.05

        # Distance-based reward scales
        self.DISTANCE_SCALE = 0.1
        self.APPROACH_REWARD_SCALE = 5.0
        self.RETREAT_PENALTY_SCALE = 0.15

        # Curriculum learning parameters
        self.movement_success_rate = 0.0
        self.navigation_success_rate = 0.0
        self.level_completion_rate = 0.0
        self.MOVEMENT_MASTERY_THRESHOLD = 0.7
        self.NAVIGATION_MASTERY_THRESHOLD = 0.8

        # Stage-specific reward scaling
        self.movement_scale = 1.0
        self.navigation_scale = 0.5
        self.completion_scale = 0.2

        # Historical tracking
        self.velocity_history = deque(maxlen=10)
        self.prev_distance_to_switch = float('inf')
        self.prev_distance_to_exit = float('inf')

        # Skill tracking
        self.demonstrated_skills = {
            'precise_movement': False,
            'platform_landing': False,
            'momentum_control': False,
            'switch_activation': False,
            'exit_reaching': False
        }

        # Movement evaluator
        self.movement_evaluator = MovementEvaluator(timestep)

        # Potential-based shaping parameters
        self.gamma = 0.99  # Discount factor for potential-based shaping
        self.prev_improvement = None  # For TD learning
        self.min_distance_to_switch = float('inf')  # For progress tracking
        self.min_distance_to_exit = float('inf')  # For progress tracking

        # Previous potential for shaping rewards
        self.prev_potential = None

        self.first_switch_distance_update = True

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _evaluate_movement_quality(self,
                                   movement_vector: np.ndarray,
                                   movement_magnitude: float,
                                   is_grounded: bool,
                                   was_in_air: bool) -> float:
        """
        Evaluate the quality of agent's movement, considering precision and control.
        Returns a scaled reward based on movement characteristics.
        """
        reward = 0.0

        # Precise movement reward
        if 0 < movement_magnitude < self.FINE_DISTANCE_THRESHOLD:
            reward += 0.1
            self.demonstrated_skills['precise_movement'] = True

        # Platform landing reward
        if was_in_air and is_grounded:
            reward += 0.5
            self.demonstrated_skills['platform_landing'] = True

        # Movement consistency reward
        if len(self.velocity_history) >= 2:
            prev_velocity = self.velocity_history[-1]
            if np.linalg.norm(prev_velocity) > 0 and np.linalg.norm(movement_vector) > 0:
                direction_consistency = np.dot(movement_vector, prev_velocity) / (
                    np.linalg.norm(movement_vector) *
                    np.linalg.norm(prev_velocity)
                )
                reward += 0.3 * direction_consistency
                if direction_consistency > 0.8:
                    self.demonstrated_skills['momentum_control'] = True

        return reward * self.movement_scale

    def _calculate_potential(self, state: Dict[str, Any]) -> float:
        """
        Calculate state potential for reward shaping using properly scaled distances for N++.

        Designed for N++'s actual scales:
        - Time ranges from 0 to 999
        - Distances can range up to 1000 units
        - Potential should reflect meaningful progress while staying numerically stable

        The potential increases as the agent:
        1. Maintains higher time remaining (efficiency)
        2. Gets closer to current objective (switch or exit)
        3. Activates the switch (major milestone)
        """
        # Scale time component to a 0-10 range
        # Using 999 as max time, so divide by 100 to get 0-10 scale
        time_potential = (state['time_remaining'] / 100)

        # Initialize base potential with normalized time
        potential = time_potential

        # Calculate distance-based potential
        if not state['switch_activated']:
            # Distance to switch
            distance_to_switch = self._calculate_distance(
                state['player_x'], state['player_y'],
                state['switch_x'], state['switch_y']
            )

            # Convert distance to a potential that peaks at 20 when very close
            # and approaches 0 at maximum distance
            # Using exponential decay to emphasize closer positions
            switch_potential = 20.0 * np.exp(-distance_to_switch / 250.0)

            # Add moderate bonus for improvement
            if hasattr(self, 'prev_distance_to_switch') and self.prev_distance_to_switch is not None:
                # Scale improvement bonus based on the size of improvement
                # Capped at ±2 to avoid overwhelming base potential
                improvement = self.prev_distance_to_switch - distance_to_switch
                improvement_bonus = 2.0 * np.tanh(improvement / 100.0)
                switch_potential += improvement_bonus

            potential += switch_potential

        else:
            # Similar logic for exit distance
            distance_to_exit = self._calculate_distance(
                state['player_x'], state['player_y'],
                state['exit_door_x'], state['exit_door_y']
            )

            # Higher base potential for exit phase (peaks at 25)
            # Plus fixed bonus for having activated switch
            exit_potential = 25.0 * np.exp(-distance_to_exit / 250.0) + 20.0

            # Add improvement bonus for exit approach
            if hasattr(self, 'prev_distance_to_exit') and self.prev_distance_to_exit is not None:
                improvement = self.prev_distance_to_exit - distance_to_exit
                improvement_bonus = 2.0 * np.tanh(improvement / 100.0)
                exit_potential += improvement_bonus

            potential += exit_potential

        return potential

    def _evaluate_navigation_quality(self, curr_distance: float, prev_distance: float) -> float:
        """
        Navigation evaluation using temporal difference learning principles.

        Rewards are based on relative improvement rather than absolute distances,
        with additional bonuses for consistent progress.
        """
        # Handle first evaluation
        if prev_distance is None or prev_distance == float('inf'):
            return 0.0

        # Calculate relative improvement
        relative_improvement = (
            prev_distance - curr_distance) / (prev_distance + 1e-6)

        # Apply non-linear scaling to emphasize significant improvements
        # Square root maintains sign while reducing the magnitude of large changes
        scaled_improvement = np.sign(
            relative_improvement) * np.sqrt(abs(relative_improvement))

        # Add momentum bonus for consistent progress
        momentum_bonus = 0.0
        if self.prev_improvement is not None:
            # Reward consistent direction of improvement
            momentum_bonus = 0.5 * (
                1.0 if np.sign(relative_improvement) == np.sign(self.prev_improvement)
                else -0.2
            )

        # Store current improvement for next iteration
        self.prev_improvement = relative_improvement

        # Combine scaled improvement with momentum bonus
        reward = scaled_improvement + momentum_bonus

        return reward * self.navigation_scale

    def calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any], action_taken: int) -> float:
        """
        Reward calculation incorporating hierarchical structure,
        potential-based shaping, and temporal difference learning.
        """
        # Level 1: Basic survival and immediate penalties
        if obs.get('player_dead', False):
            print("Player Died!")
            return self.DEATH_PENALTY

        if obs['time_remaining'] <= 0:
            return self.TIMEOUT_PENALTY

        # Initialize total reward
        reward = 0.0

        # Level 2: Movement quality and control
        movement_vector = np.array([
            obs['player_x'] - prev_obs['player_x'],
            obs['player_y'] - prev_obs['player_y']
        ])
        movement_magnitude = np.linalg.norm(movement_vector)

        # Evaluate basic movement quality
        movement_reward = self._evaluate_movement_quality(
            movement_vector,
            movement_magnitude,
            not obs['in_air'],
            prev_obs['in_air']
        )
        reward += movement_reward

        # Add movement success evaluation
        movement_success = self.movement_evaluator.evaluate_movement_success(
            current_state=obs,
            previous_state=prev_obs,
            action_taken=action_taken
        )
        if movement_success['overall_success']:
            reward += self.movement_scale * 1.0

        # Level 3: Navigation and objective completion
        curr_distance_to_switch = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['switch_x'], obs['switch_y']
        )
        curr_distance_to_exit = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['exit_door_x'], obs['exit_door_y']
        )

        # Calculate navigation rewards using TD learning
        if not obs['switch_activated']:
            navigation_reward = self._evaluate_navigation_quality(
                curr_distance_to_switch,
                self.prev_distance_to_switch
            )
            reward += navigation_reward

            # Handle minimum distance tracking for switch
            if self.min_distance_to_switch is None:
                # Initialize on first update
                self.min_distance_to_switch = curr_distance_to_switch
            elif curr_distance_to_switch < self.min_distance_to_switch:
                # Only give improvement reward after first update
                if not self.first_switch_distance_update:
                    reward += (self.min_distance_to_switch -
                               curr_distance_to_switch) * 0.5
                self.min_distance_to_switch = curr_distance_to_switch
                self.first_switch_distance_update = False

        else:
            # Handle switch activation event
            if not prev_obs['switch_activated']:
                reward += self.SWITCH_ACTIVATION_REWARD
                print("Switch Activated!")
                self.demonstrated_skills['switch_activation'] = True
                # Reset exit distance tracking
                self.min_distance_to_exit = None
                self.first_exit_distance_update = True

            # Navigate to exit
            navigation_reward = self._evaluate_navigation_quality(
                curr_distance_to_exit,
                self.prev_distance_to_exit
            )
            reward += navigation_reward

        # Calculate potential-based shaping reward
        current_potential = self._calculate_potential(obs)
        shaping_reward = 0.0
        if self.prev_potential is not None:
            shaping_reward = self.gamma * current_potential - self.prev_potential
            reward += shaping_reward
        self.prev_potential = current_potential

        # Level 4: Time management and optimization
        time_diff = obs['time_remaining'] - prev_obs['time_remaining']
        if time_diff > 0:  # Collected gold
            reward += self.GOLD_COLLECTION_REWARD * time_diff
        else:  # Normal time decrease
            time_penalty_scale = 1.0 / max(obs['time_remaining'], 1.0)
            reward += self.BASE_TIME_PENALTY * min(time_penalty_scale, 5.0)

        # Level 5: Level completion
        if 'retry level' in obs.get('begin_retry_text', '').lower():
            reward += self.TERMINAL_REWARD
            self.demonstrated_skills['exit_reaching'] = True

        # Update tracking variables
        self.velocity_history.append(movement_vector)
        self.prev_distance_to_switch = curr_distance_to_switch
        self.prev_distance_to_exit = curr_distance_to_exit

        # Log reward components on one line
        print(f"Reward: {reward:.2f} | "
              f"Movement: {movement_reward:.2f} | "
              f"Navigation: {navigation_reward:.2f} | "
              f"Shaping: {shaping_reward:.2f} | "
              f"Potential: {current_potential:.2f} | "
              f"Time: {time_diff:.2f}")
        return reward

    def update_progression_metrics(self):
        """
        Update the agent's progression metrics after each episode.
        This affects the scaling of different reward components for curriculum learning.
        """
        # alpha = 0.1  # Exponential moving average factor

        # # Update success rates
        # self.movement_success_rate = (1 - alpha) * self.movement_success_rate + \
        #     alpha * float(self.demonstrated_skills['precise_movement'] and
        #                   self.demonstrated_skills['platform_landing'])

        # self.navigation_success_rate = (1 - alpha) * self.navigation_success_rate + \
        #     alpha * float(self.demonstrated_skills['switch_activation'])

        # self.level_completion_rate = (1 - alpha) * self.level_completion_rate + \
        #     alpha * float(self.demonstrated_skills['exit_reaching'])

        # # Adjust reward scales based on progression
        # if self.movement_success_rate > self.MOVEMENT_MASTERY_THRESHOLD:
        #     self.movement_scale *= 0.95
        #     self.navigation_scale *= 1.05

        # if self.navigation_success_rate > self.NAVIGATION_MASTERY_THRESHOLD:
        #     self.navigation_scale *= 0.95
        #     self.completion_scale *= 1.05
        # Noop: this is causing very large navigation forwards,
        # noop for now

    def reset(self):
        """Reset the reward calculator's progression metrics."""
        self.movement_success_rate = 0.0
        self.navigation_success_rate = 0.0
        self.level_completion_rate = 0.0
        self.movement_scale = 1.0
        self.navigation_scale = 0.5
        self.completion_scale = 0.2
        self.demonstrated_skills = {
            'precise_movement': False,
            'platform_landing': False,
            'momentum_control': False,
            'switch_activation': False,
            'exit_reaching': False
        }
        self.velocity_history.clear()
        self.prev_distance_to_switch = float('inf')
        self.prev_distance_to_exit = float('inf')
        self.prev_improvement = None
        self.min_distance_to_switch = float('inf')
        self.min_distance_to_exit = float('inf')
        self.prev_potential = None
        self.first_switch_distance_update = True
