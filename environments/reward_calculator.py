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
        self.APPROACH_REWARD_SCALE = 3
        self.RETREAT_PENALTY_SCALE = 0.2

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
        self.prev_distance_to_switch = None
        self.prev_distance_to_exit = None

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
            if np.linalg.norm(prev_velocity) > 0:
                direction_consistency = np.dot(movement_vector, prev_velocity) / (
                    np.linalg.norm(movement_vector) *
                    np.linalg.norm(prev_velocity)
                )
                reward += 0.3 * direction_consistency
                if direction_consistency > 0.8:
                    self.demonstrated_skills['momentum_control'] = True

        return reward * self.movement_scale

    def _evaluate_navigation_quality(self,
                                     curr_distance: float,
                                     prev_distance: float) -> float:
        """
        Evaluate the quality of navigation towards objectives.
        Returns a scaled reward based on approach efficiency.
        """
        reward = 0.0
        distance_progress = prev_distance - curr_distance

        if distance_progress > 0:  # Moving closer
            reward += distance_progress * self.APPROACH_REWARD_SCALE
            if curr_distance < self.FINE_DISTANCE_THRESHOLD:
                reward += 0.5
        else:  # Moving away
            reward += distance_progress * self.RETREAT_PENALTY_SCALE

        # Add inverse distance component
        reward += 1.0 / (1.0 + curr_distance) * self.DISTANCE_SCALE

        return reward * self.navigation_scale

    def calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any], action_taken: int) -> float:
        """
        Calculate the total reward based on the current learning stage and observations.
        Implements curriculum learning by scaling different reward components based on
        demonstrated skills and mastery levels.
        """
        reward = 0.0

        # Calculate current distances
        curr_distance_to_switch = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['switch_x'], obs['switch_y']
        )
        curr_distance_to_exit = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['exit_door_x'], obs['exit_door_y']
        )

        # Calculate movement characteristics
        movement_vector = np.array([
            obs['player_x'] - prev_obs['player_x'],
            obs['player_y'] - prev_obs['player_y']
        ])
        movement_magnitude = np.linalg.norm(movement_vector)

        # Basic time management rewards
        time_penalty_scale = 1.0 / max(obs['time_remaining'], 1.0)
        reward += self.BASE_TIME_PENALTY * min(time_penalty_scale, 5.0)

        time_diff = obs['time_remaining'] - prev_obs['time_remaining']
        if time_diff > 0:
            reward += self.GOLD_COLLECTION_REWARD * time_diff

        # Stage 1: Movement Mastery
        movement_reward = self._evaluate_movement_quality(
            movement_vector,
            movement_magnitude,
            not obs['in_air'],
            prev_obs['in_air']
        )
        reward += movement_reward

        movement_success = self.movement_evaluator.evaluate_movement_success(
            current_state=obs,
            previous_state=prev_obs,
            action_taken=action_taken
        )

        # Use evaluation in reward calculation
        if movement_success['overall_success']:
            reward += self.movement_scale * 1.0

        # Use individual metrics for more granular rewards
        reward += movement_success['metrics']['precision'] * 0.3
        reward += movement_success['metrics']['landing'] * 0.5

        # Stage 2: Navigation
        navigation_reward = 0.0
        if self.demonstrated_skills['precise_movement'] and self.demonstrated_skills['platform_landing']:
            if not obs['switch_activated']:
                navigation_reward = self._evaluate_navigation_quality(
                    curr_distance_to_switch,
                    self.prev_distance_to_switch
                )
                print(f"Navigation Reward: {navigation_reward:.2f}")
                reward += navigation_reward
            else:
                if not prev_obs['switch_activated']:
                    reward += self.SWITCH_ACTIVATION_REWARD
                    print("Switch Activated!")
                    self.demonstrated_skills['switch_activation'] = True

                navigation_reward = self._evaluate_navigation_quality(
                    curr_distance_to_exit,
                    self.prev_distance_to_exit
                )
                print(f"Navigation Reward: {navigation_reward:.2f}")
                reward += navigation_reward

        # Stage 3: Optimization (after demonstrating basic competence)
        if all(self.demonstrated_skills.values()):
            reward *= self.completion_scale  # Scale rewards to encourage optimization

        # Terminal states
        if obs['time_remaining'] <= 0:
            reward += self.TIMEOUT_PENALTY
        if obs.get('player_dead', False):
            print("Player Died!")
            reward += self.DEATH_PENALTY
        if 'retry level' in obs.get('begin_retry_text', '').lower():
            reward += self.TERMINAL_REWARD
            self.demonstrated_skills['exit_reaching'] = True

        # Update tracking variables
        self.velocity_history.append(movement_vector)
        self.prev_distance_to_switch = curr_distance_to_switch
        self.prev_distance_to_exit = curr_distance_to_exit

        # Log our reward
        print(
            f"Total Reward: {reward:.2f} | Movement: {movement_reward:.2f} | Navigation: {navigation_reward:.2f}")

        return reward

    def update_progression_metrics(self, episode_info: Dict[str, Any]):
        """
        Update the agent's progression metrics after each episode.
        This affects the scaling of different reward components for curriculum learning.
        """
        alpha = 0.1  # Exponential moving average factor

        # Update success rates
        self.movement_success_rate = (1 - alpha) * self.movement_success_rate + \
            alpha * float(self.demonstrated_skills['precise_movement'] and
                          self.demonstrated_skills['platform_landing'])

        self.navigation_success_rate = (1 - alpha) * self.navigation_success_rate + \
            alpha * float(self.demonstrated_skills['switch_activation'])

        self.level_completion_rate = (1 - alpha) * self.level_completion_rate + \
            alpha * float(self.demonstrated_skills['exit_reaching'])

        # Adjust reward scales based on progression
        if self.movement_success_rate > self.MOVEMENT_MASTERY_THRESHOLD:
            self.movement_scale *= 0.95
            self.navigation_scale *= 1.05

        if self.navigation_success_rate > self.NAVIGATION_MASTERY_THRESHOLD:
            self.navigation_scale *= 0.95
            self.completion_scale *= 1.05
