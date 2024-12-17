"""Movement reward calculator for evaluating movement quality and control."""
from typing import Dict, Any
import numpy as np
from collections import deque
from npp_rl.environments.movement_evaluator import MovementEvaluator
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator


class MovementRewardCalculator(BaseRewardCalculator):
    """Handles calculation of movement-related rewards."""

    BASE_MOVEMENT_REWARD = 0.005

    def __init__(self, movement_evaluator: MovementEvaluator):
        """Initialize movement reward calculator.

        Args:
            movement_evaluator: Evaluator for movement success metrics
        """
        super().__init__()
        self.movement_evaluator = movement_evaluator
        self.velocity_history = deque(maxlen=10)
        self.movement_skills = {
            'precise_landing': 0.0,
            'momentum_control': 0.0,
            'obstacle_avoidance': 0.0
        }

    def evaluate_movement_quality(self,
                                  movement_vector: np.ndarray,
                                  movement_magnitude: float,
                                  is_grounded: bool,
                                  was_in_air: bool,
                                  movement_scale: float) -> float:
        """Evaluate the quality of agent's movement.

        Args:
            movement_vector: Vector representing movement
            movement_magnitude: Magnitude of movement
            is_grounded: Whether agent is on ground
            was_in_air: Whether agent was in air in previous state
            movement_scale: Current movement reward scale

        Returns:
            float: Scaled reward based on movement characteristics
        """
        reward = 0.0

        # Precise movement reward
        if 0 < movement_magnitude < self.FINE_DISTANCE_THRESHOLD:
            reward += 0.05

        # Platform landing reward
        if was_in_air and is_grounded:
            reward += 0.25

        # Movement consistency reward
        if len(self.velocity_history) >= 2:
            prev_velocity = self.velocity_history[-1]
            if np.linalg.norm(prev_velocity) > 0 and np.linalg.norm(movement_vector) > 0:
                direction_consistency = np.dot(movement_vector, prev_velocity) / (
                    np.linalg.norm(movement_vector) *
                    np.linalg.norm(prev_velocity)
                )
                reward += 0.15 * direction_consistency

        return reward * movement_scale

    def calculate_movement_reward(self,
                                  curr_state: Dict[str, Any],
                                  prev_state: Dict[str, Any],
                                  action_taken: int,
                                  movement_scale: float) -> float:
        """Calculate comprehensive movement reward.

        Args:
            curr_state: Current game state
            prev_state: Previous game state
            action_taken: Action index taken
            movement_scale: Current movement reward scale

        Returns:
            float: Total movement reward
        """
        movement_vector = self.calculate_movement_vector(
            curr_state, prev_state)
        movement_magnitude = np.linalg.norm(movement_vector)

        # Basic movement quality reward
        reward = self.evaluate_movement_quality(
            movement_vector,
            movement_magnitude,
            not curr_state['in_air'],
            prev_state['in_air'],
            movement_scale
        )

        # Movement success evaluation
        movement_success = self.movement_evaluator.evaluate_movement_success(
            current_state=curr_state,
            previous_state=prev_state,
            action_taken=action_taken
        )

        # Base movement reward for controlled movement
        if movement_magnitude > 0.1 and movement_success['metrics']['precision'] > 0.5:
            reward += self.BASE_MOVEMENT_REWARD

        # Precision rewards
        precision_score = movement_success['metrics']['precision']
        if precision_score > 0.8:
            reward += self.BASE_MOVEMENT_REWARD * 2.0
            self.movement_skills['precise_landing'] = min(
                1.0, self.movement_skills['precise_landing'] + 0.1)
        elif precision_score > 0.6:
            reward += self.BASE_MOVEMENT_REWARD * 1.5

        # Landing rewards
        if prev_state['in_air'] and not curr_state['in_air']:
            landing_quality = movement_success['metrics']['landing']
            if landing_quality > 0.8:
                reward += self.BASE_MOVEMENT_REWARD * 3.0
                self.movement_skills['precise_landing'] = min(
                    1.0, self.movement_skills['precise_landing'] + 0.2)
            elif landing_quality > 0.5:
                reward += self.BASE_MOVEMENT_REWARD * 2.0

        # Momentum rewards
        momentum_score = movement_success['metrics']['momentum']
        if momentum_score > 0.7:
            reward += self.BASE_MOVEMENT_REWARD * 1.5
            self.movement_skills['momentum_control'] = min(
                1.0, self.movement_skills['momentum_control'] + 0.1)

        # Apply skill-based scaling
        skill_multiplier = 1.0 + (
            self.movement_skills['precise_landing'] +
            self.movement_skills['momentum_control']
        ) / 2.0
        reward *= skill_multiplier

        if movement_success['overall_success']:
            reward += movement_scale * 1.0

        # Update velocity history
        self.velocity_history.append(movement_vector)

        return reward

    def reset(self):
        """Reset internal state for new episode."""
        self.velocity_history.clear()
        self.movement_skills = {
            'precise_landing': 0.0,
            'momentum_control': 0.0,
            'obstacle_avoidance': 0.0
        }
