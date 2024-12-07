import numpy as np
from typing import Dict, Any
from collections import deque


class MovementEvaluator:
    """
    Evaluates the quality and success of agent movements in a platformer environment.

    This system analyzes movement across multiple aspects:
    1. Control Precision: How well the agent maintains intended trajectories
    2. Landing Quality: The agent's ability to land safely and accurately
    3. Momentum Management: Efficient use of physics mechanics
    4. Objective Progress: Whether movements contribute to game goals
    """

    def __init__(self):
        # Constants for movement evaluation
        self.STABLE_VELOCITY_THRESHOLD = 0.1  # Threshold for controlled movement
        self.SAFE_LANDING_VELOCITY = 15.0     # Maximum safe landing speed
        self.MIN_MOVEMENT_THRESHOLD = 0.05     # Minimum meaningful movement
        self.TRAJECTORY_HISTORY_SIZE = 30      # Frames to track for trajectory analysis

        # Movement quality thresholds
        self.PRECISION_THRESHOLD = 0.85        # Required precision for success
        self.LANDING_SUCCESS_THRESHOLD = 0.7   # Required landing success rate
        self.MOMENTUM_EFFICIENCY_THRESHOLD = 0.8  # Required momentum efficiency

        # Historical tracking
        self.trajectory_history = deque(maxlen=self.TRAJECTORY_HISTORY_SIZE)
        # Track recent landing success
        self.landing_attempts = deque(maxlen=10)
        self.movement_segments = deque(maxlen=20)  # Track movement segments

    def evaluate_movement_success(self,
                                  current_state: Dict[str, float],
                                  previous_state: Dict[str, float],
                                  action_taken: int) -> Dict[str, Any]:
        """
        Comprehensively evaluates movement success based on multiple criteria.

        Args:
            current_state: Current game state including position, velocity
            previous_state: Previous game state for comparison
            action_taken: The action that led to current state

        Returns:
            Dictionary containing success metrics and overall evaluation
        """
        # Calculate basic movement metrics
        movement_vector = np.array([
            current_state['player_x'] - previous_state['player_x'],
            current_state['player_y'] - previous_state['player_y']
        ])
        velocity_vector = np.array([
            current_state['velocity_x'],
            current_state['velocity_y']
        ])
        movement_magnitude = np.linalg.norm(movement_vector)

        # Update trajectory history
        self.trajectory_history.append({
            'position': np.array([current_state['player_x'], current_state['player_y']]),
            'velocity': velocity_vector,
            'action': action_taken,
            'in_air': current_state['in_air']
        })

        # Evaluate different aspects of movement
        precision_score = self._evaluate_precision(
            movement_vector, action_taken)
        landing_score = self._evaluate_landing(current_state, previous_state)
        momentum_score = self._evaluate_momentum_efficiency(
            velocity_vector, movement_vector)
        progress_score = self._evaluate_objective_progress(
            current_state, previous_state)

        # Calculate segment success if we have enough history
        if len(self.trajectory_history) >= 2:
            segment_success = self._evaluate_movement_segment()
            self.movement_segments.append(segment_success)

        # Combine scores with weighted importance
        movement_success_metrics = {
            'precision': precision_score,
            'landing': landing_score,
            'momentum': momentum_score,
            'progress': progress_score,
            'segment_success': np.mean(list(self.movement_segments)) if self.movement_segments else 0.0
        }

        # Calculate overall success
        overall_success = self._calculate_overall_success(
            movement_success_metrics)

        return {
            'metrics': movement_success_metrics,
            'overall_success': overall_success,
            'movement_magnitude': movement_magnitude,
            'has_meaningful_movement': movement_magnitude > self.MIN_MOVEMENT_THRESHOLD
        }

    def _evaluate_precision(self, movement_vector: np.ndarray, action_taken: int) -> float:
        """
        Evaluates how precisely the agent controls its movement.

        Considers:
        - Direction consistency with intended action
        - Stability of movement
        - Absence of unnecessary oscillations
        """
        if len(self.trajectory_history) < 2:
            return 0.0

        # Get intended direction based on action
        intended_direction = self._get_intended_direction(action_taken)

        # Calculate direction alignment
        if np.linalg.norm(movement_vector) > self.MIN_MOVEMENT_THRESHOLD:
            actual_direction = movement_vector / \
                np.linalg.norm(movement_vector)
            direction_alignment = np.dot(actual_direction, intended_direction)
        else:
            direction_alignment = 1.0 if action_taken == 0 else 0.0  # Reward stillness for NOOP

        # Check for oscillations
        oscillation_penalty = self._calculate_oscillation_penalty()

        return max(0, direction_alignment - oscillation_penalty)

    def _evaluate_landing(self, current_state: Dict[str, float],
                          previous_state: Dict[str, float]) -> float:
        """
        Evaluates the quality of platform landings.

        Considers:
        - Landing velocity (softer is better)
        - Landing stability (no immediate bounces)
        - Landing precision (centered on platform)
        """
        # Detect landing event
        if previous_state['in_air'] and not current_state['in_air']:
            landing_velocity = abs(current_state['velocity_y'])

            # Calculate landing score
            velocity_factor = max(
                0, 1 - landing_velocity / self.SAFE_LANDING_VELOCITY)
            stability_factor = self._calculate_landing_stability()
            precision_factor = self._calculate_landing_precision(current_state)

            landing_score = (velocity_factor +
                             stability_factor + precision_factor) / 3
            self.landing_attempts.append(landing_score)

            return landing_score

        return 1.0 if not current_state['in_air'] else 0.0

    def _evaluate_momentum_efficiency(self, velocity_vector: np.ndarray,
                                      movement_vector: np.ndarray) -> float:
        """
        Evaluates how efficiently the agent uses momentum.

        Considers:
        - Conservation of momentum when beneficial
        - Appropriate speed management
        - Effective use of game physics
        """
        if np.linalg.norm(velocity_vector) < self.STABLE_VELOCITY_THRESHOLD:
            return 1.0  # Perfect score for stable low velocity

        # Calculate momentum efficiency
        velocity_direction = velocity_vector / np.linalg.norm(velocity_vector)
        movement_direction = movement_vector / \
            (np.linalg.norm(movement_vector) + 1e-7)

        # Dot product shows alignment of velocity and movement
        momentum_alignment = np.dot(velocity_direction, movement_direction)

        # Consider speed appropriateness for current situation
        speed_efficiency = self._evaluate_speed_appropriateness(
            velocity_vector)

        return (momentum_alignment + speed_efficiency) / 2

    def _evaluate_objective_progress(self, current_state: Dict[str, float],
                                     previous_state: Dict[str, float]) -> float:
        """
        Evaluates if movement is contributing to level objectives.

        Considers:
        - Progress toward switch/exit
        - Efficient path taking
        - Avoidance of unnecessary movements
        """
        if not current_state['switch_activated']:
            # Calculate progress toward switch
            current_distance_to_switch = np.linalg.norm(np.array([
                current_state['player_x'] - current_state['switch_x'],
                current_state['player_y'] - current_state['switch_y']
            ]))
            previous_distance_to_switch = np.linalg.norm(np.array([
                previous_state['player_x'] - previous_state['switch_x'],
                previous_state['player_y'] - previous_state['switch_y']
            ]))

            progress = (previous_distance_to_switch -
                        current_distance_to_switch)
            return np.clip(progress + 1.0, 0.0, 1.0)
        else:
            # Calculate progress toward exit
            current_distance_to_exit = np.linalg.norm(np.array([
                current_state['player_x'] - current_state['exit_x'],
                current_state['player_y'] - current_state['exit_y']
            ]))
            previous_distance_to_exit = np.linalg.norm(np.array([
                previous_state['player_x'] - previous_state['exit_x'],
                previous_state['player_y'] - previous_state['exit_y']
            ]))

            progress = (previous_distance_to_exit - current_distance_to_exit)
            return np.clip(progress + 1.0, 0.0, 1.0)

    def _calculate_overall_success(self, metrics: Dict[str, float]) -> float:
        """
        Calculates overall movement success from individual metrics.

        Uses weighted average with emphasis on critical aspects:
        - Precision: 30%
        - Landing: 25%
        - Momentum: 25%
        - Progress: 20%
        """
        weights = {
            'precision': 0.30,
            'landing': 0.25,
            'momentum': 0.25,
            'progress': 0.20
        }

        weighted_sum = sum(metrics[key] * weights[key]
                           for key in weights.keys())

        # Movement is considered successful if it exceeds our threshold
        return weighted_sum > 0.85
