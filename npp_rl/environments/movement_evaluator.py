import numpy as np
from typing import Dict, Any
from collections import deque
from npp_rl.util.util import calculate_velocity
from npp_rl.environments.constants import TIMESTEP


class MovementEvaluator:
    """
    Evaluates the quality and success of agent movements in the platformer environment N++.

    Analyzes movement across multiple aspects:
    1. Control Precision: How well the agent maintains intended trajectories
    2. Landing Quality: The agent's ability to land safely and accurately
    3. Momentum Management: Efficient use of physics mechanics
    4. Objective Progress: Whether movements contribute to game goals
    """

    # Constants for movement evaluation
    STABLE_VELOCITY_THRESHOLD = 0.1  # Threshold for controlled movement
    SAFE_LANDING_VELOCITY = 15.0     # Maximum safe landing speed
    MIN_MOVEMENT_THRESHOLD = 0.05     # Minimum meaningful movement
    TRAJECTORY_HISTORY_SIZE = 30      # Frames to track for trajectory analysis

    # Movement quality thresholds
    PRECISION_THRESHOLD = 0.85        # Required precision for success
    LANDING_SUCCESS_THRESHOLD = 0.7   # Required landing success rate
    MOMENTUM_EFFICIENCY_THRESHOLD = 0.8  # Required momentum efficiency

    # Movement vectors
    LEFT = np.array([-1.0, 0.0])
    RIGHT = np.array([1.0, 0.0])

    # Y axis origin is the top left corner, so up is negative
    UP = np.array([0.0, -1.0])

    # Map actions to intended directions
    ACTION_TO_DIRECTION_VECTOR = {
        0: np.zeros(2),  # NOOP
        1: LEFT,
        2: RIGHT,
        3: UP,
        # Divide by sqrt(2) to normalize diagonal vectors
        4: (LEFT + UP) / np.sqrt(2),
        5: (RIGHT + UP) / np.sqrt(2)
    }

    def __init__(self):
        # Historical tracking
        self.trajectory_history = deque(maxlen=self.TRAJECTORY_HISTORY_SIZE)

        # Track recent landing success
        self.landing_attempts = deque(maxlen=10)

        # Track movement segments
        self.movement_segments = deque(maxlen=20)

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
        vx, vy = calculate_velocity(
            current_state['player_x'], current_state['player_y'],
            previous_state['player_x'], previous_state['player_y'],
            TIMESTEP
        )
        velocity_vector = np.array([vx, vy])
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

    def _get_intended_direction(self, action_taken: int) -> np.ndarray:
        """
        Determines the intended movement direction based on the action taken.

        In a platformer, each action implies an intended direction of movement.
        This method converts discrete actions into directional vectors that 
        represent the expected movement direction.

        Args:
            action_taken: Integer representing the action
                0: NOOP
                1: Move Left
                2: Move Right
                3: Jump
                4: Jump Left
                5: Jump Right

        Returns:
            np.ndarray: 2D unit vector representing intended direction
        """

        return self.ACTION_TO_DIRECTION_VECTOR.get(action_taken, np.zeros(2))

    def _calculate_oscillation_penalty(self) -> float:
        """
        Calculates a penalty for oscillating or jittery movement.

        Oscillation in platformers usually indicates poor control or indecision.
        This method detects rapid direction changes and penalizes them.

        Returns:
            float: Penalty value between 0.0 (no oscillation) and 1.0 (high oscillation)
        """
        if len(self.trajectory_history) < 3:
            return 0.0

        # Get recent positions
        positions = [t['position'] for t in self.trajectory_history]

        # Calculate direction changes
        direction_changes = 0
        total_segments = len(positions) - 2

        for i in range(len(positions) - 2):
            # Calculate vectors between consecutive positions
            v1 = positions[i+1] - positions[i]
            v2 = positions[i+2] - positions[i+1]

            # Skip near-zero movements to avoid noise
            if np.linalg.norm(v1) < self.MIN_MOVEMENT_THRESHOLD or \
                    np.linalg.norm(v2) < self.MIN_MOVEMENT_THRESHOLD:
                total_segments -= 1
                continue

            # Normalize vectors
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

            # Calculate angle between movements
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Count significant direction changes (more than 90 degrees)
            if angle > np.pi/2:
                direction_changes += 1

        # Avoid division by zero
        if total_segments == 0:
            return 0.0

        # Calculate penalty (0.0 to 1.0)
        return min(1.0, direction_changes / total_segments)

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

    def _calculate_landing_stability(self) -> float:
        """
        Evaluates the stability of landing maneuvers.

        A stable landing in a platformer involves:
        - Controlled vertical velocity
        - Minimal horizontal velocity changes
        - Maintaining position after landing

        Returns:
            float: Stability score between 0.0 (unstable) and 1.0 (perfectly stable)
        """
        if len(self.trajectory_history) < 3:
            return 0.0

        # Get the landing moment and subsequent frames
        landing_frame = None
        for i in range(len(self.trajectory_history) - 1):
            if self.trajectory_history[i]['in_air'] and \
                    not self.trajectory_history[i + 1]['in_air']:
                landing_frame = i + 1
                break

        if landing_frame is None or landing_frame >= len(self.trajectory_history) - 1:
            return 0.0

        # Calculate stability metrics
        total_score = 0.0
        num_metrics = 0

        # 1. Vertical velocity at landing
        landing_velocity = self.trajectory_history[landing_frame]['velocity']
        vertical_speed = abs(landing_velocity[1])
        vertical_stability = max(
            0.0, 1.0 - vertical_speed / self.SAFE_LANDING_VELOCITY)
        total_score += vertical_stability
        num_metrics += 1

        # 2. Horizontal velocity consistency
        pre_landing = self.trajectory_history[landing_frame - 1]['velocity'][0]
        post_landing = self.trajectory_history[landing_frame +
                                               1]['velocity'][0]
        velocity_change = abs(post_landing - pre_landing)
        horizontal_stability = max(
            0.0, 1.0 - velocity_change / self.STABLE_VELOCITY_THRESHOLD)
        total_score += horizontal_stability
        num_metrics += 1

        # 3. Position stability after landing
        landing_pos = self.trajectory_history[landing_frame]['position']
        next_pos = self.trajectory_history[landing_frame + 1]['position']
        position_drift = np.linalg.norm(next_pos - landing_pos)
        position_stability = max(
            0.0, 1.0 - position_drift / self.MIN_MOVEMENT_THRESHOLD)
        total_score += position_stability
        num_metrics += 1

        return total_score / num_metrics

    def _calculate_landing_precision(self, current_state: Dict[str, float]) -> float:
        """
        Evaluates how precisely the agent lands on platforms.

        Precise landings in platformers involve:
        - Landing near platform centers
        - Appropriate approach angle
        - Controlled landing speed

        Args:
            current_state: Current game state including position and velocity

        Returns:
            float: Precision score between 0.0 (imprecise) and 1.0 (perfect precision)
        """
        if len(self.trajectory_history) < 2:
            return 0.0

        # Get landing trajectory
        landing_trajectory = None
        for i in range(len(self.trajectory_history) - 1):
            if self.trajectory_history[i]['in_air'] and \
                    not self.trajectory_history[i + 1]['in_air']:
                landing_trajectory = (self.trajectory_history[i],
                                      self.trajectory_history[i + 1])
                break

        if landing_trajectory is None:
            return 0.0

        total_score = 0.0
        num_metrics = 0

        # 1. Landing velocity angle
        landing_velocity = landing_trajectory[1]['velocity']
        approach_angle = abs(np.arctan2(
            landing_velocity[1], landing_velocity[0]))
        # Prefer near-vertical landing approaches
        # 1.0 for vertical, 0.0 for horizontal
        angle_score = np.cos(approach_angle)
        total_score += angle_score
        num_metrics += 1

        # 2. Landing speed control
        landing_speed = np.linalg.norm(landing_velocity)
        speed_score = max(0.0, 1.0 - landing_speed /
                          self.SAFE_LANDING_VELOCITY)
        total_score += speed_score
        num_metrics += 1

        # 3. Position control during landing
        pre_landing_pos = landing_trajectory[0]['position']
        landing_pos = landing_trajectory[1]['position']
        landing_movement = np.linalg.norm(landing_pos - pre_landing_pos)
        movement_score = max(0.0, 1.0 - landing_movement /
                             self.MIN_MOVEMENT_THRESHOLD)
        total_score += movement_score
        num_metrics += 1

        return total_score / num_metrics

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
            _, vy = calculate_velocity(
                current_state['player_x'], current_state['player_y'],
                previous_state['player_x'], previous_state['player_y'],
                TIMESTEP
            )
            landing_velocity = abs(vy)

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
        # speed_efficiency = self._evaluate_speed_appropriateness(
        #     velocity_vector)
        # Use 1 for now
        speed_efficiency = 1

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
                current_state['player_x'] - current_state['exit_door_x'],
                current_state['player_y'] - current_state['exit_door_y']
            ]))
            previous_distance_to_exit = np.linalg.norm(np.array([
                previous_state['player_x'] - previous_state['exit_door_x'],
                previous_state['player_y'] - previous_state['exit_door_y']
            ]))

            progress = (previous_distance_to_exit - current_distance_to_exit)
            return np.clip(progress + 1.0, 0.0, 1.0)

    def _evaluate_movement_segment(self) -> float:
        """
        Evaluates the quality of recent movement sequences based on the trajectory history.

        This method analyzes short sequences of movements to determine if they represent
        successful platforming behavior. It considers factors like:
        - Movement consistency and purpose
        - Appropriate use of jumps
        - Effective platform transitions
        - Progress toward objectives

        Returns:
            float: Success score for the movement segment, ranging from 0.0 to 1.0
        """
        # We need at least 2 trajectory points to evaluate a segment
        if len(self.trajectory_history) < 2:
            return 0.0

        # Get the recent trajectory points
        recent_trajectories = list(self.trajectory_history)[-2:]

        # Calculate segment metrics
        total_score = 0.0
        num_metrics = 0

        # 1. Evaluate movement consistency
        velocities = [np.array([t['velocity'] for t in recent_trajectories])]
        if len(velocities) >= 2:
            velocity_consistency = np.abs(
                np.corrcoef(velocities[:-1], velocities[1:])[0, 1]
            )
            total_score += velocity_consistency
            num_metrics += 1

        # 2. Check for purposeful jumping
        for i in range(len(recent_trajectories) - 1):
            current = recent_trajectories[i]
            next_point = recent_trajectories[i + 1]

            # Jumping should generally move upward
            if not current['in_air'] and next_point['in_air']:
                vertical_velocity = next_point['velocity'][1]
                if vertical_velocity > 0:  # Positive vertical velocity indicates upward movement
                    jump_score = min(vertical_velocity /
                                     self.STABLE_VELOCITY_THRESHOLD, 1.0)
                    total_score += jump_score
                    num_metrics += 1

        # 3. Evaluate landing preparation
        if recent_trajectories[-2]['in_air'] and not recent_trajectories[-1]['in_air']:
            landing_velocity = abs(recent_trajectories[-1]['velocity'][1])
            landing_score = max(0, 1.0 - landing_velocity /
                                self.SAFE_LANDING_VELOCITY)
            total_score += landing_score
            num_metrics += 1

        # 4. Check movement magnitude
        positions = [np.array([t['position'] for t in recent_trajectories])]
        movement_magnitude = np.linalg.norm(positions[-1] - positions[0])
        if movement_magnitude > self.MIN_MOVEMENT_THRESHOLD:
            magnitude_score = min(movement_magnitude /
                                  (self.STABLE_VELOCITY_THRESHOLD * 2), 1.0)
            total_score += magnitude_score
            num_metrics += 1

        # 5. Evaluate action consistency
        actions = [t['action'] for t in recent_trajectories]
        if len(actions) >= 2:
            # Check if actions form a coherent sequence (not random button mashing)
            action_changes = sum(1 for i in range(
                len(actions)-1) if actions[i] != actions[i+1])
            action_consistency = max(0, 1.0 - action_changes / len(actions))
            total_score += action_consistency
            num_metrics += 1

        # Calculate final score
        if num_metrics == 0:
            return 0.0

        return total_score / num_metrics

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

    def reset(self):
        """Resets the evaluator for a new episode."""
        self.trajectory_history.clear()
        self.landing_attempts.clear()
        self.movement_segments.clear()
