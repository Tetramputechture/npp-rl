"""
Trajectory calculator for physics-based edge validation in N++ levels.

This module calculates jump trajectories using actual N++ physics constants
to determine movement feasibility, energy costs, and timing requirements.
"""

import math
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum

from nclone.constants import (
    GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED,
    JUMP_FLAT_GROUND_Y, MAX_JUMP_DURATION
)

# Physics calculation constants
VERTICAL_MOVEMENT_THRESHOLD = 1e-6
MIN_HORIZONTAL_VELOCITY = 0.1
ENERGY_COST_BASE = 10.0
ENERGY_COST_JUMP_MULTIPLIER = 2.0
SUCCESS_PROBABILITY_BASE = 0.8
SUCCESS_PROBABILITY_DISTANCE_FACTOR = 0.001
SUCCESS_PROBABILITY_HIGH_BASE = 0.95
DISTANCE_PENALTY_DIVISOR = 100.0
DISTANCE_PENALTY_MAX = 0.3
HEIGHT_PENALTY_DIVISOR = 50.0
HEIGHT_PENALTY_MAX = 0.2
VELOCITY_PENALTY_MAX = 0.2
TIME_PENALTY_DIVISOR = 30.0
TIME_PENALTY_MAX = 0.1
SUCCESS_PROBABILITY_MIN = 0.1
VELOCITY_MARGIN_MULTIPLIER = 2
JUMP_THRESHOLD_Y = -1.0
JUMP_THRESHOLD_VELOCITY = 0.5
DEFAULT_TRAJECTORY_POINTS = 10
DEFAULT_MINIMUM_TIME = 1.0


class MovementState(IntEnum):
    """Ninja movement states from sim_mechanics_doc.md"""
    IMMOBILE = 0
    RUNNING = 1
    GROUND_SLIDING = 2
    JUMPING = 3
    FALLING = 4
    WALL_SLIDING = 5
    WALL_JUMPING = 6
    LAUNCH_PAD = 7
    AIRBORNE = 8


@dataclass
class TrajectoryResult:
    """Result of trajectory calculation."""
    feasible: bool
    time_of_flight: float
    energy_cost: float
    success_probability: float
    min_velocity: float
    max_velocity: float
    requires_jump: bool
    requires_wall_contact: bool
    trajectory_points: List[Tuple[float, float]]


class TrajectoryCalculator:
    """
    Calculates physics-based trajectories for N++ ninja movement.

    Uses actual N++ physics constants to determine movement feasibility,
    energy costs, and timing requirements for graph edge validation.
    """

    def __init__(self):
        """Initialize trajectory calculator with N++ physics constants."""
        pass

    def calculate_jump_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        ninja_state: Optional[MovementState] = None
    ) -> TrajectoryResult:
        """
        Calculate quadratic trajectory for jump movement.

        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            ninja_state: Current ninja movement state

        Returns:
            TrajectoryResult with feasibility and physics parameters
        """
        x0, y0 = start_pos
        x1, y1 = end_pos

        # Calculate displacement
        dx = x1 - x0
        dy = y1 - y0
        distance = math.sqrt(dx*dx + dy*dy)

        # Determine gravity based on movement state
        gravity = self._get_gravity_for_state(ninja_state)

        # Check if horizontal movement is within speed limits
        if abs(dx) > MAX_HOR_SPEED * MAX_JUMP_DURATION:
            return TrajectoryResult(
                feasible=False, time_of_flight=0.0, energy_cost=float('inf'),
                success_probability=0.0, min_velocity=0.0, max_velocity=0.0,
                requires_jump=True, requires_wall_contact=False, trajectory_points=[]
            )

        # Calculate required time of flight for horizontal displacement
        if abs(dx) < VERTICAL_MOVEMENT_THRESHOLD:  # Vertical movement
            time_of_flight = self._calculate_vertical_time(dy, gravity)
        else:
            # Assume constant horizontal velocity
            horizontal_velocity = dx / MAX_JUMP_DURATION if abs(dx) > 0 else 0
            if abs(horizontal_velocity) > MAX_HOR_SPEED:
                horizontal_velocity = math.copysign(MAX_HOR_SPEED, horizontal_velocity)

            time_of_flight = abs(dx) / max(abs(horizontal_velocity), MIN_HORIZONTAL_VELOCITY)

        # Calculate required initial vertical velocity
        # Using kinematic equation: y = y0 + v0*t + 0.5*g*t²
        # Solving for v0: v0 = (y - y0 - 0.5*g*t²) / t
        if time_of_flight > 0:
            initial_vy = (dy - 0.5 * gravity * time_of_flight * time_of_flight) / time_of_flight
        else:
            initial_vy = 0

        # Check if initial velocity is achievable
        max_initial_vy = abs(JUMP_FLAT_GROUND_Y)
        if abs(initial_vy) > max_initial_vy * VELOCITY_MARGIN_MULTIPLIER:
            return TrajectoryResult(
                feasible=False, time_of_flight=time_of_flight, energy_cost=float('inf'),
                success_probability=0.0, min_velocity=abs(initial_vy), max_velocity=max_initial_vy,
                requires_jump=True, requires_wall_contact=False, trajectory_points=[]
            )

        # Calculate trajectory points for collision checking
        trajectory_points = self._generate_trajectory_points(
            start_pos, dx/time_of_flight if time_of_flight > 0 else 0,
            initial_vy, gravity, time_of_flight
        )

        # Calculate energy cost (based on initial velocity magnitude)
        horizontal_velocity = dx / time_of_flight if time_of_flight > 0 else 0
        velocity_magnitude = math.sqrt(
            horizontal_velocity*horizontal_velocity + initial_vy*initial_vy
        )
        energy_cost = velocity_magnitude / MAX_HOR_SPEED

        # Calculate success probability (based on trajectory difficulty)
        success_probability = self._calculate_success_probability(
            distance, abs(dy), velocity_magnitude, time_of_flight
        )

        # Determine movement requirements
        requires_jump = dy < JUMP_THRESHOLD_Y or abs(initial_vy) > JUMP_THRESHOLD_VELOCITY
        requires_wall_contact = ninja_state == MovementState.WALL_SLIDING

        return TrajectoryResult(
            feasible=True,
            time_of_flight=time_of_flight,
            energy_cost=energy_cost,
            success_probability=success_probability,
            min_velocity=min(abs(horizontal_velocity), abs(initial_vy)),
            max_velocity=max(abs(horizontal_velocity), abs(initial_vy)),
            requires_jump=requires_jump,
            requires_wall_contact=requires_wall_contact,
            trajectory_points=trajectory_points
        )

    def validate_trajectory_clearance(
        self,
        trajectory_points: List[Tuple[float, float]],
        level_data: dict
    ) -> bool:
        """
        Check if trajectory clears all obstacles using ninja radius.

        Args:
            trajectory_points: List of (x, y) points along trajectory
            level_data: Level collision data

        Returns:
            True if trajectory is clear, False if blocked
        """
        # This would integrate with nclone's collision system
        # For now, return True as a placeholder
        # In full implementation, would use sweep_circle_vs_tiles
        return True

    def _get_gravity_for_state(self, ninja_state: Optional[MovementState]) -> float:
        """Get appropriate gravity constant based on ninja state."""
        if ninja_state in [MovementState.JUMPING, MovementState.WALL_JUMPING]:
            return GRAVITY_JUMP
        else:
            return GRAVITY_FALL

    def _calculate_vertical_time(self, dy: float, gravity: float) -> float:
        """Calculate time for purely vertical movement."""
        if abs(dy) < VERTICAL_MOVEMENT_THRESHOLD:
            return MIN_HORIZONTAL_VELOCITY  # Minimum time for adjacent cells

        # For upward movement, use jump physics
        if dy < 0:
            # Solve: dy = v0*t + 0.5*g*t² where v0 is initial jump velocity
            v0 = abs(JUMP_FLAT_GROUND_Y)
            # Quadratic formula: 0.5*g*t² + v0*t - |dy| = 0
            discriminant = v0*v0 + 2*gravity*abs(dy)
            if discriminant >= 0:
                return (-v0 + math.sqrt(discriminant)) / gravity

        # For downward movement, assume starting from rest
        # dy = 0.5*g*t²
        return math.sqrt(2 * abs(dy) / gravity) if gravity > 0 else DEFAULT_MINIMUM_TIME

    def _generate_trajectory_points(
        self,
        start_pos: Tuple[float, float],
        vx: float,
        vy: float,
        gravity: float,
        time_of_flight: float,
        num_points: int = DEFAULT_TRAJECTORY_POINTS
    ) -> List[Tuple[float, float]]:
        """Generate points along the trajectory for collision checking."""
        points = []
        x0, y0 = start_pos

        for i in range(num_points + 1):
            t = (i / num_points) * time_of_flight
            x = x0 + vx * t
            y = y0 + vy * t + 0.5 * gravity * t * t
            points.append((x, y))

        return points

    def _calculate_success_probability(
        self,
        distance: float,
        height_diff: float,
        velocity_magnitude: float,
        time_of_flight: float
    ) -> float:
        """Calculate probability of successful trajectory execution."""
        # Base probability starts high for simple movements
        base_prob = SUCCESS_PROBABILITY_HIGH_BASE

        # Reduce probability for longer distances
        distance_penalty = min(distance / DISTANCE_PENALTY_DIVISOR, DISTANCE_PENALTY_MAX)

        # Reduce probability for large height differences
        height_penalty = min(abs(height_diff) / HEIGHT_PENALTY_DIVISOR, HEIGHT_PENALTY_MAX)

        # Reduce probability for high velocities
        velocity_penalty = min(velocity_magnitude / MAX_HOR_SPEED, VELOCITY_PENALTY_MAX)

        # Reduce probability for long flight times
        time_penalty = min(time_of_flight / TIME_PENALTY_DIVISOR, TIME_PENALTY_MAX)

        success_prob = (
            base_prob - distance_penalty - height_penalty - velocity_penalty - time_penalty
        )
        return max(SUCCESS_PROBABILITY_MIN, min(1.0, success_prob))
