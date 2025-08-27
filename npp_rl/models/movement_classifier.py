"""
Movement classification for N++ ninja physics.

This module classifies movement types and calculates physics parameters
based on ninja state and level geometry.
"""

import math
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum

from nclone.constants import (
    MAX_HOR_SPEED, GROUND_ACCEL,
    JUMP_FLAT_GROUND_Y, JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y
)

# Movement classification constants
MOVEMENT_THRESHOLD = 1e-6
WALK_SPEED_THRESHOLD = 0.5
JUMP_VELOCITY_THRESHOLD = 0.3
MIN_HORIZONTAL_VELOCITY = 0.1
WALL_CONTACT_DISTANCE = 15.0
LAUNCH_PAD_VELOCITY_MULTIPLIER = 1.5
HORIZONTAL_MOVEMENT_THRESHOLD = 2.0
UPWARD_MOVEMENT_THRESHOLD = -5.0
DOWNWARD_MOVEMENT_THRESHOLD = 5.0
DEFAULT_TIME_ESTIMATE = 1.0
DEFAULT_DIFFICULTY = 1.0
GRAVITY_APPROXIMATE = 0.067
JUMP_TIME_FALLBACK = 5.0
JUMP_ENERGY_BASE = 1.5
HEIGHT_FACTOR_DIVISOR = 50.0
HEIGHT_FACTOR_MAX = 2.0
DISTANCE_FACTOR_DIVISOR = 100.0
DISTANCE_FACTOR_MAX = 1.5
JUMP_DIFFICULTY_DIVISOR = 3.0
FALL_ENERGY_BASE = 0.5
FALL_ENERGY_DISTANCE_DIVISOR = 100.0
FALL_ENERGY_DISTANCE_MAX = 0.5
WALL_SLIDE_SPEED_DIVISOR = 20.0
WALL_SLIDE_MIN_TIME = 0.5
WALL_SLIDE_ENERGY_COST = 1.2
WALL_SLIDE_DIFFICULTY = 0.7
WALL_JUMP_ENERGY_BASE = 2.0
WALL_JUMP_DIFFICULTY = 0.8
LAUNCH_PAD_BOOST_FACTOR = 1.7
LAUNCH_PAD_GRAVITY_DIVISOR = 0.1
LAUNCH_PAD_MIN_TIME = 1.0
LAUNCH_PAD_ENERGY_COST = 0.3
LAUNCH_PAD_DIFFICULTY = 0.4
LAUNCH_PAD_DISTANCE_THRESHOLD = 100.0


class MovementType(IntEnum):
    """Types of movement between graph nodes."""
    WALK = 0        # Horizontal ground movement
    JUMP = 1        # Upward trajectory movement
    FALL = 2        # Downward gravity movement
    WALL_SLIDE = 3  # Wall contact movement
    WALL_JUMP = 4   # Wall-assisted jump
    LAUNCH_PAD = 5  # Launch pad boost


class NinjaState:
    """Simplified ninja state for movement classification."""

    def __init__(
        self,
        movement_state: int = 0,
        velocity: Tuple[float, float] = (0.0, 0.0),
        position: Tuple[float, float] = (0.0, 0.0),
        ground_contact: bool = True,
        wall_contact: bool = False
    ):
        self.movement_state = movement_state
        self.velocity = velocity
        self.position = position
        self.ground_contact = ground_contact
        self.wall_contact = wall_contact


class MovementClassifier:
    """
    Classifies movement types and calculates physics parameters.

    Uses ninja movement state constants from sim_mechanics_doc.md:
    States: 0=Immobile, 1=Running, 2=Ground Sliding, 3=Jumping, 4=Falling, 5=Wall Sliding
    """

    def __init__(self):
        """Initialize movement classifier with N++ physics constants."""
        pass

    def classify_movement(
        self,
        src_pos: Tuple[float, float],
        tgt_pos: Tuple[float, float],
        ninja_state: Optional[NinjaState] = None,
        level_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[MovementType, Dict[str, float]]:
        """
        Classify movement type and calculate physics parameters.

        Args:
            src_pos: Source position (x, y)
            tgt_pos: Target position (x, y)
            ninja_state: Current ninja state
            level_data: Level geometry data

        Returns:
            Tuple of (MovementType, physics_parameters_dict)
        """
        x0, y0 = src_pos
        x1, y1 = tgt_pos

        # Calculate displacement
        dx = x1 - x0
        dy = y1 - y0
        distance = math.sqrt(dx*dx + dy*dy)

        # Initialize physics parameters
        physics_params = {
            'distance': distance,
            'height_diff': dy,
            'horizontal_distance': abs(dx),
            'required_velocity': 0.0,
            'energy_cost': 1.0,
            'time_estimate': DEFAULT_TIME_ESTIMATE,
            'difficulty': DEFAULT_DIFFICULTY
        }

        # Classify based on movement characteristics
        movement_type = self._determine_movement_type(
            dx, dy, ninja_state, level_data
        )

        # Calculate type-specific physics parameters
        physics_params.update(
            self._calculate_physics_parameters(movement_type, dx, dy, ninja_state)
        )

        return movement_type, physics_params

    def _determine_movement_type(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState],
        level_data: Optional[Dict[str, Any]]
    ) -> MovementType:
        """Determine the primary movement type based on displacement and state."""

        # Check for wall-based movement first
        if ninja_state and ninja_state.wall_contact:
            if abs(dy) > abs(dx) and dy < 0:  # Upward wall movement
                return MovementType.WALL_JUMP
            else:
                return MovementType.WALL_SLIDE

        # Check for launch pad movement (would need level data analysis)
        if level_data and self._is_launch_pad_movement(dx, dy, level_data):
            return MovementType.LAUNCH_PAD

        # Classify based on vertical displacement
        if abs(dy) < HORIZONTAL_MOVEMENT_THRESHOLD:  # Mostly horizontal movement
            return MovementType.WALK
        elif dy < UPWARD_MOVEMENT_THRESHOLD:  # Significant upward movement
            return MovementType.JUMP
        elif dy > DOWNWARD_MOVEMENT_THRESHOLD:  # Significant downward movement
            return MovementType.FALL
        else:
            # Mixed movement - choose based on dominant component
            if abs(dx) > abs(dy):
                return MovementType.WALK
            elif dy < 0:
                return MovementType.JUMP
            else:
                return MovementType.FALL

    def _calculate_physics_parameters(
        self,
        movement_type: MovementType,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate physics parameters specific to movement type."""

        params = {}

        if movement_type == MovementType.WALK:
            params.update(self._calculate_walk_parameters(dx, dy, ninja_state))
        elif movement_type == MovementType.JUMP:
            params.update(self._calculate_jump_parameters(dx, dy, ninja_state))
        elif movement_type == MovementType.FALL:
            params.update(self._calculate_fall_parameters(dx, dy, ninja_state))
        elif movement_type == MovementType.WALL_SLIDE:
            params.update(self._calculate_wall_slide_parameters(dx, dy, ninja_state))
        elif movement_type == MovementType.WALL_JUMP:
            params.update(self._calculate_wall_jump_parameters(dx, dy, ninja_state))
        elif movement_type == MovementType.LAUNCH_PAD:
            params.update(self._calculate_launch_pad_parameters(dx, dy, ninja_state))

        return params

    def _calculate_walk_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for walking movement."""
        horizontal_distance = abs(dx)

        # Estimate time based on acceleration and max speed
        if horizontal_distance <= 0:
            time_estimate = MIN_HORIZONTAL_VELOCITY
        else:
            # Time to reach max speed: t = v / a
            accel_time = MAX_HOR_SPEED / GROUND_ACCEL
            accel_distance = 0.5 * GROUND_ACCEL * accel_time * accel_time

            if horizontal_distance <= accel_distance:
                # Pure acceleration phase
                time_estimate = math.sqrt(2 * horizontal_distance / GROUND_ACCEL)
            else:
                # Acceleration + constant speed
                remaining_distance = horizontal_distance - accel_distance
                constant_time = remaining_distance / MAX_HOR_SPEED
                time_estimate = accel_time + constant_time

        required_velocity = min(
            horizontal_distance / max(time_estimate, MIN_HORIZONTAL_VELOCITY),
            MAX_HOR_SPEED
        )
        energy_cost = required_velocity / MAX_HOR_SPEED
        difficulty = min(energy_cost, 1.0)

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': time_estimate,
            'difficulty': difficulty
        }

    def _calculate_jump_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for jumping movement."""
        # Use standard jump velocity as baseline
        initial_vy = abs(JUMP_FLAT_GROUND_Y)

        # Estimate time of flight for upward movement
        # Using: t = (-v0 + sqrt(v0² + 2*g*|dy|)) / g
        gravity = GRAVITY_APPROXIMATE
        if abs(dy) > 0:
            discriminant = initial_vy*initial_vy + 2*gravity*abs(dy)
            time_estimate = (
                (-initial_vy + math.sqrt(discriminant)) / gravity
                if discriminant >= 0 else JUMP_TIME_FALLBACK
            )
        else:
            time_estimate = 2 * initial_vy / gravity

        # Required horizontal velocity
        required_velocity = abs(dx) / max(time_estimate, MIN_HORIZONTAL_VELOCITY)
        required_velocity = min(required_velocity, MAX_HOR_SPEED)

        # Energy cost increases with height and distance
        height_factor = min(abs(dy) / HEIGHT_FACTOR_DIVISOR, HEIGHT_FACTOR_MAX)
        distance_factor = min(abs(dx) / DISTANCE_FACTOR_DIVISOR, DISTANCE_FACTOR_MAX)
        energy_cost = JUMP_ENERGY_BASE + height_factor + distance_factor

        difficulty = min(energy_cost / JUMP_DIFFICULTY_DIVISOR, 1.0)

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': time_estimate,
            'difficulty': difficulty
        }

    def _calculate_fall_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for falling movement."""
        # Time to fall: t = sqrt(2*h/g)
        gravity = GRAVITY_APPROXIMATE
        time_estimate = math.sqrt(2 * abs(dy) / gravity) if abs(dy) > 0 else MIN_HORIZONTAL_VELOCITY

        # Required horizontal velocity
        required_velocity = abs(dx) / max(time_estimate, MIN_HORIZONTAL_VELOCITY)
        required_velocity = min(required_velocity, MAX_HOR_SPEED)

        # Falling is generally easier than jumping
        energy_cost = FALL_ENERGY_BASE + min(
            abs(dx) / FALL_ENERGY_DISTANCE_DIVISOR, FALL_ENERGY_DISTANCE_MAX
        )
        difficulty = min(energy_cost, 1.0)

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': time_estimate,
            'difficulty': difficulty
        }

    def _calculate_wall_slide_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for wall sliding movement."""
        # Wall sliding has reduced gravity effect
        # Slower than free fall
        time_estimate = max(abs(dy) / WALL_SLIDE_SPEED_DIVISOR, WALL_SLIDE_MIN_TIME)
        required_velocity = abs(dx) / max(time_estimate, MIN_HORIZONTAL_VELOCITY)
        energy_cost = WALL_SLIDE_ENERGY_COST  # Moderate energy cost
        difficulty = WALL_SLIDE_DIFFICULTY   # Moderate difficulty

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': time_estimate,
            'difficulty': difficulty
        }

    def _calculate_wall_jump_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for wall jumping movement."""
        # Wall jumps have specific velocity components
        initial_vx = abs(JUMP_WALL_REGULAR_X)
        initial_vy = abs(JUMP_WALL_REGULAR_Y)

        # Estimate time based on vertical component
        gravity = GRAVITY_APPROXIMATE
        time_estimate = 2 * initial_vy / gravity

        required_velocity = max(initial_vx, initial_vy)
        energy_cost = WALL_JUMP_ENERGY_BASE  # Wall jumps are energy intensive
        difficulty = WALL_JUMP_DIFFICULTY   # High difficulty

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': time_estimate,
            'difficulty': difficulty
        }

    def _calculate_launch_pad_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for launch pad movement."""
        # Launch pads provide significant velocity boost
        boost_factor = LAUNCH_PAD_BOOST_FACTOR
        # High initial velocity
        time_estimate = max(
            math.sqrt(abs(dy) / LAUNCH_PAD_GRAVITY_DIVISOR), LAUNCH_PAD_MIN_TIME
        )
        required_velocity = MAX_HOR_SPEED * boost_factor
        energy_cost = LAUNCH_PAD_ENERGY_COST  # Launch pads reduce energy cost
        difficulty = LAUNCH_PAD_DIFFICULTY   # Easier with launch pad assistance

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': time_estimate,
            'difficulty': difficulty
        }

    def _is_launch_pad_movement(
        self,
        dx: float,
        dy: float,
        level_data: Dict[str, Any]
    ) -> bool:
        """Check if movement involves a launch pad (placeholder)."""
        # This would analyze level_data for launch pad entities
        # For now, detect based on large displacement
        distance = math.sqrt(dx*dx + dy*dy)
        return distance > LAUNCH_PAD_DISTANCE_THRESHOLD  # Large movements might involve launch pads
