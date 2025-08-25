"""
Movement classification for N++ ninja physics.

This module classifies movement types and calculates physics parameters
based on ninja state and level geometry.
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum

# Import N++ physics constants
try:
    from nclone.constants import (
        MAX_HOR_SPEED, NINJA_RADIUS, GROUND_ACCEL, AIR_ACCEL,
        JUMP_FLAT_GROUND_Y, JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y
    )
except ImportError:
    # Fallback constants
    MAX_HOR_SPEED = 3.333333333333333
    NINJA_RADIUS = 10
    GROUND_ACCEL = 0.06666666666666665
    AIR_ACCEL = 0.04444444444444444
    JUMP_FLAT_GROUND_Y = -2
    JUMP_WALL_REGULAR_X = 1
    JUMP_WALL_REGULAR_Y = -1.4


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
        self.max_hor_speed = MAX_HOR_SPEED
        self.ninja_radius = NINJA_RADIUS
        self.ground_accel = GROUND_ACCEL
        self.air_accel = AIR_ACCEL
        self.jump_flat_ground_y = JUMP_FLAT_GROUND_Y
        self.jump_wall_regular_x = JUMP_WALL_REGULAR_X
        self.jump_wall_regular_y = JUMP_WALL_REGULAR_Y
    
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
            'time_estimate': 1.0,
            'difficulty': 1.0
        }
        
        # Classify based on movement characteristics
        movement_type = self._determine_movement_type(
            dx, dy, distance, ninja_state, level_data
        )
        
        # Calculate type-specific physics parameters
        physics_params.update(
            self._calculate_physics_parameters(movement_type, dx, dy, distance, ninja_state)
        )
        
        return movement_type, physics_params
    
    def _determine_movement_type(
        self,
        dx: float,
        dy: float,
        distance: float,
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
        if abs(dy) < 2.0:  # Mostly horizontal movement
            return MovementType.WALK
        elif dy < -5.0:  # Significant upward movement
            return MovementType.JUMP
        elif dy > 5.0:  # Significant downward movement
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
        distance: float,
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
            time_estimate = 0.1
        else:
            # Time to reach max speed: t = v / a
            accel_time = self.max_hor_speed / self.ground_accel
            accel_distance = 0.5 * self.ground_accel * accel_time * accel_time
            
            if horizontal_distance <= accel_distance:
                # Pure acceleration phase
                time_estimate = math.sqrt(2 * horizontal_distance / self.ground_accel)
            else:
                # Acceleration + constant speed
                remaining_distance = horizontal_distance - accel_distance
                constant_time = remaining_distance / self.max_hor_speed
                time_estimate = accel_time + constant_time
        
        required_velocity = min(horizontal_distance / max(time_estimate, 0.1), self.max_hor_speed)
        energy_cost = required_velocity / self.max_hor_speed
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
        initial_vy = abs(self.jump_flat_ground_y)
        
        # Estimate time of flight for upward movement
        # Using: t = (-v0 + sqrt(v0² + 2*g*|dy|)) / g
        # Simplified with g ≈ 0.067
        gravity = 0.067  # Approximate gravity
        if abs(dy) > 0:
            discriminant = initial_vy*initial_vy + 2*gravity*abs(dy)
            time_estimate = (-initial_vy + math.sqrt(discriminant)) / gravity if discriminant >= 0 else 5.0
        else:
            time_estimate = 2 * initial_vy / gravity
        
        # Required horizontal velocity
        required_velocity = abs(dx) / max(time_estimate, 0.1)
        required_velocity = min(required_velocity, self.max_hor_speed)
        
        # Energy cost increases with height and distance
        height_factor = min(abs(dy) / 50.0, 2.0)
        distance_factor = min(abs(dx) / 100.0, 1.5)
        energy_cost = 1.5 + height_factor + distance_factor
        
        difficulty = min(energy_cost / 3.0, 1.0)
        
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
        gravity = 0.067
        time_estimate = math.sqrt(2 * abs(dy) / gravity) if abs(dy) > 0 else 0.1
        
        # Required horizontal velocity
        required_velocity = abs(dx) / max(time_estimate, 0.1)
        required_velocity = min(required_velocity, self.max_hor_speed)
        
        # Falling is generally easier than jumping
        energy_cost = 0.5 + min(abs(dx) / 100.0, 0.5)
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
        time_estimate = max(abs(dy) / 20.0, 0.5)  # Slower than free fall
        required_velocity = abs(dx) / max(time_estimate, 0.1)
        energy_cost = 1.2  # Moderate energy cost
        difficulty = 0.7   # Moderate difficulty
        
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
        initial_vx = abs(self.jump_wall_regular_x)
        initial_vy = abs(self.jump_wall_regular_y)
        
        # Estimate time based on vertical component
        gravity = 0.067
        time_estimate = 2 * initial_vy / gravity
        
        required_velocity = max(initial_vx, initial_vy)
        energy_cost = 2.0  # Wall jumps are energy intensive
        difficulty = 0.8   # High difficulty
        
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
        boost_factor = 1.7  # From constants
        time_estimate = max(math.sqrt(abs(dy) / 0.1), 1.0)  # High initial velocity
        required_velocity = self.max_hor_speed * boost_factor
        energy_cost = 0.3  # Launch pads reduce energy cost
        difficulty = 0.4   # Easier with launch pad assistance
        
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
        return distance > 100.0  # Large movements might involve launch pads