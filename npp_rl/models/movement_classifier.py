"""
Movement classification for N++ ninja physics.

This module classifies movement types and calculates physics parameters
based on ninja state and level geometry.
"""

import math
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum

from nclone.constants import (
    MAX_HOR_SPEED, GROUND_ACCEL, AIR_ACCEL,
    JUMP_FLAT_GROUND_Y, JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X, JUMP_WALL_SLIDE_Y,
    NINJA_RADIUS, JUMP_LAUNCH_PAD_BOOST_SCALAR, JUMP_LAUNCH_PAD_BOOST_FACTOR,
    GRAVITY_FALL, GRAVITY_JUMP, MAX_JUMP_DURATION,
    DRAG_REGULAR, DRAG_SLOW, FRICTION_GROUND, FRICTION_WALL,
    MAX_SURVIVABLE_IMPACT
)
from nclone.entity_classes.entity_launch_pad import EntityLaunchPad
from nclone.physics import overlap_circle_vs_circle, map_orientation_to_vector

# Movement classification constants
MOVEMENT_THRESHOLD = 1e-6
WALK_SPEED_THRESHOLD = 0.5
JUMP_VELOCITY_THRESHOLD = 0.3
MIN_HORIZONTAL_VELOCITY = 0.1
WALL_CONTACT_DISTANCE = 15.0
LAUNCH_PAD_VELOCITY_MULTIPLIER = 1.5

# Approximate gravity for backward compatibility
GRAVITY_APPROXIMATE = (GRAVITY_FALL + GRAVITY_JUMP) / 2
HORIZONTAL_MOVEMENT_THRESHOLD = 2.0
UPWARD_MOVEMENT_THRESHOLD = -5.0
DOWNWARD_MOVEMENT_THRESHOLD = 5.0
DEFAULT_TIME_ESTIMATE = 1.0
DEFAULT_DIFFICULTY = 1.0
# Use actual N++ gravity constants instead of approximation
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
        # Cache for static entity data (positions never change during level)
        self._entity_cache = {}
        self._current_level_id = None

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
        """Calculate parameters for jumping movement using actual N++ physics."""
        # Use actual N++ jump velocity
        initial_vy = abs(JUMP_FLAT_GROUND_Y)  # 2.0 pixels/frame

        # Use actual N++ gravity constants
        # During jump, gravity is reduced (GRAVITY_JUMP = 0.0111...)
        # During fall, gravity is higher (GRAVITY_FALL = 0.0666...)
        gravity_up = GRAVITY_JUMP
        gravity_down = GRAVITY_FALL

        # Calculate time of flight using actual physics
        if abs(dy) > 0:
            if dy < 0:  # Upward movement
                # Time to reach peak: t_up = v0 / g_up
                t_up = initial_vy / gravity_up
                # Height at peak: h_peak = v0² / (2 * g_up)
                h_peak = (initial_vy * initial_vy) / (2 * gravity_up)
                
                if abs(dy) <= h_peak:
                    # Jump doesn't reach peak, solve: dy = v0*t - 0.5*g*t²
                    # Using quadratic formula: t = (v0 ± sqrt(v0² - 2*g*dy)) / g
                    discriminant = initial_vy*initial_vy - 2*gravity_up*abs(dy)
                    if discriminant >= 0:
                        time_estimate = (initial_vy - math.sqrt(discriminant)) / gravity_up
                    else:
                        time_estimate = JUMP_TIME_FALLBACK
                else:
                    # Jump reaches peak and continues falling
                    remaining_height = abs(dy) - h_peak
                    # Time to fall remaining height: t = sqrt(2*h/g)
                    t_down = math.sqrt(2 * remaining_height / gravity_down)
                    time_estimate = t_up + t_down
            else:  # Downward movement (falling)
                # Pure fall: t = sqrt(2*h/g)
                time_estimate = math.sqrt(2 * abs(dy) / gravity_down)
        else:
            # Horizontal jump
            time_estimate = 2 * initial_vy / gravity_up

        # Account for air resistance and drag
        time_estimate *= (1.0 / DRAG_REGULAR)  # Adjust for drag effects

        # Required horizontal velocity considering air acceleration
        if time_estimate > 0:
            required_velocity = abs(dx) / time_estimate
            # Limit by air acceleration capabilities
            max_air_velocity = AIR_ACCEL * time_estimate
            required_velocity = min(required_velocity, min(MAX_HOR_SPEED, max_air_velocity))
        else:
            required_velocity = MAX_HOR_SPEED

        # Energy cost based on actual physics requirements
        height_factor = min(abs(dy) / HEIGHT_FACTOR_DIVISOR, HEIGHT_FACTOR_MAX)
        distance_factor = min(abs(dx) / DISTANCE_FACTOR_DIVISOR, DISTANCE_FACTOR_MAX)
        
        # Account for air vs ground movement energy difference
        air_movement_penalty = 1.2 if not (ninja_state and ninja_state.ground_contact) else 1.0
        energy_cost = (JUMP_ENERGY_BASE + height_factor + distance_factor) * air_movement_penalty

        # Difficulty increases with precision requirements and physics constraints
        velocity_difficulty = required_velocity / MAX_HOR_SPEED
        timing_difficulty = min(time_estimate / MAX_JUMP_DURATION, 1.0)
        difficulty = min((energy_cost + velocity_difficulty + timing_difficulty) / JUMP_DIFFICULTY_DIVISOR, 1.0)

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
        """Calculate parameters for wall jumping movement using actual N++ physics."""
        # Determine wall jump type based on movement characteristics
        if abs(dy) > abs(dx) and dy < 0:
            # Regular wall jump (away from wall, upward)
            initial_vx = JUMP_WALL_REGULAR_X  # 1.0 pixels/frame
            initial_vy = abs(JUMP_WALL_REGULAR_Y)  # 1.4 pixels/frame
        else:
            # Wall slide jump (along wall)
            initial_vx = JUMP_WALL_SLIDE_X  # 2/3 pixels/frame
            initial_vy = abs(JUMP_WALL_SLIDE_Y)  # 1.0 pixels/frame

        # Use actual N++ gravity for wall jump physics
        gravity_up = GRAVITY_JUMP
        gravity_down = GRAVITY_FALL

        # Calculate time of flight
        if abs(dy) > 0:
            if dy < 0:  # Upward wall jump
                # Time to reach peak
                t_up = initial_vy / gravity_up
                h_peak = (initial_vy * initial_vy) / (2 * gravity_up)
                
                if abs(dy) <= h_peak:
                    # Wall jump doesn't reach full peak
                    discriminant = initial_vy*initial_vy - 2*gravity_up*abs(dy)
                    if discriminant >= 0:
                        time_estimate = (initial_vy - math.sqrt(discriminant)) / gravity_up
                    else:
                        time_estimate = JUMP_TIME_FALLBACK
                else:
                    # Wall jump reaches peak and falls
                    remaining_height = abs(dy) - h_peak
                    t_down = math.sqrt(2 * remaining_height / gravity_down)
                    time_estimate = t_up + t_down
            else:  # Downward wall movement
                time_estimate = math.sqrt(2 * abs(dy) / gravity_down)
        else:
            # Horizontal wall jump
            time_estimate = 2 * initial_vy / gravity_up

        # Account for wall friction effects
        time_estimate *= (1.0 / FRICTION_WALL)

        # Required velocity considering wall jump physics
        horizontal_velocity_needed = abs(dx) / max(time_estimate, MIN_HORIZONTAL_VELOCITY)
        
        # Wall jumps have specific velocity constraints
        required_velocity = max(
            horizontal_velocity_needed,
            math.sqrt(initial_vx*initial_vx + initial_vy*initial_vy)
        )
        
        # Limit by maximum survivable impact and air acceleration
        max_safe_velocity = MAX_SURVIVABLE_IMPACT
        required_velocity = min(required_velocity, max_safe_velocity)

        # Energy cost for wall jumps (higher due to precision requirements)
        base_energy = WALL_JUMP_ENERGY_BASE
        
        # Additional cost for complex wall jump sequences
        precision_penalty = 1.0
        if ninja_state and ninja_state.wall_contact:
            # Wall jump from wall contact requires more precision
            precision_penalty = 1.3
            
        energy_cost = base_energy * precision_penalty

        # Difficulty based on timing precision and velocity requirements
        velocity_difficulty = required_velocity / MAX_HOR_SPEED
        timing_difficulty = 1.0 - min(time_estimate / MAX_JUMP_DURATION, 1.0)  # Shorter time = harder
        wall_difficulty_base = WALL_JUMP_DIFFICULTY
        
        difficulty = min(wall_difficulty_base + velocity_difficulty + timing_difficulty, 1.0)

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': max(time_estimate, MIN_HORIZONTAL_VELOCITY),
            'difficulty': difficulty
        }

    def _calculate_launch_pad_parameters(
        self,
        dx: float,
        dy: float,
        ninja_state: Optional[NinjaState]
    ) -> Dict[str, float]:
        """Calculate parameters for launch pad movement using actual N++ physics."""
        # Use actual launch pad boost values from nclone
        base_boost = EntityLaunchPad.BOOST  # 36/7 ≈ 5.14 pixels/frame
        
        # Calculate boost velocity components (with scaling factors from constants)
        boost_velocity = base_boost * JUMP_LAUNCH_PAD_BOOST_FACTOR  # 2/3 scaling
        
        # Estimate time based on actual trajectory physics
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Estimate time based on boost velocity and gravity effects
            # For upward movement, account for gravity deceleration
            if dy < 0:  # Upward movement
                # Use actual N++ gravity for launch pad trajectory
                gravity = GRAVITY_FALL  # Launch pad trajectories use fall gravity
                initial_velocity = boost_velocity
                discriminant = initial_velocity**2 + 2 * gravity * abs(dy)
                if discriminant >= 0:
                    time_estimate = (-initial_velocity + math.sqrt(discriminant)) / gravity
                else:
                    time_estimate = LAUNCH_PAD_MIN_TIME
            else:  # Downward or horizontal movement
                time_estimate = distance / boost_velocity
        else:
            time_estimate = LAUNCH_PAD_MIN_TIME
            
        # Launch pads provide high velocity with reduced energy cost
        required_velocity = boost_velocity
        energy_cost = LAUNCH_PAD_ENERGY_COST  # Much lower than normal movement
        difficulty = LAUNCH_PAD_DIFFICULTY   # Easier with launch pad assistance

        return {
            'required_velocity': required_velocity,
            'energy_cost': energy_cost,
            'time_estimate': max(time_estimate, LAUNCH_PAD_MIN_TIME),
            'difficulty': difficulty
        }

    def _is_launch_pad_movement(
        self,
        dx: float,
        dy: float,
        level_data: Dict[str, Any]
    ) -> bool:
        """Check if movement involves a launch pad using cached entity data."""
        if not level_data:
            return False
            
        # Cache static entities for performance (only done once per level)
        self._cache_static_entities(level_data)
        
        # Calculate movement vector properties
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if movement is large enough to potentially involve launch pad
        if distance < EntityLaunchPad.BOOST * 0.5:  # Minimum boost distance
            return False
            
        # Use cached launch pad data for performance
        launch_pads = self._entity_cache.get('launch_pads', [])
        if not launch_pads:
            return False
            
        # Normalize actual movement direction
        move_dir_x = dx / distance
        move_dir_y = dy / distance
        
        # Check against all cached launch pads
        for launch_pad in launch_pads:
            boost_distance = launch_pad['boost_distance']
            
            if boost_distance > 0:
                # Use pre-calculated boost direction vectors
                boost_dir_x = launch_pad['boost_dir_x']
                boost_dir_y = launch_pad['boost_dir_y']
                
                # Calculate dot product to check alignment
                alignment = boost_dir_x * move_dir_x + boost_dir_y * move_dir_y
                
                # If movement aligns well with launch pad direction and distance is significant
                if alignment > 0.7 and distance > boost_distance * 0.3:
                    return True
                        
        return False

    def _cache_static_entities(self, level_data: Dict[str, Any]) -> None:
        """
        Cache static entity positions since they never change during a level.
        
        Static entities include:
        - Launch pads (never change position)
        - Mines (never change position) 
        - One-way platforms (never change position)
        - Switches (never change position)
        """
        level_id = level_data.get('level_id', id(level_data))
        
        # Only recache if level changed
        if self._current_level_id == level_id:
            return
            
        self._current_level_id = level_id
        self._entity_cache = {
            'launch_pads': [],
            'mines': [],
            'switches': [],
            'one_way_platforms': []
        }
        
        entities = level_data.get('entities', [])
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_type = entity.get('type', None)
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            
            # Cache launch pads
            if entity_type == EntityLaunchPad.ENTITY_TYPE:
                orientation = entity.get('orientation', 0)
                normal_x, normal_y = map_orientation_to_vector(orientation)
                
                # Pre-calculate boost vectors for performance
                yboost_scale = 1
                if normal_y < 0:
                    yboost_scale = 1 - normal_y
                    
                expected_boost_x = normal_x * EntityLaunchPad.BOOST * JUMP_LAUNCH_PAD_BOOST_FACTOR
                expected_boost_y = normal_y * EntityLaunchPad.BOOST * yboost_scale * JUMP_LAUNCH_PAD_BOOST_FACTOR
                boost_distance = math.sqrt(expected_boost_x**2 + expected_boost_y**2)
                
                self._entity_cache['launch_pads'].append({
                    'x': entity_x,
                    'y': entity_y,
                    'orientation': orientation,
                    'boost_x': expected_boost_x,
                    'boost_y': expected_boost_y,
                    'boost_distance': boost_distance,
                    'boost_dir_x': expected_boost_x / boost_distance if boost_distance > 0 else 0,
                    'boost_dir_y': expected_boost_y / boost_distance if boost_distance > 0 else 0
                })
            
            # Cache other static entities for future use
            # TODO: Add mine, switch, platform caching as needed

    def classify_movement_sequence(
        self,
        positions: list[Tuple[float, float]],
        ninja_states: Optional[list[NinjaState]] = None,
        level_data: Optional[Dict[str, Any]] = None
    ) -> list[Tuple[MovementType, Dict[str, float]]]:
        """
        Classify a sequence of movements for complex movement chains.
        
        Args:
            positions: List of (x, y) positions in sequence
            ninja_states: Optional list of ninja states for each position
            level_data: Level geometry data
            
        Returns:
            List of (MovementType, physics_parameters) for each movement segment
        """
        if len(positions) < 2:
            return []
            
        movements = []
        
        for i in range(len(positions) - 1):
            src_pos = positions[i]
            tgt_pos = positions[i + 1]
            
            # Get ninja state for this segment
            ninja_state = ninja_states[i] if ninja_states and i < len(ninja_states) else None
            
            # Classify individual movement
            movement_type, params = self.classify_movement(
                src_pos, tgt_pos, ninja_state, level_data
            )
            
            # Adjust parameters for movement chains
            if i > 0:
                # Previous movement affects current movement
                prev_movement_type, prev_params = movements[i - 1]
                params = self._adjust_for_movement_chain(
                    movement_type, params, prev_movement_type, prev_params
                )
                
            movements.append((movement_type, params))
            
        return movements
        
    def _adjust_for_movement_chain(
        self,
        current_type: MovementType,
        current_params: Dict[str, float],
        prev_type: MovementType,
        prev_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust movement parameters based on previous movement in chain."""
        adjusted_params = current_params.copy()
        
        # Chain bonuses and penalties
        if prev_type == MovementType.WALL_JUMP and current_type == MovementType.WALL_JUMP:
            # Wall jump chains are more difficult
            adjusted_params['difficulty'] = min(adjusted_params['difficulty'] * 1.2, 1.0)
            adjusted_params['energy_cost'] *= 1.1
            
        elif prev_type == MovementType.LAUNCH_PAD:
            # Launch pad provides momentum for next movement
            adjusted_params['energy_cost'] *= 0.9  # Easier follow-up
            adjusted_params['required_velocity'] *= 1.1  # Higher velocity available
            
        elif prev_type == MovementType.JUMP and current_type == MovementType.WALL_SLIDE:
            # Jump to wall slide is a common sequence
            adjusted_params['difficulty'] *= 0.95  # Slightly easier
            
        elif prev_type == MovementType.WALL_SLIDE and current_type == MovementType.WALL_JUMP:
            # Wall slide to wall jump is natural
            adjusted_params['energy_cost'] *= 0.9
            adjusted_params['time_estimate'] *= 0.95  # Faster transition
            
        # Momentum conservation effects
        prev_velocity = prev_params.get('required_velocity', 0)
        if prev_velocity > MAX_HOR_SPEED * 0.8:  # High velocity from previous movement
            # Momentum helps with current movement
            velocity_bonus = min(prev_velocity * 0.1, MAX_HOR_SPEED * 0.2)
            adjusted_params['required_velocity'] = max(
                0, adjusted_params['required_velocity'] - velocity_bonus
            )
            
        return adjusted_params
