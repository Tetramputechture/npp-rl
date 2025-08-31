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
    GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED, AIR_ACCEL, GROUND_ACCEL,
    JUMP_FLAT_GROUND_Y, MAX_JUMP_DURATION, NINJA_RADIUS,
    DRAG_REGULAR, DRAG_SLOW, FRICTION_GROUND, FRICTION_WALL,
    JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X, JUMP_WALL_SLIDE_Y
)
from nclone.physics import sweep_circle_vs_tiles

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
        if not trajectory_points or len(trajectory_points) < 2:
            return True
        
        if level_data is None:
            return True  # No level data, assume clear
            
        # Get simulation object from level_data if available
        sim = level_data.get('sim', None)
        if not sim:
            # If no simulation object, try basic tile-based validation
            return self._validate_trajectory_basic(trajectory_points, level_data)
            
        # Use nclone's sweep_circle_vs_tiles for accurate collision detection
        for i in range(len(trajectory_points) - 1):
            x0, y0 = trajectory_points[i]
            x1, y1 = trajectory_points[i + 1]
            
            # Calculate movement vector
            dx = x1 - x0
            dy = y1 - y0
            
            # Use sweep_circle_vs_tiles to check for collisions
            collision_result = sweep_circle_vs_tiles(sim, x0, y0, dx, dy, NINJA_RADIUS)
            
            # If collision_result indicates a collision, trajectory is blocked
            if collision_result and collision_result.get('collision', False):
                return False
                
        return True
        
    def _validate_trajectory_basic(
        self,
        trajectory_points: List[Tuple[float, float]],
        level_data: dict
    ) -> bool:
        """
        Basic trajectory validation without full physics simulation.
        
        Args:
            trajectory_points: List of (x, y) points along trajectory
            level_data: Level data containing tile information
            
        Returns:
            True if trajectory appears clear, False if likely blocked
        """
        # Get tile data from level_data
        tiles = level_data.get('tiles', None)
        if tiles is None:
            return True  # No tile data available, assume clear
        
        # Handle numpy arrays properly
        if hasattr(tiles, 'size') and tiles.size == 0:
            return True  # Empty array, assume clear
            
        # Check each point along trajectory for tile collisions
        for point in trajectory_points:
            # Handle both tuple (x, y) and dict {'x': x, 'y': y} formats
            if isinstance(point, dict):
                x, y = point['x'], point['y']
            else:
                x, y = point
            # Convert world coordinates to tile coordinates
            tile_x = int(x // 24)  # TILE_PIXEL_SIZE = 24
            tile_y = int(y // 24)
            
            # Check if tile exists and is solid
            if self._is_solid_tile(tiles, tile_x, tile_y):
                # Check if ninja circle overlaps with solid tile
                tile_world_x = tile_x * 24 + 12  # Center of tile
                tile_world_y = tile_y * 24 + 12
                
                # Distance from ninja center to tile center
                dist_sq = (x - tile_world_x)**2 + (y - tile_world_y)**2
                
                # If ninja radius overlaps with tile (approximate as circle vs square)
                if dist_sq < (NINJA_RADIUS + 12)**2:  # 12 is half tile size
                    return False
                    
        return True
        
    def _is_solid_tile(self, tiles: any, tile_x: int, tile_y: int) -> bool:
        """Check if a tile at given coordinates is solid."""
        try:
            # Handle different tile data formats
            if hasattr(tiles, '__getitem__'):
                if hasattr(tiles, 'shape') and len(tiles.shape) == 2:
                    # NumPy array format
                    if 0 <= tile_y < tiles.shape[0] and 0 <= tile_x < tiles.shape[1]:
                        return tiles[tile_y, tile_x] != 0  # 0 = empty, non-zero = solid
                elif isinstance(tiles, (list, tuple)):
                    # List format
                    if 0 <= tile_y < len(tiles) and 0 <= tile_x < len(tiles[tile_y]):
                        return tiles[tile_y][tile_x] != 0
                elif isinstance(tiles, dict):
                    # Dictionary format
                    return tiles.get((tile_x, tile_y), 0) != 0
            return False
        except (IndexError, KeyError, AttributeError):
            return False  # Assume empty if can't access

    def calculate_momentum_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        initial_velocity: Tuple[float, float],
        ninja_state: Optional[MovementState] = None,
        level_data: Optional[dict] = None
    ) -> TrajectoryResult:
        """
        Calculate trajectory considering initial momentum and physics constraints.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            initial_velocity: Initial velocity (vx, vy)
            ninja_state: Current ninja movement state
            level_data: Level geometry data for collision detection
            
        Returns:
            TrajectoryResult with momentum-dependent physics parameters
        """
        x0, y0 = start_pos
        x1, y1 = end_pos
        vx0, vy0 = initial_velocity
        
        dx = x1 - x0
        dy = y1 - y0
        
        # Get appropriate physics constants based on ninja state
        gravity = self._get_gravity_for_state(ninja_state)
        drag = self._get_drag_for_state(ninja_state)
        accel = self._get_acceleration_for_state(ninja_state)
        
        # Calculate trajectory with momentum conservation
        trajectory_points = []
        feasible = True
        
        # Simulate trajectory step by step
        dt = 0.1  # Time step for simulation
        max_time = 10.0  # Maximum simulation time
        
        x, y = x0, y0
        vx, vy = vx0, vy0
        t = 0.0
        
        trajectory_points.append((x, y))
        
        while t < max_time:
            # Apply physics forces
            # Gravity
            vy += gravity * dt
            
            # Drag (air resistance)
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > 0:
                drag_force = drag * speed
                vx -= (vx / speed) * drag_force * dt
                vy -= (vy / speed) * drag_force * dt
            
            # Apply acceleration constraints
            if ninja_state in [MovementState.RUNNING, MovementState.GROUND_SLIDING]:
                # Ground movement - can accelerate horizontally
                if abs(vx) < MAX_HOR_SPEED:
                    target_vx = MAX_HOR_SPEED if dx > 0 else -MAX_HOR_SPEED
                    accel_dir = 1 if target_vx > vx else -1
                    vx += accel_dir * accel * dt
                    vx = max(-MAX_HOR_SPEED, min(MAX_HOR_SPEED, vx))
            elif ninja_state in [MovementState.JUMPING, MovementState.FALLING, MovementState.AIRBORNE]:
                # Air movement - limited horizontal acceleration
                if abs(vx) < MAX_HOR_SPEED:
                    target_vx = MAX_HOR_SPEED if dx > 0 else -MAX_HOR_SPEED
                    accel_dir = 1 if target_vx > vx else -1
                    vx += accel_dir * AIR_ACCEL * dt
                    vx = max(-MAX_HOR_SPEED, min(MAX_HOR_SPEED, vx))
            
            # Update position
            x += vx * dt
            y += vy * dt
            t += dt
            
            trajectory_points.append((x, y))
            
            # Check if we've reached the target (within tolerance)
            dist_to_target = math.sqrt((x - x1)**2 + (y - y1)**2)
            if dist_to_target < NINJA_RADIUS:
                break
                
            # Check if we've overshot significantly
            if abs(x - x1) > abs(dx) * 2 or abs(y - y1) > abs(dy) * 2:
                feasible = False
                break
        
        # Validate trajectory against level geometry
        if feasible and level_data:
            feasible = self.validate_trajectory_clearance(trajectory_points, level_data)
        
        # Calculate physics parameters
        time_of_flight = t
        
        # Energy cost based on required velocity changes and time
        initial_speed = math.sqrt(vx0*vx0 + vy0*vy0)
        final_speed = math.sqrt(vx*vx + vy*vy)
        energy_cost = ENERGY_COST_BASE + abs(final_speed - initial_speed) * ENERGY_COST_JUMP_MULTIPLIER
        
        # Success probability based on trajectory complexity
        success_probability = self._calculate_success_probability(
            dx, dy, time_of_flight, initial_speed, ninja_state
        )
        
        # Velocity requirements
        max_velocity = max(initial_speed, final_speed)
        min_velocity = min(initial_speed, final_speed)
        
        # Movement requirements
        requires_jump = abs(dy) > JUMP_THRESHOLD_Y or ninja_state in [
            MovementState.JUMPING, MovementState.WALL_JUMPING
        ]
        requires_wall_contact = ninja_state in [
            MovementState.WALL_SLIDING, MovementState.WALL_JUMPING
        ]
        
        return TrajectoryResult(
            feasible=feasible,
            time_of_flight=time_of_flight,
            energy_cost=energy_cost,
            success_probability=success_probability,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            requires_jump=requires_jump,
            requires_wall_contact=requires_wall_contact,
            trajectory_points=trajectory_points
        )

    def calculate_wall_jump_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        wall_normal: Tuple[float, float],
        ninja_state: Optional[MovementState] = None,
        level_data: Optional[dict] = None
    ) -> TrajectoryResult:
        """
        Calculate trajectory for wall jump movement with proper physics.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            wall_normal: Wall normal vector (nx, ny)
            ninja_state: Current ninja movement state
            level_data: Level geometry data
            
        Returns:
            TrajectoryResult for wall jump movement
        """
        x0, y0 = start_pos
        x1, y1 = end_pos
        nx, ny = wall_normal
        
        dx = x1 - x0
        dy = y1 - y0
        
        # Determine wall jump type based on movement direction
        if abs(dy) > abs(dx) and dy < 0:
            # Regular wall jump (away from wall, upward)
            initial_vx = JUMP_WALL_REGULAR_X * (-nx)  # Away from wall
            initial_vy = -abs(JUMP_WALL_REGULAR_Y)  # Upward
        else:
            # Wall slide jump (along wall)
            initial_vx = JUMP_WALL_SLIDE_X * (-nx)
            initial_vy = -abs(JUMP_WALL_SLIDE_Y)
        
        # Use momentum trajectory calculation with wall jump initial velocity
        return self.calculate_momentum_trajectory(
            start_pos, end_pos, (initial_vx, initial_vy),
            MovementState.WALL_JUMPING, level_data
        )

    def _get_drag_for_state(self, ninja_state: Optional[MovementState]) -> float:
        """Get appropriate drag constant based on ninja state."""
        if ninja_state in [MovementState.GROUND_SLIDING, MovementState.WALL_SLIDING]:
            return DRAG_SLOW
        else:
            return DRAG_REGULAR
            
    def _get_acceleration_for_state(self, ninja_state: Optional[MovementState]) -> float:
        """Get appropriate acceleration constant based on ninja state."""
        if ninja_state in [MovementState.RUNNING, MovementState.GROUND_SLIDING]:
            return GROUND_ACCEL
        else:
            return AIR_ACCEL

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
        dx: float,
        dy: float,
        time_of_flight: float,
        velocity: float,
        ninja_state: Optional[MovementState] = None
    ) -> float:
        """
        Calculate success probability based on movement complexity.
        
        Args:
            dx: Horizontal displacement
            dy: Vertical displacement
            time_of_flight: Time required for movement
            velocity: Required velocity
            ninja_state: Current ninja state
            
        Returns:
            Success probability between 0.0 and 1.0
        """
        # Base success probability
        base_prob = SUCCESS_PROBABILITY_BASE
        
        # Adjust based on movement type
        if ninja_state in [MovementState.WALL_JUMPING, MovementState.WALL_SLIDING]:
            base_prob = SUCCESS_PROBABILITY_BASE * 0.8  # Wall movements are harder
        elif ninja_state == MovementState.LAUNCH_PAD:
            base_prob = SUCCESS_PROBABILITY_HIGH_BASE  # Launch pads are easier
        
        # Distance penalty
        distance = math.sqrt(dx*dx + dy*dy)
        distance_penalty = min(distance / DISTANCE_PENALTY_DIVISOR, DISTANCE_PENALTY_MAX)
        
        # Height penalty (upward movement is harder)
        height_penalty = 0.0
        if dy < 0:  # Upward movement
            height_penalty = min(abs(dy) / HEIGHT_PENALTY_DIVISOR, HEIGHT_PENALTY_MAX)
        
        # Velocity penalty (high velocity requirements are harder)
        velocity_penalty = 0.0
        if velocity > MAX_HOR_SPEED * 0.8:
            velocity_penalty = min((velocity - MAX_HOR_SPEED * 0.8) / MAX_HOR_SPEED, VELOCITY_PENALTY_MAX)
        
        # Time penalty (very short or very long movements are harder)
        time_penalty = 0.0
        if time_of_flight < 1.0:
            time_penalty = min((1.0 - time_of_flight) / TIME_PENALTY_DIVISOR, TIME_PENALTY_MAX)
        elif time_of_flight > MAX_JUMP_DURATION:
            time_penalty = min((time_of_flight - MAX_JUMP_DURATION) / TIME_PENALTY_DIVISOR, TIME_PENALTY_MAX)
        
        # Calculate final probability
        final_prob = base_prob - distance_penalty - height_penalty - velocity_penalty - time_penalty
        
        return max(final_prob, SUCCESS_PROBABILITY_MIN)
