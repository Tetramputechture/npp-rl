"""
Trajectory calculator for physics-based edge validation in N++ levels.

This module calculates jump trajectories using actual N++ physics constants
to determine movement feasibility, energy costs, and timing requirements.
"""

import math
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum

from nclone.constants.entity_types import EntityType
from nclone.constants import (
    GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED, AIR_ACCEL, GROUND_ACCEL,
    JUMP_FLAT_GROUND_Y, MAX_JUMP_DURATION, NINJA_RADIUS,
    DRAG_REGULAR, DRAG_SLOW, JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y,
    JUMP_WALL_SLIDE_X, JUMP_WALL_SLIDE_Y,
    TILE_PIXEL_SIZE,
    # Movement and trajectory constants
    VERTICAL_MOVEMENT_THRESHOLD, MIN_HORIZONTAL_VELOCITY,
    ENERGY_COST_BASE, ENERGY_COST_JUMP_MULTIPLIER,
    SUCCESS_PROBABILITY_BASE, SUCCESS_PROBABILITY_HIGH_BASE, DISTANCE_PENALTY_DIVISOR, DISTANCE_PENALTY_MAX,
    HEIGHT_PENALTY_DIVISOR, HEIGHT_PENALTY_MAX, VELOCITY_PENALTY_MAX,
    TIME_PENALTY_DIVISOR, TIME_PENALTY_MAX, SUCCESS_PROBABILITY_MIN,
    VELOCITY_MARGIN_MULTIPLIER, JUMP_THRESHOLD_Y, JUMP_THRESHOLD_VELOCITY,
    DEFAULT_TRAJECTORY_POINTS, DEFAULT_MINIMUM_TIME,
    # Win condition constants
    SWITCH_DOOR_MAX_DISTANCE, WIN_CONDITION_SWITCH_BONUS,
    WIN_CONDITION_EXIT_BONUS, WIN_CONDITION_DOOR_BONUS, WIN_CONDITION_DOOR_PROXIMITY
)
from nclone.physics import sweep_circle_vs_tiles

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
        # Cache for static level geometry (tiles never change during level)
        self._tile_cache = {}
        self._current_level_id = None

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
            distance, abs(dy), velocity_magnitude, time_of_flight,
            ninja_state, start_pos, end_pos
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
        Basic trajectory validation using cached level geometry.
        
        Args:
            trajectory_points: List of (x, y) points along trajectory
            level_data: Level data containing tile information
            
        Returns:
            True if trajectory appears clear, False if likely blocked
        """
        # Cache level geometry for performance (only done once per level)
        self._cache_level_geometry(level_data)
        
        # If no cached tiles, assume clear
        if not self._tile_cache:
            return True
            
        # Check each point along trajectory for tile collisions using cache
        for point in trajectory_points:
            # Handle both tuple (x, y) and dict {'x': x, 'y': y} formats
            if isinstance(point, dict):
                x, y = point['x'], point['y']
            else:
                x, y = point
            # Convert world coordinates to tile coordinates
            tile_x = int(x // TILE_PIXEL_SIZE)
            tile_y = int(y // TILE_PIXEL_SIZE)
            
            # Check if tile is solid using cache (much faster lookup)
            if (tile_x, tile_y) in self._tile_cache:
                # Check if ninja circle overlaps with solid tile
                tile_world_x = tile_x * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2  # Center of tile
                tile_world_y = tile_y * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                
                # Distance from ninja center to tile center
                dist_sq = (x - tile_world_x)**2 + (y - tile_world_y)**2
                
                # If ninja radius overlaps with tile (approximate as circle vs square)
                if dist_sq < (NINJA_RADIUS + 12)**2:  # 12 is half tile size
                    return False
                    
        return True

    def _cache_level_geometry(self, level_data: dict) -> None:
        """
        Cache static level geometry since tiles never change during a level.
        
        Args:
            level_data: Level data containing tile information
        """
        level_id = level_data.get('level_id', id(level_data))
        
        # Only recache if level changed
        if self._current_level_id == level_id:
            return
            
        self._current_level_id = level_id
        tiles = level_data.get('tiles', None)
        
        if tiles is None:
            self._tile_cache = {}
            return
            
        # Pre-process tile data into a fast lookup format
        self._tile_cache = {}
        
        # Handle different tile data formats and cache solid tiles
        if hasattr(tiles, '__getitem__'):
            if hasattr(tiles, 'shape') and len(tiles.shape) == 2:
                # NumPy array format - use actual dimensions (but expect 25x44 for real levels)
                height, width = tiles.shape
                for tile_y in range(height):
                    for tile_x in range(width):
                        if tiles[tile_y, tile_x] != 0:  # Solid tile
                            self._tile_cache[(tile_x, tile_y)] = True
            elif isinstance(tiles, (list, tuple)):
                # List/tuple format
                for tile_y, row in enumerate(tiles):
                    if hasattr(row, '__getitem__'):
                        for tile_x, tile_value in enumerate(row):
                            if tile_value != 0:  # Solid tile
                                self._tile_cache[(tile_x, tile_y)] = True
            elif isinstance(tiles, dict):
                # Dict format - copy solid tiles
                for (tile_x, tile_y), tile_value in tiles.items():
                    if tile_value != 0:
                        self._tile_cache[(tile_x, tile_y)] = True
        
    def _is_solid_tile(self, tiles: any, tile_x: int, tile_y: int) -> bool:
        """Check if a tile at given coordinates is solid."""
        try:
            # Handle different tile data formats
            if hasattr(tiles, '__getitem__'):
                if hasattr(tiles, 'shape') and len(tiles.shape) == 2:
                    # NumPy array format - use actual dimensions (but expect 25x44 for real levels)
                    height, width = tiles.shape
                    if 0 <= tile_y < height and 0 <= tile_x < width:
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

    def _analyze_win_conditions(self, level_data: dict) -> dict:
        """
        Analyze win conditions for trajectory planning optimization.
        
        Identifies switch→door sequences, exit requirements, and path constraints
        to optimize trajectory planning for goal-oriented movement.
        
        Args:
            level_data: Level data containing entities
            
        Returns:
            Dict containing win condition analysis:
            - 'exits': List of exit positions and states
            - 'switches': List of switch positions and states  
            - 'doors': List of door positions and states
            - 'switch_door_pairs': Identified switch→door relationships
            - 'completion_requirements': Steps needed to complete level
        """
        if not level_data or 'entities' not in level_data:
            return {}
            
        entities = level_data['entities']
        win_analysis = {
            'exits': [],
            'switches': [],
            'doors': [],
            'switch_door_pairs': [],
            'completion_requirements': []
        }
        
        # Collect all relevant entities
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_type = entity.get('type', None)
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            entity_state = entity.get('state', 0)
            
            # Collect exits
            if entity_type == EntityType.EXIT_DOOR:
                win_analysis['exits'].append({
                    'position': (entity_x, entity_y),
                    'state': entity_state,
                    'accessible': entity_state > 0  # Assume state > 0 means accessible
                })
                
            # Collect switches
            elif entity_type == EntityType.EXIT_SWITCH:
                win_analysis['switches'].append({
                    'position': (entity_x, entity_y),
                    'state': entity_state,
                    'activated': entity_state > 0
                })
                
            # Collect doors
            elif entity_type in [EntityType.REGULAR_DOOR, EntityType.LOCKED_DOOR]:
                win_analysis['doors'].append({
                    'position': (entity_x, entity_y),
                    'state': entity_state,
                    'type': entity_type,
                    'open': entity_state > 0  # Assume state > 0 means open
                })
        
        # Analyze switch→door relationships (simplified heuristic)
        for switch in win_analysis['switches']:
            switch_x, switch_y = switch['position']
            
            # Find closest door to each switch (simple proximity heuristic)
            closest_door = None
            min_distance = float('inf')
            
            for door in win_analysis['doors']:
                door_x, door_y = door['position']
                distance = math.sqrt((switch_x - door_x)**2 + (switch_y - door_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_door = door
                    
            if closest_door and min_distance < SWITCH_DOOR_MAX_DISTANCE:
                win_analysis['switch_door_pairs'].append({
                    'switch': switch,
                    'door': closest_door,
                    'distance': min_distance
                })
        
        # Determine completion requirements
        unactivated_switches = [s for s in win_analysis['switches'] if not s['activated']]
        inaccessible_exits = [e for e in win_analysis['exits'] if not e['accessible']]
        
        if unactivated_switches:
            win_analysis['completion_requirements'].append({
                'type': 'activate_switches',
                'count': len(unactivated_switches),
                'positions': [s['position'] for s in unactivated_switches]
            })
            
        if inaccessible_exits:
            win_analysis['completion_requirements'].append({
                'type': 'unlock_exits',
                'count': len(inaccessible_exits),
                'positions': [e['position'] for e in inaccessible_exits]
            })
        
        return win_analysis

    def _calculate_win_condition_trajectory_bonus(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: dict
    ) -> float:
        """
        Calculate trajectory bonus based on win condition progress.
        
        Trajectories that move toward switches, exits, or complete win conditions
        receive bonus scores to guide the agent toward level completion.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            level_data: Level data for win condition analysis
            
        Returns:
            Bonus score (0.0 to 1.0) for win condition progress
        """
        win_analysis = self._analyze_win_conditions(level_data)
        
        if not win_analysis:
            return 0.0
            
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        bonus = 0.0
        
        # Bonus for moving toward unactivated switches
        unactivated_switches = [s for s in win_analysis['switches'] if not s['activated']]
        if unactivated_switches:
            for switch in unactivated_switches:
                switch_x, switch_y = switch['position']
                
                # Distance from start and end to switch
                start_dist = math.sqrt((start_x - switch_x)**2 + (start_y - switch_y)**2)
                end_dist = math.sqrt((end_x - switch_x)**2 + (end_y - switch_y)**2)
                
                # Bonus if moving closer to switch
                if end_dist < start_dist:
                    improvement = (start_dist - end_dist) / max(start_dist, 1.0)
                    bonus += improvement * WIN_CONDITION_SWITCH_BONUS
        
        # Bonus for moving toward accessible exits
        accessible_exits = [e for e in win_analysis['exits'] if e['accessible']]
        if accessible_exits:
            for exit_info in accessible_exits:
                exit_x, exit_y = exit_info['position']
                
                # Distance from start and end to exit
                start_dist = math.sqrt((start_x - exit_x)**2 + (start_y - exit_y)**2)
                end_dist = math.sqrt((end_x - exit_x)**2 + (end_y - exit_y)**2)
                
                # Bonus if moving closer to exit
                if end_dist < start_dist:
                    improvement = (start_dist - end_dist) / max(start_dist, 1.0)
                    bonus += improvement * WIN_CONDITION_EXIT_BONUS
        
        # Bonus for completing switch→door sequences
        for pair in win_analysis['switch_door_pairs']:
            switch_pos = pair['switch']['position']
            door_pos = pair['door']['position']
            
            # If switch is activated and door is now accessible
            if pair['switch']['activated'] and pair['door']['open']:
                # Bonus for being near the opened door
                door_x, door_y = door_pos
                end_dist = math.sqrt((end_x - door_x)**2 + (end_y - door_y)**2)
                
                if end_dist < WIN_CONDITION_DOOR_PROXIMITY:
                    bonus += WIN_CONDITION_DOOR_BONUS
        
        return min(bonus, 1.0)  # Cap at 100% bonus

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
            dx, dy, time_of_flight, initial_speed, ninja_state,
            start_pos, end_pos, level_data
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
        ninja_state: Optional[MovementState] = None,
        start_pos: Optional[Tuple[float, float]] = None,
        end_pos: Optional[Tuple[float, float]] = None,
        level_data: Optional[dict] = None
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
        
        # Win condition bonus (if positions and level data available)
        win_condition_bonus = 0.0
        if start_pos and end_pos and level_data:
            win_condition_bonus = self._calculate_win_condition_trajectory_bonus(
                start_pos, end_pos, level_data
            )
        
        # Calculate final probability with win condition awareness
        final_prob = base_prob - distance_penalty - height_penalty - velocity_penalty - time_penalty + win_condition_bonus
        
        return max(final_prob, SUCCESS_PROBABILITY_MIN)
    
    def calculate_bounce_block_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        ninja_state: Optional[MovementState] = None,
        level_data: Optional[dict] = None
    ) -> TrajectoryResult:
        """
        Calculate trajectory for bounce block interactions.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            ninja_state: Current ninja movement state
            level_data: Level data including bounce blocks
            
        Returns:
            TrajectoryResult with bounce block physics
        """
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:
            return self._create_zero_trajectory_result()
        
        # Find bounce blocks along the trajectory
        bounce_blocks = self._find_trajectory_bounce_blocks(start_pos, end_pos, level_data)
        
        if not bounce_blocks:
            # No bounce blocks - fall back to regular trajectory
            return self.calculate_jump_trajectory(start_pos, end_pos, ninja_state, level_data)
        
        # Calculate bounce block enhanced trajectory
        return self._calculate_enhanced_bounce_trajectory(
            start_pos, end_pos, bounce_blocks, ninja_state, level_data
        )
    
    def calculate_bounce_chain_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        ninja_state: Optional[MovementState] = None,
        level_data: Optional[dict] = None
    ) -> TrajectoryResult:
        """
        Calculate trajectory for chained bounce block interactions.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            ninja_state: Current ninja movement state
            level_data: Level data including bounce blocks
            
        Returns:
            TrajectoryResult with chained bounce block physics
        """
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:
            return self._create_zero_trajectory_result()
        
        # Find chainable bounce blocks
        bounce_blocks = self._find_chainable_bounce_blocks(start_pos, end_pos, level_data)
        
        if len(bounce_blocks) < 2:
            # Not enough blocks for chaining - fall back to single block
            return self.calculate_bounce_block_trajectory(start_pos, end_pos, ninja_state, level_data)
        
        # Calculate chained trajectory
        return self._calculate_chained_bounce_trajectory(
            start_pos, end_pos, bounce_blocks, ninja_state, level_data
        )
    
    def calculate_bounce_boost_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        ninja_state: Optional[MovementState] = None,
        level_data: Optional[dict] = None
    ) -> TrajectoryResult:
        """
        Calculate trajectory for repeated bounce boost mechanics.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            ninja_state: Current ninja movement state
            level_data: Level data including bounce blocks
            
        Returns:
            TrajectoryResult with repeated boost physics
        """
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:
            return self._create_zero_trajectory_result()
        
        # Find extending bounce blocks for boost
        extending_blocks = self._find_extending_bounce_blocks(start_pos, level_data)
        
        if not extending_blocks:
            # No extending blocks - fall back to regular bounce block
            return self.calculate_bounce_block_trajectory(start_pos, end_pos, ninja_state, level_data)
        
        # Calculate repeated boost trajectory
        return self._calculate_repeated_boost_trajectory(
            start_pos, end_pos, extending_blocks, ninja_state, level_data
        )
    
    def _find_trajectory_bounce_blocks(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: Optional[dict]
    ) -> List[dict]:
        """Find bounce blocks that interact with the trajectory."""
        if not level_data or not level_data.get('entities'):
            return []
        
        # Get bounce blocks from level data using centralized constant
        entities = level_data.get('entities', [])
        bounce_blocks = [e for e in entities if e.get('type') == ENTITY_TYPE_BOUNCE_BLOCK]
        
        # Use centralized utility to find blocks near trajectory
        return find_bounce_blocks_near_trajectory(start_pos, end_pos, bounce_blocks)
    
    def _find_chainable_bounce_blocks(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        level_data: Optional[dict]
    ) -> List[dict]:
        """Find bounce blocks that can be chained together."""
        trajectory_blocks = self._find_trajectory_bounce_blocks(start_pos, end_pos, level_data)
        
        if len(trajectory_blocks) < 2:
            return trajectory_blocks
        
        # Sort blocks by distance along trajectory
        def trajectory_distance(block):
            block_pos = (block.get('x', 0.0), block.get('y', 0.0))
            # Project block position onto trajectory line
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            if dx == 0 and dy == 0:
                return 0.0
            
            bx = block_pos[0] - start_pos[0]
            by = block_pos[1] - start_pos[1]
            t = max(0, min(1, (bx*dx + by*dy) / (dx*dx + dy*dy)))
            return t
        
        trajectory_blocks.sort(key=trajectory_distance)
        
        # Filter blocks that are within chaining distance
        chainable = [trajectory_blocks[0]]  # Always include first block
        
        for i in range(1, len(trajectory_blocks)):
            prev_block = chainable[-1]
            curr_block = trajectory_blocks[i]
            
            prev_pos = (prev_block.get('x', 0.0), prev_block.get('y', 0.0))
            curr_pos = (curr_block.get('x', 0.0), curr_block.get('y', 0.0))
            
            distance = math.sqrt(
                (curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2
            )
            
            if distance <= BOUNCE_BLOCK_CHAIN_DISTANCE:
                chainable.append(curr_block)
        
        return chainable
    
    def _find_extending_bounce_blocks(
        self,
        start_pos: Tuple[float, float],
        level_data: Optional[dict]
    ) -> List[dict]:
        """Find bounce blocks in extending state near start position."""
        if not level_data or not level_data.get('entities'):
            return []
        
        extending_blocks = []
        for entity in level_data.get('entities', []):
            if (entity.get('type') == 17 and  # Bounce block type
                entity.get('bounce_state') == BounceBlockState.EXTENDING):
                
                block_x = entity.get('x', 0.0)
                block_y = entity.get('y', 0.0)
                distance = math.sqrt(
                    (block_x - start_pos[0])**2 + (block_y - start_pos[1])**2
                )
                
                if distance <= BOUNCE_BLOCK_INTERACTION_RADIUS:
                    extending_blocks.append(entity)
        
        return extending_blocks
    
    def _calculate_enhanced_bounce_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        bounce_blocks: List[dict],
        ninja_state: Optional[MovementState],
        level_data: Optional[dict]
    ) -> TrajectoryResult:
        """Calculate trajectory enhanced by bounce block physics."""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate boost multiplier based on bounce block states
        boost_multiplier = self._calculate_bounce_boost_multiplier(bounce_blocks)
        
        # Enhanced initial velocity from bounce block
        if abs(dx) > abs(dy):  # Horizontal movement
            horizontal_velocity = (dx / abs(dx)) * MAX_HOR_SPEED * boost_multiplier if dx != 0 else 0
            initial_vy = dy / max(abs(dx) / horizontal_velocity, 0.1) if dx != 0 else 0
        else:  # Vertical movement
            initial_vy = -math.sqrt(2 * abs(dy) * GRAVITY_FALL) * boost_multiplier if dy < 0 else 0
            horizontal_velocity = dx / max(abs(dy) / abs(initial_vy), 0.1) if initial_vy != 0 else 0
        
        # Calculate time of flight with bounce enhancement
        if dy < 0:  # Upward movement
            gravity = GRAVITY_FALL
            discriminant = initial_vy**2 + 2 * gravity * abs(dy)
            if discriminant >= 0:
                time_of_flight = (-initial_vy + math.sqrt(discriminant)) / gravity
            else:
                time_of_flight = DEFAULT_MINIMUM_TIME
        else:  # Downward movement
            time_of_flight = distance / max(abs(horizontal_velocity), MIN_HORIZONTAL_VELOCITY)
        
        # Reduce time due to bounce block efficiency
        time_of_flight *= (1.0 / boost_multiplier)
        
        # Calculate energy cost with bounce block efficiency
        base_energy = abs(horizontal_velocity) / MAX_HOR_SPEED + abs(initial_vy) / MAX_HOR_SPEED
        energy_cost = base_energy * BOUNCE_BLOCK_ENERGY_EFFICIENCY
        
        # Enhanced success probability
        success_probability = self._calculate_success_probability(
            dx, dy, time_of_flight, max(abs(horizontal_velocity), abs(initial_vy)),
            ninja_state, start_pos, end_pos, level_data
        )
        success_probability = min(success_probability + BOUNCE_BLOCK_SUCCESS_BONUS, 1.0)
        
        # Generate trajectory points
        trajectory_points = self._generate_trajectory_points(
            start_pos, horizontal_velocity, initial_vy, GRAVITY_FALL, time_of_flight
        )
        
        return TrajectoryResult(
            feasible=True,
            time_of_flight=time_of_flight,
            energy_cost=energy_cost,
            success_probability=success_probability,
            min_velocity=min(abs(horizontal_velocity), abs(initial_vy)),
            max_velocity=max(abs(horizontal_velocity), abs(initial_vy)) * boost_multiplier,
            requires_jump=True,
            requires_wall_contact=False,
            trajectory_points=trajectory_points
        )
    
    def _calculate_chained_bounce_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        bounce_blocks: List[dict],
        ninja_state: Optional[MovementState],
        level_data: Optional[dict]
    ) -> TrajectoryResult:
        """Calculate trajectory for chained bounce blocks."""
        # Start with enhanced bounce trajectory
        result = self._calculate_enhanced_bounce_trajectory(
            start_pos, end_pos, bounce_blocks, ninja_state, level_data
        )
        
        # Apply chain bonuses
        chain_count = len(bounce_blocks)
        chain_multiplier = 1.0 + (chain_count - 1) * 0.2  # 20% bonus per additional block
        
        # Enhanced velocity and reduced time
        result.max_velocity *= chain_multiplier
        result.time_of_flight /= chain_multiplier
        
        # Slightly higher energy cost due to complexity
        result.energy_cost *= 1.1
        
        # Slightly reduced success probability due to timing requirements
        result.success_probability *= 0.95
        
        return result
    
    def _calculate_repeated_boost_trajectory(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        extending_blocks: List[dict],
        ninja_state: Optional[MovementState],
        level_data: Optional[dict]
    ) -> TrajectoryResult:
        """Calculate trajectory for repeated boost mechanics."""
        # Start with enhanced bounce trajectory
        result = self._calculate_enhanced_bounce_trajectory(
            start_pos, end_pos, extending_blocks, ninja_state, level_data
        )
        
        # Apply repeated boost bonuses
        boost_count = min(len(extending_blocks) * 3, 10)  # Max 10 boosts
        boost_decay = 0.95 ** boost_count  # Decay with each boost
        
        # Maximum velocity from repeated boosts
        result.max_velocity = MAX_HOR_SPEED * BOUNCE_BLOCK_BOOST_MAX * boost_decay
        
        # Very fast movement
        result.time_of_flight *= 0.5
        
        # Very energy efficient
        result.energy_cost *= 0.5
        
        # Moderate success probability (timing dependent)
        result.success_probability = 0.8
        
        return result
    
    def _calculate_bounce_boost_multiplier(self, bounce_blocks: List[dict]) -> float:
        """Calculate boost multiplier based on bounce block states."""
        if not bounce_blocks:
            return 1.0
        
        total_boost = 0.0
        for block in bounce_blocks:
            compression = block.get('compression_amount', 0.0)
            state = block.get('bounce_state', BounceBlockState.NEUTRAL)
            
            # Use centralized utility for individual block boost calculation
            boost = calculate_bounce_block_boost_multiplier([block])
            
            total_boost += boost
        
        # Average boost across all blocks, with diminishing returns
        avg_boost = total_boost / len(bounce_blocks)
        return min(avg_boost, BOUNCE_BLOCK_BOOST_MAX)
    
    def _point_near_line_segment(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        threshold: float
    ) -> bool:
        """Check if point is within threshold distance of line segment."""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from line start to point
        fx = px - x1
        fy = py - y1
        
        # Handle degenerate case
        if dx == 0 and dy == 0:
            distance = math.sqrt(fx*fx + fy*fy)
            return distance <= threshold
        
        # Project point onto line
        t = max(0, min(1, (fx*dx + fy*dy) / (dx*dx + dy*dy)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point on line
        dist_x = px - closest_x
        dist_y = py - closest_y
        distance = math.sqrt(dist_x*dist_x + dist_y*dist_y)
        
        return distance <= threshold
    
    def _create_zero_trajectory_result(self) -> TrajectoryResult:
        """Create a trajectory result for zero-distance movement."""
        return TrajectoryResult(
            feasible=True,
            time_of_flight=0.0,
            energy_cost=0.0,
            success_probability=1.0,
            min_velocity=0.0,
            max_velocity=0.0,
            requires_jump=False,
            requires_wall_contact=False,
            trajectory_points=[]
        )
