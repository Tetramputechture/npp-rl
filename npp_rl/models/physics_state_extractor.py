"""
Physics State Extractor for momentum-augmented node representations.

This module extracts comprehensive physics state information from ninja position,
velocity, and movement state to enhance graph node features with physics-aware
information for better pathfinding and movement prediction.
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
from nclone.constants import (
    MAX_HOR_SPEED, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
    NINJA_RADIUS, GRAVITY_FALL, GRAVITY_JUMP,
    JUMP_FLAT_GROUND_Y, JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y
)
from nclone.entity_classes.entity_launch_pad import EntityLaunchPad
from nclone.entity_classes.entity_toggle_mine import EntityToggleMine
from nclone.entity_classes.entity_drone_zap import EntityDroneZap
from nclone.entity_classes.entity_mini_drone import EntityMiniDrone
from nclone.entity_classes.entity_thwump import EntityThwump
from nclone.entity_classes.entity_shove_thwump import EntityShoveThwump
from nclone.entity_classes.entity_exit import EntityExit
from nclone.entity_classes.entity_exit_switch import EntityExitSwitch
from nclone.physics import sweep_circle_vs_tiles

GROUND_STATES = {0, 1, 2}
AIR_STATES = {3, 4}
WALL_STATES = {5}
INACTIVE_STATES = {6, 7, 8, 9}

# Level geometry constants
TYPICAL_LEVEL_SIZE = 1000.0  # Pixels, for distance normalization
PROXIMITY_THRESHOLD = 100.0  # Distance threshold for entity proximity
HAZARD_PROXIMITY_THRESHOLD = 50.0  # Closer threshold for hazard detection

NORMALIZED_HEIGHT_DIVISOR = MAP_TILE_HEIGHT * TILE_PIXEL_SIZE

class PhysicsStateExtractor:
    """
    Extracts comprehensive physics state for node features.
    
    This class processes ninja position, velocity, and movement state to generate
    physics-aware features that can be incorporated into graph node representations.
    The extracted features include velocity components, energy calculations,
    contact states, and movement capabilities.
    
    Optimized for single ninja per map assumption and multi-exit path finding.
    """
    
    def __init__(self):
        """Initialize physics state extractor with caching for level data."""
        # Cache for static level data (optimized for single ninja assumption)
        self._level_cache = {}
        self._current_level_id = None
        # Cache for dynamic entity states (updated each frame)
        self._dynamic_entity_cache = {}

    def extract_ninja_physics_state(
        self,
        ninja_position: Tuple[float, float],
        ninja_velocity: Tuple[float, float],
        ninja_state: Dict[str, Any],
        level_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract comprehensive physics state for node features.
        
        Args:
            ninja_position: Current ninja position (x, y) in pixels
            ninja_velocity: Current ninja velocity (vx, vy) in pixels/frame
            ninja_state: Ninja state dictionary containing movement_state and buffers
            level_data: Level data for context (optional)
            
        Returns:
            Array of 31 physics features (optimized for single ninja):
            [0-1]: Normalized velocity (vx, vy)
            [2]: Velocity magnitude
            [3]: Movement state (normalized)
            [4-6]: Contact flags (ground, wall, airborne)
            [7-8]: Momentum direction (normalized)
            [9]: Kinetic energy (normalized)
            [10]: Potential energy (normalized)
            [11-15]: Input buffers (jump, floor, wall, launch_pad, input_state)
            [16-17]: Physics capabilities (can_jump, can_wall_jump)
            [18-19]: Contact normal vector (nx, ny)
            [20-22]: Entity proximity (launch_pad, hazard, collectible)
            [23-25]: Advanced buffer states (wall_slide_buffer, air_time, ground_time)
            [26-27]: Physics constraints (max_jump_height, remaining_air_accel)
            [28-30]: Multi-exit path finding (closest_exit_distance, switch_completion_ratio, path_efficiency)
        """
        vx, vy = ninja_velocity
        x, y = ninja_position
        
        # Cache level data for single ninja optimization
        if level_data:
            self._cache_level_data(level_data)
            # Update dynamic entity states (must be done each frame)
            self._update_dynamic_entities(level_data)
        
        # Extract movement state (default to 0 if not available)
        movement_state = ninja_state.get('movement_state', 0)
        if isinstance(movement_state, (list, tuple)):
            movement_state = movement_state[0] if len(movement_state) > 0 else 0
        
        # Normalize velocity components
        vx_norm = np.clip(vx / MAX_HOR_SPEED, -1.0, 1.0)
        vy_norm = np.clip(vy / MAX_HOR_SPEED, -1.0, 1.0)
        
        # Calculate velocity magnitude (normalized)
        vel_magnitude = math.sqrt(vx*vx + vy*vy) / MAX_HOR_SPEED
        vel_magnitude = np.clip(vel_magnitude, 0.0, 2.0)  # Allow some overspeed
        
        # Movement state (normalized to [0,1])
        movement_state_norm = np.clip(movement_state / 9.0, 0.0, 1.0)
        
        # Contact state detection based on movement state
        ground_contact = 1.0 if movement_state in GROUND_STATES else 0.0
        wall_contact = 1.0 if movement_state in WALL_STATES else 0.0
        airborne = 1.0 if movement_state in AIR_STATES else 0.0
        
        # Momentum direction (normalized)
        if vel_magnitude > 0.01:
            momentum_x = vx / (vel_magnitude * MAX_HOR_SPEED)
            momentum_y = vy / (vel_magnitude * MAX_HOR_SPEED)
        else:
            momentum_x = momentum_y = 0.0
        
        # Energy calculations (normalized)
        kinetic_energy = 0.5 * (vx*vx + vy*vy) / (MAX_HOR_SPEED * MAX_HOR_SPEED)
        kinetic_energy = np.clip(kinetic_energy, 0.0, 2.0)  # Allow some overspeed
        
        # Potential energy (normalized height from bottom)
        potential_energy = np.clip((NORMALIZED_HEIGHT_DIVISOR - y) / NORMALIZED_HEIGHT_DIVISOR, 0.0, 1.0)
        
        # Input buffers from ninja state (normalized)
        # These may not always be available, so provide defaults
        jump_buffer = ninja_state.get('jump_buffer', 0)
        if isinstance(jump_buffer, (list, tuple)):
            jump_buffer = jump_buffer[0] if len(jump_buffer) > 0 else 0
        jump_buffer = np.clip(jump_buffer / 5.0, 0.0, 1.0)
        
        floor_buffer = ninja_state.get('floor_buffer', 0)
        if isinstance(floor_buffer, (list, tuple)):
            floor_buffer = floor_buffer[0] if len(floor_buffer) > 0 else 0
        floor_buffer = np.clip(floor_buffer / 5.0, 0.0, 1.0)
        
        wall_buffer = ninja_state.get('wall_buffer', 0)
        if isinstance(wall_buffer, (list, tuple)):
            wall_buffer = wall_buffer[0] if len(wall_buffer) > 0 else 0
        wall_buffer = np.clip(wall_buffer / 5.0, 0.0, 1.0)
        
        launch_pad_buffer = ninja_state.get('launch_pad_buffer', 0)
        if isinstance(launch_pad_buffer, (list, tuple)):
            launch_pad_buffer = launch_pad_buffer[0] if len(launch_pad_buffer) > 0 else 0
        launch_pad_buffer = np.clip(launch_pad_buffer / 4.0, 0.0, 1.0)
        
        # Input state (jump input active)
        jump_input = ninja_state.get('jump_input', False)
        if isinstance(jump_input, (list, tuple)):
            jump_input = jump_input[0] if len(jump_input) > 0 else False
        input_state = 1.0 if jump_input else 0.0
        
        # Physics capabilities based on current state
        # Can jump if on ground, or have jump buffer, or can wall jump
        can_jump = 1.0 if (ground_contact > 0.5 or jump_buffer > 0.0 or 
                          (wall_contact > 0.5 and wall_buffer > 0.0)) else 0.0
        
        # Can wall jump if touching wall or have wall buffer
        can_wall_jump = 1.0 if (wall_contact > 0.5 or wall_buffer > 0.0) else 0.0
        
        # Extract contact normal vector
        contact_normal_x, contact_normal_y = self._extract_contact_normal(
            ninja_position, ninja_state, level_data
        )
        
        # Extract entity proximity information
        launch_pad_proximity, hazard_proximity, collectible_proximity = self._extract_entity_proximity(
            ninja_position, level_data
        )
        
        # Extract advanced buffer states
        wall_slide_buffer, air_time, ground_time = self._extract_advanced_buffers(ninja_state)
        
        # Extract physics constraints
        max_jump_height, remaining_air_accel = self._extract_physics_constraints(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Extract multi-exit path finding features (single ninja optimization)
        closest_exit_distance, switch_completion_ratio, path_efficiency = self._get_multi_exit_features(
            ninja_position
        )
        
        return np.array([
            vx_norm, vy_norm, vel_magnitude, movement_state_norm,
            ground_contact, wall_contact, airborne,
            momentum_x, momentum_y, kinetic_energy, potential_energy,
            jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state,
            can_jump, can_wall_jump,
            contact_normal_x, contact_normal_y,
            launch_pad_proximity, hazard_proximity, collectible_proximity,
            wall_slide_buffer, air_time, ground_time,
            max_jump_height, remaining_air_accel,
            closest_exit_distance, switch_completion_ratio, path_efficiency
        ], dtype=np.float32)
    
    def get_feature_names(self) -> list:
        """
        Get names of the physics features for debugging and analysis.
        
        Returns:
            List of feature names corresponding to the extracted features
        """
        return [
            'velocity_x_norm', 'velocity_y_norm', 'velocity_magnitude', 'movement_state',
            'ground_contact', 'wall_contact', 'airborne',
            'momentum_x', 'momentum_y', 'kinetic_energy', 'potential_energy',
            'jump_buffer', 'floor_buffer', 'wall_buffer', 'launch_pad_buffer', 'input_state',
            'can_jump', 'can_wall_jump',
            'contact_normal_x', 'contact_normal_y',
            'launch_pad_proximity', 'hazard_proximity', 'collectible_proximity',
            'wall_slide_buffer', 'air_time', 'ground_time',
            'max_jump_height', 'remaining_air_accel',
            'closest_exit_distance', 'switch_completion_ratio', 'path_efficiency'
        ]
    
    def validate_physics_state(self, physics_features: np.ndarray) -> bool:
        """
        Validate that physics features are within expected ranges.
        
        Args:
            physics_features: Array of physics features to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        if len(physics_features) != 31:
            return False
        
        # Check velocity components are normalized
        if abs(physics_features[0]) > 1.1 or abs(physics_features[1]) > 1.1:
            return False
        
        # Check contact flags are binary
        for i in [4, 5, 6, 15, 16, 17]:
            if not (0.0 <= physics_features[i] <= 1.0):
                return False
        
        # Check energy values are non-negative
        if physics_features[9] < 0 or physics_features[10] < 0:
            return False
        
        return True

    def _extract_contact_normal(
        self,
        ninja_position: Tuple[float, float],
        ninja_state: Dict[str, Any],
        level_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Extract contact normal vector from collision information.
        
        Args:
            ninja_position: Current ninja position
            ninja_state: Ninja state dictionary
            level_data: Level data for collision detection
            
        Returns:
            Normalized contact normal vector (nx, ny)
        """
        x, y = ninja_position
        
        # Try to get contact normal from ninja state if available
        contact_normal = ninja_state.get('contact_normal', None)
        if contact_normal and len(contact_normal) >= 2:
            nx, ny = contact_normal[0], contact_normal[1]
            # Normalize the vector
            length = math.sqrt(nx*nx + ny*ny)
            if length > 0:
                return nx / length, ny / length
        
        # If no contact normal available, estimate from movement state and level geometry
        movement_state = ninja_state.get('movement_state', 0)
        
        if movement_state in WALL_STATES:
            # Wall contact - estimate normal by checking nearby tiles
            if level_data and 'tiles' in level_data:
                return self._estimate_wall_normal(ninja_position, level_data['tiles'])
        elif movement_state in GROUND_STATES:
            # Ground contact - normal points upward
            return 0.0, -1.0
        
        # No contact or unknown contact
        return 0.0, 0.0
    
    def _estimate_wall_normal(
        self,
        ninja_position: Tuple[float, float],
        tiles: Any
    ) -> Tuple[float, float]:
        """Estimate wall normal by checking nearby solid tiles."""
        x, y = ninja_position
        tile_x = int(x // TILE_PIXEL_SIZE)
        tile_y = int(y // TILE_PIXEL_SIZE)
        
        # Check tiles around ninja position
        normal_x, normal_y = 0.0, 0.0
        
        # Check left and right
        if self._is_solid_tile_safe(tiles, tile_x - 1, tile_y):
            normal_x += 1.0  # Wall on left, normal points right
        if self._is_solid_tile_safe(tiles, tile_x + 1, tile_y):
            normal_x -= 1.0  # Wall on right, normal points left
            
        # Check up and down
        if self._is_solid_tile_safe(tiles, tile_x, tile_y - 1):
            normal_y += 1.0  # Wall above, normal points down
        if self._is_solid_tile_safe(tiles, tile_x, tile_y + 1):
            normal_y -= 1.0  # Wall below, normal points up
        
        # Normalize
        length = math.sqrt(normal_x*normal_x + normal_y*normal_y)
        if length > 0:
            return normal_x / length, normal_y / length
        
        return 0.0, 0.0
    
    def _is_solid_tile_safe(self, tiles: Any, tile_x: int, tile_y: int) -> bool:
        """Safely check if a tile is solid without throwing exceptions."""
        try:
            if hasattr(tiles, 'shape') and len(tiles.shape) == 2:
                if 0 <= tile_y < tiles.shape[0] and 0 <= tile_x < tiles.shape[1]:
                    return tiles[tile_y, tile_x] != 0
            elif isinstance(tiles, (list, tuple)):
                if 0 <= tile_y < len(tiles) and 0 <= tile_x < len(tiles[tile_y]):
                    return tiles[tile_y][tile_x] != 0
            elif isinstance(tiles, dict):
                return tiles.get((tile_x, tile_y), 0) != 0
        except (IndexError, KeyError, AttributeError):
            pass
        return False

    def _update_dynamic_entities(self, level_data: Dict[str, Any]) -> None:
        """
        Update dynamic entity states that can change during gameplay.
        
        This method should be called each time physics state is extracted to ensure
        entity states (like toggle mine states, switch activations) are current.
        
        Args:
            level_data: Current level data with entity states
        """
        if not level_data:
            return
            
        self._dynamic_entity_cache = {
            'toggle_mines': [],
            'switches': [],
            'drones': [],
            'thwumps': []
        }
        
        entities = level_data.get('entities', [])
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_type = entity.get('type', None)
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            entity_state = entity.get('state', 0)
            
            # Cache toggle mines (state changes: 1=safe, 0=deadly after ninja visit)
            if entity_type == EntityToggleMine.ENTITY_TYPE:
                self._dynamic_entity_cache['toggle_mines'].append({
                    'x': entity_x,
                    'y': entity_y,
                    'state': entity_state,
                    'is_deadly': entity_state == 0,  # Deadly when state 0
                    'entity': entity
                })
            
            # Cache switches (activation state changes)
            elif entity_type == EntityExitSwitch.ENTITY_TYPE:
                self._dynamic_entity_cache['switches'].append({
                    'x': entity_x,
                    'y': entity_y,
                    'state': entity_state,
                    'activated': entity_state > 0,
                    'entity': entity
                })
            
            # Cache drones (position and direction can change)
            elif entity_type in [EntityDroneZap.ENTITY_TYPE, EntityMiniDrone.ENTITY_TYPE]:
                self._dynamic_entity_cache['drones'].append({
                    'x': entity_x,
                    'y': entity_y,
                    'state': entity_state,
                    'type': entity_type,
                    'entity': entity
                })
            
            # Cache thwumps (position and state can change)
            elif entity_type in [EntityThwump.ENTITY_TYPE, EntityShoveThwump.ENTITY_TYPE]:
                self._dynamic_entity_cache['thwumps'].append({
                    'x': entity_x,
                    'y': entity_y,
                    'state': entity_state,
                    'type': entity_type,
                    'entity': entity
                })

    def _extract_entity_proximity(
        self,
        ninja_position: Tuple[float, float],
        level_data: Optional[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Extract proximity to different entity types.
        
        Args:
            ninja_position: Current ninja position
            level_data: Level data containing entities
            
        Returns:
            Tuple of (launch_pad_proximity, hazard_proximity, collectible_proximity)
        """
        if not level_data or 'entities' not in level_data:
            return 0.0, 0.0, 0.0
        
        x, y = ninja_position
        entities = level_data['entities']
        
        launch_pad_proximity = 0.0
        hazard_proximity = 0.0
        collectible_proximity = 0.0
        
        proximity_threshold = PROXIMITY_THRESHOLD
        
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            entity_type = entity.get('type', None)
            
            # Calculate distance
            dist = math.sqrt((x - entity_x)**2 + (y - entity_y)**2)
            
            if dist < proximity_threshold:
                # Normalize proximity (closer = higher value)
                proximity = 1.0 - (dist / proximity_threshold)
                
                if entity_type == EntityLaunchPad.ENTITY_TYPE:  # Launch pad
                    launch_pad_proximity = max(launch_pad_proximity, proximity)
                elif self._is_hazardous_entity(entity, ninja_position):
                    hazard_proximity = max(hazard_proximity, proximity)
                elif entity_type in [6, 7, 8]:  # Collectibles (gold, switches, etc.)
                    collectible_proximity = max(collectible_proximity, proximity)
        
        return launch_pad_proximity, hazard_proximity, collectible_proximity

    def _is_hazardous_entity(self, entity: Dict[str, Any], ninja_position: Optional[Tuple[float, float]] = None) -> bool:
        """
        Determine if an entity is hazardous based on precise N++ hazard definition.
        
        A hazard is defined as:
        - A toggle mine that is toggled (in state 0)
        - Any side of any kind of drone
        - The dangerous side of a thwump (requires ninja position)
        - Any side of a shove thwump once activated
        
        Args:
            entity: Entity dictionary with type, state, and other properties
            ninja_position: Optional ninja position for orientation-based hazard detection
            
        Returns:
            True if entity is currently hazardous
        """
        entity_type = entity.get('type', None)
        entity_state = entity.get('state', 0)
        
        # Toggle mine in toggled state (state 0) is hazardous
        if entity_type in [EntityToggleMine.ENTITY_TYPE, 21]:  # Types 1 and 21
            return entity_state == 0  # Toggled (deadly) state
            
        # All drone types are always hazardous on any side
        if entity_type in [EntityDroneZap.ENTITY_TYPE, EntityMiniDrone.ENTITY_TYPE]:
            return True
            
        # Thwumps - need to check orientation and state for dangerous side
        if entity_type == EntityThwump.ENTITY_TYPE:
            # Thwumps are dangerous on the side they can move toward
            # Based on N++ mechanics, thwumps move in their orientation direction
            orientation = entity.get('orientation', 0)
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            
            # Calculate ninja position relative to thwump
            ninja_x, ninja_y = ninja_position
            relative_x = ninja_x - entity_x
            relative_y = ninja_y - entity_y
            
            # Map orientation to dangerous direction
            # 0=right, 1=down, 2=left, 3=up (standard N++ orientations)
            if orientation == 0:  # Right-facing thwump
                return relative_x > 0  # Dangerous on right side
            elif orientation == 1:  # Down-facing thwump
                return relative_y > 0  # Dangerous on bottom side
            elif orientation == 2:  # Left-facing thwump
                return relative_x < 0  # Dangerous on left side
            elif orientation == 3:  # Up-facing thwump
                return relative_y < 0  # Dangerous on top side
            else:
                # Unknown orientation, assume dangerous
                return True
            
        # Shove thwumps when activated are hazardous on any side
        if entity_type == EntityShoveThwump.ENTITY_TYPE:
            # Shove thwumps are hazardous when activated (state > 0)
            return entity_state > 0
            
        # Regular mines (if they exist) are always hazardous
        if entity_type == 2:  # Regular mine type
            return True
            
        return False

    def _cache_level_data(self, level_data: Dict[str, Any]) -> None:
        """
        Cache static level data for single ninja optimization.
        
        Since there's only one ninja per map, we can cache:
        - Exit positions and states
        - Switch positions and required states
        - Static entity positions
        - Level geometry bounds
        
        Args:
            level_data: Level data dictionary
        """
        if not level_data:
            return
            
        level_id = level_data.get('level_id', id(level_data))
        
        # Only recache if level changed
        if self._current_level_id == level_id:
            return
            
        self._current_level_id = level_id
        self._level_cache = {
            'exits': [],
            'switches': [],
            'switch_door_pairs': [],
            'level_bounds': None
        }
        
        entities = level_data.get('entities', [])
        
        # Cache exits and switches for multi-exit path finding
        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            entity_type = entity.get('type', None)
            entity_x = entity.get('x', 0)
            entity_y = entity.get('y', 0)
            entity_state = entity.get('state', 0)
            
            # Cache exit positions
            if entity_type == EntityExit.ENTITY_TYPE:
                self._level_cache['exits'].append({
                    'position': (entity_x, entity_y),
                    'state': entity_state,
                    'entity': entity
                })
                
            # Cache switch positions and states
            elif entity_type == EntityExitSwitch.ENTITY_TYPE:
                self._level_cache['switches'].append({
                    'position': (entity_x, entity_y),
                    'state': entity_state,
                    'entity': entity
                })
        
        # Calculate level bounds for normalization
        if entities:
            min_x = min(e.get('x', 0) for e in entities if isinstance(e, dict))
            max_x = max(e.get('x', 0) for e in entities if isinstance(e, dict))
            min_y = min(e.get('y', 0) for e in entities if isinstance(e, dict))
            max_y = max(e.get('y', 0) for e in entities if isinstance(e, dict))
            self._level_cache['level_bounds'] = (min_x, max_x, min_y, max_y)

    def _get_multi_exit_features(self, ninja_position: Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Extract multi-exit path finding features for single ninja optimization.
        
        Args:
            ninja_position: Current ninja position (x, y)
            
        Returns:
            Tuple of (closest_exit_distance, switch_completion_ratio, path_efficiency)
        """
        if not self._level_cache.get('exits'):
            return 0.0, 0.0, 0.0
            
        x, y = ninja_position
        
        # Find closest exit distance (normalized)
        closest_exit_distance = float('inf')
        for exit_info in self._level_cache['exits']:
            exit_x, exit_y = exit_info['position']
            dist = math.sqrt((x - exit_x)**2 + (y - exit_y)**2)
            closest_exit_distance = min(closest_exit_distance, dist)
            
        # Normalize distance
        if closest_exit_distance == float('inf'):
            closest_exit_distance = 0.0
        else:
            # Normalize by typical level size
            closest_exit_distance = max(0.0, 1.0 - (closest_exit_distance / TYPICAL_LEVEL_SIZE))
        
        # Calculate switch completion ratio
        total_switches = len(self._level_cache['switches'])
        if total_switches == 0:
            switch_completion_ratio = 1.0  # No switches needed
        else:
            # Count activated switches (assuming state > 0 means activated)
            activated_switches = sum(1 for s in self._level_cache['switches'] if s['state'] > 0)
            switch_completion_ratio = activated_switches / total_switches
        
        # Calculate path efficiency (distance to closest switch if not all activated)
        path_efficiency = 1.0
        if switch_completion_ratio < 1.0:
            # Find closest unactivated switch
            closest_switch_distance = float('inf')
            for switch_info in self._level_cache['switches']:
                if switch_info['state'] == 0:  # Unactivated
                    switch_x, switch_y = switch_info['position']
                    dist = math.sqrt((x - switch_x)**2 + (y - switch_y)**2)
                    closest_switch_distance = min(closest_switch_distance, dist)
            
            if closest_switch_distance != float('inf'):
                path_efficiency = max(0.0, 1.0 - (closest_switch_distance / TYPICAL_LEVEL_SIZE))
        
        return closest_exit_distance, switch_completion_ratio, path_efficiency

    def _extract_advanced_buffers(
        self,
        ninja_state: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Extract advanced buffer states.
        
        Args:
            ninja_state: Ninja state dictionary
            
        Returns:
            Tuple of (wall_slide_buffer, air_time, ground_time)
        """
        # Wall slide buffer (time since last wall contact)
        wall_slide_buffer = ninja_state.get('wall_slide_buffer', 0)
        if isinstance(wall_slide_buffer, (list, tuple)):
            wall_slide_buffer = wall_slide_buffer[0] if len(wall_slide_buffer) > 0 else 0
        wall_slide_buffer = np.clip(wall_slide_buffer / 10.0, 0.0, 1.0)
        
        # Air time (frames spent in air)
        air_time = ninja_state.get('air_time', 0)
        if isinstance(air_time, (list, tuple)):
            air_time = air_time[0] if len(air_time) > 0 else 0
        air_time = np.clip(air_time / 60.0, 0.0, 1.0)  # Normalize to ~1 second
        
        # Ground time (frames spent on ground)
        ground_time = ninja_state.get('ground_time', 0)
        if isinstance(ground_time, (list, tuple)):
            ground_time = ground_time[0] if len(ground_time) > 0 else 0
        ground_time = np.clip(ground_time / 60.0, 0.0, 1.0)  # Normalize to ~1 second
        
        return wall_slide_buffer, air_time, ground_time

    def _extract_physics_constraints(
        self,
        ninja_position: Tuple[float, float],
        ninja_velocity: Tuple[float, float],
        ninja_state: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Extract physics constraints and capabilities.
        
        Args:
            ninja_position: Current ninja position
            ninja_velocity: Current ninja velocity
            ninja_state: Ninja state dictionary
            
        Returns:
            Tuple of (max_jump_height, remaining_air_accel)
        """
        x, y = ninja_position
        vx, vy = ninja_velocity
        
        # Calculate maximum jump height from current position
        # Using kinematic equation: h = v²/(2g) where v is current upward velocity
        if vy < 0:  # Moving upward
            max_jump_height = (vy * vy) / (2 * GRAVITY_JUMP)
            max_jump_height = np.clip(max_jump_height / 100.0, 0.0, 1.0)  # Normalize
        else:
            # Use standard jump height if not currently jumping upward
            standard_jump_height = (JUMP_FLAT_GROUND_Y * JUMP_FLAT_GROUND_Y) / (2 * GRAVITY_JUMP)
            max_jump_height = np.clip(standard_jump_height / 100.0, 0.0, 1.0)
        
        # Calculate remaining air acceleration capability
        # Based on current horizontal velocity vs maximum
        current_speed = abs(vx)
        remaining_accel = max(0.0, MAX_HOR_SPEED - current_speed) / MAX_HOR_SPEED
        
        return max_jump_height, remaining_accel