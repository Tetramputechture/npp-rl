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
from nclone.physics import sweep_circle_vs_tiles

GROUND_STATES = {0, 1, 2}
AIR_STATES = {3, 4}
WALL_STATES = {5}
INACTIVE_STATES = {6, 7, 8, 9}

NORMALIZED_HEIGHT_DIVISOR = MAP_TILE_HEIGHT * TILE_PIXEL_SIZE

class PhysicsStateExtractor:
    """
    Extracts comprehensive physics state for node features.
    
    This class processes ninja position, velocity, and movement state to generate
    physics-aware features that can be incorporated into graph node representations.
    The extracted features include velocity components, energy calculations,
    contact states, and movement capabilities.
    """

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
            Array of 28 physics features:
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
        """
        vx, vy = ninja_velocity
        x, y = ninja_position
        
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
        
        return np.array([
            vx_norm, vy_norm, vel_magnitude, movement_state_norm,
            ground_contact, wall_contact, airborne,
            momentum_x, momentum_y, kinetic_energy, potential_energy,
            jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state,
            can_jump, can_wall_jump,
            contact_normal_x, contact_normal_y,
            launch_pad_proximity, hazard_proximity, collectible_proximity,
            wall_slide_buffer, air_time, ground_time,
            max_jump_height, remaining_air_accel
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
            'max_jump_height', 'remaining_air_accel'
        ]
    
    def validate_physics_state(self, physics_features: np.ndarray) -> bool:
        """
        Validate that physics features are within expected ranges.
        
        Args:
            physics_features: Array of physics features to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        if len(physics_features) != 28:
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
        
        proximity_threshold = 100.0  # pixels
        
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
                elif entity_type in [1, 2, 3, 4, 5]:  # Hazards (mines, turrets, etc.)
                    hazard_proximity = max(hazard_proximity, proximity)
                elif entity_type in [6, 7, 8]:  # Collectibles (gold, switches, etc.)
                    collectible_proximity = max(collectible_proximity, proximity)
        
        return launch_pad_proximity, hazard_proximity, collectible_proximity

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