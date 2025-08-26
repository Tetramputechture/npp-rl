"""
Physics State Extractor for momentum-augmented node representations.

This module extracts comprehensive physics state information from ninja position,
velocity, and movement state to enhance graph node features with physics-aware
information for better pathfinding and movement prediction.
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Import physics constants from nclone
import sys
import os
nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nclone')
if os.path.exists(nclone_path) and nclone_path not in sys.path:
    sys.path.insert(0, nclone_path)

try:
    from nclone.constants import (
        MAX_HOR_SPEED, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE,
        GRAVITY_FALL, GRAVITY_JUMP
    )
except ImportError:
    # Fallback constants if import fails
    MAX_HOR_SPEED = 3.333333333333333
    MAP_TILE_HEIGHT = 23
    TILE_PIXEL_SIZE = 24
    GRAVITY_FALL = 0.06666666666666665
    GRAVITY_JUMP = 0.01111111111111111


class PhysicsStateExtractor:
    """
    Extracts comprehensive physics state for node features.
    
    This class processes ninja position, velocity, and movement state to generate
    physics-aware features that can be incorporated into graph node representations.
    The extracted features include velocity components, energy calculations,
    contact states, and movement capabilities.
    """
    
    def __init__(self):
        """Initialize the physics state extractor with N++ constants."""
        self.max_hor_speed = MAX_HOR_SPEED  # 3.333 pixels/frame
        self.level_height = MAP_TILE_HEIGHT * TILE_PIXEL_SIZE  # 552 pixels
        self.gravity_fall = GRAVITY_FALL
        self.gravity_jump = GRAVITY_JUMP
        
        # Movement state mappings based on sim_mechanics_doc.md
        self.ground_states = {0, 1, 2}  # Immobile, Running, Ground Sliding
        self.air_states = {3, 4}        # Jumping, Falling
        self.wall_states = {5}          # Wall Sliding
        self.inactive_states = {6, 7, 8, 9}  # Dead, Awaiting Death, Celebrating, Disabled
    
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
            Array of 18 physics features:
            [0-1]: Normalized velocity (vx, vy)
            [2]: Velocity magnitude
            [3]: Movement state (normalized)
            [4-6]: Contact flags (ground, wall, airborne)
            [7-8]: Momentum direction (normalized)
            [9]: Kinetic energy (normalized)
            [10]: Potential energy (normalized)
            [11-15]: Input buffers (jump, floor, wall, launch_pad, input_state)
            [16-17]: Physics capabilities (can_jump, can_wall_jump)
        """
        vx, vy = ninja_velocity
        x, y = ninja_position
        
        # Extract movement state (default to 0 if not available)
        movement_state = ninja_state.get('movement_state', 0)
        if isinstance(movement_state, (list, tuple)):
            movement_state = movement_state[0] if len(movement_state) > 0 else 0
        
        # Normalize velocity components
        vx_norm = np.clip(vx / self.max_hor_speed, -1.0, 1.0)
        vy_norm = np.clip(vy / self.max_hor_speed, -1.0, 1.0)
        
        # Calculate velocity magnitude (normalized)
        vel_magnitude = math.sqrt(vx*vx + vy*vy) / self.max_hor_speed
        vel_magnitude = np.clip(vel_magnitude, 0.0, 2.0)  # Allow some overspeed
        
        # Movement state (normalized to [0,1])
        movement_state_norm = np.clip(movement_state / 9.0, 0.0, 1.0)
        
        # Contact state detection based on movement state
        ground_contact = 1.0 if movement_state in self.ground_states else 0.0
        wall_contact = 1.0 if movement_state in self.wall_states else 0.0
        airborne = 1.0 if movement_state in self.air_states else 0.0
        
        # Momentum direction (normalized)
        if vel_magnitude > 0.01:
            momentum_x = vx / (vel_magnitude * self.max_hor_speed)
            momentum_y = vy / (vel_magnitude * self.max_hor_speed)
        else:
            momentum_x = momentum_y = 0.0
        
        # Energy calculations (normalized)
        kinetic_energy = 0.5 * (vx*vx + vy*vy) / (self.max_hor_speed * self.max_hor_speed)
        kinetic_energy = np.clip(kinetic_energy, 0.0, 2.0)  # Allow some overspeed
        
        # Potential energy (normalized height from bottom)
        potential_energy = np.clip((self.level_height - y) / self.level_height, 0.0, 1.0)
        
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
        
        return np.array([
            vx_norm, vy_norm, vel_magnitude, movement_state_norm,
            ground_contact, wall_contact, airborne,
            momentum_x, momentum_y, kinetic_energy, potential_energy,
            jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state,
            can_jump, can_wall_jump
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
            'can_jump', 'can_wall_jump'
        ]
    
    def validate_physics_state(self, physics_features: np.ndarray) -> bool:
        """
        Validate that physics features are within expected ranges.
        
        Args:
            physics_features: Array of physics features to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        if len(physics_features) != 18:
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