"""
Tests for Physics State Extractor.

This module tests the physics state extraction functionality for momentum-augmented
node representations in the graph neural network.
"""

import pytest
import numpy as np
from npp_rl.models.physics_state_extractor import PhysicsStateExtractor


class TestPhysicsStateExtractor:
    """Test cases for PhysicsStateExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PhysicsStateExtractor()
    
    def test_initialization(self):
        """Test that the extractor initializes correctly."""
        assert self.extractor.max_hor_speed > 0
        assert self.extractor.level_height > 0
        assert len(self.extractor.ground_states) > 0
        assert len(self.extractor.air_states) > 0
        assert len(self.extractor.wall_states) > 0
    
    def test_basic_physics_extraction(self):
        """Test basic physics state extraction."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (1.0, -0.5)
        ninja_state = {
            'movement_state': 1,  # Running
            'jump_buffer': 0,
            'floor_buffer': 2,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Check feature array length
        assert len(features) == 18
        
        # Check velocity normalization
        expected_vx_norm = 1.0 / self.extractor.max_hor_speed
        expected_vy_norm = -0.5 / self.extractor.max_hor_speed
        assert abs(features[0] - expected_vx_norm) < 1e-6
        assert abs(features[1] - expected_vy_norm) < 1e-6
        
        # Check velocity magnitude
        expected_mag = np.sqrt(1.0**2 + 0.5**2) / self.extractor.max_hor_speed
        assert abs(features[2] - expected_mag) < 1e-6
        
        # Check movement state normalization
        assert abs(features[3] - 1.0/9.0) < 1e-6  # State 1 normalized
        
        # Check contact flags for running state
        assert features[4] == 1.0  # ground_contact
        assert features[5] == 0.0  # wall_contact
        assert features[6] == 0.0  # airborne
    
    def test_airborne_state(self):
        """Test physics extraction for airborne ninja."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (2.0, 1.0)  # Falling
        ninja_state = {
            'movement_state': 4,  # Falling
            'jump_buffer': 0,
            'floor_buffer': 0,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Check contact flags for falling state
        assert features[4] == 0.0  # ground_contact
        assert features[5] == 0.0  # wall_contact
        assert features[6] == 1.0  # airborne
        
        # Check movement state
        assert abs(features[3] - 4.0/9.0) < 1e-6  # State 4 normalized
    
    def test_wall_sliding_state(self):
        """Test physics extraction for wall sliding ninja."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (0.0, 2.0)  # Sliding down wall
        ninja_state = {
            'movement_state': 5,  # Wall Sliding
            'jump_buffer': 0,
            'floor_buffer': 0,
            'wall_buffer': 3,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Check contact flags for wall sliding state
        assert features[4] == 0.0  # ground_contact
        assert features[5] == 1.0  # wall_contact
        assert features[6] == 0.0  # airborne
        
        # Check wall buffer
        assert abs(features[13] - 3.0/5.0) < 1e-6  # wall_buffer normalized
        
        # Check can_wall_jump capability
        assert features[17] == 1.0  # can_wall_jump
    
    def test_momentum_direction(self):
        """Test momentum direction calculation."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (3.0, 4.0)  # 3-4-5 triangle
        ninja_state = {
            'movement_state': 1,
            'jump_buffer': 0,
            'floor_buffer': 0,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Check momentum direction (should be normalized)
        vel_mag = np.sqrt(3.0**2 + 4.0**2)
        expected_momentum_x = 3.0 / vel_mag
        expected_momentum_y = 4.0 / vel_mag
        
        assert abs(features[7] - expected_momentum_x) < 1e-6
        assert abs(features[8] - expected_momentum_y) < 1e-6
    
    def test_energy_calculations(self):
        """Test kinetic and potential energy calculations."""
        ninja_position = (100.0, 100.0)  # Near top of level
        ninja_velocity = (2.0, 1.0)
        ninja_state = {
            'movement_state': 3,  # Jumping
            'jump_buffer': 0,
            'floor_buffer': 0,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': True
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Check kinetic energy
        expected_ke = 0.5 * (2.0**2 + 1.0**2) / (self.extractor.max_hor_speed**2)
        assert abs(features[9] - expected_ke) < 1e-6
        
        # Check potential energy (high position should give high PE)
        expected_pe = (self.extractor.level_height - 100.0) / self.extractor.level_height
        assert abs(features[10] - expected_pe) < 1e-6
        
        # Check jump input state
        assert features[15] == 1.0  # input_state
    
    def test_jump_capabilities(self):
        """Test jump capability detection."""
        # Test ground jump capability
        ninja_position = (100.0, 200.0)
        ninja_velocity = (0.0, 0.0)
        ninja_state = {
            'movement_state': 0,  # Immobile on ground
            'jump_buffer': 0,
            'floor_buffer': 5,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        assert features[16] == 1.0  # can_jump (on ground)
        assert features[17] == 0.0  # can_wall_jump (not on wall)
        
        # Test wall jump capability
        ninja_state['movement_state'] = 5  # Wall sliding
        ninja_state['wall_buffer'] = 2
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        assert features[16] == 1.0  # can_jump (wall jump available)
        assert features[17] == 1.0  # can_wall_jump (on wall)
    
    def test_buffer_normalization(self):
        """Test that buffers are properly normalized."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (0.0, 0.0)
        ninja_state = {
            'movement_state': 1,
            'jump_buffer': 5,      # Max jump buffer
            'floor_buffer': 3,
            'wall_buffer': 4,
            'launch_pad_buffer': 4,  # Max launch pad buffer
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        assert features[11] == 1.0  # jump_buffer normalized (5/5)
        assert abs(features[12] - 3.0/5.0) < 1e-6  # floor_buffer normalized
        assert abs(features[13] - 4.0/5.0) < 1e-6  # wall_buffer normalized
        assert features[14] == 1.0  # launch_pad_buffer normalized (4/4)
    
    def test_velocity_clamping(self):
        """Test that extreme velocities are properly clamped."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (10.0, -15.0)  # Very high velocities
        ninja_state = {
            'movement_state': 4,  # Falling
            'jump_buffer': 0,
            'floor_buffer': 0,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Velocity components should be clamped to [-1, 1]
        assert -1.1 <= features[0] <= 1.1  # vx_norm
        assert -1.1 <= features[1] <= 1.1  # vy_norm
        
        # Velocity magnitude should be reasonable
        assert features[2] >= 0.0
        assert features[2] <= 2.1  # Allow some overspeed
    
    def test_feature_validation(self):
        """Test the feature validation method."""
        # Valid features
        valid_features = np.array([
            0.5, -0.3, 0.6, 0.2,  # velocity and state
            1.0, 0.0, 0.0,        # contact flags
            0.8, -0.6, 0.4, 0.7,  # momentum and energy
            0.2, 0.4, 0.0, 0.0, 0.0,  # buffers
            1.0, 0.0              # capabilities
        ], dtype=np.float32)
        
        assert self.extractor.validate_physics_state(valid_features)
        
        # Invalid features (wrong length)
        invalid_features = np.array([0.5, -0.3], dtype=np.float32)
        assert not self.extractor.validate_physics_state(invalid_features)
        
        # Invalid features (velocity out of range)
        invalid_velocity = valid_features.copy()
        invalid_velocity[0] = 2.0  # Too high
        assert not self.extractor.validate_physics_state(invalid_velocity)
        
        # Invalid features (negative energy)
        invalid_energy = valid_features.copy()
        invalid_energy[9] = -0.5  # Negative kinetic energy
        assert not self.extractor.validate_physics_state(invalid_energy)
    
    def test_feature_names(self):
        """Test that feature names are provided correctly."""
        names = self.extractor.get_feature_names()
        assert len(names) == 18
        assert 'velocity_x_norm' in names
        assert 'kinetic_energy' in names
        assert 'can_jump' in names
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with None values
        ninja_position = (100.0, 200.0)
        ninja_velocity = (0.0, 0.0)
        ninja_state = {}  # Empty state
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Should not crash and return valid array
        assert len(features) == 18
        assert np.all(np.isfinite(features))
        
        # Test with list/tuple values in state
        ninja_state = {
            'movement_state': [3],  # List instead of int
            'jump_buffer': (2,),    # Tuple instead of int
            'jump_input': [True]    # List instead of bool
        }
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state
        )
        
        # Should handle gracefully
        assert len(features) == 18
        assert np.all(np.isfinite(features))