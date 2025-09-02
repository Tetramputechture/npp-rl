"""
Comprehensive unit tests for physics integration components.

Tests the movement classifier, trajectory calculator, and physics state extractor
with actual N++ physics integration.
"""

import unittest
import numpy as np
import math
from unittest.mock import Mock, patch

from npp_rl.models.movement_classifier import MovementClassifier, MovementType
from npp_rl.models.trajectory_calculator import TrajectoryCalculator, MovementState, TrajectoryResult
from npp_rl.models.physics_state_extractor import PhysicsStateExtractor

from nclone.constants import (
    MAX_HOR_SPEED, NINJA_RADIUS, JUMP_FLAT_GROUND_Y,
    JUMP_WALL_REGULAR_X, JUMP_WALL_REGULAR_Y,
    GRAVITY_FALL, GRAVITY_JUMP
)
from nclone.entity_classes.entity_launch_pad import EntityLaunchPad


class TestMovementClassifier(unittest.TestCase):
    """Test the enhanced movement classifier with N++ physics integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()
        
        # Mock ninja state
        self.ninja_state = Mock()
        self.ninja_state.ground_contact = True
        self.ninja_state.wall_contact = False
        self.ninja_state.air_contact = False
        
        # Mock level data with launch pad
        self.level_data = {
            'entities': [
                {
                    'type': EntityType.LAUNCH_PAD,
                    'x': 100.0,
                    'y': 100.0,
                    'orientation': 0  # Pointing right
                }
            ],
            'tiles': np.zeros((20, 20))  # Empty level
        }
    
    def test_classify_walk_movement(self):
        """Test classification of walking movement."""
        start_pos = (0.0, 100.0)
        end_pos = (30.0, 100.0)  # Short horizontal movement, away from launch pad
        
        # Use level data without launch pad to avoid interference
        simple_level_data = {
            'entities': [],
            'tiles': np.zeros((20, 20))
        }
        
        movement_type, params = self.classifier.classify_movement(
            start_pos, end_pos, self.ninja_state, simple_level_data
        )
        
        self.assertEqual(movement_type, MovementType.WALK)
        self.assertIn('required_velocity', params)
        self.assertIn('energy_cost', params)
        self.assertIn('time_estimate', params)
        self.assertIn('difficulty', params)
        
        # Verify physics parameters are reasonable
        self.assertGreater(params['required_velocity'], 0)
        self.assertLessEqual(params['required_velocity'], MAX_HOR_SPEED)
        self.assertGreater(params['energy_cost'], 0)
        self.assertGreater(params['time_estimate'], 0)
        self.assertGreaterEqual(params['difficulty'], 0)
        self.assertLessEqual(params['difficulty'], 1.0)
    
    def test_classify_jump_movement(self):
        """Test classification of jumping movement."""
        start_pos = (0.0, 100.0)
        end_pos = (30.0, 50.0)  # Upward diagonal movement
        
        movement_type, params = self.classifier.classify_movement(
            start_pos, end_pos, self.ninja_state, self.level_data
        )
        
        self.assertEqual(movement_type, MovementType.JUMP)
        
        # Verify jump physics parameters
        self.assertGreater(params['required_velocity'], 0)
        self.assertGreater(params['energy_cost'], 1.0)  # Jumps cost more energy
        self.assertGreater(params['time_estimate'], 0)
    
    def test_classify_wall_jump_movement(self):
        """Test classification of wall jump movement."""
        # Set ninja state to wall contact
        wall_ninja_state = Mock()
        wall_ninja_state.wall_contact = True
        wall_ninja_state.ground_contact = False
        wall_ninja_state.air_contact = False
        
        start_pos = (0.0, 100.0)
        end_pos = (20.0, 80.0)  # Away from wall, upward - smaller movement
        
        # Use simple level data
        simple_level_data = {
            'entities': [],
            'tiles': np.zeros((20, 20))
        }
        
        movement_type, params = self.classifier.classify_movement(
            start_pos, end_pos, wall_ninja_state, simple_level_data
        )
        
        # Should be wall slide or wall jump
        self.assertIn(movement_type, [MovementType.WALL_SLIDE, MovementType.WALL_JUMP])
        
        # Wall movements should have reasonable difficulty and energy cost
        self.assertGreater(params['difficulty'], 0.0)
        self.assertGreater(params['energy_cost'], 1.0)
    
    def test_classify_launch_pad_movement(self):
        """Test classification of launch pad movement."""
        start_pos = (95.0, 105.0)  # Near launch pad
        end_pos = (200.0, 50.0)   # Large movement that aligns with launch pad
        
        movement_type, params = self.classifier.classify_movement(
            start_pos, end_pos, self.ninja_state, self.level_data
        )
        
        self.assertEqual(movement_type, MovementType.LAUNCH_PAD)
        
        # Launch pad movements should have lower energy cost
        self.assertLess(params['energy_cost'], 1.0)
        self.assertLess(params['difficulty'], 0.5)
    
    def test_movement_sequence_classification(self):
        """Test classification of movement sequences."""
        positions = [
            (0.0, 100.0),
            (30.0, 100.0),  # Walk - shorter distance
            (50.0, 80.0),   # Jump
            (70.0, 60.0)    # Continue jump
        ]
        
        # Use simple level data
        simple_level_data = {
            'entities': [],
            'tiles': np.zeros((20, 20))
        }
        
        ninja_states = [self.ninja_state] * len(positions)
        
        movements = self.classifier.classify_movement_sequence(
            positions, ninja_states, simple_level_data
        )
        
        self.assertEqual(len(movements), 3)  # 3 movement segments
        
        # Verify we get reasonable movement types
        movement_types = [m[0] for m in movements]
        self.assertTrue(all(mt in [MovementType.WALK, MovementType.JUMP, MovementType.WALL_SLIDE] 
                           for mt in movement_types))
    
    def test_movement_chain_adjustments(self):
        """Test that movement chains apply appropriate adjustments."""
        # Create a wall jump chain
        positions = [
            (0.0, 100.0),
            (20.0, 80.0),   # First wall movement
            (40.0, 60.0)    # Second wall movement
        ]
        
        wall_ninja_state = Mock()
        wall_ninja_state.wall_contact = True
        wall_ninja_state.ground_contact = False
        wall_ninja_state.air_contact = False
        
        # Use simple level data
        simple_level_data = {
            'entities': [],
            'tiles': np.zeros((20, 20))
        }
        
        ninja_states = [wall_ninja_state] * len(positions)
        
        movements = self.classifier.classify_movement_sequence(
            positions, ninja_states, simple_level_data
        )
        
        # Verify we have movements and they have difficulty parameters
        self.assertGreater(len(movements), 1)
        for movement_type, params in movements:
            self.assertIn('difficulty', params)
            self.assertGreaterEqual(params['difficulty'], 0.0)


class TestTrajectoryCalculator(unittest.TestCase):
    """Test the enhanced trajectory calculator with physics integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = TrajectoryCalculator()
        
        # Mock level data
        self.level_data = {
            'tiles': np.zeros((20, 20)),  # Empty level
            'sim': None  # No simulation object for basic tests
        }
    
    def test_calculate_jump_trajectory(self):
        """Test basic jump trajectory calculation."""
        start_pos = (0.0, 100.0)
        end_pos = (50.0, 50.0)
        
        result = self.calculator.calculate_jump_trajectory(
            start_pos, end_pos, MovementState.JUMPING
        )
        
        self.assertIsInstance(result, TrajectoryResult)
        self.assertTrue(result.feasible)
        self.assertGreater(result.time_of_flight, 0)
        self.assertGreater(result.energy_cost, 0)
        self.assertGreater(result.success_probability, 0)
        self.assertLessEqual(result.success_probability, 1.0)
        self.assertTrue(result.requires_jump)
        self.assertGreater(len(result.trajectory_points), 0)
    
    def test_calculate_momentum_trajectory(self):
        """Test momentum-dependent trajectory calculation."""
        start_pos = (0.0, 100.0)
        end_pos = (100.0, 80.0)
        initial_velocity = (2.0, -1.0)  # Moving right and up
        
        # Use level data without sim object to avoid collision detection
        simple_level_data = {
            'tiles': np.zeros((20, 20))
        }
        
        result = self.calculator.calculate_momentum_trajectory(
            start_pos, end_pos, initial_velocity,
            MovementState.AIRBORNE, simple_level_data
        )
        
        self.assertIsInstance(result, TrajectoryResult)
        self.assertGreater(result.time_of_flight, 0)
        self.assertGreater(len(result.trajectory_points), 1)
        
        # Verify trajectory follows physics
        points = result.trajectory_points
        self.assertEqual(points[0], start_pos)
        
        # Check that trajectory moves in expected direction initially
        if len(points) > 1:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            self.assertGreater(dx, 0)  # Should move right initially
    
    def test_calculate_wall_jump_trajectory(self):
        """Test wall jump trajectory calculation."""
        start_pos = (0.0, 100.0)
        end_pos = (50.0, 70.0)
        wall_normal = (-1.0, 0.0)  # Wall on the right, normal points left
        
        # Use level data without sim object
        simple_level_data = {
            'tiles': np.zeros((20, 20))
        }
        
        result = self.calculator.calculate_wall_jump_trajectory(
            start_pos, end_pos, wall_normal,
            MovementState.WALL_JUMPING, simple_level_data
        )
        
        self.assertIsInstance(result, TrajectoryResult)
        self.assertTrue(result.requires_wall_contact)
        self.assertTrue(result.requires_jump)
        
        # Wall jump should have reasonable physics parameters
        self.assertGreater(result.min_velocity, 0)
        self.assertLessEqual(result.max_velocity, MAX_HOR_SPEED * 2)  # Allow some overspeed
    
    def test_trajectory_collision_detection(self):
        """Test trajectory collision detection integration."""
        start_pos = (0.0, 100.0)
        end_pos = (50.0, 50.0)
        
        result = self.calculator.calculate_jump_trajectory(
            start_pos, end_pos, MovementState.JUMPING
        )
        
        # Test with empty level (should be clear)
        empty_level_data = {
            'tiles': np.zeros((10, 10))
        }
        
        is_clear = self.calculator.validate_trajectory_clearance(
            result.trajectory_points, empty_level_data
        )
        
        self.assertTrue(is_clear)
        
        # Test with solid level (should detect collision)
        solid_level_data = {
            'tiles': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        }
        
        is_clear = self.calculator.validate_trajectory_clearance(
            result.trajectory_points, solid_level_data
        )
        
        self.assertFalse(is_clear)
    
    def test_basic_tile_collision_detection(self):
        """Test basic tile-based collision detection fallback."""
        # Create level with solid tiles
        level_data = {
            'tiles': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # All solid tiles as list
        }
        
        trajectory_points = [
            (0.0, 0.0),
            (24.0, 24.0),  # Should intersect with solid tile
            (48.0, 48.0)
        ]
        
        is_clear = self.calculator.validate_trajectory_clearance(
            trajectory_points, level_data
        )
        
        self.assertFalse(is_clear)  # Should detect collision with solid tiles
    
    def test_success_probability_calculation(self):
        """Test success probability calculation factors."""
        # Test easy movement
        easy_result = self.calculator.calculate_jump_trajectory(
            (0.0, 100.0), (20.0, 95.0), MovementState.JUMPING
        )
        
        # Test difficult movement
        difficult_result = self.calculator.calculate_jump_trajectory(
            (0.0, 100.0), (200.0, 20.0), MovementState.JUMPING
        )
        
        # Difficult movement should have lower success probability
        self.assertGreater(easy_result.success_probability, difficult_result.success_probability)


class TestPhysicsStateExtractor(unittest.TestCase):
    """Test the enhanced physics state extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PhysicsStateExtractor()
        
        # Mock ninja state with comprehensive data
        self.ninja_state = {
            'movement_state': 1,  # Running
            'jump_buffer': 3,
            'floor_buffer': 2,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'jump_input': False,
            'wall_slide_buffer': 0,
            'air_time': 0,
            'ground_time': 30
        }
        
        # Mock level data
        self.level_data = {
            'entities': [
                {
                    'type': EntityType.LAUNCH_PAD,
                    'x': 150.0,
                    'y': 100.0,
                    'orientation': 0
                }
            ],
            'tiles': np.zeros((20, 20))
        }
    
    def test_extract_basic_physics_features(self):
        """Test extraction of basic physics features."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (2.0, -1.0)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Should return 31 features (with multi-exit path finding)
        self.assertEqual(len(features), 31)
        
        # Check velocity normalization
        self.assertAlmostEqual(features[0], 2.0 / MAX_HOR_SPEED, places=3)  # vx_norm
        self.assertAlmostEqual(features[1], -1.0 / MAX_HOR_SPEED, places=3)  # vy_norm
        
        # Check velocity magnitude
        expected_magnitude = math.sqrt(2.0*2.0 + 1.0*1.0) / MAX_HOR_SPEED
        self.assertAlmostEqual(features[2], expected_magnitude, places=3)
        
        # Check movement state normalization
        self.assertAlmostEqual(features[3], 1.0 / 9.0, places=3)
        
        # Check contact flags (running state = ground contact)
        self.assertEqual(features[4], 1.0)  # ground_contact
        self.assertEqual(features[5], 0.0)  # wall_contact
        self.assertEqual(features[6], 0.0)  # airborne
    
    def test_extract_energy_calculations(self):
        """Test kinetic and potential energy calculations."""
        ninja_position = (100.0, 200.0)
        ninja_velocity = (3.0, 0.0)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Check kinetic energy calculation
        expected_ke = 0.5 * (3.0*3.0) / (MAX_HOR_SPEED * MAX_HOR_SPEED)
        self.assertAlmostEqual(features[9], expected_ke, places=3)
        
        # Check potential energy (normalized height)
        self.assertGreaterEqual(features[10], 0.0)
        self.assertLessEqual(features[10], 1.0)
    
    def test_extract_buffer_states(self):
        """Test extraction of buffer states."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (0.0, 0.0)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Check buffer normalization
        self.assertAlmostEqual(features[11], 3.0 / 5.0, places=3)  # jump_buffer
        self.assertAlmostEqual(features[12], 2.0 / 5.0, places=3)  # floor_buffer
        self.assertEqual(features[13], 0.0)  # wall_buffer
        self.assertEqual(features[14], 0.0)  # launch_pad_buffer
        self.assertEqual(features[15], 0.0)  # input_state
    
    def test_extract_physics_capabilities(self):
        """Test extraction of physics capabilities."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (0.0, 0.0)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Should be able to jump (ground contact + jump buffer)
        self.assertEqual(features[16], 1.0)  # can_jump
        
        # Should not be able to wall jump (no wall contact)
        self.assertEqual(features[17], 0.0)  # can_wall_jump
    
    def test_extract_contact_normal(self):
        """Test contact normal extraction."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (0.0, 0.0)
        
        # Test ground contact
        ground_state = self.ninja_state.copy()
        ground_state['movement_state'] = 1  # Running (ground contact)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ground_state, self.level_data
        )
        
        # Ground contact should have upward normal
        self.assertEqual(features[18], 0.0)   # contact_normal_x
        self.assertEqual(features[19], -1.0)  # contact_normal_y (upward)
    
    def test_extract_entity_proximity(self):
        """Test entity proximity extraction."""
        ninja_position = (140.0, 100.0)  # Near launch pad at (150, 100)
        ninja_velocity = (0.0, 0.0)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Should detect launch pad proximity
        self.assertGreater(features[20], 0.0)  # launch_pad_proximity
        self.assertEqual(features[21], 0.0)    # hazard_proximity
        self.assertEqual(features[22], 0.0)    # collectible_proximity
    
    def test_extract_advanced_buffers(self):
        """Test advanced buffer state extraction."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (0.0, 0.0)
        
        # Set advanced buffer states
        advanced_state = self.ninja_state.copy()
        advanced_state['wall_slide_buffer'] = 5
        advanced_state['air_time'] = 30
        advanced_state['ground_time'] = 60
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, advanced_state, self.level_data
        )
        
        # Check advanced buffer normalization
        self.assertAlmostEqual(features[23], 5.0 / 10.0, places=3)  # wall_slide_buffer
        self.assertAlmostEqual(features[24], 30.0 / 60.0, places=3)  # air_time
        self.assertAlmostEqual(features[25], 1.0, places=3)  # ground_time (clamped)
    
    def test_extract_physics_constraints(self):
        """Test physics constraints extraction."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (1.0, -2.0)  # Moving up
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Check max jump height calculation
        self.assertGreater(features[26], 0.0)  # max_jump_height
        self.assertLessEqual(features[26], 1.0)
        
        # Check remaining air acceleration
        expected_remaining = (MAX_HOR_SPEED - 1.0) / MAX_HOR_SPEED
        self.assertAlmostEqual(features[27], expected_remaining, places=3)
    
    def test_feature_validation(self):
        """Test physics feature validation."""
        ninja_position = (100.0, 100.0)
        ninja_velocity = (2.0, -1.0)
        
        features = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Validate features
        is_valid = self.extractor.validate_physics_state(features)
        self.assertTrue(is_valid)
        
        # Test invalid features
        invalid_features = features.copy()
        invalid_features[0] = 2.0  # Invalid velocity (too high)
        
        is_valid = self.extractor.validate_physics_state(invalid_features)
        self.assertFalse(is_valid)
    
    def test_get_feature_names(self):
        """Test feature names retrieval."""
        feature_names = self.extractor.get_feature_names()
        
        self.assertEqual(len(feature_names), 31)
        self.assertIn('velocity_x_norm', feature_names)
        self.assertIn('contact_normal_x', feature_names)
        self.assertIn('launch_pad_proximity', feature_names)
        self.assertIn('max_jump_height', feature_names)


if __name__ == '__main__':
    unittest.main()