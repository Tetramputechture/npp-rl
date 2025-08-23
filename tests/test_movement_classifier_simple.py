"""
Simple tests for MovementClassifier.

This module tests the movement type classification functionality
for N++ RL graph edge features.
"""

import unittest
from npp_rl.models.movement_classifier import MovementClassifier, MovementType, NinjaState


class TestMovementClassifier(unittest.TestCase):
    """Test cases for MovementClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()
    
    def test_classify_walk_movement(self):
        """Test classification of walking movement."""
        # Horizontal movement
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 100.0)
        )
        
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
        
        # Should classify horizontal movement as WALK
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_jump_movement(self):
        """Test classification of jumping movement."""
        # Upward movement
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 50.0)
        )
        
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
        
        # Should classify upward movement as JUMP
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_classify_fall_movement(self):
        """Test classification of falling movement."""
        # Downward movement
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 150.0)
        )
        
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
        
        # Should classify downward movement as FALL
        self.assertEqual(movement_type, MovementType.FALL)
    
    def test_classify_with_ninja_state(self):
        """Test classification with ninja state."""
        ninja_state = NinjaState(
            movement_state=1,  # Running
            velocity=(50.0, 0.0),
            position=(100.0, 100.0),
            ground_contact=True
        )
        
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 100.0),
            ninja_state=ninja_state
        )
        
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
    
    def test_classify_zero_movement(self):
        """Test classification with zero movement."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(100.0, 100.0)
        )
        
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
    
    def test_movement_type_enum_values(self):
        """Test that MovementType enum has expected values."""
        expected_types = ['WALK', 'JUMP', 'FALL', 'WALL_SLIDE', 'WALL_JUMP', 'LAUNCH_PAD']
        
        for type_name in expected_types:
            self.assertTrue(hasattr(MovementType, type_name))
        
        # Test that enum values are integers (IntEnum)
        self.assertIsInstance(MovementType.WALK, int)
        self.assertIsInstance(MovementType.JUMP, int)
        self.assertIsInstance(MovementType.FALL, int)
        
        # Test specific values
        self.assertEqual(MovementType.WALK, 0)
        self.assertEqual(MovementType.JUMP, 1)
        self.assertEqual(MovementType.FALL, 2)
        self.assertEqual(MovementType.WALL_SLIDE, 3)
        self.assertEqual(MovementType.WALL_JUMP, 4)
        self.assertEqual(MovementType.LAUNCH_PAD, 5)
    
    def test_ninja_state_creation(self):
        """Test NinjaState creation and attributes."""
        state = NinjaState(
            movement_state=3,
            velocity=(25.0, -50.0),
            position=(200.0, 150.0),
            ground_contact=False,
            wall_contact=True
        )
        
        self.assertEqual(state.movement_state, 3)
        self.assertEqual(state.velocity, (25.0, -50.0))
        self.assertEqual(state.position, (200.0, 150.0))
        self.assertFalse(state.ground_contact)
        self.assertTrue(state.wall_contact)
    
    def test_ninja_state_defaults(self):
        """Test NinjaState default values."""
        state = NinjaState()
        
        self.assertEqual(state.movement_state, 0)
        self.assertEqual(state.velocity, (0.0, 0.0))
        self.assertEqual(state.position, (0.0, 0.0))
        self.assertTrue(state.ground_contact)
        self.assertFalse(state.wall_contact)


class TestMovementClassifierEdgeCases(unittest.TestCase):
    """Test edge cases for MovementClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()
    
    def test_classify_with_none_ninja_state(self):
        """Test classification with None ninja state."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 50.0),
            ninja_state=None
        )
        
        # Should handle None state gracefully
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
    
    def test_classify_with_level_data(self):
        """Test classification with level data."""
        level_data = {
            'tiles': [[0 for _ in range(10)] for _ in range(10)]
        }
        
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 100.0),
            level_data=level_data
        )
        
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
    
    def test_classify_negative_coordinates(self):
        """Test classification with negative coordinates."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(-100.0, -100.0),
            tgt_pos=(-50.0, -150.0)
        )
        
        # Should handle negative coordinates
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
    
    def test_classify_large_distance(self):
        """Test classification with large distance."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(0.0, 0.0),
            tgt_pos=(1000.0, 1000.0)
        )
        
        # Should handle large distances
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)
    
    def test_parameters_structure(self):
        """Test that returned parameters have expected structure."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 50.0)
        )
        
        # Parameters should be a dictionary
        self.assertIsInstance(params, dict)
        
        # Should contain physics-related parameters
        # (Exact keys depend on implementation)
        for key, value in params.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, (int, float, bool))


if __name__ == '__main__':
    unittest.main()