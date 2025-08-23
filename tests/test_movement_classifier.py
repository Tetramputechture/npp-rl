"""
Tests for MovementClassifier.

This module tests the movement type classification functionality
for N++ RL graph edge features.
"""

import unittest
from npp_rl.models.movement_classifier import MovementClassifier, MovementType


class TestMovementClassifier(unittest.TestCase):
    """Test cases for MovementClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()
    
    def test_classify_walk_movement(self):
        """Test classification of walking movement."""
        # Horizontal movement with low velocity
        movement_type = self.classifier.classify_movement(
            velocity=(50.0, 0.0),
            ninja_state=0,  # Standing/walking state
            dx=100.0,
            dy=0.0
        )
        
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_jump_movement(self):
        """Test classification of jumping movement."""
        # Upward movement with negative vertical velocity
        movement_type = self.classifier.classify_movement(
            velocity=(50.0, -100.0),
            ninja_state=1,  # Jumping state
            dx=50.0,
            dy=-100.0
        )
        
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_classify_fall_movement(self):
        """Test classification of falling movement."""
        # Downward movement with positive vertical velocity
        movement_type = self.classifier.classify_movement(
            velocity=(20.0, 80.0),
            ninja_state=2,  # Falling state
            dx=20.0,
            dy=100.0
        )
        
        self.assertEqual(movement_type, MovementType.FALL)
    
    def test_classify_wall_slide_movement(self):
        """Test classification of wall sliding movement."""
        # Slow downward movement against wall
        movement_type = self.classifier.classify_movement(
            velocity=(0.0, 30.0),
            ninja_state=3,  # Wall sliding state
            dx=0.0,
            dy=50.0
        )
        
        self.assertEqual(movement_type, MovementType.WALL_SLIDE)
    
    def test_classify_wall_jump_movement(self):
        """Test classification of wall jumping movement."""
        # Horizontal movement away from wall with upward velocity
        movement_type = self.classifier.classify_movement(
            velocity=(100.0, -50.0),
            ninja_state=4,  # Wall jumping state
            dx=100.0,
            dy=-30.0
        )
        
        self.assertEqual(movement_type, MovementType.WALL_JUMP)
    
    def test_classify_launch_pad_movement(self):
        """Test classification of launch pad movement."""
        # Very high velocity movement
        movement_type = self.classifier.classify_movement(
            velocity=(200.0, -200.0),
            ninja_state=5,  # Launch pad state
            dx=200.0,
            dy=-200.0
        )
        
        self.assertEqual(movement_type, MovementType.LAUNCH_PAD)
    
    def test_classify_with_none_velocity(self):
        """Test classification with None velocity."""
        movement_type = self.classifier.classify_movement(
            velocity=None,
            ninja_state=0,
            dx=50.0,
            dy=0.0
        )
        
        # Should default to WALK for horizontal movement
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_with_none_state(self):
        """Test classification with None ninja state."""
        movement_type = self.classifier.classify_movement(
            velocity=(50.0, -100.0),
            ninja_state=None,
            dx=50.0,
            dy=-100.0
        )
        
        # Should classify based on velocity and displacement
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_classify_zero_movement(self):
        """Test classification with zero movement."""
        movement_type = self.classifier.classify_movement(
            velocity=(0.0, 0.0),
            ninja_state=0,
            dx=0.0,
            dy=0.0
        )
        
        # Should default to WALK for no movement
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_small_movement(self):
        """Test classification with very small movement."""
        movement_type = self.classifier.classify_movement(
            velocity=(1.0, 0.5),
            ninja_state=0,
            dx=2.0,
            dy=1.0
        )
        
        # Should classify as WALK for small movements
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_large_horizontal_movement(self):
        """Test classification with large horizontal movement."""
        movement_type = self.classifier.classify_movement(
            velocity=(150.0, 10.0),
            ninja_state=0,
            dx=300.0,
            dy=20.0
        )
        
        # Should classify as WALK for primarily horizontal movement
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_large_vertical_upward_movement(self):
        """Test classification with large upward movement."""
        movement_type = self.classifier.classify_movement(
            velocity=(30.0, -120.0),
            ninja_state=1,
            dx=30.0,
            dy=-200.0
        )
        
        # Should classify as JUMP for large upward movement
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_classify_large_vertical_downward_movement(self):
        """Test classification with large downward movement."""
        movement_type = self.classifier.classify_movement(
            velocity=(20.0, 100.0),
            ninja_state=2,
            dx=20.0,
            dy=200.0
        )
        
        # Should classify as FALL for large downward movement
        self.assertEqual(movement_type, MovementType.FALL)
    
    def test_classify_diagonal_movement(self):
        """Test classification with diagonal movement."""
        # Equal horizontal and vertical components
        movement_type = self.classifier.classify_movement(
            velocity=(70.0, -70.0),
            ninja_state=1,
            dx=100.0,
            dy=-100.0
        )
        
        # Should classify as JUMP for upward diagonal movement
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_classify_edge_case_velocities(self):
        """Test classification with edge case velocities."""
        # Test with very high velocity (launch pad threshold)
        movement_type = self.classifier.classify_movement(
            velocity=(180.0, -180.0),
            ninja_state=0,
            dx=180.0,
            dy=-180.0
        )
        
        # Should classify as LAUNCH_PAD for very high velocities
        self.assertEqual(movement_type, MovementType.LAUNCH_PAD)
    
    def test_classify_wall_interaction_states(self):
        """Test classification with wall interaction states."""
        # Test wall slide with appropriate state
        movement_type = self.classifier.classify_movement(
            velocity=(0.0, 40.0),
            ninja_state=3,  # Wall sliding
            dx=0.0,
            dy=60.0
        )
        
        self.assertEqual(movement_type, MovementType.WALL_SLIDE)
        
        # Test wall jump with appropriate state
        movement_type = self.classifier.classify_movement(
            velocity=(80.0, -40.0),
            ninja_state=4,  # Wall jumping
            dx=120.0,
            dy=-60.0
        )
        
        self.assertEqual(movement_type, MovementType.WALL_JUMP)


class TestMovementClassifierEdgeCases(unittest.TestCase):
    """Test edge cases for MovementClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()
    
    def test_classify_with_invalid_state(self):
        """Test classification with invalid ninja state."""
        movement_type = self.classifier.classify_movement(
            velocity=(50.0, 0.0),
            ninja_state=999,  # Invalid state
            dx=50.0,
            dy=0.0
        )
        
        # Should still classify based on velocity/displacement
        self.assertEqual(movement_type, MovementType.WALK)
    
    def test_classify_with_negative_velocities(self):
        """Test classification with negative velocities."""
        movement_type = self.classifier.classify_movement(
            velocity=(-50.0, -100.0),
            ninja_state=1,
            dx=-50.0,
            dy=-100.0
        )
        
        # Should handle negative velocities correctly
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_classify_inconsistent_velocity_displacement(self):
        """Test classification with inconsistent velocity and displacement."""
        # Velocity suggests jump, displacement suggests fall
        movement_type = self.classifier.classify_movement(
            velocity=(50.0, -100.0),  # Upward velocity
            ninja_state=0,
            dx=50.0,
            dy=100.0  # Downward displacement
        )
        
        # Should prioritize velocity over displacement
        self.assertEqual(movement_type, MovementType.JUMP)
    
    def test_movement_type_enum_values(self):
        """Test that MovementType enum has expected values."""
        expected_types = ['WALK', 'JUMP', 'FALL', 'WALL_SLIDE', 'WALL_JUMP', 'LAUNCH_PAD']
        
        for type_name in expected_types:
            self.assertTrue(hasattr(MovementType, type_name))
        
        # Test that enum values are strings
        self.assertIsInstance(MovementType.WALK, str)
        self.assertIsInstance(MovementType.JUMP, str)
        self.assertIsInstance(MovementType.FALL, str)


if __name__ == '__main__':
    unittest.main()