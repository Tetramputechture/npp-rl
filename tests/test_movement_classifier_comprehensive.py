"""
Comprehensive tests for MovementClassifier.

This module provides complete test coverage for the movement type classification
functionality for N++ RL graph edge features.
"""

import unittest
import numpy as np

from npp_rl.models.movement_classifier import (
    MovementClassifier,
    MovementType,
    NinjaState
)


class TestMovementClassifier(unittest.TestCase):
    """Comprehensive test cases for MovementClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()

    def test_classify_walk_movement(self):
        """Test classification of walking movement."""
        # Horizontal movement on same level
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 100.0)
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Should classify horizontal movement as WALK
        self.assertEqual(movement_type, MovementType.WALK)

        # Verify parameters structure
        for key, value in params.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, (int, float, bool))

    def test_classify_jump_movement(self):
        """Test classification of jumping movement."""
        # Upward movement requiring jump
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

    def test_classify_wall_slide_movement(self):
        """Test classification of wall sliding movement."""
        # Create ninja state with wall contact
        ninja_state = NinjaState(
            movement_state=5,  # Wall sliding state
            velocity=(0.0, 25.0),  # Downward velocity
            position=(100.0, 100.0),
            ground_contact=False,
            wall_contact=True
        )

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(100.0, 150.0),
            ninja_state=ninja_state
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Should classify as wall slide when wall contact is true
        self.assertEqual(movement_type, MovementType.WALL_SLIDE)

    def test_classify_wall_jump_movement(self):
        """Test classification of wall jumping movement."""
        # Create ninja state for wall jump
        ninja_state = NinjaState(
            movement_state=6,  # Wall jumping state
            velocity=(50.0, -30.0),  # Upward and outward velocity
            position=(100.0, 100.0),
            ground_contact=False,
            wall_contact=True
        )

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 70.0),
            ninja_state=ninja_state
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Should classify as wall-related movement (wall jump or wall slide)
        self.assertIn(movement_type, [MovementType.WALL_JUMP, MovementType.WALL_SLIDE])

    def test_classify_launch_pad_movement(self):
        """Test classification of launch pad movement."""
        # Create ninja state for launch pad
        ninja_state = NinjaState(
            movement_state=7,  # Launch pad state
            velocity=(100.0, -100.0),  # High velocity
            position=(100.0, 100.0),
            ground_contact=False,
            wall_contact=False
        )

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(300.0, 50.0),
            ninja_state=ninja_state
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Should classify as high-velocity movement (launch pad or jump)
        self.assertIn(movement_type, [MovementType.LAUNCH_PAD, MovementType.JUMP])

    def test_classify_with_ninja_state(self):
        """Test classification with various ninja states."""
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

        # Should use ninja state information in classification
        self.assertEqual(movement_type, MovementType.WALK)

    def test_classify_zero_movement(self):
        """Test classification with zero movement."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(100.0, 100.0)
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Zero movement should still return valid classification
        # (Implementation may vary - could be WALK or special handling)

    def test_classify_with_level_data(self):
        """Test classification with level geometry data."""
        level_data = {
            'tiles': np.zeros((20, 20), dtype=int)
        }
        # Add some walls
        level_data['tiles'][10:12, 5:15] = 1

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 100.0),
            level_data=level_data
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

    def test_parameters_structure(self):
        """Test that returned parameters have expected structure."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 50.0)
        )

        # Parameters should be a dictionary
        self.assertIsInstance(params, dict)

        # Should contain physics-related parameters
        for key, value in params.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, (int, float, bool))

            # Parameter names should be descriptive
            self.assertGreater(len(key), 0)


class TestNinjaState(unittest.TestCase):
    """Test cases for NinjaState class."""

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

    def test_ninja_state_velocity_components(self):
        """Test NinjaState velocity component access."""
        state = NinjaState(velocity=(30.0, -20.0))

        vx, vy = state.velocity
        self.assertEqual(vx, 30.0)
        self.assertEqual(vy, -20.0)

    def test_ninja_state_position_components(self):
        """Test NinjaState position component access."""
        state = NinjaState(position=(150.0, 75.0))

        x, y = state.position
        self.assertEqual(x, 150.0)
        self.assertEqual(y, 75.0)


class TestMovementClassifierEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for MovementClassifier."""

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

    def test_classify_small_movement(self):
        """Test classification with very small movement."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(100.1, 100.1)
        )

        # Should handle small movements
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

    def test_classify_with_extreme_velocities(self):
        """Test classification with extreme velocity values."""
        # Very high velocity
        ninja_state = NinjaState(velocity=(1000.0, -1000.0))

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(200.0, 50.0),
            ninja_state=ninja_state
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Very low velocity
        ninja_state = NinjaState(velocity=(0.001, -0.001))

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(100.1, 100.1),
            ninja_state=ninja_state
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

    def test_classify_inconsistent_velocity_displacement(self):
        """Test classification with inconsistent velocity and displacement."""
        # Velocity suggests rightward movement, but displacement is leftward
        ninja_state = NinjaState(velocity=(50.0, 0.0))

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(50.0, 100.0),  # Leftward displacement
            ninja_state=ninja_state
        )

        # Should handle inconsistency gracefully
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

    def test_classify_with_invalid_movement_state(self):
        """Test classification with invalid ninja movement state."""
        ninja_state = NinjaState(movement_state=999)  # Invalid state

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(150.0, 100.0),
            ninja_state=ninja_state
        )

        # Should handle invalid state gracefully
        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)


class TestMovementClassifierIntegration(unittest.TestCase):
    """Integration tests for MovementClassifier with physics system."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()

    def test_classification_consistency(self):
        """Test that classification is consistent for similar movements."""
        # Test multiple similar horizontal movements
        base_movements = [
            ((100.0, 100.0), (150.0, 100.0)),
            ((200.0, 200.0), (250.0, 200.0)),
            ((0.0, 50.0), (50.0, 50.0))
        ]

        movement_types = []
        for src_pos, tgt_pos in base_movements:
            movement_type, _ = self.classifier.classify_movement(src_pos, tgt_pos)
            movement_types.append(movement_type)

        # All horizontal movements should have same classification
        self.assertEqual(len(set(movement_types)), 1)
        self.assertEqual(movement_types[0], MovementType.WALK)

    def test_physics_parameter_calculation(self):
        """Test physics parameter calculation accuracy."""
        movement_type, params = self.classifier.classify_movement(
            src_pos=(0.0, 100.0),
            tgt_pos=(100.0, 50.0)
        )

        # Verify parameter ranges are reasonable
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # No infinite or NaN values
                self.assertFalse(np.isinf(value), f"Parameter {key} is infinite")
                self.assertFalse(np.isnan(value), f"Parameter {key} is NaN")

                # Reasonable ranges for physics parameters
                if 'velocity' in key.lower():
                    self.assertLessEqual(abs(value), 1000.0, f"Velocity {key} too high")
                elif 'time' in key.lower():
                    self.assertGreaterEqual(value, 0.0, f"Time {key} negative")
                elif 'energy' in key.lower():
                    self.assertGreaterEqual(value, 0.0, f"Energy {key} negative")

    def test_movement_state_integration(self):
        """Test integration with N++ movement states."""
        # Test all valid movement states
        valid_states = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 9 states from sim_mechanics_doc.md

        for state in valid_states:
            with self.subTest(movement_state=state):
                ninja_state = NinjaState(movement_state=state)

                movement_type, params = self.classifier.classify_movement(
                    src_pos=(100.0, 100.0),
                    tgt_pos=(150.0, 80.0),
                    ninja_state=ninja_state
                )

                self.assertIsInstance(movement_type, MovementType)
                self.assertIsInstance(params, dict)

    def test_level_geometry_integration(self):
        """Test integration with level geometry data."""
        # Create level with various tile types
        level_data = {
            'tiles': np.random.randint(0, 5, size=(20, 20))
        }

        movement_type, params = self.classifier.classify_movement(
            src_pos=(100.0, 100.0),
            tgt_pos=(200.0, 150.0),
            level_data=level_data
        )

        self.assertIsInstance(movement_type, MovementType)
        self.assertIsInstance(params, dict)

        # Should handle complex level geometry
        self.assertGreater(len(params), 0)


if __name__ == '__main__':
    unittest.main()
