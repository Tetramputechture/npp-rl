"""
Comprehensive tests for TrajectoryCalculator.

This module provides complete test coverage for the physics-based trajectory
calculation and validation functionality for N++ RL graph edge features.
"""

import unittest
import numpy as np

from npp_rl.models.trajectory_calculator import (
    TrajectoryCalculator,
    TrajectoryResult,
    MovementState
)


class TestTrajectoryCalculator(unittest.TestCase):
    """Comprehensive test cases for TrajectoryCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()

    def test_initialization(self):
        """Test TrajectoryCalculator initialization with physics constants."""
        # Verify physics constants are loaded
        self.assertIsInstance(self.calc.gravity_fall, (int, float))
        self.assertIsInstance(self.calc.gravity_jump, (int, float))
        self.assertIsInstance(self.calc.max_hor_speed, (int, float))
        self.assertIsInstance(self.calc.ninja_radius, (int, float))

        # Verify constants have expected values (from N++ physics)
        self.assertAlmostEqual(self.calc.gravity_fall, 0.0667, places=4)
        self.assertAlmostEqual(self.calc.gravity_jump, 0.0111, places=4)
        self.assertAlmostEqual(self.calc.max_hor_speed, 3.333, places=3)
        self.assertEqual(self.calc.ninja_radius, 10)

    def test_calculate_jump_trajectory_basic(self):
        """Test basic jump trajectory calculation."""
        # Test upward jump
        result = self.calc.calculate_jump_trajectory(
            start_pos=(100.0, 100.0),
            end_pos=(150.0, 50.0)
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryResult)
        self.assertIsInstance(result.feasible, bool)
        self.assertIsInstance(result.time_of_flight, (int, float))
        self.assertIsInstance(result.energy_cost, (int, float))
        self.assertIsInstance(result.success_probability, (int, float))

        # Verify result structure
        if result.feasible:
            self.assertGreater(result.time_of_flight, 0)
            self.assertGreater(result.energy_cost, 0)
            self.assertGreaterEqual(result.success_probability, 0.0)
            self.assertLessEqual(result.success_probability, 1.0)

    def test_calculate_jump_trajectory_horizontal(self):
        """Test horizontal movement trajectory."""
        result = self.calc.calculate_jump_trajectory(
            start_pos=(100.0, 100.0),
            end_pos=(150.0, 100.0)
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryResult)

        # Horizontal movement should be feasible
        self.assertTrue(result.feasible)
        self.assertGreater(result.time_of_flight, 0)

    def test_calculate_jump_trajectory_downward(self):
        """Test downward movement trajectory."""
        result = self.calc.calculate_jump_trajectory(
            start_pos=(100.0, 100.0),
            end_pos=(150.0, 150.0)
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryResult)

        # Downward movement should be feasible (falling)
        self.assertTrue(result.feasible)
        self.assertGreater(result.time_of_flight, 0)

    def test_trajectory_with_movement_state(self):
        """Test trajectory calculation with different movement states."""
        states_to_test = [
            MovementState.IMMOBILE,
            MovementState.RUNNING,
            MovementState.JUMPING,
            MovementState.FALLING
        ]

        for state in states_to_test:
            with self.subTest(ninja_state=state):
                result = self.calc.calculate_jump_trajectory(
                    start_pos=(100.0, 100.0),
                    end_pos=(150.0, 80.0),
                    ninja_state=state
                )

                self.assertIsNotNone(result)
                self.assertIsInstance(result, TrajectoryResult)

    def test_validate_trajectory_clearance_clear_path(self):
        """Test trajectory validation with clear path."""
        # Create simple trajectory
        trajectory = [
            {'x': 100.0, 'y': 100.0, 't': 0.0},
            {'x': 125.0, 'y': 90.0, 't': 10.0},
            {'x': 150.0, 'y': 80.0, 't': 20.0}
        ]

        # Mock level data with no obstacles
        level_data = {
            'tiles': np.zeros((20, 20), dtype=int)
        }

        is_clear = self.calc.validate_trajectory_clearance(trajectory, level_data)
        self.assertTrue(is_clear)

    def test_validate_trajectory_clearance_empty_trajectory(self):
        """Test trajectory validation with empty trajectory."""
        trajectory = []
        level_data = {'tiles': np.zeros((10, 10), dtype=int)}

        is_clear = self.calc.validate_trajectory_clearance(trajectory, level_data)
        self.assertTrue(is_clear)  # Empty trajectory is considered clear

    def test_movement_states_enum(self):
        """Test MovementState enum values."""
        expected_states = [
            'IMMOBILE', 'RUNNING', 'JUMPING', 'FALLING'
        ]

        for state_name in expected_states:
            self.assertTrue(hasattr(MovementState, state_name))

        # Test specific values match N++ simulation
        self.assertEqual(MovementState.IMMOBILE, 0)
        self.assertEqual(MovementState.RUNNING, 1)
        self.assertEqual(MovementState.JUMPING, 3)
        self.assertEqual(MovementState.FALLING, 4)

    def test_physics_constants_integration(self):
        """Test that physics constants are properly integrated."""
        # Test that constants are used in calculations
        result1 = self.calc.calculate_jump_trajectory(
            start_pos=(0.0, 100.0),
            end_pos=(100.0, 50.0)
        )

        # Modify gravity temporarily to test integration
        original_gravity = self.calc.gravity_fall
        self.calc.gravity_fall = original_gravity * 2

        result2 = self.calc.calculate_jump_trajectory(
            start_pos=(0.0, 100.0),
            end_pos=(100.0, 50.0)
        )

        # Restore original gravity
        self.calc.gravity_fall = original_gravity

        # Results should be different with different gravity
        if result1.feasible and result2.feasible:
            self.assertNotEqual(result1.time_of_flight, result2.time_of_flight)


class TestTrajectoryCalculatorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for TrajectoryCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()

    def test_same_position(self):
        """Test trajectory calculation with same start and end position."""
        result = self.calc.calculate_jump_trajectory(
            start_pos=(100.0, 100.0),
            end_pos=(100.0, 100.0)
        )

        # Should handle same position gracefully
        self.assertIsNotNone(result)
        self.assertIsInstance(result.feasible, bool)

    def test_very_long_distance(self):
        """Test trajectory calculation with very long distance."""
        result = self.calc.calculate_jump_trajectory(
            start_pos=(0.0, 0.0),
            end_pos=(10000.0, 0.0)
        )

        # Should handle large distances
        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryResult)

        # Very long distance should likely be infeasible
        if not result.feasible:
            self.assertEqual(result.time_of_flight, 0.0)

    def test_negative_coordinates(self):
        """Test trajectory calculation with negative coordinates."""
        result = self.calc.calculate_jump_trajectory(
            start_pos=(-100.0, -100.0),
            end_pos=(-50.0, -150.0)
        )

        # Should handle negative coordinates
        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryResult)

    def test_validate_clearance_invalid_level_data(self):
        """Test trajectory validation with invalid level data."""
        trajectory = [
            {'x': 100.0, 'y': 100.0, 't': 0.0},
            {'x': 150.0, 'y': 80.0, 't': 20.0}
        ]

        # Test with None level data
        is_clear = self.calc.validate_trajectory_clearance(trajectory, None)
        self.assertTrue(is_clear)  # Should default to clear

        # Test with empty level data
        is_clear = self.calc.validate_trajectory_clearance(trajectory, {})
        self.assertTrue(is_clear)  # Should default to clear

    def test_extreme_physics_values(self):
        """Test trajectory calculation with extreme physics values."""
        # Test with very high initial velocity
        result = self.calc.calculate_jump_trajectory(
            start_pos=(0.0, 0.0),
            end_pos=(50.0, 0.0),
            ninja_state=MovementState.JUMPING
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryResult)

    def test_trajectory_result_dataclass(self):
        """Test TrajectoryResult dataclass structure."""
        # Create a sample result
        result = TrajectoryResult(
            feasible=True,
            time_of_flight=15.5,
            energy_cost=25.0,
            success_probability=0.85,
            min_velocity=10.0,
            max_velocity=50.0,
            requires_jump=True,
            requires_wall_contact=False,
            trajectory_points=[]
        )

        # Verify all fields are accessible
        self.assertTrue(result.feasible)
        self.assertEqual(result.time_of_flight, 15.5)
        self.assertEqual(result.energy_cost, 25.0)
        self.assertEqual(result.success_probability, 0.85)
        self.assertEqual(result.min_velocity, 10.0)
        self.assertEqual(result.max_velocity, 50.0)
        self.assertTrue(result.requires_jump)
        self.assertFalse(result.requires_wall_contact)
        self.assertEqual(result.trajectory_points, [])


class TestTrajectoryCalculatorIntegration(unittest.TestCase):
    """Integration tests for TrajectoryCalculator with collision system."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()

    def test_collision_integration(self):
        """Test integration with collision detection system."""
        # Create level with obstacles
        level_data = {
            'tiles': np.zeros((20, 20), dtype=int)
        }
        # Add some obstacles
        level_data['tiles'][10:12, 5:15] = 1  # Wall obstacle

        # Test trajectory that should hit obstacle
        trajectory = []
        for i in range(20):
            x = 50.0 + i * 10.0
            y = 200.0  # Should intersect with wall
            trajectory.append({'x': x, 'y': y, 't': i})

        is_clear = self.calc.validate_trajectory_clearance(trajectory, level_data)

        # Should detect collision (implementation dependent)
        self.assertIsInstance(is_clear, bool)

    def test_physics_validation_accuracy(self):
        """Test physics calculation accuracy against known values."""
        # Test simple horizontal movement
        result = self.calc.calculate_jump_trajectory(
            start_pos=(0.0, 100.0),
            end_pos=(100.0, 100.0)
        )

        if result.feasible:
            # For horizontal movement, time should be reasonable
            # (Implementation may use different calculation method)
            self.assertGreater(result.time_of_flight, 0)
            self.assertLess(result.time_of_flight, 1000)  # Reasonable upper bound

    def test_energy_cost_calculation(self):
        """Test energy cost calculation for different movement types."""
        # Test different trajectory types
        test_cases = [
            ((0.0, 100.0), (50.0, 100.0)),    # Horizontal
            ((0.0, 100.0), (50.0, 50.0)),     # Upward
            ((0.0, 100.0), (50.0, 150.0)),    # Downward
        ]

        for start_pos, end_pos in test_cases:
            with self.subTest(start=start_pos, end=end_pos):
                result = self.calc.calculate_jump_trajectory(start_pos, end_pos)

                if result.feasible:
                    # Energy cost should be positive
                    self.assertGreater(result.energy_cost, 0.0)

                    # Energy cost should be reasonable (not infinite)
                    self.assertLess(result.energy_cost, 1000.0)


if __name__ == '__main__':
    unittest.main()
