"""
Tests for TrajectoryCalculator physics validation.

This module tests the physics-based trajectory calculation and validation
functionality for N++ RL graph edge features.
"""

import unittest
import numpy as np
import math
from unittest.mock import Mock, patch

from npp_rl.models.trajectory_calculator import TrajectoryCalculator


class TestTrajectoryCalculator(unittest.TestCase):
    """Test cases for TrajectoryCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()
    
    def test_calculate_jump_trajectory_basic(self):
        """Test basic jump trajectory calculation."""
        # Test upward jump
        result = self.calc.calculate_jump_trajectory(
            start_pos=(100.0, 100.0),
            end_pos=(150.0, 50.0)
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.feasible, bool)
        self.assertIsInstance(result.time_of_flight, (int, float))
        self.assertIsInstance(result.energy_cost, (int, float))
        
        # Should be feasible for reasonable jump
        if result.feasible:
            self.assertGreater(result.time_of_flight, 0)
            self.assertGreater(result.energy_cost, 0)
    
    def test_calculate_jump_trajectory_horizontal(self):
        """Test horizontal movement trajectory."""
        trajectory = self.calc.calculate_jump_trajectory(
            x0=0.0, y0=100.0, vx0=100.0, vy0=0.0
        )
        
        self.assertIsNotNone(trajectory)
        
        # Should move horizontally and fall due to gravity
        x_values = [point['x'] for point in trajectory]
        y_values = [point['y'] for point in trajectory]
        
        self.assertGreater(max(x_values), 0.0)  # Should move right
        self.assertGreater(max(y_values), 100.0)  # Should fall down
    
    def test_calculate_jump_trajectory_downward(self):
        """Test downward movement trajectory."""
        trajectory = self.calc.calculate_jump_trajectory(
            x0=100.0, y0=100.0, vx0=0.0, vy0=50.0
        )
        
        self.assertIsNotNone(trajectory)
        
        # Should fall straight down
        y_values = [point['y'] for point in trajectory]
        self.assertGreater(max(y_values), 100.0)  # Should move down
    
    def test_validate_trajectory_clearance_clear_path(self):
        """Test trajectory validation with clear path."""
        # Create simple trajectory
        trajectory = [
            {'x': 0.0, 'y': 100.0, 'vx': 50.0, 'vy': 0.0, 't': 0.0},
            {'x': 50.0, 'y': 100.0, 'vx': 50.0, 'vy': 10.0, 't': 1.0},
            {'x': 100.0, 'y': 110.0, 'vx': 50.0, 'vy': 20.0, 't': 2.0}
        ]
        
        # Mock level data with no obstacles
        level_data = {
            'tiles': [[0 for _ in range(50)] for _ in range(30)]  # All empty tiles
        }
        
        success_prob = self.calc.validate_trajectory_clearance(
            trajectory, level_data, 100.0, 110.0
        )
        
        self.assertGreater(success_prob, 0.8)  # Should have high success probability
    
    def test_validate_trajectory_clearance_with_obstacles(self):
        """Test trajectory validation with obstacles."""
        # Create trajectory that goes through obstacles
        trajectory = [
            {'x': 0.0, 'y': 100.0, 'vx': 50.0, 'vy': 0.0, 't': 0.0},
            {'x': 50.0, 'y': 100.0, 'vx': 50.0, 'vy': 0.0, 't': 1.0},
            {'x': 100.0, 'y': 100.0, 'vx': 50.0, 'vy': 0.0, 't': 2.0}
        ]
        
        # Mock level data with obstacles in the path
        level_data = {
            'tiles': [[1 if 2 <= i <= 4 and 4 <= j <= 6 else 0 
                      for j in range(50)] for i in range(30)]
        }
        
        success_prob = self.calc.validate_trajectory_clearance(
            trajectory, level_data, 100.0, 100.0
        )
        
        self.assertLess(success_prob, 0.5)  # Should have low success probability
    
    def test_validate_trajectory_clearance_empty_trajectory(self):
        """Test trajectory validation with empty trajectory."""
        success_prob = self.calc.validate_trajectory_clearance(
            [], {}, 100.0, 100.0
        )
        
        self.assertEqual(success_prob, 0.0)  # Should fail for empty trajectory
    
    def test_physics_constants_integration(self):
        """Test that physics constants are properly integrated."""
        # Test that gravity affects trajectory
        trajectory1 = self.calc.calculate_jump_trajectory(
            x0=0.0, y0=100.0, vx0=100.0, vy0=0.0
        )
        
        # Should have multiple points showing gravity effect
        self.assertGreater(len(trajectory1), 5)
        
        # Y velocity should increase over time due to gravity
        vy_values = [point['vy'] for point in trajectory1]
        self.assertGreater(vy_values[-1], vy_values[0])  # Should accelerate downward
    
    def test_trajectory_time_step_consistency(self):
        """Test that trajectory time steps are consistent."""
        trajectory = self.calc.calculate_jump_trajectory(
            x0=0.0, y0=100.0, vx0=100.0, vy0=-50.0
        )
        
        if len(trajectory) > 1:
            # Check time step consistency
            time_steps = [trajectory[i+1]['t'] - trajectory[i]['t'] 
                         for i in range(len(trajectory)-1)]
            
            # All time steps should be approximately equal
            avg_dt = sum(time_steps) / len(time_steps)
            for dt in time_steps:
                self.assertAlmostEqual(dt, avg_dt, places=3)
    
    def test_trajectory_physics_accuracy(self):
        """Test trajectory physics accuracy against analytical solution."""
        # Test simple projectile motion
        x0, y0 = 0.0, 100.0
        vx0, vy0 = 100.0, -100.0
        
        trajectory = self.calc.calculate_jump_trajectory(x0, y0, vx0, vy0)
        
        if len(trajectory) > 1:
            # Check a point in the middle of trajectory
            mid_point = trajectory[len(trajectory)//2]
            t = mid_point['t']
            
            # Analytical solution (using N++ physics constants)
            from nclone.constants import GRAVITY_FALL
            expected_x = x0 + vx0 * t
            expected_y = y0 + vy0 * t + 0.5 * GRAVITY_FALL * t * t
            
            # Allow some tolerance for numerical integration
            self.assertAlmostEqual(mid_point['x'], expected_x, delta=5.0)
            self.assertAlmostEqual(mid_point['y'], expected_y, delta=5.0)


class TestTrajectoryCalculatorEdgeCases(unittest.TestCase):
    """Test edge cases for TrajectoryCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = TrajectoryCalculator()
    
    def test_zero_velocity(self):
        """Test trajectory calculation with zero velocity."""
        trajectory = self.calc.calculate_jump_trajectory(
            x0=100.0, y0=100.0, vx0=0.0, vy0=0.0
        )
        
        # Should still generate trajectory (falling due to gravity)
        self.assertIsNotNone(trajectory)
        self.assertGreater(len(trajectory), 0)
    
    def test_very_high_velocity(self):
        """Test trajectory calculation with very high velocity."""
        trajectory = self.calc.calculate_jump_trajectory(
            x0=0.0, y0=100.0, vx0=1000.0, vy0=-1000.0
        )
        
        # Should handle high velocities gracefully
        self.assertIsNotNone(trajectory)
        self.assertGreater(len(trajectory), 0)
    
    def test_negative_coordinates(self):
        """Test trajectory calculation with negative starting coordinates."""
        trajectory = self.calc.calculate_jump_trajectory(
            x0=-100.0, y0=-100.0, vx0=50.0, vy0=-50.0
        )
        
        # Should handle negative coordinates
        self.assertIsNotNone(trajectory)
        self.assertGreater(len(trajectory), 0)
    
    def test_validate_clearance_invalid_level_data(self):
        """Test trajectory validation with invalid level data."""
        trajectory = [
            {'x': 0.0, 'y': 100.0, 'vx': 50.0, 'vy': 0.0, 't': 0.0}
        ]
        
        # Test with None level data
        success_prob = self.calc.validate_trajectory_clearance(
            trajectory, None, 100.0, 100.0
        )
        self.assertEqual(success_prob, 0.0)
        
        # Test with empty level data
        success_prob = self.calc.validate_trajectory_clearance(
            trajectory, {}, 100.0, 100.0
        )
        self.assertEqual(success_prob, 0.0)


if __name__ == '__main__':
    unittest.main()