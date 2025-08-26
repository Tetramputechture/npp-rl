"""
Tests for Momentum Tracker.

This module tests the momentum tracking functionality for temporal physics reasoning
in the reinforcement learning agent.
"""

import pytest
import numpy as np
from npp_rl.models.momentum_tracker import MomentumTracker


class TestMomentumTracker:
    """Test cases for MomentumTracker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = MomentumTracker(history_length=5)
    
    def test_initialization(self):
        """Test that the tracker initializes correctly."""
        assert self.tracker.history_length == 5
        assert len(self.tracker.velocity_history) == 0
        assert len(self.tracker.position_history) == 0
        assert self.tracker.frame_count == 0
        assert not self.tracker.is_ready()
    
    def test_update_history(self):
        """Test updating position and velocity history."""
        positions = [(100.0, 200.0), (102.0, 198.0), (104.0, 196.0)]
        velocities = [(2.0, -2.0), (2.0, -2.0), (2.0, -2.0)]
        
        for pos, vel in zip(positions, velocities):
            self.tracker.update(pos, vel)
        
        assert len(self.tracker.position_history) == 3
        assert len(self.tracker.velocity_history) == 3
        assert self.tracker.frame_count == 3
        assert self.tracker.is_ready()  # Has enough data for predictions
    
    def test_history_length_limit(self):
        """Test that history respects maximum length."""
        # Add more entries than history_length
        for i in range(10):
            self.tracker.update((i, i), (1.0, 1.0))
        
        # Should only keep the last 5 entries
        assert len(self.tracker.position_history) == 5
        assert len(self.tracker.velocity_history) == 5
        assert self.tracker.position_history[-1] == (9, 9)  # Last entry
        assert self.tracker.position_history[0] == (5, 5)   # First kept entry
    
    def test_acceleration_calculation(self):
        """Test acceleration calculation from velocity history."""
        # Add constant velocity (zero acceleration)
        velocities = [(2.0, -1.0), (2.0, -1.0), (2.0, -1.0)]
        for i, vel in enumerate(velocities):
            self.tracker.update((i, i), vel)
        
        ax, ay = self.tracker.get_acceleration()
        assert abs(ax) < 1e-6  # Should be zero
        assert abs(ay) < 1e-6  # Should be zero
        
        # Add accelerating velocity
        self.tracker.update((3, 3), (3.0, -2.0))  # Increased velocity
        ax, ay = self.tracker.get_acceleration()
        assert abs(ax - 1.0) < 1e-6  # ax = 3.0 - 2.0 = 1.0
        assert abs(ay - (-1.0)) < 1e-6  # ay = -2.0 - (-1.0) = -1.0
    
    def test_average_acceleration(self):
        """Test average acceleration calculation."""
        # Add varying velocities
        velocities = [(1.0, 0.0), (2.0, -1.0), (4.0, -3.0), (7.0, -6.0)]
        for i, vel in enumerate(velocities):
            self.tracker.update((i, i), vel)
        
        # Calculate average acceleration over last 3 frames
        avg_ax, avg_ay = self.tracker.get_average_acceleration(window_size=3)
        
        # Expected accelerations from last 3 velocities: (2,-2), (3,-3)
        # Average: (2.5, -2.5)
        expected_ax = (2.0 + 3.0) / 2.0
        expected_ay = ((-2.0) + (-3.0)) / 2.0
        
        assert abs(avg_ax - expected_ax) < 1e-6
        assert abs(avg_ay - expected_ay) < 1e-6
    
    def test_position_prediction(self):
        """Test future position prediction."""
        # Set up constant velocity motion
        positions = [(0.0, 0.0), (2.0, -1.0), (4.0, -2.0)]
        velocities = [(2.0, -1.0), (2.0, -1.0), (2.0, -1.0)]
        
        for pos, vel in zip(positions, velocities):
            self.tracker.update(pos, vel)
        
        # Predict 3 frames ahead without gravity
        future_pos = self.tracker.predict_future_position(frames_ahead=3, use_gravity=False)
        
        # Expected: current_pos + velocity * time
        # (4.0, -2.0) + (2.0, -1.0) * 3 = (10.0, -5.0)
        expected_x = 4.0 + 2.0 * 3
        expected_y = -2.0 + (-1.0) * 3
        
        assert abs(future_pos[0] - expected_x) < 1e-6
        assert abs(future_pos[1] - expected_y) < 1e-6
    
    def test_position_prediction_with_gravity(self):
        """Test future position prediction with gravity."""
        # Set up motion with initial upward velocity
        positions = [(0.0, 100.0), (1.0, 98.0), (2.0, 96.0)]
        velocities = [(1.0, -2.0), (1.0, -2.0), (1.0, -2.0)]
        
        for pos, vel in zip(positions, velocities):
            self.tracker.update(pos, vel)
        
        # Predict with gravity
        future_pos = self.tracker.predict_future_position(frames_ahead=2, use_gravity=True)
        
        # Should include gravity effect
        current_pos = positions[-1]
        current_vel = velocities[-1]
        gravity_accel = (0.0, self.tracker.gravity_fall)
        
        t = 2
        expected_x = current_pos[0] + current_vel[0] * t
        expected_y = current_pos[1] + current_vel[1] * t + 0.5 * gravity_accel[1] * t * t
        
        assert abs(future_pos[0] - expected_x) < 1e-6
        # Y should be affected by gravity
        assert future_pos[1] > expected_y - 0.5 * gravity_accel[1] * t * t
    
    def test_velocity_prediction(self):
        """Test future velocity prediction."""
        # Set up accelerating motion
        velocities = [(1.0, 0.0), (2.0, -0.5), (3.0, -1.0)]
        for i, vel in enumerate(velocities):
            self.tracker.update((i, i), vel)
        
        # Predict velocity 2 frames ahead
        future_vel = self.tracker.predict_velocity_at_time(frames_ahead=2, use_gravity=False)
        
        # Current acceleration: (1.0, -0.5)
        # Future velocity: (3.0, -1.0) + (1.0, -0.5) * 2 = (5.0, -2.0)
        expected_vx = 3.0 + 1.0 * 2
        expected_vy = -1.0 + (-0.5) * 2
        
        assert abs(future_vel[0] - expected_vx) < 1e-6
        assert abs(future_vel[1] - expected_vy) < 1e-6
    
    def test_velocity_clamping(self):
        """Test that predicted velocities are clamped to reasonable bounds."""
        # Set up very high acceleration
        velocities = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        for i, vel in enumerate(velocities):
            self.tracker.update((i, i), vel)
        
        # Predict far into future
        future_vel = self.tracker.predict_velocity_at_time(frames_ahead=10)
        
        # Horizontal velocity should be clamped
        max_allowed = self.tracker.max_hor_speed * 1.5
        assert abs(future_vel[0]) <= max_allowed
    
    def test_movement_stability(self):
        """Test movement stability calculation."""
        # Stable motion (constant velocity)
        stable_velocities = [(2.0, -1.0)] * 5
        for i, vel in enumerate(stable_velocities):
            self.tracker.update((i, i), vel)
        
        stability = self.tracker.get_movement_stability()
        assert stability > 0.8  # Should be high stability
        
        # Reset and test chaotic motion
        self.tracker.reset()
        chaotic_velocities = [(1.0, 0.0), (-2.0, 3.0), (0.5, -1.5), (3.0, 2.0), (-1.0, -2.0)]
        for i, vel in enumerate(chaotic_velocities):
            self.tracker.update((i, i), vel)
        
        stability = self.tracker.get_movement_stability()
        assert stability < 0.5  # Should be low stability
    
    def test_momentum_features(self):
        """Test momentum feature extraction."""
        # Set up some motion history
        positions = [(0.0, 0.0), (1.0, -1.0), (3.0, -3.0), (6.0, -6.0)]
        velocities = [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)]
        
        for pos, vel in zip(positions, velocities):
            self.tracker.update(pos, vel)
        
        features = self.tracker.get_momentum_features()
        
        # Should return 10 features
        assert len(features) == 10
        
        # All features should be finite
        assert np.all(np.isfinite(features))
        
        # Features should be normalized
        assert np.all(np.abs(features) <= 2.1)  # Allow some reasonable bounds
        
        # Speed trend should be positive (accelerating)
        assert features[9] > 0  # speed_trend
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # No data
        assert self.tracker.get_acceleration() == (0.0, 0.0)
        assert self.tracker.get_movement_stability() == 0.5  # Neutral
        
        # Only one data point
        self.tracker.update((0.0, 0.0), (1.0, 1.0))
        assert self.tracker.get_acceleration() == (0.0, 0.0)
        
        # Prediction should return current position
        pred_pos = self.tracker.predict_future_position()
        assert pred_pos == (0.0, 0.0)
    
    def test_reset(self):
        """Test tracker reset functionality."""
        # Add some data
        for i in range(3):
            self.tracker.update((i, i), (1.0, 1.0))
        
        assert len(self.tracker.position_history) == 3
        assert self.tracker.frame_count == 3
        
        # Reset
        self.tracker.reset()
        
        assert len(self.tracker.position_history) == 0
        assert len(self.tracker.velocity_history) == 0
        assert self.tracker.frame_count == 0
        assert not self.tracker.is_ready()
    
    def test_history_length_getter(self):
        """Test getting current history length."""
        assert self.tracker.get_history_length() == 0
        
        self.tracker.update((0, 0), (1, 1))
        assert self.tracker.get_history_length() == 1
        
        for i in range(10):
            self.tracker.update((i, i), (1, 1))
        
        # Should be capped at history_length
        assert self.tracker.get_history_length() == 5
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zero velocities
        for i in range(3):
            self.tracker.update((i, i), (0.0, 0.0))
        
        # Should handle gracefully
        features = self.tracker.get_momentum_features()
        assert len(features) == 10
        assert np.all(np.isfinite(features))
        
        # Future velocity should be close to zero (with some gravity effect)
        assert abs(features[6]) < 0.2  # future_vx should be small
        assert abs(features[7]) < 0.5  # future_vy might have gravity effect
        
        # Test prediction with zero velocity
        future_pos = self.tracker.predict_future_position()
        current_pos = self.tracker.position_history[-1]
        # Should be close to current position (only gravity effect)
        assert abs(future_pos[0] - current_pos[0]) < 1e-6