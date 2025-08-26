"""
Momentum Tracker for temporal physics reasoning.

This module tracks ninja momentum history over time to enable prediction of future
positions and acceleration patterns. It provides temporal context for physics-aware
decision making in the reinforcement learning agent.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, List

# Import physics constants
import sys
import os
nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nclone')
if os.path.exists(nclone_path) and nclone_path not in sys.path:
    sys.path.insert(0, nclone_path)

try:
    from nclone.constants import GRAVITY_FALL, GRAVITY_JUMP, MAX_HOR_SPEED
except ImportError:
    # Fallback constants
    GRAVITY_FALL = 0.06666666666666665
    GRAVITY_JUMP = 0.01111111111111111
    MAX_HOR_SPEED = 3.333333333333333


class MomentumTracker:
    """
    Track ninja momentum history for temporal physics reasoning.
    
    This class maintains a rolling history of ninja position and velocity data
    to enable prediction of future movement patterns and acceleration analysis.
    It supports kinematic predictions based on recent movement history.
    """
    
    def __init__(self, history_length: int = 10):
        """
        Initialize momentum tracker.
        
        Args:
            history_length: Number of frames to keep in history
        """
        self.history_length = history_length
        self.velocity_history = deque(maxlen=history_length)
        self.position_history = deque(maxlen=history_length)
        self.timestamp_history = deque(maxlen=history_length)
        
        # Physics constants
        self.gravity_fall = GRAVITY_FALL
        self.gravity_jump = GRAVITY_JUMP
        self.max_hor_speed = MAX_HOR_SPEED
        
        # Current frame counter
        self.frame_count = 0
    
    def update(self, position: Tuple[float, float], velocity: Tuple[float, float]):
        """
        Update momentum history with new position and velocity data.
        
        Args:
            position: Current ninja position (x, y) in pixels
            velocity: Current ninja velocity (vx, vy) in pixels/frame
        """
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        self.timestamp_history.append(self.frame_count)
        self.frame_count += 1
    
    def get_acceleration(self) -> Tuple[float, float]:
        """
        Calculate current acceleration from velocity history.
        
        Returns:
            Acceleration (ax, ay) in pixels/frame²
        """
        if len(self.velocity_history) < 2:
            return (0.0, 0.0)
        
        recent_vel = self.velocity_history[-1]
        prev_vel = self.velocity_history[-2]
        
        ax = recent_vel[0] - prev_vel[0]
        ay = recent_vel[1] - prev_vel[1]
        
        return (ax, ay)
    
    def get_average_acceleration(self, window_size: int = 3) -> Tuple[float, float]:
        """
        Calculate average acceleration over a window of recent frames.
        
        Args:
            window_size: Number of recent frames to average over
            
        Returns:
            Average acceleration (ax, ay) in pixels/frame²
        """
        if len(self.velocity_history) < 2:
            return (0.0, 0.0)
        
        if len(self.velocity_history) < window_size:
            window_size = len(self.velocity_history)
        
        accelerations = []
        # Calculate accelerations for the last window_size-1 transitions
        start_idx = len(self.velocity_history) - window_size
        for i in range(start_idx + 1, len(self.velocity_history)):
            curr_vel = self.velocity_history[i]
            prev_vel = self.velocity_history[i-1]
            ax = curr_vel[0] - prev_vel[0]
            ay = curr_vel[1] - prev_vel[1]
            accelerations.append((ax, ay))
        
        if not accelerations:
            return (0.0, 0.0)
        
        avg_ax = sum(acc[0] for acc in accelerations) / len(accelerations)
        avg_ay = sum(acc[1] for acc in accelerations) / len(accelerations)
        
        return (avg_ax, avg_ay)
    
    def predict_future_position(self, frames_ahead: int = 5, use_gravity: bool = True) -> Tuple[float, float]:
        """
        Predict ninja position N frames in the future using kinematic equations.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            use_gravity: Whether to include gravity in prediction
            
        Returns:
            Predicted position (x, y) in pixels
        """
        if len(self.position_history) == 0:
            return (0.0, 0.0)
        
        if len(self.velocity_history) < 2:
            # No velocity history, assume stationary
            return self.position_history[-1]
        
        current_pos = self.position_history[-1]
        current_vel = self.velocity_history[-1]
        acceleration = self.get_average_acceleration()
        
        # Apply gravity if requested
        if use_gravity:
            # Use fall gravity as default (more conservative)
            gravity_accel = (0.0, self.gravity_fall)
            acceleration = (acceleration[0] + gravity_accel[0], 
                          acceleration[1] + gravity_accel[1])
        
        # Kinematic prediction: pos = pos0 + vel*t + 0.5*acc*t²
        t = frames_ahead
        future_x = current_pos[0] + current_vel[0] * t + 0.5 * acceleration[0] * t * t
        future_y = current_pos[1] + current_vel[1] * t + 0.5 * acceleration[1] * t * t
        
        return (future_x, future_y)
    
    def predict_velocity_at_time(self, frames_ahead: int = 5, use_gravity: bool = True) -> Tuple[float, float]:
        """
        Predict ninja velocity N frames in the future.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            use_gravity: Whether to include gravity in prediction
            
        Returns:
            Predicted velocity (vx, vy) in pixels/frame
        """
        if len(self.velocity_history) == 0:
            return (0.0, 0.0)
        
        current_vel = self.velocity_history[-1]
        acceleration = self.get_average_acceleration()
        
        # Apply gravity if requested
        if use_gravity:
            gravity_accel = (0.0, self.gravity_fall)
            acceleration = (acceleration[0] + gravity_accel[0], 
                          acceleration[1] + gravity_accel[1])
        
        # Velocity prediction: vel = vel0 + acc*t
        t = frames_ahead
        future_vx = current_vel[0] + acceleration[0] * t
        future_vy = current_vel[1] + acceleration[1] * t
        
        # Clamp horizontal velocity to reasonable bounds
        future_vx = np.clip(future_vx, -self.max_hor_speed * 1.5, self.max_hor_speed * 1.5)
        
        return (future_vx, future_vy)
    
    def get_movement_stability(self) -> float:
        """
        Calculate movement stability based on velocity variance.
        
        Returns:
            Stability score from 0.0 (chaotic) to 1.0 (stable)
        """
        if len(self.velocity_history) < 3:
            return 0.5  # Neutral stability with insufficient data
        
        # Calculate velocity variance
        velocities = np.array(list(self.velocity_history))
        vel_variance = np.var(velocities, axis=0)
        
        # Combine x and y variance
        total_variance = np.sum(vel_variance)
        
        # Convert to stability score (lower variance = higher stability)
        # Use exponential decay to map variance to [0,1], with more sensitivity
        stability = np.exp(-total_variance / (self.max_hor_speed ** 2 * 0.5))
        
        return float(np.clip(stability, 0.0, 1.0))
    
    def get_momentum_features(self) -> np.ndarray:
        """
        Extract momentum-based features for use in neural networks.
        
        Returns:
            Array of momentum features:
            [0-1]: Current acceleration (ax, ay)
            [2-3]: Average acceleration (ax, ay)
            [4-5]: Predicted position change (dx, dy) in 5 frames
            [6-7]: Predicted velocity (vx, vy) in 5 frames
            [8]: Movement stability score
            [9]: Speed trend (increasing/decreasing)
        """
        if len(self.velocity_history) < 2:
            return np.zeros(10, dtype=np.float32)
        
        # Current acceleration
        curr_accel = self.get_acceleration()
        
        # Average acceleration
        avg_accel = self.get_average_acceleration()
        
        # Predicted position change
        current_pos = self.position_history[-1] if self.position_history else (0, 0)
        future_pos = self.predict_future_position(frames_ahead=5)
        pos_change = (future_pos[0] - current_pos[0], future_pos[1] - current_pos[1])
        
        # Predicted velocity
        future_vel = self.predict_velocity_at_time(frames_ahead=5)
        
        # Movement stability
        stability = self.get_movement_stability()
        
        # Speed trend (positive = accelerating, negative = decelerating)
        if len(self.velocity_history) >= 3:
            recent_speeds = [np.sqrt(vx*vx + vy*vy) for vx, vy in list(self.velocity_history)[-3:]]
            speed_trend = (recent_speeds[-1] - recent_speeds[0]) / 2.0
            speed_trend = np.clip(speed_trend / self.max_hor_speed, -1.0, 1.0)
        else:
            speed_trend = 0.0
        
        # Normalize features with proper clamping
        features = np.array([
            np.clip(curr_accel[0] / self.max_hor_speed, -2.0, 2.0),  # Normalized acceleration x
            np.clip(curr_accel[1] / self.max_hor_speed, -2.0, 2.0),  # Normalized acceleration y
            np.clip(avg_accel[0] / self.max_hor_speed, -2.0, 2.0),   # Normalized avg acceleration x
            np.clip(avg_accel[1] / self.max_hor_speed, -2.0, 2.0),   # Normalized avg acceleration y
            np.clip(pos_change[0] / (5 * self.max_hor_speed), -2.0, 2.0),  # Normalized position change x
            np.clip(pos_change[1] / (5 * self.max_hor_speed), -2.0, 2.0),  # Normalized position change y
            np.clip(future_vel[0] / self.max_hor_speed, -2.0, 2.0),  # Normalized future velocity x
            np.clip(future_vel[1] / self.max_hor_speed, -2.0, 2.0),  # Normalized future velocity y
            stability,                           # Movement stability [0,1]
            speed_trend                          # Speed trend [-1,1]
        ], dtype=np.float32)
        
        return features
    
    def reset(self):
        """Reset the momentum tracker, clearing all history."""
        self.velocity_history.clear()
        self.position_history.clear()
        self.timestamp_history.clear()
        self.frame_count = 0
    
    def get_history_length(self) -> int:
        """Get the current length of stored history."""
        return len(self.position_history)
    
    def is_ready(self) -> bool:
        """Check if tracker has enough data for meaningful predictions."""
        return len(self.velocity_history) >= 2