import numpy as np
from collections import deque
import cv2
from typing import Dict, Any
from npp_rl.environments.spatial_memory import SpatialMemoryTracker


class ObservationProcessor:
    """Processes raw game observations into the format needed by the agent"""

    def __init__(self, frame_stack: int = 4, max_velocity: float = 20000.0):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)  # Recent frames
        self.max_velocity = max_velocity
        self.spatial_memory = SpatialMemoryTracker()
        self.image_size = 84  # Increased from 84x84 to 84x84

        # Initialize frame storage with smaller fixed intervals
        # This helps capture more immediate temporal dependencies
        # Fixed intervals for 3 historical frames
        self.historical_intervals = [2, 4, 8]
        self.frame_history = deque(maxlen=max(self.historical_intervals))

        # Store movement vectors for velocity calculation
        self.movement_history = deque(maxlen=16)  # Store recent movements
        self.prev_frame = None
        self.prev_gray = None  # Store previous grayscale frame for optical flow

        # Initialize empty hazard and goal maps
        self.hazard_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.switch_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.exit_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw frame to grayscale, resize, normalize, and extract motion features"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # type: ignore

        # Apply edge detection to highlight level features
        edges = cv2.Canny(frame, 100, 200)  # type: ignore
        frame = cv2.addWeighted(frame, 0.7, edges, 0.3, 0)  # type: ignore

        # Resize with interpolation
        frame = cv2.resize(
            frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)  # type: ignore

        # Enhance contrast
        frame = cv2.equalizeHist(frame)  # type: ignore

        # Update previous frame
        self.prev_gray = frame.copy()

        # Normalize frame
        frame = frame.astype(np.float32) / 255.0

        return frame

    def get_numerical_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process numerical features including spatial memory metrics"""
        # Get spatial memory metrics
        if 'player_x' in obs and 'player_y' in obs:
            memory_maps = self.spatial_memory.get_exploration_maps(
                obs['player_x'], obs['player_y'])
        else:
            memory_maps = {
                'recent_visits': np.zeros((88, 88)),
                'visit_frequency': np.zeros((88, 88)),
                'area_exploration': np.zeros((88, 88)),
                'transitions': np.zeros((88, 88))
            }

        # Resize memory maps from 88x88 to 84x84 while preserving information
        resized_maps = {
            key: cv2.resize(
                value, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            for key, value in memory_maps.items()
        }

        # Stack memory maps without time remaining
        features = np.stack([
            resized_maps['recent_visits'],
            resized_maps['visit_frequency'],
            resized_maps['area_exploration'],
            resized_maps['transitions']
        ], axis=-1)

        return features.astype(np.float32)

    def create_goal_heatmap(self, x: float, y: float, sigma: float = 5.0) -> np.ndarray:
        """Create a Gaussian heatmap centered on a goal location."""
        x_grid = np.linspace(0, self.image_size-1, self.image_size)
        y_grid = np.linspace(0, self.image_size-1, self.image_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Scale coordinates to image size (now 1:1 since both are 84x84)
        x_scaled = x
        y_scaled = y

        # Create Gaussian heatmap with tighter sigma since we're using actual coordinates
        heatmap = np.exp(-((xx - x_scaled) ** 2 +
                         (yy - y_scaled) ** 2) / (2 * sigma ** 2))
        return heatmap.astype(np.float32)

    def process_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process full observation with temporal features"""
        # Process current frame
        frame = self.preprocess_frame(obs['screen'])

        # Update frame histories
        self.frames.append(frame)
        self.frame_history.append(frame)

        # Fill frame stack if needed
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)

        # Stack recent frames (4 frames)
        recent_frames = [f[..., np.newaxis] for f in self.frames]

        # Add historical frames at fixed intervals (3 frames)
        historical_frames = []
        for interval in self.historical_intervals:
            if len(self.frame_history) >= interval:
                historical_frame = self.frame_history[-interval]
            else:
                historical_frame = frame
            historical_frames.append(historical_frame[..., np.newaxis])

        # Get numerical features (spatial memory + time)
        features = self.get_numerical_features(obs)

        # Update hazard map from level data if available
        if 'level_data' in obs and obs['level_data'] is not None:
            self.hazard_map = obs['level_data'].hazard_map

        # Create goal heatmaps - coordinates are already in 84x84 space
        if not obs['switch_activated']:
            # When switch is not activated, make switch location the goal
            self.switch_map = self.create_goal_heatmap(
                obs['switch_x'], obs['switch_y'])
            self.exit_map = np.zeros_like(self.switch_map)  # Clear exit map
        else:
            # When switch is activated, make exit door the goal
            self.switch_map = np.zeros_like(self.exit_map)  # Clear switch map
            self.exit_map = self.create_goal_heatmap(
                obs['exit_door_x'], obs['exit_door_y'])

        # Add hazard and goal channels
        hazard_channel = self.hazard_map[..., np.newaxis]
        switch_channel = self.switch_map[..., np.newaxis]
        exit_channel = self.exit_map[..., np.newaxis]

        # Combine everything
        final_observation = np.concatenate(
            recent_frames + historical_frames +
            [features, hazard_channel, switch_channel, exit_channel],
            axis=-1
        )

        # Assert our values are in the correct range
        assert np.all(final_observation >= 0) and np.all(
            final_observation <= 1), "Observation values are out of range"

        return np.clip(final_observation, 0, 1).astype(np.float32)

    def reset(self) -> None:
        """Reset processor state"""
        self.frames.clear()
        self.frame_history.clear()
        self.movement_history.clear()
        self.spatial_memory.reset()
        self.prev_frame = None
        self.prev_gray = None
        self.hazard_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.switch_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.exit_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
