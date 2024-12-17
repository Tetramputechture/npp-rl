import numpy as np
from collections import deque
import cv2
from typing import Dict, Any
from npp_rl.environments.spatial_memory import SpatialMemoryTracker


class ObservationProcessor:
    """Processes raw game observations into the format needed by the agent"""

    def __init__(self, frame_stack: int = 4):
        self.frame_stack = frame_stack
        self.image_size = 84

        # Store frames with fixed intervals (current frame, 5th frame back, 9th frame back, 13th frame back)
        self.frame_intervals = [0, 4, 8, 12]  # Intervals for 4 frames
        # Store enough frames for our intervals
        self.frame_history = deque(maxlen=max(self.frame_intervals) + 1)

        # Velocity normalization constant
        self.max_velocity = 100.0  # Maximum velocity for normalization
        self.spatial_memory = SpatialMemoryTracker()

        # Store movement vectors for velocity calculation
        self.movement_history = deque(maxlen=16)  # Store recent movements
        self.prev_frame = None

        # Initialize goal maps
        self.switch_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.exit_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)

        # Initialize player state channels
        self.player_state = np.zeros(
            (self.image_size, self.image_size, 6), dtype=np.float32)

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

        # Normalize frame
        frame = frame.astype(np.float32) / 255.0

        return frame

    def get_distance_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Calculate normalized distances to objectives"""
        player_x = obs.get('player_x', 0)
        player_y = obs.get('player_y', 0)

        # Calculate Euclidean distances
        switch_dist = np.sqrt(
            (player_x - obs['switch_x'])**2 + (player_y - obs['switch_y'])**2)
        exit_dist = np.sqrt(
            (player_x - obs['exit_door_x'])**2 + (player_y - obs['exit_door_y'])**2)

        # Normalize by diagonal of level
        level_diagonal = np.sqrt(
            obs['level_width']**2 + obs['level_height']**2)
        return np.array([switch_dist/level_diagonal, exit_dist/level_diagonal], dtype=np.float32)

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

        # Stack memory maps
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

        # Scale coordinates to image size
        x_scaled = x
        y_scaled = y

        # Create Gaussian heatmap with tighter sigma since we're using actual coordinates
        heatmap = np.exp(-((xx - x_scaled) ** 2 +
                         (yy - y_scaled) ** 2) / (2 * sigma ** 2))
        return heatmap.astype(np.float32)

    def get_player_state_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize player position and velocity features"""
        # Get player position and normalize to [0,1]
        player_x = obs.get('player_x', 0) / obs.get('level_width', 1)
        player_y = obs.get('player_y', 0) / obs.get('level_height', 1)

        # Get player velocity and normalize to [0, 1]
        player_vx = np.clip(obs.get('player_vx', 0) / self.max_velocity, -1, 1)
        player_vy = np.clip(obs.get('player_vy', 0) / self.max_velocity, -1, 1)
        player_vx = (player_vx + 1) / 2
        player_vy = (player_vy + 1) / 2

        # Get in_air status (0 or 1)
        in_air = float(obs.get('in_air', False))

        # Get walled status (0 or 1)
        walled = float(obs.get('walled', False))

        # Create feature maps for each value
        player_state = np.zeros(
            (self.image_size, self.image_size, 6), dtype=np.float32)
        player_state[..., 0] = player_x  # Normalized X position
        player_state[..., 1] = player_y  # Normalized Y position
        player_state[..., 2] = player_vx  # Normalized X velocity
        player_state[..., 3] = player_vy  # Normalized Y velocity
        player_state[..., 4] = in_air     # In air status
        player_state[..., 5] = walled     # Walled status

        return player_state

    def process_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process full observation with temporal features"""
        # Process current frame
        frame = self.preprocess_frame(obs['screen'])

        # Update frame history
        self.frame_history.append(frame)

        # Fill frame history if needed
        while len(self.frame_history) < max(self.frame_intervals) + 1:
            self.frame_history.append(frame)

        # Stack frames at specified intervals
        stacked_frames = []
        for interval in self.frame_intervals:
            if len(self.frame_history) > interval:
                historical_frame = self.frame_history[-interval-1]
            else:
                historical_frame = frame
            stacked_frames.append(historical_frame[..., np.newaxis])

        # Get numerical features (spatial memory)
        features = self.get_numerical_features(obs)

        # Get player state features
        player_state = self.get_player_state_features(obs)

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
        switch_channel = self.switch_map[..., np.newaxis]
        exit_channel = self.exit_map[..., np.newaxis]

        # Combine everything
        final_observation = np.concatenate(
            stacked_frames + [features, player_state,
                              switch_channel, exit_channel],
            axis=-1
        )

        # Assert our values are in the correct range
        assert np.all(final_observation >= 0) and np.all(
            final_observation <= 1), "Observation values are out of range"

        return np.clip(final_observation, 0, 1).astype(np.float32)

    def reset(self) -> None:
        """Reset processor state"""
        self.frame_history.clear()
        self.movement_history.clear()
        self.spatial_memory.reset()
        self.prev_frame = None
        self.switch_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.exit_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
