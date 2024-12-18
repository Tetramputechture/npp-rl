import numpy as np
from collections import deque
import cv2
from typing import Dict, Any
from npp_rl.environments.constants import OBSERVATION_IMAGE_SIZE, LEVEL_WIDTH, LEVEL_HEIGHT, MAX_VELOCITY


class ObservationProcessor:
    """Processes raw game observations into frame stacks and normalized feature vectors."""

    def __init__(self):
        self.image_size = OBSERVATION_IMAGE_SIZE

        # Store frames with fixed intervals
        self.frame_intervals = [0, 4, 8, 12]
        self.frame_history = deque(maxlen=max(self.frame_intervals) + 1)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw frame to grayscale and resize with improved accuracy"""
        # Convert to grayscale using more accurate weights if needed
        if len(frame.shape) == 3:
            # Using BT.601 standard for RGB to grayscale conversion
            frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

        # Resize with better quality interpolation
        frame = cv2.resize(
            frame, (self.image_size, self.image_size),
            interpolation=cv2.INTER_LANCZOS4)

        # Normalize to [0, 255] range and convert to uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def process_player_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process and normalize player state information."""
        # Normalize positions between 0 and 1
        pos_x = obs['player_x'] / LEVEL_WIDTH
        pos_y = obs['player_y'] / LEVEL_HEIGHT

        # Normalize velocities to [-1, 1] then to [0, 1]
        vel_x = (np.clip(obs['player_vx'], -MAX_VELOCITY,
                 MAX_VELOCITY) / MAX_VELOCITY + 1) / 2
        vel_y = (np.clip(obs['player_vy'], -MAX_VELOCITY,
                 MAX_VELOCITY) / MAX_VELOCITY + 1) / 2

        # Boolean states are already 0 or 1
        in_air = float(obs['in_air'])
        walled = float(obs['walled'])

        return np.array([pos_x, pos_y, vel_x, vel_y, in_air, walled], dtype=np.float32)

    def process_goal_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process and normalize goal-related features."""
        # Calculate distances to switch and exit
        switch_pos = np.array([obs['switch_x'], obs['switch_y']])
        exit_pos = np.array([obs['exit_door_x'], obs['exit_door_y']])
        player_pos = np.array([obs['player_x'], obs['player_y']])

        # Calculate Euclidean distances and normalize by diagonal of level
        level_diagonal = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
        switch_dist = np.linalg.norm(player_pos - switch_pos) / level_diagonal
        exit_dist = np.linalg.norm(player_pos - exit_pos) / level_diagonal

        # Switch activation is already boolean (0 or 1)
        switch_activated = float(obs['switch_activated'])

        return np.array([switch_dist, exit_dist, switch_activated], dtype=np.float32)

    def process_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process observation into frame stack and feature vectors."""
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

        # Process player state and goal features
        player_state = self.process_player_state(obs)
        goal_features = self.process_goal_features(obs)

        return {
            'visual': np.concatenate(stacked_frames, axis=-1),
            'player_state': player_state,
            'goal_features': goal_features
        }

    def reset(self) -> None:
        """Reset processor state."""
        self.frame_history.clear()
