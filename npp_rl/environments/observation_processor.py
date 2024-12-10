import numpy as np
from collections import deque
import cv2
from typing import Dict, Any
from npp_rl.environments.spatial_memory import SpatialMemoryTracker
from npp_rl.util.util import calculate_velocity
from npp_rl.environments.constants import TIMESTEP


class ObservationProcessor:
    """Processes raw game observations into the format needed by the agent"""

    def __init__(self, frame_stack: int = 4, max_velocity: float = 20000.0):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)  # Recent frames
        self.max_velocity = max_velocity
        self.spatial_memory = SpatialMemoryTracker()

        # Initialize historical frame storage
        self.frame_history = deque(maxlen=1024)  # Store up to 1024 frames
        # Indices for historical frames we want to keep (8, 16, 32, 64, 128, 256, 512, 1024)
        self.historical_indices = [8, 16, 32, 64, 128, 256, 512, 1024]

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw frame to grayscale, resize, and normalize"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply edge detection to highlight level features
        edges = cv2.Canny(frame, 100, 200)
        frame = cv2.addWeighted(frame, 0.7, edges, 0.3, 0)

        # Resize with interpolation
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Enhance contrast
        frame = cv2.equalizeHist(frame)

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    def get_numerical_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process numerical features into grouped format"""
        # Only include time remaining feature
        features = np.array([
            obs['time_remaining'] / 600.0  # Normalize time remaining to [0,1]
        ], dtype=np.float32)

        # Broadcast to spatial dimensions (84x84)
        return np.broadcast_to(
            features.reshape((1, 1, -1)),
            (84, 84, features.shape[0])
        )

    def process_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process full observation including frame stack and numerical features"""
        # Process current frame
        frame = self.preprocess_frame(obs['screen'])

        # Update recent frames
        self.frames.append(frame)

        # Update historical frames
        self.frame_history.append(frame)

        # Fill frame stack if needed
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)

        # Fill historical frames if needed
        while len(self.frame_history) < max(self.historical_indices):
            self.frame_history.append(frame)

        # Stack recent frames
        frames_list = [f[..., np.newaxis] for f in self.frames]

        # Add historical frames
        for idx in self.historical_indices:
            if idx < len(self.frame_history):
                historical_frame = self.frame_history[-idx]
            else:
                historical_frame = frame
            frames_list.append(historical_frame[..., np.newaxis])

        # Concatenate all frames
        stacked_frames = np.concatenate(frames_list, axis=-1)

        # Get numerical features
        features = self.get_numerical_features(obs)

        # Combine everything
        final_observation = np.concatenate([stacked_frames, features], axis=2)
        return np.clip(final_observation, 0, 1).astype(np.float32)

    def reset(self) -> None:
        """Reset processor state"""
        self.frames.clear()
        self.frame_history.clear()
        self.spatial_memory.reset()
