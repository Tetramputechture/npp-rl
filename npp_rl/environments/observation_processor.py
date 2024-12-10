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
        self.frames = deque(maxlen=frame_stack)
        self.max_velocity = max_velocity
        self.spatial_memory = SpatialMemoryTracker()

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

    def get_numerical_features(self, obs: Dict[str, Any],
                               prev_obs: Dict[str, Any]) -> np.ndarray:
        """Process numerical features into grouped format"""
        # Update spatial memory
        self.spatial_memory.update(obs['player_x'], obs['player_y'])

        # Calculate velocities
        vx, vy = calculate_velocity(
            obs['player_x'], obs['player_y'],
            prev_obs['player_x'], prev_obs['player_y'],
            TIMESTEP
        )

        # Normalize velocities
        normalized_vx = (vx + self.max_velocity) / (2 * self.max_velocity)
        normalized_vy = (vy + self.max_velocity) / (2 * self.max_velocity)

        # Get exploration maps
        exploration_maps = self.spatial_memory.get_exploration_maps(
            obs['player_x'], obs['player_y'])

        # Calculate features
        position_features = np.array([
            (obs['player_x'] - 63) / (1217 - 63),
            (obs['player_y'] - 171) / (791 - 171),
            normalized_vx,
            normalized_vy
        ], dtype=np.float32)

        objective_features = np.array([
            (obs['exit_door_x'] - obs['player_x'] + 1258) / 2516,
            (obs['exit_door_y'] - obs['player_y'] + 802) / 1604,
            (obs['switch_x'] - obs['player_x'] + 1258) / 2516,
            (obs['switch_y'] - obs['player_y'] + 802) / 1604
        ], dtype=np.float32)

        state_features = np.array([
            obs['time_remaining'] / 600.0,
            float(obs['switch_activated']),
            float(obs['in_air'])
        ], dtype=np.float32)

        exploration_features = np.array([
            1.0 - exploration_maps['recent_visits'][42, 42],
            np.mean(exploration_maps['visit_frequency'][40:44, 40:44]),
            exploration_maps['area_exploration'][42, 42],
            np.mean(exploration_maps['transitions'][40:44, 40:44])
        ], dtype=np.float32)

        # Process mine features
        mine_features = np.array([
            obs['closest_mine_distance'] / 1000.0,  # Normalize distance
            (np.arctan2(obs['closest_mine_vector'][1], obs['closest_mine_vector']
             [0]) + np.pi) / (2 * np.pi),  # Normalize angle to [0,1]
            # Relative velocity towards mine
            np.dot(np.array([normalized_vx, normalized_vy]),
                   obs['closest_mine_vector'])
        ], dtype=np.float32)

        # Combine all features
        features = np.concatenate([
            position_features,
            objective_features,
            state_features,
            exploration_features,
            mine_features
        ])

        return np.broadcast_to(
            features.reshape((1, 1, -1)),
            (84, 84, features.shape[0])
        )

    def process_observation(self, obs: Dict[str, Any],
                            prev_obs: Dict[str, Any],
                            action: int = None) -> np.ndarray:
        """Process full observation including frame stack and numerical features"""
        # Process current frame
        frame = self.preprocess_frame(obs['screen'])
        self.frames.append(frame)

        # Fill frame stack if needed
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)

        # Stack frames along new axis first
        frames_list = [f[..., np.newaxis] for f in self.frames]
        stacked_frames = np.concatenate(frames_list, axis=-1)

        # Get numerical features
        features = self.get_numerical_features(obs, prev_obs)

        # Combine everything
        final_observation = np.concatenate([stacked_frames, features], axis=2)
        return np.clip(final_observation, 0, 1).astype(np.float32)

    def reset(self) -> None:
        """Reset processor state"""
        self.frames.clear()
        self.spatial_memory.reset()
