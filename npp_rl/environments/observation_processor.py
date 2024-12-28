"""This class handles the processing of the game state into a format that can be used for training.

We get our formatted game observation from the NPlusPlus get_observation() method, which has these keys:

    'screen' -- the current frame of the game
    'game_state' -- the current game state
        - Concatenated array of ninja state and entity states (only exit and switch)
    'player_dead' -- whether the player has died
    'player_won' -- whether the player has won
    'player_x' -- the x position of the player
    'player_y' -- the y position of the player
    'switch_activated' -- whether the switch has been activated
    'switch_x' -- the x position of the switch
    'switch_y' -- the y position of the switch
    'exit_door_x' -- the x position of the exit door
    'exit_door_y' -- the y position of the exit door
    'time_remaining' -- the time remaining in the simulation before truncation

This class should handle returning:

- The base map (single frame OBSERVATION_IMAGE_WIDTH x OBSERVATION_IMAGE_HEIGHT)
  - This is the current screen frame, processed to grayscale
- The frame centered on the player (PLAYER_FRAME_WIDTH x PLAYER_FRAME_HEIGHT)
  - This covers the player's movement in detail
- The game state
    - Ninja state (12 values)
        - Position normalized
        - Speed normalized
        - Airborn boolean
        - Walled boolean
        - Jump duration normalized
        - Facing normalized
        - Tilt angle normalized
        - Applied gravity normalized
        - Applied drag normalized
        - Applied friction normalized
    - Exit and switch entity states
    - Time remaining
    - Vector between ninja and switch
    - Vector between ninja and exit door
"""
import numpy as np
import os
from collections import deque
import cv2
from typing import Dict, Any, Tuple
from npp_rl.environments.constants import (
    OBSERVATION_IMAGE_WIDTH, OBSERVATION_IMAGE_HEIGHT,
    PLAYER_FRAME_WIDTH, PLAYER_FRAME_HEIGHT,
    FRAME_INTERVALS, LEVEL_WIDTH, LEVEL_HEIGHT,
    MAX_VELOCITY
)


def frame_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale with shape (H, W, 1).

    This is an optimized version that:
    1. Uses direct array indexing instead of np.dot
    2. Handles different input shapes
    3. Avoids unnecessary operations if input is already grayscale
    """
    # If already grayscale, return as is
    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[-1] == 1):
        return frame if len(frame.shape) == 3 else frame[..., np.newaxis]

    # Ensure we're only taking first 3 channels if more exist
    frame = frame[..., :3]

    # Use direct array operations instead of np.dot
    # Weights for RGB to grayscale conversion (same as before)
    grayscale = (0.2989 * frame[..., 0] +
                 0.5870 * frame[..., 1] +
                 0.1140 * frame[..., 2])

    return grayscale[..., np.newaxis]  # Add channel dimension


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to desired dimensions."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)


def clip_frame(frame: np.ndarray) -> np.ndarray:
    """Clip frame to [0, 255] range and convert to uint8."""
    return np.clip(frame, 0, 255).astype(np.uint8)


def normalize_position(x: float, y: float) -> Tuple[float, float]:
    """Normalize position coordinates to [-1, 1] range."""
    return (
        (x / LEVEL_WIDTH) * 2 - 1,
        (y / LEVEL_HEIGHT) * 2 - 1
    )


def normalize_velocity(vx: float, vy: float) -> Tuple[float, float]:
    """Normalize velocity components to [-1, 1] range."""
    return (
        np.clip(vx / MAX_VELOCITY, -1, 1),
        np.clip(vy / MAX_VELOCITY, -1, 1)
    )


def calculate_vector(from_x: float, from_y: float, to_x: float, to_y: float) -> Tuple[float, float]:
    """Calculate normalized vector between two points."""
    dx = to_x - from_x
    dy = to_y - from_y
    magnitude = np.sqrt(dx * dx + dy * dy)
    if magnitude == 0:
        return (0.0, 0.0)
    return (dx / magnitude, dy / magnitude)


class ObservationProcessor:
    """Processes raw game observations into frame stacks and normalized feature vectors."""

    def __init__(self, enable_frame_stack: bool = False):
        # Keep only 3 frames in history: current, last, and second to last
        self.frame_history = deque(maxlen=3)
        self.enable_frame_stack = enable_frame_stack

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw frame to grayscale and resize with improved accuracy"""
        if len(frame.shape) == 3 and frame.shape[-1] != 1:
            frame = frame_to_grayscale(frame)
        frame = resize_frame(frame, OBSERVATION_IMAGE_WIDTH,
                             OBSERVATION_IMAGE_HEIGHT)
        frame = clip_frame(frame)
        if len(frame.shape) == 2:
            # Ensure we have a channel dimension
            frame = frame[..., np.newaxis]
        return frame

    def frame_around_player(self, frame: np.ndarray, player_x: float, player_y: float) -> np.ndarray:
        """Crop the frame to a rectangle centered on the player."""
        # Frame is already grayscale at this point
        player_frame = frame

        # Calculate the starting and ending coordinates for the crop
        start_x = int(player_x - PLAYER_FRAME_WIDTH // 2)
        end_x = int(player_x + PLAYER_FRAME_WIDTH // 2)
        start_y = int(player_y - PLAYER_FRAME_HEIGHT // 2)
        end_y = int(player_y + PLAYER_FRAME_HEIGHT // 2)

        # Ensure the crop is within the frame boundaries
        start_x = max(0, start_x)
        end_x = min(frame.shape[0], end_x)
        start_y = max(0, start_y)
        end_y = min(frame.shape[1], end_y)

        # Get the cropped frame
        player_frame = player_frame[start_x:end_x, start_y:end_y]

        # Calculate padding needed on each side
        top_pad = max(0, PLAYER_FRAME_HEIGHT // 2 - int(player_y))
        bottom_pad = max(0, PLAYER_FRAME_HEIGHT -
                         player_frame.shape[1] - top_pad)
        left_pad = max(0, PLAYER_FRAME_WIDTH // 2 - int(player_x))
        right_pad = max(0, PLAYER_FRAME_WIDTH -
                        player_frame.shape[0] - left_pad)

        # Pad the frame
        player_frame = cv2.copyMakeBorder(
            player_frame,
            top_pad, bottom_pad,
            left_pad, right_pad,
            cv2.BORDER_CONSTANT,
            value=0
        )

        if len(player_frame.shape) == 2:
            # Ensure we have a channel dimension
            player_frame = player_frame[..., np.newaxis]

        return player_frame

    def process_base_map(self, screen: np.ndarray) -> np.ndarray:
        """Process the base map (screen) into a standardized format.

        Returns a grayscale version of the current screen frame at the standard observation dimensions.
        Frame is already in grayscale at this point.
        """
        # Our simulator logic pads each edge of the map with 24 pixel tiles
        # We need to remove these padding tiles to get the actual map
        screen = screen[24:-24, 24:-24]

        # Resize to standard dimensions
        base_map = resize_frame(
            screen,
            OBSERVATION_IMAGE_WIDTH,
            OBSERVATION_IMAGE_HEIGHT
        )

        if len(base_map.shape) == 2:
            # Ensure we have a channel dimension
            base_map = base_map[..., np.newaxis]
        return base_map

    def process_game_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process game state into normalized feature vector."""
        # Calculate normalized vectors to objectives
        to_switch = calculate_vector(
            obs['player_x'], obs['player_y'],
            obs['switch_x'], obs['switch_y']
        )
        to_exit = calculate_vector(
            obs['player_x'], obs['player_y'],
            obs['exit_door_x'], obs['exit_door_y']
        )

        # Extract the original game state which contains ninja state and entity states
        game_state = obs['game_state']

        # Combine all features
        processed_state = np.concatenate([
            game_state,  # Original ninja and entity states
            [obs['time_remaining']],  # Time remaining
            [*to_switch, *to_exit]  # Vectors to objectives
        ]).astype(np.float32)

        return processed_state

    def process_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process observation into frame stack, base map, and feature vectors."""
        # Convert to grayscale once at the start
        screen = obs['screen']
        if len(screen.shape) == 3 and screen.shape[-1] != 1:
            screen = frame_to_grayscale(screen)

        # Process current frame using already grayscale screen
        player_frame = self.frame_around_player(
            screen,
            obs['player_x'],
            obs['player_y']
        )

        result = {
            'base_frame': self.process_base_map(screen),
            'game_state': self.process_game_state(obs)
        }

        if self.enable_frame_stack:
            # Update frame history with grayscale frame
            self.frame_history.append(screen)

            # Fill frame history if needed
            while len(self.frame_history) < 3:
                self.frame_history.append(screen)

            # Get player-centered frames for current and previous frames
            player_frames = []
            # Reverse to get [current, last, second_to_last]
            for frame in reversed(self.frame_history):
                player_frame = self.frame_around_player(
                    frame,  # Frame is already grayscale
                    obs['player_x'],
                    obs['player_y']
                )
                # Ensure each frame has shape (H, W, 1)
                if len(player_frame.shape) == 2:
                    player_frame = player_frame[..., np.newaxis]
                player_frames.append(player_frame)

            # Stack frames along channel dimension
            result['player_frame'] = np.concatenate(player_frames, axis=-1)

            # Verify we have exactly 3 channels
            assert result['player_frame'].shape[
                -1] == 3, f"Expected 3 channels, got {result['player_frame'].shape[-1]}"
        else:
            # Ensure single frame has shape (H, W, 1)
            if len(player_frame.shape) == 2:
                player_frame = player_frame[..., np.newaxis]
            result['player_frame'] = player_frame

        # Write our player frame to a file if the obs['sim_frame'] is 1
        if obs['sim_frame'] == 1:
            cv2.imwrite('player_frame.png', result['player_frame'])
            cv2.imwrite('base_frame.png', result['base_frame'])

        return result

    def reset(self) -> None:
        """Reset processor state."""
        self.frame_history.clear()
