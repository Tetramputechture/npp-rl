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
    """Convert frame to grayscale with shape (H, W, 1)."""
    grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
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

    def __init__(self):
        self.frame_history = deque(maxlen=max(FRAME_INTERVALS) + 1)

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
        # Convert to grayscale if needed
        player_frame = frame
        if len(frame.shape) == 3 and frame.shape[-1] != 1:
            player_frame = frame_to_grayscale(frame)

        # Calculate the starting and ending coordinates for the crop
        start_x = int(player_x - PLAYER_FRAME_WIDTH // 2)
        end_x = int(player_x + PLAYER_FRAME_WIDTH // 2)
        start_y = int(player_y - PLAYER_FRAME_HEIGHT // 2)
        end_y = int(player_y + PLAYER_FRAME_HEIGHT // 2)

        # Ensure the crop is within the frame boundaries
        start_x = max(0, start_x)
        end_x = min(frame.shape[1], end_x)
        start_y = max(0, start_y)
        end_y = min(frame.shape[0], end_y)

        # Get the cropped frame
        player_frame = player_frame[start_y:end_y, start_x:end_x]

        # Calculate padding needed on each side
        top_pad = max(0, PLAYER_FRAME_HEIGHT // 2 - int(player_y))
        bottom_pad = max(0, PLAYER_FRAME_HEIGHT -
                         player_frame.shape[0] - top_pad)
        left_pad = max(0, PLAYER_FRAME_WIDTH // 2 - int(player_x))
        right_pad = max(0, PLAYER_FRAME_WIDTH -
                        player_frame.shape[1] - left_pad)

        # Pad the frame
        player_frame = cv2.copyMakeBorder(
            player_frame,
            top_pad, bottom_pad,
            left_pad, right_pad,
            cv2.BORDER_CONSTANT,
            value=0
        )

        clipped_frame = clip_frame(player_frame)

        if len(clipped_frame.shape) == 2:
            # Ensure we have a channel dimension
            clipped_frame = clipped_frame[..., np.newaxis]

        return clipped_frame

    def process_base_map(self, screen: np.ndarray) -> np.ndarray:
        """Process the base map (screen) into a standardized format.

        Returns a grayscale version of the current screen frame at the standard observation dimensions.
        """
        # Convert to grayscale if needed
        if len(screen.shape) == 3 and screen.shape[-1] != 1:
            screen = frame_to_grayscale(screen)

        # Our simulator logic pads each edge of the map with 24 pixel tiles
        # We need to remove these padding tiles to get the actual map
        screen = screen[24:-24, 24:-24]

        # Resize to standard dimensions
        base_map = resize_frame(
            screen,
            OBSERVATION_IMAGE_WIDTH,
            OBSERVATION_IMAGE_HEIGHT
        )

        base_map = clip_frame(base_map)
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
        # Process current frame
        player_frame = self.frame_around_player(
            obs['screen'],
            obs['player_x'],
            obs['player_y']
        )

        # Frame stacking logic. We don't need this since we are only using the current frame now,
        # with an LSTM head on our feature extractor.
        # # Update frame history
        # self.frame_history.append(frame)

        # # Fill frame history if needed
        # while len(self.frame_history) < max(FRAME_INTERVALS) + 1:
        #     self.frame_history.append(frame)

        # # Get player-centered frames at specified intervals
        # player_frames = []
        # for interval in FRAME_INTERVALS:
        #     historical_frame = self.frame_history[-interval-1]
        #     player_frame = self.frame_around_player(
        #         historical_frame,
        #         obs['player_x'],
        #         obs['player_y']
        #     )
        #     player_frames.append(player_frame[..., np.newaxis])

        # Process base map
        base_map = self.process_base_map(obs['screen'])

        # Process game state
        game_state = self.process_game_state(obs)

        # Print our shapes
        # print(
        #     f"Player frames shape: {player_frame.shape}")
        # print(f"Base map shape: {base_map.shape}")
        # print(f"Game state shape: {game_state.shape}")

        return {
            'player_frame': player_frame,
            'base_frame': base_map,
            'game_state': game_state
        }

    def reset(self) -> None:
        """Reset processor state."""
        self.frame_history.clear()
