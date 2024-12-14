import numpy as np
from collections import deque
import cv2
from typing import Dict, Any


class ObservationProcessor:
    """Processes raw game observations into the format needed by the agent"""

    def __init__(self, frame_stack: int = 4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)  # Recent frames
        self.image_size = 84

        # Initialize frame storage with smaller fixed intervals
        # This helps capture more immediate temporal dependencies
        # Fixed intervals for 3 historical frames
        self.historical_intervals = [2, 4, 8]
        self.frame_history = deque(maxlen=max(self.historical_intervals))

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

        # Combine everything
        final_observation = np.concatenate(
            recent_frames + historical_frames,
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
