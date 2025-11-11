"""Video recording utility for evaluation episodes.

Handles frame capture and MP4 video encoding for trained agents.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import imageio

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records episodes as MP4 videos."""

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        codec: str = "libx264",
        quality: Optional[int] = 8,
    ):
        """Initialize video recorder.

        Args:
            output_path: Path to save video file
            fps: Frames per second (default: 30)
            codec: Video codec (default: libx264)
            quality: Video quality, lower is better (default: 8)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.frames: List[np.ndarray] = []
        self.is_recording = False

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def start_recording(self) -> None:
        """Start recording frames."""
        self.frames = []
        self.is_recording = True
        logger.debug(f"Started recording video: {self.output_path}")

    def record_frame(self, frame: np.ndarray) -> None:
        """Record a single frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
        """
        if not self.is_recording:
            print("Attempted to record frame but recording not started")
            return

        if frame is None:
            print("Received None frame, skipping")
            return

        # Make a copy and handle transposition if needed
        # Based on nclone/run_multiple_headless.py, frames may need transposition
        frame_copy = frame.copy()

        # Check if frame needs transposition (W, H, C) -> (H, W, C)
        # Typically gym environments return (H, W, C) but check dimensions
        if len(frame_copy.shape) == 3:
            # If width > height and it looks transposed, fix it
            if frame_copy.shape[0] < frame_copy.shape[1]:
                # Likely (W, H, C), transpose to (H, W, C)
                frame_copy = frame_copy.transpose(1, 0, 2)

        # Rotate video 90 degrees counter-clockwise
        frame_copy = np.rot90(frame_copy, k=3)

        self.frames.append(frame_copy)

    def stop_recording(self, save: bool = True) -> bool:
        """Stop recording and optionally save video.

        Args:
            save: Whether to save the video (default: True)

        Returns:
            True if video was saved successfully, False otherwise
        """
        self.is_recording = False

        if not save or len(self.frames) == 0:
            logger.debug(f"Discarding {len(self.frames)} frames (save={save})")
            self.frames = []
            return False

        return self._save_video()

    def _save_video(self) -> bool:
        """Save recorded frames as MP4 video.

        Returns:
            True if successful, False otherwise
        """
        if len(self.frames) == 0:
            print("No frames to save")
            return False

        try:
            logger.debug(
                f"Saving video: {self.output_path} "
                f"({len(self.frames)} frames @ {self.fps} fps)"
            )

            # Use imageio to write video
            with imageio.get_writer(
                str(self.output_path),
                fps=self.fps,
                codec=self.codec,
                quality=self.quality,
                format="FFMPEG",
            ) as writer:
                for frame in self.frames:
                    writer.append_data(frame)

            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Saved video: {self.output_path.name} "
                f"({len(self.frames)} frames, {file_size_mb:.2f} MB)"
            )

            self.frames = []
            return True

        except Exception as e:
            print(f"Failed to save video {self.output_path}: {e}")
            self.frames = []
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Save video if no exception occurred
        save = exc_type is None
        self.stop_recording(save=save)


def create_video_recorder(
    output_path: str,
    fps: int = 30,
    codec: str = "libx264",
    quality: Optional[int] = 8,
) -> Optional[VideoRecorder]:
    """Create video recorder with error handling.

    Args:
        output_path: Path to save video
        fps: Frames per second
        codec: Video codec
        quality: Video quality

    Returns:
        VideoRecorder instance or None if imageio not available
    """
    try:
        return VideoRecorder(
            output_path=output_path,
            fps=fps,
            codec=codec,
            quality=quality,
        )
    except Exception as e:
        print(f"Failed to create video recorder: {e}")
        return None
