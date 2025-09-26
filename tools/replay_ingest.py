#!/usr/bin/env python3
"""
Human Replay Data Ingestion Tool

This tool converts raw human replay data (JSONL format) into structured datasets
compatible with the N++ RL training pipeline. It processes replay files and
produces NPZ or Parquet shards with observations, actions, and metadata.

It can also generate videos of replays using the actual NppEnvironment rendering.

Usage:
    python tools/replay_ingest.py --input raw_replays/ --output processed/ --format npz
    python tools/replay_ingest.py --input raw_replays/ --output-video replay.mp4 --generate-video
    python tools/replay_ingest.py --help
"""

import argparse
import json
import logging
import numpy as np
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
nclone_path = project_root.parent / "nclone"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(nclone_path))

# Import nclone modules after path setup
try:
    from nclone.gym_environment.npp_environment import NppEnvironment
    from nclone.replay import BinaryReplayParser
except ImportError as e:
    print(f"Error importing nclone modules: {e}")
    print("Make sure nclone is in the correct location relative to npp-rl")
    sys.exit(1)

logger = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    """Structured representation of a single replay frame."""

    timestamp: float
    level_id: str
    frame_number: int
    player_position: Tuple[float, float]
    player_velocity: Tuple[float, float]
    player_state: Dict[str, Any]
    player_inputs: Dict[str, bool]
    entities: List[Dict[str, Any]]
    level_bounds: Dict[str, int]
    meta: Dict[str, Any]


@dataclass
class ProcessedSample:
    """Processed sample ready for training."""

    observation: Dict[str, np.ndarray]
    action: int
    meta: Dict[str, Any]


class ActionMapper:
    """Maps raw player inputs to discrete action space."""

    ACTION_MAPPING = {
        (False, False, False): 0,  # no_action
        (True, False, False): 1,  # left
        (False, True, False): 2,  # right
        (False, False, True): 3,  # jump
        (True, False, True): 4,  # left_jump
        (False, True, True): 5,  # right_jump
    }

    ACTION_NAMES = {
        0: "no_action",
        1: "left",
        2: "right",
        3: "jump",
        4: "left_jump",
        5: "right_jump",
    }

    @classmethod
    def map_inputs_to_action(cls, inputs: Dict[str, bool]) -> int:
        """Convert player inputs to discrete action index."""
        left = inputs.get("left", False)
        right = inputs.get("right", False)
        jump = inputs.get("jump", False)

        # Handle invalid combinations (left + right)
        if left and right:
            left = right = False

        key = (left, right, jump)
        return cls.ACTION_MAPPING.get(key, 0)  # Default to no_action

    @classmethod
    def get_action_name(cls, action: int) -> str:
        """Get human-readable action name."""
        return cls.ACTION_NAMES.get(action, "unknown")


class ReplayValidator:
    """Validates replay data for quality and consistency."""

    @staticmethod
    def validate_frame(frame_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single replay frame.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required top-level fields
        required_fields = [
            "timestamp",
            "level_id",
            "frame_number",
            "player_state",
            "player_inputs",
            "entities",
            "level_bounds",
            "meta",
        ]

        for field in required_fields:
            if field not in frame_data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Validate timestamp
        if (
            not isinstance(frame_data["timestamp"], (int, float))
            or frame_data["timestamp"] <= 0
        ):
            errors.append("Invalid timestamp")

        # Validate frame number
        if (
            not isinstance(frame_data["frame_number"], int)
            or frame_data["frame_number"] < 0
        ):
            errors.append("Invalid frame_number")

        # Validate player state
        player_state = frame_data["player_state"]
        if "position" not in player_state or "velocity" not in player_state:
            errors.append("Missing player position or velocity")
        else:
            pos = player_state["position"]
            vel = player_state["velocity"]

            if not all(isinstance(pos.get(k), (int, float)) for k in ["x", "y"]):
                errors.append("Invalid player position coordinates")

            if not all(isinstance(vel.get(k), (int, float)) for k in ["x", "y"]):
                errors.append("Invalid player velocity coordinates")

        # Validate player inputs
        player_inputs = frame_data["player_inputs"]
        input_keys = ["left", "right", "jump", "restart"]
        for key in input_keys:
            if key not in player_inputs or not isinstance(player_inputs[key], bool):
                errors.append(f"Invalid or missing player input: {key}")

        # Validate entities
        if not isinstance(frame_data["entities"], list):
            errors.append("Entities must be a list")
        else:
            for i, entity in enumerate(frame_data["entities"]):
                if not isinstance(entity, dict):
                    errors.append(f"Entity {i} must be a dictionary")
                    continue

                if "type" not in entity or "position" not in entity:
                    errors.append(f"Entity {i} missing type or position")

        # Validate level bounds
        level_bounds = frame_data["level_bounds"]
        if not all(
            isinstance(level_bounds.get(k), int) and level_bounds.get(k) > 0
            for k in ["width", "height"]
        ):
            errors.append("Invalid level bounds")

        return len(errors) == 0, errors

    @staticmethod
    def validate_trajectory(frames: List[ReplayFrame]) -> Tuple[bool, List[str]]:
        """
        Validate a complete trajectory for consistency.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not frames:
            return False, ["Empty trajectory"]

        # Check temporal consistency
        timestamps = [f.timestamp for f in frames]
        if not all(
            timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
        ):
            errors.append("Timestamps not monotonically increasing")

        # Check frame number consistency
        frame_numbers = [f.frame_number for f in frames]
        if frame_numbers != list(range(len(frame_numbers))):
            errors.append("Frame numbers not sequential")

        # Check level consistency
        level_ids = set(f.level_id for f in frames)
        if len(level_ids) > 1:
            errors.append("Multiple level IDs in single trajectory")

        return len(errors) == 0, errors


class ObservationProcessor:
    """Processes raw replay data into environment-compatible observations."""

    def __init__(self):
        """
        Initialize observation processor.
        """
        self.env = None  # Will be created lazily for observation processing

    def _get_env(self) -> NppEnvironment:
        """Get or create environment instance for observation processing."""
        if self.env is None:
            self.env = NppEnvironment(
                render_mode="rgb_array",
            )
        return self.env

    def process_frame(self, frame: ReplayFrame) -> Optional[ProcessedSample]:
        """
        Process a single replay frame into a training sample.

        Args:
            frame: Raw replay frame data

        Returns:
            Processed sample or None if processing fails
        """
        try:
            # Map inputs to action
            action = ActionMapper.map_inputs_to_action(frame.player_inputs)

            # Create mock observation (in real implementation, this would render the actual game state)
            # For now, we create placeholder observations with correct shapes
            observation = self._create_mock_observation(frame)

            # Create metadata
            meta = {
                "timestamp": frame.timestamp,
                "level_id": frame.level_id,
                "frame_number": frame.frame_number,
                "quality_score": frame.meta.get("quality_score", 0.5),
                "session_id": frame.meta.get("session_id", "unknown"),
            }

            return ProcessedSample(observation=observation, action=action, meta=meta)

        except Exception as e:
            logger.error(f"Failed to process frame {frame.frame_number}: {e}")
            return None

    def _create_mock_observation(self, frame: ReplayFrame) -> Dict[str, np.ndarray]:
        """
        Create mock observation for testing purposes.

        In a real implementation, this would:
        1. Set up the game state based on frame data
        2. Render player and global views
        3. Compute game state vector

        For now, we create placeholder data with correct shapes.
        """
        # Mock player frame (64x64x3)
        player_frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Mock global view (128x128x3)
        global_view = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        # Mock game state vector
        game_state_dim = 31

        # Create realistic game state based on frame data
        game_state = np.zeros(game_state_dim, dtype=np.float32)

        # Fill in known values from frame data
        pos = frame.player_position
        vel = frame.player_velocity
        bounds = frame.level_bounds

        # Normalize position to [0, 1]
        game_state[0] = pos[0] / bounds["width"]
        game_state[1] = pos[1] / bounds["height"]

        # Normalize velocity to [-1, 1] (assuming max velocity of 10)
        game_state[2] = np.clip(vel[0] / 10.0, -1, 1)
        game_state[3] = np.clip(vel[1] / 10.0, -1, 1)

        # Player state flags
        game_state[4] = float(frame.player_state.get("on_ground", False))
        game_state[5] = float(frame.player_state.get("wall_sliding", False))
        game_state[6] = frame.player_state.get("jump_time_remaining", 0.0)

        # Fill remaining with mock data
        game_state[7:] = np.random.random(game_state_dim - 7) * 0.1

        return {
            "player_frame": player_frame,
            "global_view": global_view,
            "game_state": game_state,
        }


class VideoGenerator:
    """Generates videos from replay data using NppEnvironment rendering."""

    def __init__(self, fps: int = 60, resolution: Tuple[int, int] = (1056, 600)):
        """
        Initialize video generator.

        Args:
            fps: Frames per second for output video
            resolution: Video resolution (width, height)
        """
        self.fps = fps
        self.resolution = resolution
        self.env = None

    def _get_env(self, custom_map_path: Optional[str] = None) -> NppEnvironment:
        """Get or create environment instance for rendering."""
        if self.env is None:
            self.env = NppEnvironment(
                render_mode="rgb_array",
                enable_animation=True,
                enable_debug_overlay=False,
                custom_map_path=custom_map_path,
            )
        return self.env

    def generate_video_from_jsonl(
        self,
        replay_file: Path,
        output_video: Path,
        custom_map_path: Optional[str] = None,
    ) -> bool:
        """
        Generate video from JSONL replay file.

        Args:
            replay_file: Path to JSONL replay file
            output_video: Output video file path
            custom_map_path: Optional custom map to load

        Returns:
            True if video generation succeeded
        """
        try:
            # Load replay frames
            frames = self._load_jsonl_replay(replay_file)
            if not frames:
                logger.error("No valid frames found in replay file")
                return False

            # Initialize environment
            env = self._get_env(custom_map_path)
            env.reset()

            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                logger.info(f"Generating {len(frames)} frames...")

                # Generate frame images
                for i, frame in enumerate(frames):
                    try:
                        # Map frame inputs to action
                        action = ActionMapper.map_inputs_to_action(frame.player_inputs)

                        # Step environment
                        obs, reward, terminated, truncated, info = env.step(action)

                        # Render frame
                        frame_image = env.render()
                        if frame_image is not None:
                            # Save frame as PNG
                            frame_file = temp_path / f"frame_{i:06d}.png"
                            self._save_frame_as_png(frame_image, frame_file)

                        if terminated or truncated:
                            logger.info(f"Episode ended at frame {i}")
                            break

                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {e}")
                        continue

                # Generate video using ffmpeg
                return self._create_video_from_frames(temp_path, output_video)

        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            return False

    def generate_video_from_binary_replay(
        self, binary_replay_dir: Path, output_video: Path
    ) -> bool:
        """
        Generate video from binary replay directory.

        Args:
            binary_replay_dir: Directory containing binary replay files
            output_video: Output video file path

        Returns:
            True if video generation succeeded
        """
        try:
            # Parse binary replay to JSONL first
            parser = BinaryReplayParser()

            if not parser.detect_trace_mode(binary_replay_dir):
                logger.error("Directory does not contain trace mode files")
                return False

            # Load replay data
            inputs_list, map_data = parser.load_inputs_and_map(binary_replay_dir)

            if not inputs_list:
                logger.error("No input data found in binary replay")
                return False

            # Use first input sequence
            inputs = inputs_list[0]
            level_id = binary_replay_dir.name
            session_id = f"{level_id}_video"

            # Simulate replay to get frames
            frames_data = parser.simulate_replay(inputs, map_data, level_id, session_id)

            # Convert to ReplayFrame objects
            frames = []
            for frame_data in frames_data:
                frame = ReplayFrame(
                    timestamp=frame_data["timestamp"],
                    level_id=frame_data["level_id"],
                    frame_number=frame_data["frame_number"],
                    player_position=(
                        frame_data["player_state"]["position"]["x"],
                        frame_data["player_state"]["position"]["y"],
                    ),
                    player_velocity=(
                        frame_data["player_state"]["velocity"]["x"],
                        frame_data["player_state"]["velocity"]["y"],
                    ),
                    player_state=frame_data["player_state"],
                    player_inputs=frame_data["player_inputs"],
                    entities=frame_data["entities"],
                    level_bounds=frame_data["level_bounds"],
                    meta=frame_data["meta"],
                )
                frames.append(frame)

            # Generate video from frames
            return self._generate_video_from_frames(frames, output_video, map_data)

        except Exception as e:
            logger.error(f"Failed to generate video from binary replay: {e}")
            return False

    def _load_jsonl_replay(self, replay_file: Path) -> List[ReplayFrame]:
        """Load replay frames from JSONL file."""
        frames = []
        validator = ReplayValidator()

        try:
            with open(replay_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        frame_data = json.loads(line.strip())

                        # Validate frame
                        is_valid, errors = validator.validate_frame(frame_data)
                        if not is_valid:
                            logger.warning(
                                f"Frame {line_num} validation failed: {errors}"
                            )
                            continue

                        # Convert to ReplayFrame
                        frame = ReplayFrame(
                            timestamp=frame_data["timestamp"],
                            level_id=frame_data["level_id"],
                            frame_number=frame_data["frame_number"],
                            player_position=(
                                frame_data["player_state"]["position"]["x"],
                                frame_data["player_state"]["position"]["y"],
                            ),
                            player_velocity=(
                                frame_data["player_state"]["velocity"]["x"],
                                frame_data["player_state"]["velocity"]["y"],
                            ),
                            player_state=frame_data["player_state"],
                            player_inputs=frame_data["player_inputs"],
                            entities=frame_data["entities"],
                            level_bounds=frame_data["level_bounds"],
                            meta=frame_data["meta"],
                        )
                        frames.append(frame)

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")

        except Exception as e:
            logger.error(f"Failed to load replay file {replay_file}: {e}")

        return frames

    def _generate_video_from_frames(
        self,
        frames: List[ReplayFrame],
        output_video: Path,
        map_data: Optional[List[int]] = None,
    ) -> bool:
        """Generate video from replay frames using environment rendering."""
        try:
            # Initialize environment
            env = self._get_env()
            env.reset()

            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                logger.info(f"Generating {len(frames)} frames...")

                # Generate frame images
                for i, frame in enumerate(frames):
                    try:
                        # Map frame inputs to action
                        action = ActionMapper.map_inputs_to_action(frame.player_inputs)

                        # Step environment
                        obs, reward, terminated, truncated, info = env.step(action)

                        # Render frame
                        frame_image = env.render()
                        if frame_image is not None:
                            # Save frame as PNG
                            frame_file = temp_path / f"frame_{i:06d}.png"
                            self._save_frame_as_png(frame_image, frame_file)

                        if terminated or truncated:
                            logger.info(f"Episode ended at frame {i}")
                            break

                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {e}")
                        continue

                # Generate video using ffmpeg
                return self._create_video_from_frames(temp_path, output_video)

        except Exception as e:
            logger.error(f"Failed to generate video from frames: {e}")
            return False

    def _save_frame_as_png(self, frame_array: np.ndarray, output_path: Path):
        """Save frame array as PNG image."""
        try:
            from PIL import Image

            # Handle different frame formats
            if len(frame_array.shape) == 3:
                if frame_array.shape[2] == 1:
                    # Single channel - squeeze to 2D
                    frame_2d = np.squeeze(frame_array, axis=2)
                    image = Image.fromarray(frame_2d.astype(np.uint8), mode="L")
                elif frame_array.shape[2] == 3:
                    # RGB format
                    image = Image.fromarray(frame_array.astype(np.uint8), mode="RGB")
                elif frame_array.shape[2] == 4:
                    # RGBA format
                    image = Image.fromarray(frame_array.astype(np.uint8), mode="RGBA")
                else:
                    logger.warning(
                        f"Unsupported frame format with {frame_array.shape[2]} channels"
                    )
                    return
            elif len(frame_array.shape) == 2:
                # Already 2D grayscale
                image = Image.fromarray(frame_array.astype(np.uint8), mode="L")
            else:
                logger.warning(f"Unsupported frame shape {frame_array.shape}")
                return

            image.save(output_path)

        except Exception as e:
            logger.error(f"Failed to save frame as PNG: {e}")

    def _create_video_from_frames(self, frames_dir: Path, output_video: Path) -> bool:
        """Create video from frame images using ffmpeg."""
        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(
                    "ffmpeg not found. Please install ffmpeg to generate videos."
                )
                return False

            # Create output directory if it doesn't exist
            output_video.parent.mkdir(parents=True, exist_ok=True)

            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate",
                str(self.fps),
                "-i",
                str(frames_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
                str(output_video),
            ]

            logger.info(f"Running ffmpeg: {' '.join(cmd)}")

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Video generated successfully: {output_video}")
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return False


class ReplayIngester:
    """Main class for ingesting and processing replay data."""

    def __init__(self, output_format: str = "npz"):
        """
        Initialize replay ingester.

        Args:
            output_format: 'npz' or 'parquet' output format
        """
        self.output_format = output_format
        self.processor = ObservationProcessor()
        self.validator = ReplayValidator()

        # Statistics
        self.stats = {
            "files_processed": 0,
            "frames_processed": 0,
            "frames_valid": 0,
            "frames_invalid": 0,
            "trajectories_processed": 0,
            "samples_generated": 0,
        }

    def load_replay_file(self, file_path: Path) -> List[ReplayFrame]:
        """
        Load and parse a JSONL replay file.

        Args:
            file_path: Path to JSONL replay file

        Returns:
            List of parsed replay frames
        """
        frames = []

        try:
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        frame_data = json.loads(line.strip())

                        # Validate frame
                        is_valid, errors = self.validator.validate_frame(frame_data)
                        if not is_valid:
                            logger.warning(
                                f"{file_path}:{line_num} - Validation errors: {errors}"
                            )
                            self.stats["frames_invalid"] += 1
                            continue

                        # Convert to structured format
                        frame = ReplayFrame(
                            timestamp=frame_data["timestamp"],
                            level_id=frame_data["level_id"],
                            frame_number=frame_data["frame_number"],
                            player_position=(
                                frame_data["player_state"]["position"]["x"],
                                frame_data["player_state"]["position"]["y"],
                            ),
                            player_velocity=(
                                frame_data["player_state"]["velocity"]["x"],
                                frame_data["player_state"]["velocity"]["y"],
                            ),
                            player_state=frame_data["player_state"],
                            player_inputs=frame_data["player_inputs"],
                            entities=frame_data["entities"],
                            level_bounds=frame_data["level_bounds"],
                            meta=frame_data["meta"],
                        )

                        frames.append(frame)
                        self.stats["frames_valid"] += 1

                    except json.JSONDecodeError as e:
                        logger.error(f"{file_path}:{line_num} - JSON decode error: {e}")
                        self.stats["frames_invalid"] += 1
                    except Exception as e:
                        logger.error(f"{file_path}:{line_num} - Processing error: {e}")
                        self.stats["frames_invalid"] += 1

                self.stats["frames_processed"] += len(frames)

        except Exception as e:
            logger.error(f"Failed to load replay file {file_path}: {e}")

        return frames

    def process_trajectory(self, frames: List[ReplayFrame]) -> List[ProcessedSample]:
        """
        Process a trajectory of frames into training samples.

        Args:
            frames: List of replay frames forming a trajectory

        Returns:
            List of processed samples
        """
        # Validate trajectory
        is_valid, errors = self.validator.validate_trajectory(frames)
        if not is_valid:
            logger.warning(f"Trajectory validation failed: {errors}")
            return []

        samples = []
        for frame in frames:
            sample = self.processor.process_frame(frame)
            if sample is not None:
                samples.append(sample)

        self.stats["trajectories_processed"] += 1
        self.stats["samples_generated"] += len(samples)

        return samples

    def save_samples_npz(self, samples: List[ProcessedSample], output_path: Path):
        """Save processed samples to NPZ format."""
        if not samples:
            logger.warning("No samples to save")
            return

        # Separate observations, actions, and metadata
        observations = defaultdict(list)
        actions = []
        meta = defaultdict(list)

        for sample in samples:
            # Collect observations
            for key, value in sample.observation.items():
                observations[key].append(value)

            actions.append(sample.action)

            # Collect metadata
            for key, value in sample.meta.items():
                meta[key].append(value)

        # Convert to numpy arrays
        obs_arrays = {}
        for key, values in observations.items():
            obs_arrays[key] = np.stack(values)

        actions_array = np.array(actions, dtype=np.int32)

        meta_arrays = {}
        for key, values in meta.items():
            if isinstance(values[0], str):
                meta_arrays[key] = np.array(values, dtype=object)
            else:
                meta_arrays[key] = np.array(values)

        # Save to NPZ
        np.savez_compressed(
            output_path,
            observations=obs_arrays,
            actions=actions_array,
            meta=meta_arrays,
        )

        logger.info(f"Saved {len(samples)} samples to {output_path}")

    def save_samples_parquet(self, samples: List[ProcessedSample], output_path: Path):
        """Save processed samples to Parquet format."""
        # TODO: Implement Parquet saving for large datasets
        logger.warning("Parquet format not yet implemented, falling back to NPZ")
        npz_path = output_path.with_suffix(".npz")
        self.save_samples_npz(samples, npz_path)

    def process_directory(
        self, input_dir: Path, output_dir: Path, max_samples_per_file: int = 10000
    ):
        """
        Process all replay files in a directory.

        Args:
            input_dir: Directory containing JSONL replay files
            output_dir: Directory to save processed datasets
            max_samples_per_file: Maximum samples per output file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all JSONL files
        replay_files = list(input_dir.glob("**/*.jsonl"))
        logger.info(f"Found {len(replay_files)} replay files")

        all_samples = []
        file_counter = 0

        for file_path in replay_files:
            logger.info(f"Processing {file_path}")

            # Load and process file
            frames = self.load_replay_file(file_path)
            if not frames:
                continue

            samples = self.process_trajectory(frames)
            all_samples.extend(samples)

            self.stats["files_processed"] += 1

            # Save batch if we have enough samples
            if len(all_samples) >= max_samples_per_file:
                output_path = output_dir / f"batch_{file_counter:04d}.npz"

                if self.output_format == "npz":
                    self.save_samples_npz(all_samples, output_path)
                else:
                    self.save_samples_parquet(all_samples, output_path)

                all_samples = []
                file_counter += 1

        # Save remaining samples
        if all_samples:
            output_path = output_dir / f"batch_{file_counter:04d}.npz"

            if self.output_format == "npz":
                self.save_samples_npz(all_samples, output_path)
            else:
                self.save_samples_parquet(all_samples, output_path)

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("REPLAY INGESTION STATISTICS")
        print("=" * 50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Frames valid: {self.stats['frames_valid']}")
        print(f"Frames invalid: {self.stats['frames_invalid']}")
        print(f"Trajectories processed: {self.stats['trajectories_processed']}")
        print(f"Samples generated: {self.stats['samples_generated']}")

        if self.stats["frames_processed"] > 0:
            validity_rate = (
                self.stats["frames_valid"] / self.stats["frames_processed"] * 100
            )
            print(f"Validity rate: {validity_rate:.1f}%")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert human replay data to training datasets and generate videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process replay files with rich observations
  python tools/replay_ingest.py --input datasets/raw --output datasets/processed --profile rich
  
  # Generate video from JSONL replay
  python tools/replay_ingest.py --input datasets/raw/replay.jsonl --output-video replay.mp4 --generate-video
  
  # Generate video from binary replay directory
  python tools/replay_ingest.py --input replays/level_001 --output-video level_001.mp4 --generate-video --binary-replay
  
  # Dry run to validate data without processing
  python tools/replay_ingest.py --input datasets/raw --dry-run
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory with JSONL replay files or binary replay directory",
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "rich"],
        default="rich",
        help="Observation profile to use",
    )
    parser.add_argument(
        "--format", choices=["npz", "parquet"], default="npz", help="Output format"
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000, help="Maximum samples per output file"
    )
    parser.add_argument(
        "--generate-video", action="store_true", help="Generate video from replay data"
    )
    parser.add_argument("--output-video", type=Path, help="Output video file path")
    parser.add_argument(
        "--binary-replay",
        action="store_true",
        help="Input is binary replay directory (trace mode)",
    )
    parser.add_argument(
        "--fps", type=int, default=60, help="Video framerate (default: 60)"
    )
    parser.add_argument(
        "--custom-map", type=str, help="Custom map file to use for video generation"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data without processing"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Handle video generation
    if args.generate_video:
        if not args.input:
            parser.error("--input is required for video generation")
        if not args.output_video:
            parser.error("--output-video is required for video generation")

        video_generator = VideoGenerator(fps=args.fps)

        if args.binary_replay:
            # Generate video from binary replay directory
            success = video_generator.generate_video_from_binary_replay(
                args.input, args.output_video
            )
        else:
            # Generate video from JSONL replay file
            success = video_generator.generate_video_from_jsonl(
                args.input, args.output_video, args.custom_map
            )

        if success:
            logger.info(f"Video generation completed: {args.output_video}")
        else:
            logger.error("Video generation failed")
            return 1

        return 0

    # Validate arguments for data processing
    if not args.input:
        parser.error("--input is required (unless using --generate-video)")

    if not args.input.exists():
        parser.error(f"Input directory does not exist: {args.input}")

    if not args.dry_run and not args.output:
        parser.error("--output is required (unless using --dry-run)")

    # Initialize ingester
    ingester = ReplayIngester(output_format=args.format)

    if args.dry_run:
        logger.info("Dry run mode - validating data only")
        # TODO: Implement dry run validation
        logger.info("Dry run validation not yet implemented")
    else:
        # Process data
        logger.info(f"Processing replay data from {args.input}")
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Observation profile: {args.profile}")
        logger.info(f"Output format: {args.format}")

        ingester.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            max_samples_per_file=args.max_samples,
        )

    # Print statistics
    ingester.print_statistics()


if __name__ == "__main__":
    main()
