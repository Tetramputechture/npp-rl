#!/usr/bin/env python3
"""
Human Replay Data Ingestion Tool

This tool converts raw human replay data (JSONL format) into structured datasets
compatible with the N++ RL training pipeline. It processes replay files and
produces NPZ or Parquet shards with observations, actions, and metadata.

For video generation from replay data, use the nclone video generator:
    python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4

Usage:
    python tools/replay_ingest.py --input raw_replays/ --output processed/ --format npz
    python tools/replay_ingest.py --help
"""

import argparse
import json
import logging
import numpy as np
import sys
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
        description="Convert human replay data to training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process replay files with rich observations
  python tools/replay_ingest.py --input datasets/raw --output datasets/processed --profile rich
  
  # Dry run to validate data without processing
  python tools/replay_ingest.py --input datasets/raw --dry-run

  # For video generation, use the nclone video generator:
  python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4
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

    # Validate arguments for data processing
    if not args.input:
        parser.error("--input is required")

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
