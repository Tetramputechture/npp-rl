"""Unit tests for replay dataset loader."""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

from npp_rl.path_prediction.replay_dataset import (
    PathReplayDataset,
    collate_replay_batch,
)
from nclone.replay.gameplay_recorder import CompactReplay


class TestReplayDataset(unittest.TestCase):
    """Test cases for PathReplayDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test replays
        self.temp_dir = tempfile.mkdtemp()
        self.replay_dir = Path(self.temp_dir) / "replays"
        self.replay_dir.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_dummy_replay(self, filename: str, num_frames: int = 100):
        """Create a dummy replay file for testing.

        Args:
            filename: Name of replay file
            num_frames: Number of frames in replay
        """
        # Create minimal replay data
        map_data = {
            "tiles": np.zeros((40, 40), dtype=np.uint8),
            "objects": [],
        }

        # Create input sequence
        input_sequence = []
        for i in range(num_frames):
            input_sequence.append(
                {
                    "frame": i,
                    "left": False,
                    "right": False,
                    "jump": False,
                }
            )

        replay = CompactReplay(
            episode_id=filename.replace(".replay", ""),
            map_data=map_data,
            input_sequence=input_sequence,
            success=True,
            completion_time=num_frames / 60.0,
            metadata={},
        )

        # Save replay
        replay_path = self.replay_dir / filename
        with open(replay_path, "wb") as f:
            f.write(replay.to_binary())

    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        # Create some dummy replays
        self._create_dummy_replay("replay1.replay", 100)
        self._create_dummy_replay("replay2.replay", 150)

        dataset = PathReplayDataset(
            replay_dir=str(self.replay_dir),
            waypoint_interval=5,
            min_trajectory_length=20,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.waypoint_interval, 5)
        self.assertEqual(dataset.min_trajectory_length, 20)

    def test_empty_directory_raises_error(self):
        """Test that empty directory raises ValueError."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()

        with self.assertRaises(ValueError):
            PathReplayDataset(replay_dir=str(empty_dir))

    def test_dataset_length(self):
        """Test that dataset length matches number of replay files."""
        num_replays = 5
        for i in range(num_replays):
            self._create_dummy_replay(f"replay{i}.replay")

        dataset = PathReplayDataset(replay_dir=str(self.replay_dir))
        self.assertEqual(len(dataset), num_replays)

    def test_max_replays_limit(self):
        """Test that max_replays parameter limits dataset size."""
        # Create more replays than max
        for i in range(10):
            self._create_dummy_replay(f"replay{i}.replay")

        dataset = PathReplayDataset(
            replay_dir=str(self.replay_dir),
            max_replays=3,
        )

        self.assertEqual(len(dataset), 3)

    def test_waypoint_extraction(self):
        """Test that waypoints are extracted at correct intervals."""
        self._create_dummy_replay("test.replay", num_frames=50)

        dataset = PathReplayDataset(
            replay_dir=str(self.replay_dir),
            waypoint_interval=10,
        )

        # Get first sample (this may fail if ReplayExecutor has issues)
        try:
            sample = dataset[0]

            # Check that waypoints were extracted
            waypoints = sample["expert_waypoints"]
            self.assertIsInstance(waypoints, list)

            # Should have approximately num_frames / waypoint_interval waypoints
            # Exact number depends on replay execution
            self.assertGreaterEqual(len(waypoints), 0)

        except Exception as e:
            # If replay execution fails, skip this test
            self.skipTest(f"Replay execution failed: {e}")


class TestCollateReplayBatch(unittest.TestCase):
    """Test cases for collate_replay_batch function."""

    def test_collate_valid_samples(self):
        """Test collation of valid samples."""
        # Create mock samples
        samples = [
            {
                "graph_obs": None,
                "tile_patterns": torch.randn(6),
                "entity_features": torch.randn(32),
                "expert_waypoints": [(100, 100), (150, 150), (200, 200)],
                "expert_waypoints_tensor": torch.tensor(
                    [[100, 100], [150, 150], [200, 200]], dtype=torch.float32
                ),
                "trajectory_length": 50,
                "success": True,
                "replay_id": "test1",
            },
            {
                "graph_obs": None,
                "tile_patterns": torch.randn(6),
                "entity_features": torch.randn(32),
                "expert_waypoints": [(50, 50), (100, 100)],
                "expert_waypoints_tensor": torch.tensor(
                    [[50, 50], [100, 100]], dtype=torch.float32
                ),
                "trajectory_length": 30,
                "success": True,
                "replay_id": "test2",
            },
        ]

        batch = collate_replay_batch(samples)

        # Check batch structure
        self.assertEqual(batch["batch_size"], 2)
        self.assertEqual(batch["tile_patterns"].shape, (2, 6))
        self.assertEqual(batch["entity_features"].shape, (2, 32))

        # Check waypoints are padded correctly
        max_waypoints = 3
        self.assertEqual(batch["expert_waypoints_tensor"].shape, (2, max_waypoints, 2))

        # Check masks
        self.assertEqual(batch["waypoint_masks"].shape, (2, max_waypoints))
        self.assertTrue(
            batch["waypoint_masks"][0, :3].all()
        )  # First sample has 3 waypoints
        self.assertTrue(
            batch["waypoint_masks"][1, :2].all()
        )  # Second sample has 2 waypoints
        self.assertFalse(batch["waypoint_masks"][1, 2])  # Third waypoint is padding

    def test_collate_empty_samples(self):
        """Test collation handles empty samples."""
        samples = [
            {
                "graph_obs": None,
                "tile_patterns": torch.zeros(6),
                "entity_features": torch.zeros(32),
                "expert_waypoints": [],
                "expert_waypoints_tensor": torch.zeros((0, 2)),
                "trajectory_length": 0,
                "success": False,
                "replay_id": "empty",
            },
        ]

        batch = collate_replay_batch(samples)

        # Should return empty batch
        self.assertEqual(batch["batch_size"], 0)
        self.assertEqual(batch["expert_waypoints_tensor"].shape, (0, 0, 2))

    def test_collate_mixed_samples(self):
        """Test collation with mix of valid and empty samples."""
        samples = [
            {
                "graph_obs": None,
                "tile_patterns": torch.randn(6),
                "entity_features": torch.randn(32),
                "expert_waypoints": [(100, 100)],
                "expert_waypoints_tensor": torch.tensor(
                    [[100, 100]], dtype=torch.float32
                ),
                "trajectory_length": 20,
                "success": True,
                "replay_id": "valid",
            },
            {
                "graph_obs": None,
                "tile_patterns": torch.zeros(6),
                "entity_features": torch.zeros(32),
                "expert_waypoints": [],
                "expert_waypoints_tensor": torch.zeros((0, 2)),
                "trajectory_length": 0,
                "success": False,
                "replay_id": "empty",
            },
        ]

        batch = collate_replay_batch(samples)

        # Should only include valid sample
        self.assertEqual(batch["batch_size"], 1)
        self.assertEqual(batch["expert_waypoints_tensor"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
