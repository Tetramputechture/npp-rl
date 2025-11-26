#!/usr/bin/env python3
"""Quick test script to verify replay dataset loading works."""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from npp_rl.data.replay_dataset import PathReplayDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test loading replays from dataset."""

    replay_dir = "/home/tetra/projects/nclone/datasets/path-replays"

    logger.info(f"Testing dataset loading from {replay_dir}")

    # Create dataset with minimal settings
    dataset = PathReplayDataset(
        replay_dir=replay_dir,
        waypoint_interval=10,  # Larger interval = fewer waypoints
        min_trajectory_length=10,  # Lower threshold
        enable_rendering=False,
        max_replays=3,  # Just test 3 replays
    )

    logger.info(f"Dataset created with {len(dataset)} replays")

    # Try loading first sample
    logger.info("\n" + "=" * 60)
    logger.info("Loading sample 0...")
    logger.info("=" * 60)

    sample = dataset[0]

    logger.info("\nSample 0 loaded successfully!")
    logger.info(f"  Trajectory length: {sample['trajectory_length']}")
    logger.info(f"  Success: {sample['success']}")
    logger.info(f"  Num waypoints: {len(sample['expert_waypoints'])}")
    logger.info(f"  Waypoint tensor shape: {sample['expert_waypoints_tensor'].shape}")
    logger.info(f"  Tile patterns shape: {sample['tile_patterns'].shape}")
    logger.info(f"  Entity features shape: {sample['entity_features'].shape}")

    # Try loading second sample
    logger.info("\n" + "=" * 60)
    logger.info("Loading sample 1...")
    logger.info("=" * 60)

    sample = dataset[1]

    logger.info("\nSample 1 loaded successfully!")
    logger.info(f"  Trajectory length: {sample['trajectory_length']}")
    logger.info(f"  Num waypoints: {len(sample['expert_waypoints'])}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Dataset loading test passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_dataset_loading()
