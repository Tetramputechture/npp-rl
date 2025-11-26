#!/usr/bin/env python3
"""Test dataset loading with DEBUG logging to identify hanging issues."""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from npp_rl.data.replay_dataset import PathReplayDataset

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test loading replays from dataset with debug logging."""

    replay_dir = "/home/tetra/projects/nclone/datasets/path-replays"

    logger.info(f"Testing dataset loading from {replay_dir}")
    logger.info("=" * 60)

    # Create dataset with minimal settings
    dataset = PathReplayDataset(
        replay_dir=replay_dir,
        waypoint_interval=10,  # Larger interval = fewer waypoints
        min_trajectory_length=10,  # Lower threshold
        enable_rendering=False,
        max_replays=5,  # Test 5 replays
    )

    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_dataset_loading()
