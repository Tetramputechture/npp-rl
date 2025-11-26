#!/usr/bin/env python3
"""Script to help collect diverse replay data for path prediction training.

This script provides utilities and guidance for collecting high-quality replay data
with sufficient diversity for training a robust path predictor.

Key strategies:
1. Record replays from multiple map types/difficulties
2. Include both successful completions and failures
3. Ensure varied path lengths and complexities
4. Label replays with metadata for stratified sampling

Usage:
    # Interactive recording with existing agent
    python scripts/collect_replays_diverse.py \
        --mode agent \
        --agent-path trained_agents/ppo_latest/best_model.zip \
        --output-dir datasets/path-replays-diverse \
        --num-episodes 500 \
        --max-steps 1000
    
    # Manual recording (requires human gameplay)
    python scripts/collect_replays_diverse.py \
        --mode manual \
        --maps-dir ../nclone/nclone/maps/test-maps \
        --output-dir datasets/path-replays-manual \
        --num-episodes 100
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project roots to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
nclone_root = project_root.parent / "nclone"
sys.path.insert(0, str(nclone_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect diverse replay data for path prediction training"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["agent", "manual", "info"],
        default="info",
        help="Collection mode: agent (use trained agent), manual (human gameplay), info (show guidance)",
    )
    parser.add_argument(
        "--agent-path",
        type=str,
        help="Path to trained agent model (.zip file) for agent mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/path-replays-diverse",
        help="Output directory for replay files",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=500,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--maps-dir",
        type=str,
        help="Directory containing maps for manual mode",
    )
    
    return parser.parse_args()


def show_collection_guidance():
    """Display guidance on replay collection strategies."""
    print("=" * 80)
    print("REPLAY COLLECTION GUIDANCE FOR PATH PREDICTION TRAINING")
    print("=" * 80)
    print()
    print("TARGET: Collect 500-1000 high-quality, diverse replays")
    print()
    print("RECOMMENDED DATA COMPOSITION:")
    print("  - 60-70% successful completions (reached exit)")
    print("  - 30-40% partial completions (made progress but failed)")
    print("  - Mix of difficulty levels")
    print("  - Varied path lengths (100-1000 steps)")
    print("  - Multiple map generators/types")
    print()
    print("QUALITY CRITERIA:")
    print("  ✓ Minimum path length: 20 steps (5 waypoints with interval=5)")
    print("  ✓ Clear progress toward goal")
    print("  ✓ Valid player movements (no stuck/glitched states)")
    print("  ✓ Diverse spatial coverage of map")
    print()
    print("COLLECTION METHODS:")
    print()
    print("Method 1: Use Existing Trained Agent")
    print("  - Fastest method for large-scale collection")
    print("  - Run existing RL agent on diverse maps")
    print("  - Command:")
    print("    python scripts/collect_replays_diverse.py \\")
    print("        --mode agent \\")
    print("        --agent-path trained_agents/ppo_latest/best_model.zip \\")
    print("        --output-dir datasets/path-replays \\")
    print("        --num-episodes 500")
    print()
    print("Method 2: Record Human Gameplay")
    print("  - Higher quality but slower")
    print("  - Play N++ levels while recording")
    print("  - Use built-in recording from base_environment.py")
    print()
    print("Method 3: Use Existing Replay Datasets")
    print("  - Check if you already have replays from RL training")
    print("  - Location: datasets/replays or similar")
    print("  - Filter for quality and diversity")
    print()
    print("DATA AUGMENTATION:")
    print("  - Horizontal flipping (automatic, 2x multiplier)")
    print("  - Waypoint jitter (automatic)")
    print("  - Effective dataset size = actual replays × 2-3")
    print("  - With 500 replays + augmentation ≈ 1000-1500 effective samples")
    print()
    print("VERIFICATION:")
    print("  After collection, verify data quality:")
    print("    python scripts/verify_replay_dataset.py \\")
    print("        --replay-dir datasets/path-replays \\")
    print("        --min-trajectory-length 20 \\")
    print("        --report-path replay_quality_report.json")
    print()
    print("=" * 80)
    print()
    print("NEXT STEPS:")
    print("  1. Collect replays using one of the methods above")
    print("  2. Verify data quality")
    print("  3. Train path predictor:")
    print("     python scripts/train_path_predictor.py \\")
    print("         --replay-dir datasets/path-replays \\")
    print("         --output-dir path-pred-trained \\")
    print("         --num-epochs 100 \\")
    print("         --batch-size 64")
    print("=" * 80)


def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == "info":
        show_collection_guidance()
        return
    
    if args.mode == "agent":
        logger.error("Agent-based collection not yet implemented")
        logger.info("For now, use existing replay datasets from RL training")
        logger.info("Example replay locations:")
        logger.info("  - datasets/replays/")
        logger.info("  - ../nclone/datasets/expert-demos/")
        logger.info("")
        logger.info("Or record manually using NPlayHeadless with recording enabled")
        return
    
    if args.mode == "manual":
        logger.error("Manual collection not yet implemented")
        logger.info("To record manually:")
        logger.info("1. Use the base environment with recording enabled")
        logger.info("2. Play levels via the environment")
        logger.info("3. Replays will be saved automatically")
        return


if __name__ == "__main__":
    main()

