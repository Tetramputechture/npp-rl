#!/usr/bin/env python3
"""List available architectures for training."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.architecture_configs import (
    ARCHITECTURE_REGISTRY,
    get_architecture_config,
)


def main():
    """List all available architectures with descriptions."""
    parser = argparse.ArgumentParser(description="List available NPP-RL architectures")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Display detailed descriptions for each architecture",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Available NPP-RL Architectures")
    print("=" * 70)
    print()

    for i, arch_name in enumerate(sorted(ARCHITECTURE_REGISTRY.keys()), 1):
        try:
            config = get_architecture_config(arch_name)
            print(f"{i}. {arch_name}")
            print(f"   Description: {config.description}")

            if args.detailed:
                print(f"   Detailed Description: {config.detailed_description}")

            print(f"   Features dim: {config.features_dim}")

            # Show which modalities are enabled
            modalities = []
            if config.modalities.use_player_frame:
                modalities.append("player frame")
            if config.modalities.use_global_view:
                modalities.append("global view")
            if config.modalities.use_graph:
                modalities.append(f"graph ({config.graph.architecture.value})")
            if config.modalities.use_game_state:
                modalities.append("game state")
            if config.modalities.use_reachability:
                modalities.append("reachability")

            print(f"   Modalities: {', '.join(modalities)}")
            print()

        except Exception as e:
            print(f"{i}. {arch_name}")
            print(f"   Error loading config: {e}")
            print()

    print("=" * 70)
    print(f"Total architectures: {len(ARCHITECTURE_REGISTRY)}")
    print("=" * 70)
    print()
    print("Usage in train_and_compare.py:")
    print("  --architectures full_hgt vision_free gat")
    print()


if __name__ == "__main__":
    main()
