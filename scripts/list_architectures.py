#!/usr/bin/env python3
"""List available architectures for training."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.optimization.architecture_configs import (
    ARCHITECTURE_REGISTRY,
    get_architecture_config
)


def main():
    """List all available architectures with descriptions."""
    print("=" * 70)
    print("Available NPP-RL Architectures")
    print("=" * 70)
    print()
    
    for i, arch_name in enumerate(sorted(ARCHITECTURE_REGISTRY.keys()), 1):
        try:
            config = get_architecture_config(arch_name)
            print(f"{i}. {arch_name}")
            print(f"   Description: {config.name}")
            print(f"   Features dim: {config.features_dim}")
            
            # Show which modalities are enabled
            modalities = []
            if config.use_temporal_cnn:
                modalities.append("temporal frames")
            if config.use_global_cnn:
                modalities.append("global view")
            if config.use_graph:
                modalities.append(f"graph ({config.graph_type})")
            if config.use_game_state:
                modalities.append("game state")
            if config.use_reachability:
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


if __name__ == '__main__':
    main()
