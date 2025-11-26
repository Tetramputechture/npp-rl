#!/usr/bin/env python3
"""Quick test to verify feature dimensions match model expectations."""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from npp_rl.data.replay_dataset import PathReplayDataset
from npp_rl.path_prediction.multipath_predictor import create_multipath_predictor

print("Testing feature dimensions...")

# Create dataset with one replay
dataset = PathReplayDataset(
    replay_dir="/home/tetra/projects/nclone/datasets/path-replays",
    max_replays=1,
    waypoint_interval=10,
)

print(f"\nDataset created with {len(dataset)} replays")

# Load one sample
sample = dataset[0]

print("\nSample features:")
print(f"  Tile patterns shape: {sample['tile_patterns'].shape}")
print(f"  Entity features shape: {sample['entity_features'].shape}")
print("  Expected: tile=64, entity=32")

# Create predictor
predictor_config = {
    "graph_feature_dim": 256,
    "tile_pattern_dim": 64,
    "entity_feature_dim": 32,
    "num_path_candidates": 4,
    "max_waypoints": 20,
    "hidden_dim": 512,
}

predictor = create_multipath_predictor(predictor_config)
print("\nPredictor created")

# Test forward pass
batch_size = 2
graph_obs = torch.randn(batch_size, 256)
tile_patterns = torch.randn(batch_size, 64)
entity_features = torch.randn(batch_size, 32)

print(f"\nTesting forward pass with batch_size={batch_size}")
print(f"  graph_obs: {graph_obs.shape}")
print(f"  tile_patterns: {tile_patterns.shape}")
print(f"  entity_features: {entity_features.shape}")

try:
    outputs = predictor(
        graph_obs=graph_obs,
        tile_patterns=tile_patterns,
        entity_features=entity_features,
    )
    print("\n✓ Forward pass successful!")
    print(f"  Output waypoints shape: {outputs['waypoints'].shape}")
    print(f"  Output confidences shape: {outputs['confidences'].shape}")
except Exception as e:
    print(f"\n✗ Forward pass failed: {e}")
    sys.exit(1)

print("\n✓ All dimension tests passed!")
