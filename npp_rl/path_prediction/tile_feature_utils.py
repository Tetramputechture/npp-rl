"""Utilities for extracting tile features consistently across training and inference.

This module ensures tile pattern features are created identically whether we're:
1. Training the path predictor from observations
2. Visualizing predictions from simulator
3. Running evaluation on levels
4. Using the predictor in RL training
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


def extract_tile_features_from_tiles(
    tiles: np.ndarray,
    target_dim: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract tile pattern features directly from tile array.

    This extracts statistical and structural features from the tile layout
    that are useful for path prediction.

    Features extracted:
    - Tile type distribution (frequency of each tile type)
    - Structural features (walls, platforms, slopes, etc.)
    - Density and connectivity metrics

    Args:
        tiles: 2D numpy array of tile IDs (height x width)
        target_dim: Target dimension for output feature vector (default 64)
        device: Device to place tensor on

    Returns:
        Feature tensor of shape [target_dim]
    """
    features = torch.zeros(target_dim, device=device)

    if tiles is None or tiles.size == 0:
        return features

    height, width = tiles.shape
    total_tiles = height * width

    # Feature 0-9: Tile type distribution (top 10 most important tile types)
    # Type 0: Empty, Type 1: Full solid, Types 2-5: Half tiles, etc.
    important_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i, tile_type in enumerate(important_types):
        if i < target_dim:
            count = np.sum(tiles == tile_type)
            features[i] = count / total_tiles

    # Feature 10-15: Structural features
    if target_dim > 10:
        # 10: Solid tile ratio (type 1)
        features[10] = np.sum(tiles == 1) / total_tiles

        # 11: Empty tile ratio (type 0)
        features[11] = np.sum(tiles == 0) / total_tiles

        # 12: Half tile ratio (types 2-5)
        half_tiles = np.isin(tiles, [2, 3, 4, 5])
        features[12] = np.sum(half_tiles) / total_tiles

        # 13: Slope tile ratio (types 6-9)
        slope_tiles = np.isin(tiles, [6, 7, 8, 9])
        features[13] = np.sum(slope_tiles) / total_tiles

        # 14: Curved tile ratio (types 10-17: quarter circles and pipes)
        curved_tiles = np.isin(tiles, range(10, 18))
        features[14] = np.sum(curved_tiles) / total_tiles

        # 15: Complex tile ratio (types 18+: mild/steep slopes)
        complex_tiles = tiles >= 18
        features[15] = np.sum(complex_tiles) / total_tiles

    # Feature 16-23: Edge detection (walls, platforms, boundaries)
    if target_dim > 16:
        # Detect horizontal edges (platform tops)
        horizontal_edges = 0
        for y in range(height - 1):
            for x in range(width):
                if tiles[y, x] == 0 and tiles[y + 1, x] != 0:  # Empty above solid
                    horizontal_edges += 1
        features[16] = horizontal_edges / total_tiles

        # Detect vertical edges (walls)
        vertical_edges = 0
        for y in range(height):
            for x in range(width - 1):
                if tiles[y, x] == 0 and tiles[y, x + 1] != 0:  # Empty next to solid
                    vertical_edges += 1
        features[17] = vertical_edges / total_tiles

        # Detect enclosed spaces (4-neighbor empty surrounded by solid)
        enclosed = 0
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if tiles[y, x] == 0:
                    neighbors_solid = (
                        (tiles[y - 1, x] != 0)
                        + (tiles[y + 1, x] != 0)
                        + (tiles[y, x - 1] != 0)
                        + (tiles[y, x + 1] != 0)
                    )
                    if neighbors_solid >= 3:
                        enclosed += 1
        features[18] = enclosed / total_tiles

        # Level complexity metrics
        unique_tiles = len(np.unique(tiles))
        features[19] = min(unique_tiles / 38.0, 1.0)  # Tile diversity (38 total types)

        # Vertical distribution (how spread out vertically)
        non_empty_rows = np.sum(np.any(tiles != 0, axis=1))
        features[20] = non_empty_rows / height

        # Horizontal distribution
        non_empty_cols = np.sum(np.any(tiles != 0, axis=0))
        features[21] = non_empty_cols / width

        # Top-heavy vs bottom-heavy (center of mass Y)
        y_indices, x_indices = np.where(tiles != 0)
        if len(y_indices) > 0:
            center_y = np.mean(y_indices) / height
            features[22] = center_y

        # Left-heavy vs right-heavy (center of mass X)
        if len(x_indices) > 0:
            center_x = np.mean(x_indices) / width
            features[23] = center_x

    # Feature 24-31: Local pattern statistics
    if target_dim > 24:
        # Count common local patterns (3x3 neighborhoods)
        # Open spaces (3x3 all empty)
        open_spaces = 0
        # Tunnel patterns (horizontal corridors)
        tunnels = 0
        # Platform patterns
        platforms = 0

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighborhood = tiles[y - 1 : y + 2, x - 1 : x + 2]

                # Open space: all empty
                if np.all(neighborhood == 0):
                    open_spaces += 1

                # Tunnel: solid above and below, empty in middle row
                if (
                    np.all(neighborhood[0, :] != 0)
                    and np.all(neighborhood[2, :] != 0)
                    and np.all(neighborhood[1, :] == 0)
                ):
                    tunnels += 1

                # Platform: empty above, solid below
                if np.all(neighborhood[0, :] == 0) and np.any(neighborhood[2, :] != 0):
                    platforms += 1

        total_neighborhoods = max((height - 2) * (width - 2), 1)
        features[24] = open_spaces / total_neighborhoods
        features[25] = tunnels / total_neighborhoods
        features[26] = platforms / total_neighborhoods

    # Feature 32-39: Reachability proxy (if available from observation)
    # These would be filled from observation data if available
    # For now, leave as zeros and will be filled by observation data

    return features


def extract_tile_features_from_observation(
    obs: Dict[str, Any],
    target_dim: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract tile features from RL observation dictionary.

    This combines tile array features with observation-specific features
    like reachability data.

    Args:
        obs: Observation dictionary from environment
        target_dim: Target dimension for output feature vector
        device: Device to place tensor on

    Returns:
        Feature tensor of shape [target_dim]
    """
    # Start with tile-based features if tiles are in observation
    if "tiles" in obs and obs["tiles"] is not None:
        tiles = obs["tiles"]
        if isinstance(tiles, torch.Tensor):
            tiles = tiles.cpu().numpy()
        features = extract_tile_features_from_tiles(tiles, target_dim, device)
    else:
        features = torch.zeros(target_dim, device=device)

    # Overlay reachability features if available (features 32-63)
    if "reachability_features" in obs:
        reachability = obs["reachability_features"]
        if isinstance(reachability, np.ndarray):
            reachability_tensor = torch.from_numpy(reachability).float().to(device)
        else:
            reachability_tensor = torch.tensor(
                reachability, dtype=torch.float32, device=device
            )

        # Copy reachability to second half of feature vector
        start_idx = min(32, target_dim // 2)
        n = min(len(reachability_tensor), target_dim - start_idx)
        features[start_idx : start_idx + n] = reachability_tensor[:n]

    return features


def extract_tile_features_batch(
    data_list: list,
    target_dim: int = 64,
    device: str = "cpu",
    from_observations: bool = True,
) -> torch.Tensor:
    """Extract tile features for a batch.

    Args:
        data_list: List of tile arrays or observation dicts
        target_dim: Target dimension for output feature vectors
        device: Device to place tensors on
        from_observations: If True, data_list contains observations; else tile arrays

    Returns:
        Feature tensor of shape [batch_size, target_dim]
    """
    batch_features = []

    for data in data_list:
        if from_observations:
            features = extract_tile_features_from_observation(data, target_dim, device)
        else:
            features = extract_tile_features_from_tiles(data, target_dim, device)
        batch_features.append(features)

    return torch.stack(batch_features)


def debug_print_tile_features(
    tile_features: torch.Tensor,
    tiles: Optional[np.ndarray] = None,
    name: str = "Tile Features",
):
    """Print debug information about tile features.

    Args:
        tile_features: Tile feature tensor
        tiles: Original tile array (optional)
        name: Name for this feature set in output
    """
    print(f"\n=== {name} ===")
    print(f"Shape: {tile_features.shape}")
    print(f"Device: {tile_features.device}")
    print(f"Non-zero elements: {torch.count_nonzero(tile_features).item()}")
    print(
        f"Min: {tile_features.min().item():.6f}, Max: {tile_features.max().item():.6f}"
    )
    print(f"Mean: {tile_features.mean().item():.6f}")

    print("\nFeature breakdown:")
    print(f"  [0-9]   Tile type distribution: {tile_features[:10].tolist()}")
    print(f"  [10-15] Structural features: {tile_features[10:16].tolist()}")
    print(f"  [16-23] Edge/complexity: {tile_features[16:24].tolist()}")
    print(f"  [24-31] Local patterns: {tile_features[24:32].tolist()}")
    print(f"  [32+]   Reachability/other: {tile_features[32:40].tolist()}")

    if tiles is not None:
        print("\nOriginal tiles:")
        print(f"  Shape: {tiles.shape}")
        print(f"  Unique tile types: {len(np.unique(tiles))}")
        print(f"  Empty tiles: {np.sum(tiles == 0)}/{tiles.size}")
        print(f"  Solid tiles: {np.sum(tiles == 1)}/{tiles.size}")
