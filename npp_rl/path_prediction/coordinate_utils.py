"""Coordinate transformation utilities for path prediction.

This module provides centralized coordinate transformation functions to ensure
consistency across training, inference, and visualization.

N++ has three coordinate systems:

1. **World coordinates**: Include 24px border around playable area
   - Used by: Simulator, rendering
   - Range: X ∈ [0, 1104], Y ∈ [0, 648] (includes border)
   - Origin: Top-left of bordered area

2. **Tile-data coordinates**: No border, just playable area
   - Used by: Graph nodes, training data, waypoints
   - Range: X ∈ [0, 1056], Y ∈ [0, 600] (no border)
   - Origin: Top-left of playable area
   - Conversion: tile_data = world - NODE_WORLD_COORD_OFFSET

3. **Normalized coordinates**: Neural network input/output space
   - Used by: Model predictions, loss functions
   - Range: X ∈ [0, 1], Y ∈ [0, 1]
   - Conversion: normalized = tile_data / (WIDTH, HEIGHT)

Graph nodes are positioned on a 12px sub-tile grid:
- X positions: ..., -18, -6, 6, 18, 30, 42, 54, 66, ...
- Y positions: ..., -18, -6, 6, 18, 30, 42, 54, 66, ...
- Pattern: 6 + 12*k for integer k
"""

import numpy as np
import torch
from typing import Tuple, List, Union

# N++ level dimensions (interior playable area in pixels)
LEVEL_WIDTH_PIXELS = 44 * 24  # 1056 pixels (44 tiles × 24 pixels/tile)
LEVEL_HEIGHT_PIXELS = 25 * 24  # 600 pixels (25 tiles × 24 pixels/tile)

# World coordinate offset (1 tile = 24 pixels border around playable area)
NODE_WORLD_COORD_OFFSET = 24

# Graph node sub-tile grid spacing
GRAPH_NODE_GRID_SPACING = 12  # pixels


def world_to_tile_data(
    x: Union[float, int], y: Union[float, int]
) -> Tuple[float, float]:
    """Convert world coordinates to tile-data coordinates.

    World coordinates include a 24px border; tile-data coordinates do not.

    Args:
        x: X position in world coordinates
        y: Y position in world coordinates

    Returns:
        Tuple of (x, y) in tile-data coordinates
    """
    return (
        float(x) - NODE_WORLD_COORD_OFFSET,
        float(y) - NODE_WORLD_COORD_OFFSET,
    )


def tile_data_to_world(
    x: Union[float, int], y: Union[float, int]
) -> Tuple[float, float]:
    """Convert tile-data coordinates to world coordinates.

    Tile-data coordinates are just the playable area; world coordinates
    include a 24px border.

    Args:
        x: X position in tile-data coordinates
        y: Y position in tile-data coordinates

    Returns:
        Tuple of (x, y) in world coordinates
    """
    return (
        float(x) + NODE_WORLD_COORD_OFFSET,
        float(y) + NODE_WORLD_COORD_OFFSET,
    )


def normalize_coords(
    x: Union[float, int],
    y: Union[float, int],
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> Tuple[float, float]:
    """Normalize pixel coordinates to [0, 1] range.

    Args:
        x: X position in pixels (tile-data coordinates)
        y: Y position in pixels (tile-data coordinates)
        width: Level width in pixels (default: 1056)
        height: Level height in pixels (default: 600)

    Returns:
        Tuple of (x_norm, y_norm) in [0, 1] range
    """
    return (float(x) / width, float(y) / height)


def denormalize_coords(
    x_norm: float,
    y_norm: float,
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> Tuple[float, float]:
    """Denormalize [0, 1] coordinates back to pixel coordinates.

    Args:
        x_norm: Normalized X coordinate in [0, 1]
        y_norm: Normalized Y coordinate in [0, 1]
        width: Level width in pixels (default: 1056)
        height: Level height in pixels (default: 600)

    Returns:
        Tuple of (x, y) in tile-data pixel coordinates
    """
    return (x_norm * width, y_norm * height)


def normalize_waypoints(
    waypoints: List[Tuple[Union[float, int], Union[float, int]]],
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> List[Tuple[float, float]]:
    """Normalize a list of waypoints from pixels to [0, 1].

    Args:
        waypoints: List of (x, y) tuples in tile-data pixel coordinates
        width: Level width in pixels
        height: Level height in pixels

    Returns:
        List of (x_norm, y_norm) tuples in [0, 1] range
    """
    return [normalize_coords(x, y, width, height) for x, y in waypoints]


def denormalize_waypoints(
    waypoints: List[Tuple[float, float]],
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> List[Tuple[float, float]]:
    """Denormalize a list of waypoints from [0, 1] to pixels.

    Args:
        waypoints: List of (x_norm, y_norm) tuples in [0, 1] range
        width: Level width in pixels
        height: Level height in pixels

    Returns:
        List of (x, y) tuples in tile-data pixel coordinates
    """
    return [denormalize_coords(x, y, width, height) for x, y in waypoints]


def denormalize_waypoints_to_world(
    waypoints: List[Tuple[float, float]],
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> List[Tuple[float, float]]:
    """Denormalize waypoints from [0, 1] to world pixel coordinates.

    This is a convenience function that combines denormalization and
    tile-data to world coordinate conversion for visualization.

    Args:
        waypoints: List of (x_norm, y_norm) tuples in [0, 1] range
        width: Level width in pixels
        height: Level height in pixels

    Returns:
        List of (x, y) tuples in world pixel coordinates (for rendering)
    """
    # First denormalize to tile-data coordinates
    tile_data_waypoints = denormalize_waypoints(waypoints, width, height)
    # Then convert to world coordinates
    return [tile_data_to_world(x, y) for x, y in tile_data_waypoints]


def validate_normalized_coords(
    waypoints: Union[List[Tuple[float, float]], torch.Tensor, np.ndarray],
    tolerance: float = 0.1,
) -> Tuple[bool, str]:
    """Validate that coordinates are in expected normalized [0, 1] range.

    Args:
        waypoints: Waypoints to validate (list, tensor, or array)
        tolerance: Allow slight extrapolation beyond [0, 1] (default: 0.1)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if isinstance(waypoints, torch.Tensor):
        waypoints_array = waypoints.detach().cpu().numpy()
    elif isinstance(waypoints, np.ndarray):
        waypoints_array = waypoints
    else:
        waypoints_array = np.array(waypoints)

    if waypoints_array.size == 0:
        return True, "Empty waypoints (valid)"

    min_val = waypoints_array.min()
    max_val = waypoints_array.max()

    if min_val < -tolerance or max_val > 1.0 + tolerance:
        return (
            False,
            f"Coordinates out of range: min={min_val:.3f}, max={max_val:.3f} "
            f"(expected ~[0, 1] with tolerance={tolerance})",
        )

    return True, f"Valid normalized coordinates: [{min_val:.3f}, {max_val:.3f}]"


def validate_pixel_coords(
    waypoints: Union[List[Tuple[float, float]], torch.Tensor, np.ndarray],
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
    tolerance: float = 50.0,
) -> Tuple[bool, str]:
    """Validate that coordinates are in expected pixel range.

    Args:
        waypoints: Waypoints to validate (list, tensor, or array)
        width: Expected level width in pixels
        height: Expected level height in pixels
        tolerance: Allow pixels outside bounds (default: 50px)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if isinstance(waypoints, torch.Tensor):
        waypoints_array = waypoints.detach().cpu().numpy()
    elif isinstance(waypoints, np.ndarray):
        waypoints_array = waypoints
    else:
        waypoints_array = np.array(waypoints)

    if waypoints_array.size == 0:
        return True, "Empty waypoints (valid)"

    if waypoints_array.ndim == 1:
        waypoints_array = waypoints_array.reshape(-1, 2)

    xs = waypoints_array[:, 0]
    ys = waypoints_array[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if (
        x_min < -tolerance
        or x_max > width + tolerance
        or y_min < -tolerance
        or y_max > height + tolerance
    ):
        return (
            False,
            f"Coordinates out of pixel range: X=[{x_min:.1f}, {x_max:.1f}], "
            f"Y=[{y_min:.1f}, {y_max:.1f}] (expected X=[0, {width}], Y=[0, {height}] "
            f"with tolerance={tolerance}px)",
        )

    return (
        True,
        f"Valid pixel coordinates: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]",
    )


def snap_to_graph_grid(x: float, y: float) -> Tuple[int, int]:
    """Snap coordinates to graph node sub-tile grid.

    Graph nodes are positioned at: 6 + 12*k for integer k
    Examples: ..., -18, -6, 6, 18, 30, 42, 54, ...

    Args:
        x: X coordinate in pixels (tile-data space)
        y: Y coordinate in pixels (tile-data space)

    Returns:
        Tuple of (x_snap, y_snap) snapped to nearest grid point
    """
    # Graph grid is at: 6 + 12*k = 12*k + 6
    # To snap: round((x - 6) / 12) * 12 + 6
    snap_x = round((x - 6) / GRAPH_NODE_GRID_SPACING) * GRAPH_NODE_GRID_SPACING + 6
    snap_y = round((y - 6) / GRAPH_NODE_GRID_SPACING) * GRAPH_NODE_GRID_SPACING + 6
    return (int(snap_x), int(snap_y))


def convert_pixel_distance_to_normalized(
    distance_pixels: float,
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> float:
    """Convert pixel distance to normalized [0, 1] space distance.

    Uses Euclidean distance in normalized space as reference.
    For a distance D in pixels, the equivalent in normalized space
    depends on the level dimensions.

    Args:
        distance_pixels: Distance in pixels
        width: Level width in pixels
        height: Level height in pixels

    Returns:
        Approximate equivalent distance in normalized space
    """
    # Average the X and Y scaling factors
    # This is approximate but works well for isotropic distances
    x_scale = 1.0 / width
    y_scale = 1.0 / height
    avg_scale = (x_scale + y_scale) / 2.0
    return distance_pixels * avg_scale


def convert_normalized_distance_to_pixel(
    distance_normalized: float,
    width: float = LEVEL_WIDTH_PIXELS,
    height: float = LEVEL_HEIGHT_PIXELS,
) -> float:
    """Convert normalized [0, 1] space distance to pixel distance.

    Args:
        distance_normalized: Distance in normalized space
        width: Level width in pixels
        height: Level height in pixels

    Returns:
        Approximate equivalent distance in pixels
    """
    # Average the X and Y scaling factors
    avg_scale = (width + height) / 2.0
    return distance_normalized * avg_scale


# Convenience constants for common conversions
NORMALIZED_50_PIXELS = convert_pixel_distance_to_normalized(
    50.0
)  # ~0.047 for 1056px width
NORMALIZED_100_PIXELS = convert_pixel_distance_to_normalized(
    100.0
)  # ~0.095 for 1056px width
NORMALIZED_200_PIXELS = convert_pixel_distance_to_normalized(
    200.0
)  # ~0.19 for 1056px width

