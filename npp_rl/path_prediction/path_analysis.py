"""Path analysis utilities for characterizing learned path properties.

This module provides post-hoc analysis of predicted paths, computing emergent
properties rather than using predefined categories. Properties are derived from
the actual path geometry, tile proximity, and entity interactions.

COORDINATE SYSTEM: All functions expect waypoints in TILE-DATA pixel coordinates
(no 24px border). This is the coordinate system used by graph nodes and training data.
- Range: X ∈ [0, 1056], Y ∈ [0, 600]
- NOT world coordinates (which include 24px border)
- NOT normalized [0, 1] coordinates
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def compute_path_properties(
    waypoints: List[Tuple[int, int]],
    tiles: Optional[np.ndarray] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    level_width: int = 1056,
    level_height: int = 600,
) -> Dict[str, float]:
    """Compute geometric and contextual properties of a path.

    This function analyzes a path and extracts quantitative properties that
    can be used to characterize the path's strategy without prescriptive labels.

    Args:
        waypoints: List of (x, y) coordinate tuples in TILE-DATA pixel space
                   (no border, range [0, 1056] x [0, 600])
        tiles: Optional 2D numpy array of tile IDs (23 rows x 42 cols for N++)
        entities: Optional list of entity dicts with 'type', 'x', 'y' keys
                  (should also be in tile-data coordinates)
        level_width: Width of level in pixels (default: 44*24 = 1056)
        level_height: Height of level in pixels (default: 25*24 = 600)

    Returns:
        Dictionary of path properties:
        - path_length: Total Euclidean distance traveled
        - straightness: Ratio of direct distance to path length (0-1)
        - vertical_bias: Ratio of vertical to total movement (0-1)
        - horizontal_bias: Ratio of horizontal to total movement (0-1)
        - curvature: Average angular change between segments
        - wall_contact_ratio: Fraction of waypoints near walls (0-1)
        - hazard_proximity: Average distance to nearest hazard (pixels)
        - movement_complexity: Variance in segment lengths (normalized)
        - coverage: Fraction of level area traversed (0-1)
        - endpoint_distance: Distance from start to end
    """
    if not waypoints or len(waypoints) < 2:
        return _empty_properties()

    properties = {}

    # Convert to numpy for easier computation
    waypoints_array = np.array(waypoints, dtype=np.float32)

    # Basic geometric properties
    segments = np.diff(waypoints_array, axis=0)  # [num_segments, 2]
    segment_lengths = np.linalg.norm(segments, axis=1)  # [num_segments]

    # Path length: total distance traveled
    properties["path_length"] = float(np.sum(segment_lengths))

    # Endpoint distance: direct distance from start to end
    endpoint_distance = np.linalg.norm(waypoints_array[-1] - waypoints_array[0])
    properties["endpoint_distance"] = float(endpoint_distance)

    # Straightness: how direct the path is (1.0 = perfectly straight)
    if properties["path_length"] > 0:
        properties["straightness"] = endpoint_distance / properties["path_length"]
    else:
        properties["straightness"] = 1.0

    # Vertical and horizontal bias
    vertical_dist = np.sum(np.abs(segments[:, 1]))  # Sum of vertical movements
    horizontal_dist = np.sum(np.abs(segments[:, 0]))  # Sum of horizontal movements
    total_dist = vertical_dist + horizontal_dist

    if total_dist > 0:
        properties["vertical_bias"] = vertical_dist / total_dist
        properties["horizontal_bias"] = horizontal_dist / total_dist
    else:
        properties["vertical_bias"] = 0.0
        properties["horizontal_bias"] = 0.0

    # Curvature: average angular change between consecutive segments
    if len(segments) >= 2:
        angles = []
        for i in range(len(segments) - 1):
            seg1 = segments[i]
            seg2 = segments[i + 1]

            # Compute angle between segments using dot product
            len1 = np.linalg.norm(seg1)
            len2 = np.linalg.norm(seg2)

            if len1 > 0 and len2 > 0:
                cos_angle = np.dot(seg1, seg2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)

        if angles:
            properties["curvature"] = float(np.mean(angles))
        else:
            properties["curvature"] = 0.0
    else:
        properties["curvature"] = 0.0

    # Movement complexity: variance in segment lengths (normalized by mean)
    if len(segment_lengths) > 1 and np.mean(segment_lengths) > 0:
        properties["movement_complexity"] = float(
            np.std(segment_lengths) / np.mean(segment_lengths)
        )
    else:
        properties["movement_complexity"] = 0.0

    # Coverage: approximate fraction of level area covered by path
    # Create bounding box and compute area ratio
    if len(waypoints_array) > 0:
        min_x, min_y = waypoints_array.min(axis=0)
        max_x, max_y = waypoints_array.max(axis=0)
        path_area = (max_x - min_x) * (max_y - min_y)
        level_area = level_width * level_height
        properties["coverage"] = float(
            min(path_area / level_area, 1.0) if level_area > 0 else 0.0
        )
    else:
        properties["coverage"] = 0.0

    # Wall contact ratio (requires tile data)
    if tiles is not None:
        properties["wall_contact_ratio"] = _compute_wall_contact_ratio(
            waypoints_array, tiles
        )
    else:
        properties["wall_contact_ratio"] = 0.0

    # Hazard proximity (requires entity data)
    if entities is not None:
        properties["hazard_proximity"] = _compute_hazard_proximity(
            waypoints_array, entities
        )
    else:
        properties["hazard_proximity"] = float("inf")

    return properties


def _empty_properties() -> Dict[str, float]:
    """Return empty/default properties for invalid paths."""
    return {
        "path_length": 0.0,
        "straightness": 0.0,
        "vertical_bias": 0.0,
        "horizontal_bias": 0.0,
        "curvature": 0.0,
        "wall_contact_ratio": 0.0,
        "hazard_proximity": float("inf"),
        "movement_complexity": 0.0,
        "coverage": 0.0,
        "endpoint_distance": 0.0,
    }


def _compute_wall_contact_ratio(waypoints: np.ndarray, tiles: np.ndarray) -> float:
    """Compute fraction of waypoints that are near walls.

    A waypoint is considered "near wall" if it's within 1 tile (24 pixels)
    of a solid tile.

    Args:
        waypoints: Array of waypoint positions [num_waypoints, 2] in tile-data pixels
        tiles: 2D array of tile IDs [height, width]

    Returns:
        Ratio of waypoints near walls (0.0 to 1.0)
    """
    if len(waypoints) == 0 or tiles is None or tiles.size == 0:
        return 0.0

    tile_height, tile_width = tiles.shape
    tile_size = 24  # N++ uses 24x24 pixel tiles

    near_wall_count = 0

    for waypoint in waypoints:
        x, y = waypoint

        # Convert pixel position to tile coordinates
        # Waypoints are already in tile-data space (no border offset needed)
        tile_x = int(x / tile_size)
        tile_y = int(y / tile_size)

        # Check surrounding tiles (3x3 grid)
        is_near_wall = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                check_x = tile_x + dx
                check_y = tile_y + dy

                # Check bounds
                if 0 <= check_x < tile_width and 0 <= check_y < tile_height:
                    tile_id = tiles[check_y, check_x]
                    # Tile IDs 0-6 are typically solid/collidable in N++
                    # 0 = empty, 1-6 = various solid types
                    if tile_id > 0:
                        is_near_wall = True
                        break

            if is_near_wall:
                break

        if is_near_wall:
            near_wall_count += 1

    return near_wall_count / len(waypoints)


def _compute_hazard_proximity(
    waypoints: np.ndarray, entities: List[Dict[str, Any]]
) -> float:
    """Compute average distance from waypoints to nearest hazard.

    Args:
        waypoints: Array of waypoint positions [num_waypoints, 2]
        entities: List of entity dicts with 'type', 'x', 'y' keys

    Returns:
        Average distance to nearest hazard in pixels
    """
    if len(waypoints) == 0 or not entities:
        return float("inf")

    # Extract hazard positions (mines, lasers, etc.)
    # N++ entity types: 1 = mine, 21 = toggle mine, etc.
    hazard_types = {1, 21, 15, 16, 17, 18}  # Mines, lasers, etc.
    hazard_positions = []

    for entity in entities:
        if entity.get("type") in hazard_types:
            hazard_positions.append((entity.get("x", 0), entity.get("y", 0)))

    if not hazard_positions:
        return float("inf")

    hazard_array = np.array(hazard_positions, dtype=np.float32)

    # For each waypoint, find distance to nearest hazard
    min_distances = []
    for waypoint in waypoints:
        distances = np.linalg.norm(hazard_array - waypoint, axis=1)
        min_distances.append(np.min(distances))

    return float(np.mean(min_distances))


def characterize_path_strategy(properties: Dict[str, float]) -> str:
    """Generate human-readable characterization from path properties.

    Uses decision tree based on computed properties to generate emergent labels.
    These labels are descriptive, not prescriptive.

    Args:
        properties: Dictionary of path properties from compute_path_properties

    Returns:
        Human-readable strategy label
    """
    # Empty or invalid path
    if properties.get("path_length", 0) < 10:
        return "minimal"

    # Characterize based on geometric properties
    straightness = properties.get("straightness", 0.0)
    wall_contact = properties.get("wall_contact_ratio", 0.0)
    hazard_prox = properties.get("hazard_proximity", float("inf"))
    vertical_bias = properties.get("vertical_bias", 0.0)
    curvature = properties.get("curvature", 0.0)

    # Build label from properties
    labels = []

    # Geometric characterization
    if straightness > 0.85:
        labels.append("direct")
    elif straightness < 0.5:
        labels.append("indirect")

    # Wall interaction
    if wall_contact > 0.6:
        labels.append("wall-heavy")
    elif wall_contact > 0.3:
        labels.append("wall-moderate")

    # Vertical vs horizontal
    if vertical_bias > 0.65:
        labels.append("climbing")
    elif vertical_bias < 0.35:
        labels.append("traversing")

    # Risk characterization
    if hazard_prox < 50:
        labels.append("risky")
    elif hazard_prox > 150:
        labels.append("cautious")

    # Path complexity
    if curvature > 1.5:
        labels.append("complex")
    elif curvature < 0.5:
        labels.append("simple")

    # Combine labels or use generic
    if labels:
        return "-".join(labels[:3])  # Limit to 3 descriptors
    else:
        return "standard"


def compute_path_diversity(
    paths: List[List[Tuple[int, int]]],
) -> Dict[str, float]:
    """Compute diversity metrics for a set of paths.

    Measures how different the paths are from each other using multiple metrics.

    Args:
        paths: List of paths, where each path is a list of (x, y) waypoints

    Returns:
        Dictionary with diversity metrics:
        - avg_endpoint_distance: Average pairwise distance between endpoints
        - avg_trajectory_divergence: Average pairwise trajectory difference
        - coverage_diversity: Variation in level area covered
    """
    if len(paths) < 2:
        return {
            "avg_endpoint_distance": 0.0,
            "avg_trajectory_divergence": 0.0,
            "coverage_diversity": 0.0,
        }

    endpoint_distances = []
    trajectory_divergences = []

    # Compute pairwise metrics
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path_i = paths[i]
            path_j = paths[j]

            if len(path_i) >= 2 and len(path_j) >= 2:
                # Endpoint distance
                end_i = np.array(path_i[-1])
                end_j = np.array(path_j[-1])
                endpoint_dist = np.linalg.norm(end_i - end_j)
                endpoint_distances.append(endpoint_dist)

                # Trajectory divergence: average distance between path points
                # Sample points uniformly from both paths
                num_samples = min(len(path_i), len(path_j), 10)
                if num_samples >= 2:
                    indices_i = np.linspace(0, len(path_i) - 1, num_samples, dtype=int)
                    indices_j = np.linspace(0, len(path_j) - 1, num_samples, dtype=int)

                    points_i = np.array([path_i[idx] for idx in indices_i])
                    points_j = np.array([path_j[idx] for idx in indices_j])

                    divergence = np.mean(np.linalg.norm(points_i - points_j, axis=1))
                    trajectory_divergences.append(divergence)

    # Coverage diversity: variance in path bounding box sizes
    coverage_areas = []
    for path in paths:
        if len(path) >= 2:
            path_array = np.array(path)
            min_coords = path_array.min(axis=0)
            max_coords = path_array.max(axis=0)
            area = np.prod(max_coords - min_coords)
            coverage_areas.append(area)

    return {
        "avg_endpoint_distance": float(np.mean(endpoint_distances))
        if endpoint_distances
        else 0.0,
        "avg_trajectory_divergence": float(np.mean(trajectory_divergences))
        if trajectory_divergences
        else 0.0,
        "coverage_diversity": float(np.std(coverage_areas))
        if len(coverage_areas) > 1
        else 0.0,
    }


def analyze_path_set(
    paths: List[List[Tuple[int, int]]],
    tiles: Optional[np.ndarray] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Analyze a set of paths and return properties and characterizations.

    Convenience function that computes properties and characterizations for
    multiple paths at once.

    Args:
        paths: List of paths (each path is list of waypoints)
        tiles: Optional tile array for wall contact analysis
        entities: Optional entity list for hazard analysis

    Returns:
        List of dictionaries, one per path, containing:
        - properties: Dict of computed properties
        - characterization: Human-readable label
        - path_index: Index in original list
    """
    results = []

    for idx, path in enumerate(paths):
        properties = compute_path_properties(path, tiles, entities)
        characterization = characterize_path_strategy(properties)

        results.append(
            {
                "path_index": idx,
                "properties": properties,
                "characterization": characterization,
            }
        )

    return results
