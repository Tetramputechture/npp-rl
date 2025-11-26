"""Graph-based path operation utilities for validating and computing paths.

This module provides utilities to integrate neural network path predictions with
the tile-accurate adjacency graph from GraphBuilder, ensuring predicted paths
are physically traversable.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import heapq

logger = logging.getLogger(__name__)

# Lazy-load subcell lookup loader (singleton)
_subcell_lookup_loader = None


def _get_subcell_lookup_loader():
    """Get or initialize the SubcellNodeLookupLoader singleton."""
    global _subcell_lookup_loader
    if _subcell_lookup_loader is None:
        try:
            from nclone.graph.reachability.subcell_node_lookup import (
                SubcellNodeLookupLoader,
            )

            _subcell_lookup_loader = SubcellNodeLookupLoader()
        except Exception as e:
            logger.warning(f"Could not load SubcellNodeLookupLoader: {e}")
            _subcell_lookup_loader = None
    return _subcell_lookup_loader


def snap_position_to_graph_node(
    position: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Any,
    threshold: int = 12,
) -> Optional[Tuple[int, int]]:
    """Snap pixel position to nearest graph node within threshold.

    Uses precomputed subcell lookup table for O(1) performance.
    Falls back to grid-based snapping if lookup table unavailable.

    Args:
        position: Pixel position (x, y)
        adjacency: Graph adjacency dictionary
        spatial_hash: SpatialHash for fast node lookups (unused, kept for API compatibility)
        threshold: Maximum distance in pixels to snap (default: 12px = 1 sub-node)

    Returns:
        Nearest graph node position, or None if no node within threshold
    """
    if not adjacency:
        return None

    px, py = position

    # Try using precomputed subcell lookup table (O(1))
    loader = _get_subcell_lookup_loader()
    if loader:
        try:
            return loader.find_closest_node_position(
                float(px), float(py), adjacency, max_radius=float(threshold)
            )
        except Exception as e:
            logger.debug(f"Subcell lookup failed, falling back to grid snapping: {e}")

    # Fallback: Manual grid-based snapping
    # Sub-nodes are on 12px grid at positions: ..., -18, -6, 6, 18, 30, 42, ...
    snap_x = round((px - 6) / 12) * 12 + 6
    snap_y = round((py - 6) / 12) * 12 + 6

    # Check if snapped position exists in graph
    if (snap_x, snap_y) in adjacency:
        return (snap_x, snap_y)

    # Check nearby sub-nodes within threshold
    threshold_sq = threshold * threshold
    candidates = []

    # Check 3x3 grid of sub-nodes around snapped position
    for dx in [-12, 0, 12]:
        for dy in [-12, 0, 12]:
            candidate = (snap_x + dx, snap_y + dy)
            if candidate in adjacency:
                dist_sq = (candidate[0] - px) ** 2 + (candidate[1] - py) ** 2
                if dist_sq <= threshold_sq:
                    candidates.append((candidate, dist_sq))

    if not candidates:
        return None

    # Return closest candidate
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def validate_path_on_graph(
    waypoints: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Any,
) -> Tuple[bool, float]:
    """Check if waypoints form valid graph path.

    Args:
        waypoints: List of waypoint positions
        adjacency: Graph adjacency dictionary
        spatial_hash: SpatialHash for fast lookups

    Returns:
        Tuple of (is_valid, graph_distance)
        - is_valid: True if all consecutive waypoints are graph-connected
        - graph_distance: Total graph distance (0.0 if invalid)
    """
    if not waypoints or len(waypoints) < 2:
        return True, 0.0

    if not adjacency:
        return False, 0.0

    total_distance = 0.0

    for i in range(len(waypoints) - 1):
        src = waypoints[i]
        dst = waypoints[i + 1]

        # Check if source node exists
        if src not in adjacency:
            return False, 0.0

        # Check if destination is a neighbor of source
        neighbors = adjacency.get(src, [])
        found = False
        for neighbor_pos, cost in neighbors:
            if neighbor_pos == dst:
                total_distance += cost
                found = True
                break

        if not found:
            # Not directly connected, path is invalid
            return False, 0.0

    return True, total_distance


def compute_graph_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Any,
) -> Optional[List[Tuple[int, int]]]:
    """Compute shortest graph path between nodes using Dijkstra's algorithm.

    Args:
        start: Start node position
        end: End node position
        adjacency: Graph adjacency dictionary
        spatial_hash: SpatialHash for fast lookups

    Returns:
        List of node positions forming shortest path, or None if unreachable
    """
    if not adjacency or start not in adjacency or end not in adjacency:
        return None

    if start == end:
        return [start]

    # Dijkstra's algorithm with priority queue
    # Priority queue: (distance, node)
    pq = [(0.0, start)]
    distances = {start: 0.0}
    previous = {start: None}
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        # Found target
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = previous.get(node)
            path.reverse()
            return path

        # Explore neighbors
        for neighbor_pos, edge_cost in adjacency.get(current, []):
            if neighbor_pos in visited:
                continue

            new_dist = current_dist + edge_cost

            if neighbor_pos not in distances or new_dist < distances[neighbor_pos]:
                distances[neighbor_pos] = new_dist
                previous[neighbor_pos] = current
                heapq.heappush(pq, (new_dist, neighbor_pos))

    # No path found
    return None


def compute_graph_distance(
    waypoints: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Any,
) -> float:
    """Compute actual graph distance for waypoint sequence.

    For consecutive waypoints that are not directly connected, computes
    shortest path between them and sums the total distance.

    Args:
        waypoints: List of waypoint positions
        adjacency: Graph adjacency dictionary
        spatial_hash: SpatialHash for fast lookups

    Returns:
        Total graph distance, or infinity if path is impossible
    """
    if not waypoints or len(waypoints) < 2:
        return 0.0

    if not adjacency:
        return float("inf")

    total_distance = 0.0

    for i in range(len(waypoints) - 1):
        src = waypoints[i]
        dst = waypoints[i + 1]

        # Check if directly connected
        if src in adjacency:
            neighbors = adjacency.get(src, [])
            direct_cost = None
            for neighbor_pos, cost in neighbors:
                if neighbor_pos == dst:
                    direct_cost = cost
                    break

            if direct_cost is not None:
                total_distance += direct_cost
                continue

        # Not directly connected, need to find shortest path
        path = compute_graph_path(src, dst, adjacency, spatial_hash)
        if path is None:
            return float("inf")  # Unreachable

        # Compute distance along path
        for j in range(len(path) - 1):
            path_src = path[j]
            path_dst = path[j + 1]
            neighbors = adjacency.get(path_src, [])
            for neighbor_pos, cost in neighbors:
                if neighbor_pos == path_dst:
                    total_distance += cost
                    break

    return total_distance


def extract_graph_from_env(env) -> Tuple[Optional[Dict], Optional[Any]]:
    """Extract adjacency and spatial_hash from environment.

    Args:
        env: Gym environment instance (must have get_graph_data method)

    Returns:
        Tuple of (adjacency, spatial_hash), or (None, None) if unavailable
    """
    if not hasattr(env, "get_graph_data"):
        logger.warning("Environment does not have get_graph_data method")
        return None, None

    try:
        graph_data = env.get_graph_data()
        if graph_data is None:
            return None, None

        adjacency = graph_data.get("adjacency")
        spatial_hash = graph_data.get("spatial_hash")

        return adjacency, spatial_hash
    except Exception as e:
        logger.error(f"Failed to extract graph from environment: {e}")
        return None, None


def repair_path_with_graph(
    waypoints: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Any,
    snap_threshold: int = 24,
) -> List[Tuple[int, int]]:
    """Repair invalid path by snapping to graph and inserting intermediate nodes.

    Args:
        waypoints: List of waypoint positions (may be invalid)
        adjacency: Graph adjacency dictionary
        spatial_hash: SpatialHash for fast lookups
        snap_threshold: Maximum distance to snap waypoints to graph nodes

    Returns:
        Repaired path as list of graph nodes
    """
    if not waypoints or not adjacency:
        return []

    # Step 1: Snap all waypoints to nearest graph nodes
    snapped = []
    for wp in waypoints:
        node = snap_position_to_graph_node(wp, adjacency, spatial_hash, snap_threshold)
        if node and (not snapped or node != snapped[-1]):  # Avoid duplicates
            snapped.append(node)

    if len(snapped) < 2:
        return snapped

    # Step 2: Repair gaps by inserting graph path segments
    repaired = [snapped[0]]

    for i in range(1, len(snapped)):
        src = snapped[i - 1]
        dst = snapped[i]

        # Check if directly connected
        if src in adjacency:
            neighbors = adjacency.get(src, [])
            is_neighbor = any(neighbor_pos == dst for neighbor_pos, _ in neighbors)

            if is_neighbor:
                repaired.append(dst)
                continue

        # Not directly connected, find shortest path
        path = compute_graph_path(src, dst, adjacency, spatial_hash)
        if path and len(path) > 1:
            repaired.extend(path[1:])  # Skip first node (already in repaired)
        else:
            # Can't connect, skip this waypoint
            logger.debug(
                f"Could not find path between {src} and {dst}, skipping waypoint"
            )
            continue

    return repaired
