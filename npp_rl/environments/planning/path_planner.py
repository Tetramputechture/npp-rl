import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from queue import PriorityQueue
import math


@dataclass
class Node:
    x: int
    y: int
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = float('inf')  # Heuristic cost to goal
    parent: Optional['Node'] = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost


class PathPlanner:
    """High-level path planner using A* with dynamic replanning capabilities.

    This planner operates on a grid representation of the level, using tile data
    to identify traversable spaces. It generates optimal paths considering:
    1. Static obstacles (walls, spikes)
    2. Required waypoints (switch before exit)
    3. Physics-based movement constraints
    4. Dynamic replanning when new obstacles are discovered
    """

    def __init__(self, grid_size: int = 24):
        self.grid_size = grid_size
        self.tile_data = None
        self.segment_data = None
        self.grid_edges = None
        self.segment_edges = None
        self.current_path: List[Node] = []
        self.visited_nodes: Set[Tuple[int, int]] = set()
        self.clearance_radius = 12  # Required clearance in pixels
        self.cell_subdivisions = 2  # Each cell is divided into subcells for finer pathfinding

        # Cache for collision checks
        self._collision_cache = {}

    def update_collision_grid(self, tile_data: Dict, segment_data: Dict,
                              grid_edges: Dict, segment_edges: Dict):
        """Update internal collision data from simulator data structures.

        Args:
            tile_data: Dictionary mapping (x,y) coordinates to tile IDs
            segment_data: Dictionary mapping (x,y) coordinates to segments
            grid_edges: Dictionary containing horizontal and vertical grid edges
            segment_edges: Dictionary containing horizontal and vertical segment edges
        """
        self.tile_data = tile_data
        self.segment_data = segment_data
        self.grid_edges = grid_edges
        self.segment_edges = segment_edges

        # Clear collision cache
        self._collision_cache.clear()

        # Get grid dimensions from tile data
        if tile_data:
            max_x = max(x for x, _ in tile_data.keys())
            max_y = max(y for _, y in tile_data.keys())
            self.grid_dimensions = (max_x + 1, max_y + 1)
        else:
            self.grid_dimensions = (0, 0)

    def is_path_clear(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Check if a direct path between two points has sufficient clearance."""
        if not self.tile_data:
            return False

        # Convert to pixel coordinates
        x1, y1 = start
        x2, y2 = end

        # Calculate path vector
        dx = x2 - x1
        dy = y2 - y1
        path_length = math.sqrt(dx*dx + dy*dy)

        if path_length == 0:
            return True

        # Normalize direction
        dx /= path_length
        dy /= path_length

        # Check points along path with clearance
        steps = int(path_length / (self.clearance_radius/2))
        for i in range(steps + 1):
            t = i * (self.clearance_radius/2)
            check_x = x1 + dx * t
            check_y = y1 + dy * t

            if not self.has_clearance((check_x, check_y)):
                return False

        return True

    def has_clearance(self, point: Tuple[float, float]) -> bool:
        """Check if a point has sufficient clearance from obstacles."""
        x, y = point
        cell_x = int(x / self.grid_size)
        cell_y = int(y / self.grid_size)

        # Check cache
        cache_key = (cell_x, cell_y)
        if cache_key in self._collision_cache:
            return self._collision_cache[cache_key]

        # Check if cell contains solid tile
        if (cell_x, cell_y) in self.tile_data:
            tile_id = self.tile_data[(cell_x, cell_y)]
            # Tile ID 1 is a solid tile in N++
            if tile_id == 1:
                self._collision_cache[cache_key] = False
                return False

        # Check segments in surrounding cells
        radius_cells = math.ceil(self.clearance_radius / self.grid_size)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                check_x = cell_x + dx
                check_y = cell_y + dy

                # Check segments
                if (check_x, check_y) in self.segment_data and self.segment_data[(check_x, check_y)]:
                    # Calculate distance to cell center
                    cell_center_x = (check_x + 0.5) * self.grid_size
                    cell_center_y = (check_y + 0.5) * self.grid_size
                    dist = math.sqrt((x - cell_center_x) **
                                     2 + (y - cell_center_y)**2)

                    if dist < self.clearance_radius + self.grid_size/2:
                        self._collision_cache[cache_key] = False
                        return False

                # Check grid edges
                if self._check_grid_edge_collision(check_x, check_y, x, y):
                    self._collision_cache[cache_key] = False
                    return False

        self._collision_cache[cache_key] = True
        return True

    def _check_grid_edge_collision(self, cell_x: int, cell_y: int,
                                   point_x: float, point_y: float) -> bool:
        """Check if a point collides with grid edges in a cell."""
        # Check horizontal edges
        for dx in range(2):
            for dy in range(3):
                edge_key = (2*cell_x + dx, 2*cell_y + dy)
                if edge_key in self.grid_edges['horizontal'] and self.grid_edges['horizontal'][edge_key]:
                    edge_x = edge_key[0] * self.grid_size/2
                    edge_y = edge_key[1] * self.grid_size/2
                    if self._point_to_line_segment_distance(
                        (point_x, point_y),
                        (edge_x, edge_y),
                        (edge_x + self.grid_size/2, edge_y)
                    ) < self.clearance_radius:
                        return True

        # Check vertical edges
        for dx in range(3):
            for dy in range(2):
                edge_key = (2*cell_x + dx, 2*cell_y + dy)
                if edge_key in self.grid_edges['vertical'] and self.grid_edges['vertical'][edge_key]:
                    edge_x = edge_key[0] * self.grid_size/2
                    edge_y = edge_key[1] * self.grid_size/2
                    if self._point_to_line_segment_distance(
                        (point_x, point_y),
                        (edge_x, edge_y),
                        (edge_x, edge_y + self.grid_size/2)
                    ) < self.clearance_radius:
                        return True

        return False

    def _point_to_line_segment_distance(self, point: Tuple[float, float],
                                        segment_start: Tuple[float, float],
                                        segment_end: Tuple[float, float]) -> float:
        """Calculate minimum distance from point to line segment."""
        px, py = point
        x1, y1 = segment_start
        x2, y2 = segment_end

        # Calculate squared length of segment
        length_sq = (x2 - x1)**2 + (y2 - y1)**2

        if length_sq == 0:
            # Segment is a point
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # Calculate projection of point onto line
        t = max(0, min(1, ((px - x1) * (x2 - x1) +
                (py - y1) * (y2 - y1)) / length_sq))

        # Calculate closest point on segment
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        # Return distance to closest point
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def get_valid_neighbors(self, node: Node) -> List[Tuple[int, int]]:
        """Get valid neighboring positions considering clearance."""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x = node.x + dx
            new_y = node.y + dy

            # Check bounds
            if not (0 <= new_x < self.grid_dimensions[0] and
                    0 <= new_y < self.grid_dimensions[1]):
                continue

            # Check if position has clearance
            pos = self.grid_to_continuous(new_x, new_y)
            if self.has_clearance(pos):
                neighbors.append((new_x, new_y))

        return neighbors

    def find_path(self, start_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find optimal path from start to goal using A* with clearance constraints.

        The goal can either be a switch or an exit door.
        """
        if self.tile_data is None:
            return []

        # Convert positions to grid coordinates
        start_grid = self.continuous_to_grid(*start_pos)
        goal_grid = self.continuous_to_grid(*goal_pos)

        # Initialize A* structures
        open_set = PriorityQueue()
        closed_set = set()
        nodes = {}

        # Create and add start node
        start_node = Node(*start_grid, g_cost=0)
        start_node.h_cost = self.manhattan_distance(
            start_node, Node(*goal_grid))
        open_set.put((start_node.f_cost, start_grid))
        nodes[start_grid] = start_node

        while not open_set.empty():
            current_cost, current_pos = open_set.get()
            current = nodes[current_pos]

            if (current.x, current.y) == goal_grid:
                return self.reconstruct_path(current)

            closed_set.add((current.x, current.y))

            # Process neighbors with clearance check
            for neighbor_pos in self.get_valid_neighbors(current):
                if neighbor_pos in closed_set:
                    continue

                # Create neighbor node if it doesn't exist
                if neighbor_pos not in nodes:
                    nodes[neighbor_pos] = Node(*neighbor_pos)
                neighbor = nodes[neighbor_pos]

                # Calculate new cost
                tentative_g = current.g_cost + \
                    self.get_movement_cost(current, neighbor)

                if tentative_g < neighbor.g_cost:
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.manhattan_distance(
                        neighbor, Node(*goal_grid))
                    open_set.put((neighbor.f_cost, neighbor_pos))

        return []  # No path found

    def get_movement_cost(self, current: Node, neighbor: Node) -> float:
        """Calculate movement cost between nodes."""
        dx = neighbor.x - current.x
        dy = neighbor.y - current.y
        base_cost = math.sqrt(dx*dx + dy*dy)

        # Add penalty for diagonal movement
        if dx != 0 and dy != 0:
            base_cost *= 1.4

        return base_cost

    def reconstruct_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """Reconstruct path from goal node to start node."""
        path = []
        current = goal_node

        while current:
            path.append(self.grid_to_continuous(current.x, current.y))
            current = current.parent

        return list(reversed(path))

    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous coordinates to grid coordinates."""
        return (int(x / self.grid_size), int(y / self.grid_size))

    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to continuous coordinates (center of cell)."""
        return ((grid_x + 0.5) * self.grid_size, (grid_y + 0.5) * self.grid_size)

    def manhattan_distance(self, start: Node, goal: Node) -> float:
        """Calculate Manhattan distance heuristic."""
        return abs(start.x - goal.x) + abs(start.y - goal.y)

    def get_next_waypoint(self, current_pos: Tuple[float, float],
                          path: List[Tuple[float, float]],
                          lookahead: int = 3) -> Optional[Tuple[float, float]]:
        """Get next waypoint from current position with lookahead."""
        if not path:
            return None

        # Find closest point on path
        current_grid = self.continuous_to_grid(*current_pos)
        min_dist = float('inf')
        closest_idx = 0

        for i, waypoint in enumerate(path):
            waypoint_grid = self.continuous_to_grid(*waypoint)
            dist = abs(waypoint_grid[0] - current_grid[0]) + \
                abs(waypoint_grid[1] - current_grid[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Look ahead on path
        target_idx = min(closest_idx + lookahead, len(path) - 1)
        return path[target_idx]

    def update_visited_nodes(self, pos: Tuple[float, float]):
        """Update set of visited nodes for exploration tracking."""
        grid_pos = self.continuous_to_grid(*pos)
        self.visited_nodes.add(grid_pos)

    def get_exploration_progress(self) -> float:
        """Calculate exploration progress as ratio of visited to total traversable nodes."""
        if self.tile_data is None:
            return 0.0

        traversable_count = np.sum(~self.tile_data)
        if traversable_count == 0:
            return 0.0

        return len(self.visited_nodes) / traversable_count

    def _is_switch_activated(self) -> bool:
        """Check if switch has been activated. Should be overridden with actual game state."""
        return False
