from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass
from npp_rl.environments.planning.path_planner import PathPlanner


@dataclass
class WaypointMetrics:
    distance_to_waypoint: float
    progress_to_waypoint: float
    waypoint_reached: bool
    path_deviation: float
    path_clearance: float  # New metric for path clearance


class WaypointManager:
    """Manages mid-level navigation between waypoints.

    This class bridges the gap between high-level path planning and low-level
    control by:
    1. Managing waypoint sequences
    2. Tracking progress towards waypoints
    3. Providing navigation metrics for reward calculation
    4. Handling dynamic waypoint updates
    5. Ensuring paths have sufficient clearance from obstacles
    """

    def __init__(self, path_planner: PathPlanner, waypoint_radius: float = 50.0):
        self.path_planner = path_planner
        self.waypoint_radius = waypoint_radius
        self.current_path: List[Tuple[float, float]] = []
        self.current_waypoint: Optional[Tuple[float, float]] = None
        self.last_distance = float('inf')
        self.min_clearance = 12.0  # Minimum required clearance in pixels
        # Track visited waypoints
        self.visited_waypoints: List[Tuple[float, float]] = []
        # Distance threshold to consider a waypoint as backtracking
        self.backtrack_threshold = 23.0

    def is_backtracking(self, waypoint: Tuple[float, float]) -> bool:
        """Check if a waypoint represents backtracking by comparing to visited waypoints."""
        for visited in self.visited_waypoints:
            dist = np.sqrt((waypoint[0] - visited[0])
                           ** 2 + (waypoint[1] - visited[1])**2)
            if dist < self.backtrack_threshold:
                return True
        return False

    def update_path(self, new_path: List[Tuple[float, float]], validate: bool = True):
        """Update the current path and reset progress tracking.

        Args:
            new_path: List of waypoints defining the path
            validate: If True, validate path segments for clearance
        """
        if validate:
            # Validate each path segment
            valid_path = []
            for i in range(len(new_path) - 1):
                start = new_path[i]
                end = new_path[i + 1]

                # Check if direct path is clear
                if self.path_planner.is_path_clear(start, end):
                    valid_path.append(start)
                else:
                    # Find alternative path segment
                    segment_path = self.path_planner.find_path(start, end)
                    if segment_path:
                        # Exclude last point to avoid duplicates
                        valid_path.extend(segment_path[:-1])
                    else:
                        # If no valid path found, keep original segment but mark it
                        valid_path.append(start)

            # Add final point
            if new_path:
                valid_path.append(new_path[-1])

            self.current_path = valid_path
        else:
            self.current_path = new_path

        self.last_distance = float('inf')
        if self.current_path:
            # Get the first waypoint
            self.current_waypoint = self.current_path[0]

            # If we have at least two points, adjust the waypoint to be centered on the path
            if len(self.current_path) > 1:
                next_point = self.current_path[1]

                # Calculate direction vector
                dx = next_point[0] - self.current_waypoint[0]
                dy = next_point[1] - self.current_waypoint[1]

                # Normalize the direction vector
                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx /= length
                    dy /= length

                    # Move waypoint slightly forward along the path (about 1/4 of the waypoint radius)
                    offset = self.waypoint_radius * 0.25
                    self.current_waypoint = (
                        self.current_waypoint[0] + dx * offset,
                        self.current_waypoint[1] + dy * offset
                    )

    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get the current target waypoint."""
        return self.current_waypoint

    def calculate_metrics(self, current_pos: Tuple[float, float], prev_pos: Tuple[float, float]) -> WaypointMetrics:
        """Calculate navigation metrics for current position."""
        if not self.current_waypoint:
            return WaypointMetrics(
                distance_to_waypoint=float('inf'),
                progress_to_waypoint=0.0,
                waypoint_reached=False,
                path_deviation=0.0,
                path_clearance=0.0
            )

        # Calculate current distance to waypoint
        current_distance = np.sqrt(
            (current_pos[0] - self.current_waypoint[0])**2 +
            (current_pos[1] - self.current_waypoint[1])**2
        )

        # Calculate previous distance to waypoint
        prev_distance = self.last_distance if self.last_distance != float(
            'inf') else current_distance

        # Calculate progress as difference between previous and current distance
        progress = max(0.0, prev_distance - current_distance)

        # Update last distance for next calculation
        self.last_distance = current_distance

        # Check if waypoint reached - use squared distance for efficiency
        waypoint_reached = current_distance < self.waypoint_radius

        # Calculate path deviation
        path_deviation = self._calculate_path_deviation(current_pos)

        # Calculate current clearance
        clearance = self._calculate_clearance(current_pos)

        return WaypointMetrics(
            distance_to_waypoint=current_distance,
            progress_to_waypoint=progress,
            waypoint_reached=waypoint_reached,
            path_deviation=path_deviation,
            path_clearance=clearance
        )

    def _calculate_clearance(self, current_pos: Tuple[float, float]) -> float:
        """Calculate current clearance from obstacles."""
        if self.path_planner.has_clearance(current_pos):
            return self.min_clearance
        return 0.0

    def _calculate_path_deviation(self, current_pos: Tuple[float, float]) -> float:
        """Calculate minimum distance from current position to planned path."""
        if len(self.current_path) < 2:
            return 0.0

        min_deviation = float('inf')

        # Check deviation from each path segment
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i + 1]

            # Calculate deviation using point-to-line-segment distance
            deviation = self._point_to_segment_distance(current_pos, p1, p2)
            min_deviation = min(min_deviation, deviation)

        return min_deviation

    def _point_to_segment_distance(self,
                                   point: Tuple[float, float],
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
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # Calculate projection of point onto line
        t = max(0, min(1, ((px - x1) * (x2 - x1) +
                (py - y1) * (y2 - y1)) / length_sq))

        # Calculate closest point on segment
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        # Return distance to closest point
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def update_waypoint(self, current_pos: Tuple[float, float], prev_pos: Tuple[float, float], metrics: WaypointMetrics, lookahead: int = 3):
        """Update current waypoint based on progress and path clearance."""
        if not self.current_path:
            return

        if metrics.waypoint_reached:
            # Add reached waypoint to visited list
            if self.current_waypoint:
                self.visited_waypoints.append(self.current_waypoint)

            # Get next waypoint with lookahead
            next_waypoint = self.path_planner.get_next_waypoint(
                current_pos, self.current_path, lookahead)

            # Check if next waypoint would be backtracking
            if next_waypoint:
                # Try to find a path directly to the goal, skipping intermediate waypoints
                goal = self.current_path[-1] if self.current_path else None
                if goal:
                    direct_path = self.path_planner.find_path(
                        current_pos, goal)
                    if direct_path:
                        self.current_waypoint = direct_path[0]
                        self.update_path(direct_path)
                        return

            # Validate path to next waypoint
            if next_waypoint and self.path_planner.is_path_clear(current_pos, next_waypoint):
                self.current_waypoint = next_waypoint
                self.last_distance = float('inf')
            else:
                # If path is blocked, try to find alternative route
                if next_waypoint:
                    new_path = self.path_planner.find_path(
                        current_pos, next_waypoint)
                    if new_path and not any(self.is_backtracking(wp) for wp in new_path):
                        self.update_path(new_path)
