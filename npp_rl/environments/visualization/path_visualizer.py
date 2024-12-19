import numpy as np
import cv2
from typing import List, Tuple, Optional
import os
from datetime import datetime


class PathVisualizer:
    """Visualizes planned paths and waypoints overlaid on environment frames.

    Features:
    1. Path visualization with gradient coloring for direction
    2. Current waypoint highlighting
    3. Start and goal position markers
    4. Path deviation indicators
    5. Save visualizations to specified directory
    """

    def __init__(self, save_dir: str = 'training_logs'):
        """Initialize path visualizer.

        Args:
            save_dir: Base directory for saving visualizations
        """
        self.base_save_dir = save_dir
        self.session_dir = None
        self.frame_count = 0

        # Visualization constants
        self.PATH_COLOR = (0, 255, 0)  # Green
        self.WAYPOINT_COLOR = (255, 0, 0)  # Red
        self.START_COLOR = (0, 255, 255)  # Yellow
        self.GOAL_COLOR = (255, 0, 255)  # Magenta
        self.DEVIATION_COLOR = (255, 165, 0)  # Orange

        self._setup_save_directory()

    def _setup_save_directory(self):
        """Create directory structure for saving visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = os.path.join(self.base_save_dir,
                                        f'ppo_training_log/training_session-{timestamp}/paths')
        os.makedirs(self.session_dir, exist_ok=True)

    def render_path(self,
                    frame: np.ndarray,
                    planned_path: List[Tuple[float, float]],
                    current_pos: Tuple[float, float],
                    current_waypoint: Optional[Tuple[float, float]] = None,
                    goal_pos: Optional[Tuple[float, float]] = None,
                    path_deviation: float = 0.0,
                    distance_to_waypoint: float = 0.0,
                    episode_reward: float = 0.0) -> np.ndarray:
        """Render planned path and navigation information on frame.

        Args:
            frame: Environment frame to draw on
            planned_path: List of (x, y) coordinates for planned path
            current_pos: Current agent position (x, y)
            current_waypoint: Current target waypoint (x, y)
            goal_pos: Final goal position (x, y)
            path_deviation: Current deviation from planned path
            distance_to_waypoint: Current distance to waypoint
            episode_reward: Current episode reward

        Returns:
            Frame with visualization overlays
        """
        # Create copy of frame for drawing
        viz_frame = frame.copy()

        # Convert to color if grayscale
        if len(viz_frame.shape) == 2:
            viz_frame = cv2.cvtColor(viz_frame, cv2.COLOR_GRAY2BGR)

        # Draw planned path with gradient coloring
        if len(planned_path) > 1:
            for i in range(len(planned_path) - 1):
                start = tuple(map(int, planned_path[i]))
                end = tuple(map(int, planned_path[i + 1]))

                # Calculate color based on position in path
                progress = i / (len(planned_path) - 1)
                color = self._get_path_color(progress)

                cv2.line(viz_frame, start, end, color, 2)

        # Draw current position
        cv2.circle(viz_frame,
                   tuple(map(int, current_pos)),
                   5, self.START_COLOR, -1)

        # Draw current waypoint
        if current_waypoint:
            cv2.circle(viz_frame,
                       tuple(map(int, current_waypoint)),
                       3, self.WAYPOINT_COLOR, 2)

            # Draw line to current waypoint
            cv2.line(viz_frame,
                     tuple(map(int, current_pos)),
                     tuple(map(int, current_waypoint)),
                     self.WAYPOINT_COLOR, 1)

        # Draw goal position
        if goal_pos:
            cv2.circle(viz_frame,
                       tuple(map(int, goal_pos)),
                       4, self.GOAL_COLOR, -1)

        # Draw path deviation indicator
        if path_deviation > 0:
            deviation_radius = int(path_deviation)
            cv2.circle(viz_frame,
                       tuple(map(int, current_pos)),
                       deviation_radius,
                       self.DEVIATION_COLOR, 1)

        # Add text overlay with metrics
        self._add_metrics_overlay(
            viz_frame, path_deviation, episode_reward, distance_to_waypoint)

        return viz_frame

    def _get_path_color(self, progress: float) -> Tuple[int, int, int]:
        """Get color for path segment based on progress.

        Args:
            progress: Value between 0 and 1 indicating position in path

        Returns:
            BGR color tuple
        """
        # Gradient from green to blue
        return (
            int(255 * progress),  # B
            int(255 * (1 - progress)),  # G
            0  # R
        )

    def _add_metrics_overlay(self, frame: np.ndarray, path_deviation: float, episode_reward: float, distance_to_waypoint: float):
        """Add text overlay with navigation metrics."""
        cv2.putText(frame,
                    f"Path Deviation: {path_deviation:.2f} | Episode Reward: {episode_reward:.2f} | Distance to Waypoint: {distance_to_waypoint:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

    def save_visualization(self, frame: np.ndarray, episode: int, step: int):
        """Save visualization frame to disk.

        Args:
            frame: Frame to save
            episode: Current episode number
            step: Current step within episode
        """
        filename = os.path.join(self.session_dir,
                                f'path_viz_ep{episode}_step{step}.png')
        cv2.imwrite(filename, frame)
        self.frame_count += 1
