"""Exploration reward calculator for evaluating area exploration and discovery."""
from typing import Dict, Any, Set, Tuple
from collections import deque
import numpy as np
import pygame
import cairo
from npp_rl.environments.reward_calculation.base_reward_calculator import BaseRewardCalculator


class ExplorationRewardCalculator(BaseRewardCalculator):
    """Handles calculation of exploration and area-based rewards."""

    # Game environment dimensions
    TILE_SIZE = 24  # pixels
    MAP_WIDTH = 42  # tiles
    MAP_HEIGHT = 23  # tiles
    TOTAL_WIDTH = TILE_SIZE * MAP_WIDTH  # 1008 pixels
    TOTAL_HEIGHT = TILE_SIZE * MAP_HEIGHT  # 552 pixels

    # Increased exploration rewards
    EXPLORATION_REWARD = 0.01
    STUCK_PENALTY = -0.3  # Reduced from -0.5
    AREA_EXPLORATION_REWARD = 0.01
    NEW_TRANSITION_REWARD = 0.01
    BACKTRACK_PENALTY = -0.01
    LOCAL_MINIMA_PENALTY = -0.01
    AREA_REVISIT_DECAY = 0.6
    PROGRESSIVE_BACKTRACK_PENALTY = -0.01

    # Add maximum penalty caps
    MAX_BACKTRACK_PENALTY = -0.01  # Reduced from -3.0
    MAX_LOCAL_MINIMA_PENALTY = -0.01  # Reduced from -2.0
    MAX_TOTAL_PENALTY = -0.25  # Reduced from -5.0

    # Visualization colors
    UNEXPLORED_COLOR = (50, 50, 50)  # Dark gray
    EXPLORED_COLOR = (200, 200, 200)  # Light gray
    CURRENT_POS_COLOR = (255, 0, 0)  # Red
    TRANSITION_COLOR = (0, 255, 0)  # Green

    def __init__(self):
        """Initialize exploration reward calculator."""
        super().__init__()
        # Grid sizes for position tracking - aligned with tile size
        self.position_grid_size = self.TILE_SIZE  # Track per tile
        self.area_grid_size = self.TILE_SIZE * 2  # Track per 2x2 tile area

        # Exploration tracking
        self.exploration_memory = 1000
        self.stuck_threshold = 30
        self.local_minima_counter = 0
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.visited_areas: Set[Tuple[int, int]] = set()
        self.area_visit_counts = {}
        self.area_transition_points: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set(
        )

        # Path analysis
        self.prev_area = None
        self.path_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=100)
        self.distance_history = deque(maxlen=100)
        self.last_objective_distance = None

        # Training progress tracking
        self.total_steps = 0
        self.early_training_threshold = 50000  # Steps considered "early training"

        # Visualization surface
        self._init_visualization_surface()

    def _init_visualization_surface(self):
        """Initialize the visualization surface for rendering explored areas."""
        self.vis_surface = cairo.ImageSurface(
            cairo.Format.RGB24, self.TOTAL_WIDTH, self.TOTAL_HEIGHT)
        self.vis_context = cairo.Context(self.vis_surface)
        # Fill with unexplored color initially
        self.vis_context.set_source_rgb(
            *[x/255 for x in self.UNEXPLORED_COLOR])
        self.vis_context.paint()

    def render_exploration_map(self) -> np.ndarray:
        """Render a visualization of explored areas and return as a numpy array.

        Returns:
            np.ndarray: RGB image of exploration map (height, width, 3)
        """
        # Clear surface with unexplored color
        self.vis_context.set_source_rgb(
            *[x/255 for x in self.UNEXPLORED_COLOR])
        self.vis_context.paint()

        # Draw explored areas
        self.vis_context.set_source_rgb(*[x/255 for x in self.EXPLORED_COLOR])
        for pos in self.visited_positions:
            x, y = pos
            x *= self.position_grid_size
            y *= self.position_grid_size
            self.vis_context.rectangle(
                x, y, self.position_grid_size, self.position_grid_size)
        self.vis_context.fill()

        # Draw transition points
        self.vis_context.set_source_rgb(
            *[x/255 for x in self.TRANSITION_COLOR])
        self.vis_context.set_line_width(2)
        for (start, end) in self.area_transition_points:
            start_x = start[0] * self.area_grid_size + self.area_grid_size // 2
            start_y = start[1] * self.area_grid_size + self.area_grid_size // 2
            end_x = end[0] * self.area_grid_size + self.area_grid_size // 2
            end_y = end[1] * self.area_grid_size + self.area_grid_size // 2
            self.vis_context.move_to(start_x, start_y)
            self.vis_context.line_to(end_x, end_y)
        self.vis_context.stroke()

        # Convert cairo surface to numpy array
        buf = self.vis_surface.get_data()
        img_array = np.ndarray(shape=(self.TOTAL_HEIGHT, self.TOTAL_WIDTH, 4),
                               dtype=np.uint8,
                               buffer=buf)
        # Convert BGRA to RGB
        return img_array[:, :, [2, 1, 0]]

    def _get_grid_id(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous position to discrete grid position.

        Args:
            x: X position in game coordinates (pixels)
            y: Y position in game coordinates (pixels)

        Returns:
            Tuple[int, int]: Grid coordinates (tile x, tile y)
        """
        # Ensure coordinates are within map bounds
        x = max(0, min(x, self.TOTAL_WIDTH - 1))
        y = max(0, min(y, self.TOTAL_HEIGHT - 1))

        return (int(x / self.position_grid_size),
                int(y / self.position_grid_size))

    def _get_area_id(self, x: float, y: float) -> Tuple[int, int]:
        """Convert position to larger area grid position.

        Args:
            x: X position in game coordinates (pixels)
            y: Y position in game coordinates (pixels)

        Returns:
            Tuple[int, int]: Area coordinates (area x, area y)
        """
        # Ensure coordinates are within map bounds
        x = max(0, min(x, self.TOTAL_WIDTH - 1))
        y = max(0, min(y, self.TOTAL_HEIGHT - 1))

        return (int(x / self.area_grid_size),
                int(y / self.area_grid_size))

    def _check_stuck_in_local_minima(self,
                                     curr_pos: Tuple[int, int],
                                     current_distance: float) -> bool:
        """Detect if agent is stuck in a local minimum.

        Args:
            curr_pos: Current grid position (tile coordinates)
            current_distance: Current distance to objective (in pixels)

        Returns:
            bool: Whether agent is stuck in local minimum
        """
        # Add current position and distance to history
        self.position_history.append(curr_pos)
        self.distance_history.append(current_distance)

        if len(self.position_history) < self.stuck_threshold:
            return False

        # Check position variety in recent history
        recent_positions = set(list(self.position_history)
                               [-self.stuck_threshold:])
        position_variety = len(recent_positions)

        # Check distance progress - now using pixel distances
        recent_distances = list(self.distance_history)[-self.stuck_threshold:]
        min_distance = min(recent_distances)
        max_distance = max(recent_distances)
        distance_progress = max_distance - min_distance

        # Determine if stuck - adjusted thresholds for pixel coordinates
        is_stuck = (position_variety < 5 and  # Still stuck if visiting less than 5 unique tiles
                    # Less than 2 tiles of progress
                    abs(distance_progress) < self.TILE_SIZE * 2 and
                    len(recent_positions) >= self.stuck_threshold)

        if is_stuck:
            self.local_minima_counter += 1
        else:
            self.local_minima_counter = max(0, self.local_minima_counter - 1)

        return is_stuck

    def _get_penalty_scale(self) -> float:
        """Calculate penalty scaling based on training progress."""
        if self.total_steps < self.early_training_threshold:
            # Linearly increase penalties from 30% to 100% during early training
            return 0.3 + (0.7 * (self.total_steps / self.early_training_threshold))
        return 1.0

    def calculate_exploration_reward(self,
                                     curr_state: Dict[str, Any]) -> float:
        """Calculate comprehensive exploration reward.

        Args:
            curr_state: Current game state containing player position and objectives

        Returns:
            float: Calculated reward value
        """
        self.total_steps += 1
        penalty_scale = self._get_penalty_scale()
        reward = 0.0

        # Get current positions in both coordinate systems
        current_pos = (curr_state['player_x'], curr_state['player_y'])
        grid_pos = self._get_grid_id(*current_pos)
        current_area = self._get_area_id(*current_pos)

        # Calculate current objective distance in pixels
        if not curr_state['switch_activated']:
            current_distance = self.calculate_distance_to_objective(
                curr_state['player_x'], curr_state['player_y'],
                curr_state['switch_x'], curr_state['switch_y']
            )
        else:
            current_distance = self.calculate_distance_to_objective(
                curr_state['player_x'], curr_state['player_y'],
                curr_state['exit_door_x'], curr_state['exit_door_y']
            )

        # Enhanced exploration reward for new positions
        if grid_pos not in self.visited_positions:
            # Exponential decay based on distance to encourage thorough exploration
            # Normalize distance by tile size for consistent scaling
            # Decay over ~20 tiles
            distance_factor = np.exp(-current_distance / (self.TILE_SIZE * 20))
            exploration_reward = self.EXPLORATION_REWARD * \
                (1.0 + distance_factor)
            reward += exploration_reward
            self.visited_positions.add(grid_pos)

            if len(self.visited_positions) > self.exploration_memory:
                self.visited_positions.remove(
                    next(iter(self.visited_positions)))

        # Scaled backtracking penalty
        visit_count = self.area_visit_counts.get(current_area, 0)
        if visit_count > 0:
            backtrack_penalty = self.PROGRESSIVE_BACKTRACK_PENALTY * \
                (visit_count ** 1.2) * penalty_scale  # Reduced power and scaled
            reward += backtrack_penalty

        # Enhanced area transitions and exploration
        if self.prev_area != current_area:
            if current_area not in self.visited_areas:
                # Exponential reward scaling based on exploration progress
                # Assuming average of 20 areas
                exploration_progress = len(self.visited_areas) / 20
                area_reward = self.AREA_EXPLORATION_REWARD * \
                    (1.0 + np.exp(-exploration_progress))
                reward += area_reward
                self.visited_areas.add(current_area)
            else:
                visit_count = self.area_visit_counts.get(current_area, 0)
                revisit_penalty = self.AREA_EXPLORATION_REWARD * \
                    (self.AREA_REVISIT_DECAY ** (visit_count + 1)) * penalty_scale
                reward += revisit_penalty

            # Record transition points with enhanced rewards
            if self.prev_area is not None:
                transition_point = (self.prev_area, current_area)
                if transition_point not in self.area_transition_points:
                    if current_area not in list(self.path_history)[-20:]:
                        reward += self.NEW_TRANSITION_REWARD
                    self.area_transition_points.add(transition_point)

            self.area_visit_counts[current_area] = visit_count + 1

        # Scaled backtracking detection
        if len(self.path_history) > 50:
            recent_areas = list(self.path_history)[-50:]
            area_count = len(set(recent_areas))
            if area_count < 3:
                if current_area in recent_areas[:-10]:
                    backtrack_count = recent_areas.count(current_area)
                    backtrack_penalty = max(
                        self.BACKTRACK_PENALTY *
                        (backtrack_count ** 1.1) * penalty_scale,
                        self.MAX_BACKTRACK_PENALTY
                    )
                    reward += backtrack_penalty

        # Track objective progress with reduced penalties
        if self.last_objective_distance is not None:
            progress = self.last_objective_distance - current_distance
            if progress < -50:  # Moving away from objective
                reward += max(self.BACKTRACK_PENALTY * 0.2 * penalty_scale,
                              self.MAX_BACKTRACK_PENALTY)
            elif progress > 50:  # Moving towards objective
                reward += abs(self.BACKTRACK_PENALTY) * 0.2

        # Update history
        self.path_history.append(current_area)
        self.prev_area = current_area
        self.last_objective_distance = current_distance

        # Scaled local minima detection
        if self._check_stuck_in_local_minima(grid_pos, current_distance):
            penalty_scale = min(self.local_minima_counter **
                                1.1, 2.0) * self._get_penalty_scale()
            local_minima_penalty = max(
                self.LOCAL_MINIMA_PENALTY * penalty_scale,
                self.MAX_LOCAL_MINIMA_PENALTY
            )
            reward += local_minima_penalty

        # Final cap on total negative reward with scaling
        min_reward = self.MAX_TOTAL_PENALTY * penalty_scale
        reward = max(reward, min_reward)

        return reward

    def reset(self):
        """Reset internal state for new episode."""
        self.visited_positions.clear()
        self.visited_areas.clear()
        self.area_visit_counts.clear()
        self.area_transition_points.clear()
        self.prev_area = None
        self.path_history.clear()
        self.position_history.clear()
        self.distance_history.clear()
        self.local_minima_counter = 0
