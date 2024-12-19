import numpy as np
from collections import deque
import cv2
from typing import Dict, Any
from npp_rl.environments.spatial_memory import SpatialMemoryTracker
from npp_rl.environments.constants import MAX_VELOCITY, OBSERVATION_IMAGE_SIZE
from npp_rl.environments.planning.path_planner import PathPlanner


class ObservationProcessor:
    """Processes raw game observations into the format needed by the agent"""

    def __init__(self, frame_stack: int = 4):
        self.frame_stack = frame_stack
        self.image_size = OBSERVATION_IMAGE_SIZE

        # Initialize path planner for distance field computation
        self.path_planner = PathPlanner()

        # Store frames with fixed intervals
        self.frame_intervals = [0, 4, 8, 12]
        self.frame_history = deque(maxlen=max(self.frame_intervals) + 1)

        # Initialize spatial memory tracker
        self.spatial_memory = SpatialMemoryTracker()

        # Store movement vectors for velocity calculation
        self.movement_history = deque(maxlen=16)
        self.prev_frame = None

        # Initialize observation channels
        self.distance_field = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.path_visualization = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.clearance_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.static_obstacles = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)
        self.dynamic_obstacles = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw frame to grayscale, resize, normalize, and extract motion features"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # type: ignore

        # Apply edge detection to highlight level features
        edges = cv2.Canny(frame, 100, 200)  # type: ignore
        frame = cv2.addWeighted(frame, 0.7, edges, 0.3, 0)  # type: ignore

        # Resize with interpolation
        frame = cv2.resize(
            frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)  # type: ignore

        # Enhance contrast
        frame = cv2.equalizeHist(frame)  # type: ignore

        # Keep frame in uint8 range [0, 255]
        return frame.astype(np.uint8)

    def get_distance_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Calculate normalized distances to objectives"""
        player_x = obs.get('player_x', 0)
        player_y = obs.get('player_y', 0)

        # Calculate Euclidean distances
        switch_dist = np.sqrt(
            (player_x - obs['switch_x'])**2 + (player_y - obs['switch_y'])**2)
        exit_dist = np.sqrt(
            (player_x - obs['exit_door_x'])**2 + (player_y - obs['exit_door_y'])**2)

        # Normalize by diagonal of level
        level_diagonal = np.sqrt(
            obs['level_width']**2 + obs['level_height']**2)
        return np.array([switch_dist/level_diagonal, exit_dist/level_diagonal], dtype=np.float32)

    def get_numerical_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process numerical features including spatial memory metrics"""
        # Get spatial memory metrics
        if 'player_x' in obs and 'player_y' in obs:
            memory_maps = self.spatial_memory.get_exploration_maps(
                obs['player_x'], obs['player_y'])
        else:
            memory_maps = {
                'recent_visits': np.zeros((88, 88)),
                'visit_frequency': np.zeros((88, 88)),
                'area_exploration': np.zeros((88, 88)),
                'transitions': np.zeros((88, 88))
            }

        # Resize memory maps from 88x88 to 84x84 while preserving information
        resized_maps = {
            key: cv2.resize(
                value, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            for key, value in memory_maps.items()
        }

        # Stack memory maps
        features = np.stack([
            resized_maps['recent_visits'],
            resized_maps['visit_frequency'],
            resized_maps['area_exploration'],
            resized_maps['transitions']
        ], axis=-1)

        return features.astype(np.float32)

    def get_player_state_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize player position and velocity features"""
        # Get player position and normalize to [0,1]
        player_x = obs.get('player_x', 0) / obs.get('level_width', 1)
        player_y = obs.get('player_y', 0) / obs.get('level_height', 1)

        # Get player velocity and normalize to [0, 1]
        player_vx = np.clip(obs.get('player_vx', 0) / MAX_VELOCITY, -1, 1)
        player_vy = np.clip(obs.get('player_vy', 0) / MAX_VELOCITY, -1, 1)
        player_vx = (player_vx + 1) / 2
        player_vy = (player_vy + 1) / 2

        # Get in_air status (0 or 1)
        in_air = float(obs.get('in_air', False))

        # Get walled status (0 or 1)
        walled = float(obs.get('walled', False))

        # Return as 1D array
        return np.array([
            player_x, player_y,
            player_vx, player_vy,
            in_air, walled
        ], dtype=np.float32)

    def get_goal_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract goal-related features"""
        # Get player position
        player_x = obs.get('player_x', 0)
        player_y = obs.get('player_y', 0)

        # Calculate normalized distances
        switch_x = obs.get('switch_x', 0)
        switch_y = obs.get('switch_y', 0)
        exit_x = obs.get('exit_door_x', 0)
        exit_y = obs.get('exit_door_y', 0)

        # Calculate Euclidean distances and normalize by diagonal
        level_diagonal = np.sqrt(
            obs.get('level_width', 1)**2 + obs.get('level_height', 1)**2)

        switch_dist = np.sqrt((player_x - switch_x)**2 +
                              (player_y - switch_y)**2) / level_diagonal
        exit_dist = np.sqrt((player_x - exit_x)**2 +
                            (player_y - exit_y)**2) / level_diagonal

        # Get switch activation status
        switch_activated = float(obs.get('switch_activated', False))

        return np.array([switch_dist, exit_dist, switch_activated], dtype=np.float32)

    def get_pathfinding_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Generate pathfinding-related observation channels."""
        # Update path planner with current game state
        tile_data = obs.get('tile_data', {})
        segment_data = obs.get('segment_data', {})
        grid_edges = obs.get('grid_edges', {})
        segment_edges = obs.get('segment_edges', {})

        self.path_planner.update_collision_grid(
            tile_data=tile_data,
            segment_data=segment_data,
            grid_edges=grid_edges,
            segment_edges=segment_edges
        )

        # Get current position and goal
        current_pos = (obs.get('player_x', 0), obs.get('player_y', 0))
        goal_pos = (obs.get('switch_x', 0), obs.get('switch_y', 0)) if not obs.get(
            'switch_activated', False) else (obs.get('exit_door_x', 0), obs.get('exit_door_y', 0))

        # Compute optimal path
        path = self.path_planner.find_path(current_pos, goal_pos)

        # Generate distance field (normalized distances to goal along optimal path)
        self.distance_field = self._compute_distance_field(
            current_pos, goal_pos, path)

        # Generate path visualization (gaussian blur around path)
        self.path_visualization = self._visualize_path(path)

        # Generate clearance map (distance to nearest obstacle)
        self.clearance_map = self._compute_clearance_map(
            tile_data, segment_data)

        return np.stack([
            self.distance_field,
            self.path_visualization,
            self.clearance_map
        ], axis=-1)

    def get_collision_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Generate collision-related observation channels."""
        # Process static obstacles (tiles, segments)
        tile_data = obs.get('tile_data', {})
        segment_data = obs.get('segment_data', {})

        self.static_obstacles = self._process_static_obstacles(
            tile_data, segment_data)

        # Since we're not supporting dynamic objects, just return static obstacles
        return np.stack([
            self.static_obstacles,
            # Empty dynamic obstacles channel
            np.zeros_like(self.static_obstacles)
        ], axis=-1)

    def _compute_distance_field(self, current_pos: tuple, goal_pos: tuple, path: list) -> np.ndarray:
        """Compute normalized distance field to goal along optimal path."""
        distance_field = np.ones(
            (self.image_size, self.image_size), dtype=np.float32)

        if path:
            # Normalize coordinates to observation size
            path = [(int(x * self.image_size / 1032),
                     int(y * self.image_size / 576)) for x, y in path]

            # Create distance field using distance transform
            path_image = np.zeros(
                (self.image_size, self.image_size), dtype=np.uint8)
            for x, y in path:
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    path_image[y, x] = 255

            distance_field = cv2.distanceTransform(
                255 - path_image, cv2.DIST_L2, 3)
            distance_field = 1 - (distance_field / distance_field.max())

        return distance_field

    def _visualize_path(self, path: list) -> np.ndarray:
        """Create a visualization of the optimal path."""
        path_viz = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)

        if path:
            # Normalize coordinates
            path = [(int(x * self.image_size / 1032),
                     int(y * self.image_size / 576)) for x, y in path]

            # Draw path with gaussian blur
            for i in range(len(path)-1):
                pt1 = path[i]
                pt2 = path[i+1]
                cv2.line(path_viz, pt1, pt2, 1, 2)

            path_viz = cv2.GaussianBlur(path_viz, (5, 5), 1.0)

        return path_viz

    def _compute_clearance_map(self, tile_data: dict, segment_data: dict) -> np.ndarray:
        """Compute clearance map showing distance to nearest obstacle."""
        obstacle_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.uint8)

        # Mark obstacles
        for (x, y), tile_id in tile_data.items():
            if tile_id == 1:  # Solid tile
                x = int(x * self.image_size / 37)
                y = int(y * self.image_size / 21)
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    obstacle_map[y:y+2, x:x+2] = 255

        # Compute distance transform
        clearance = cv2.distanceTransform(255 - obstacle_map, cv2.DIST_L2, 3)
        return clearance / clearance.max()

    def process_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process full observation including pathfinding and collision features."""
        # Process current frame
        frame = self.preprocess_frame(obs['screen'])

        # Update frame history
        self.frame_history.append(frame)

        # Fill frame history if needed
        while len(self.frame_history) < max(self.frame_intervals) + 1:
            self.frame_history.append(frame)

        # Stack frames at specified intervals
        stacked_frames = []
        for interval in self.frame_intervals:
            if len(self.frame_history) > interval:
                historical_frame = self.frame_history[-interval-1]
            else:
                historical_frame = frame
            # Already in uint8 range [0, 255], just add channel dimension
            stacked_frames.append(historical_frame[..., np.newaxis])

        # Get numerical features
        # memory_features = self.get_numerical_features(obs)
        # pathfinding_features = self.get_pathfinding_features(obs)
        # collision_features = self.get_collision_features(obs)

        # Convert memory features to uint8
        # memory_features = (memory_features * 255).astype(np.uint8)
        # pathfinding_features = (pathfinding_features * 255).astype(np.uint8)
        # collision_features = (collision_features * 255).astype(np.uint8)

        # Get player state features
        player_state = self.get_player_state_features(obs)

        # Get goal features
        goal_features = self.get_goal_features(obs)

        # Combine all visual features
        # visual_observation = np.concatenate(
        #     stacked_frames +
        #     [memory_features] +
        #     [pathfinding_features] +
        #     [collision_features],
        #     axis=-1
        # )

        # return {
        #     'visual': visual_observation,  # Already uint8 in [0, 255] range
        #     'player_state': player_state.astype(np.float32),
        #     'goal_features': goal_features.astype(np.float32)
        # }
        return {
            'visual': stacked_frames,
            'player_state': player_state.astype(np.float32),
            'goal_features': goal_features.astype(np.float32)
        }

    def reset(self) -> None:
        """Reset processor state."""
        self.frame_history.clear()
        self.movement_history.clear()
        self.spatial_memory.reset()
        self.prev_frame = None

        # Reset pathfinding and collision features
        self.distance_field.fill(0)
        self.path_visualization.fill(0)
        self.clearance_map.fill(0)
        self.static_obstacles.fill(0)
        self.dynamic_obstacles.fill(0)

    def _process_static_obstacles(self, tile_data: dict, segment_data: dict) -> np.ndarray:
        """Process static obstacles (tiles and segments) into an obstacle map.

        Args:
            tile_data: Dictionary mapping (x,y) coordinates to tile IDs
            segment_data: Dictionary mapping (x,y) coordinates to list of segments

        Returns:
            np.ndarray: Binary obstacle map of shape (image_size, image_size)
        """
        # Initialize empty obstacle map
        obstacle_map = np.zeros(
            (self.image_size, self.image_size), dtype=np.float32)

        # Scale factors to convert from tile/segment coordinates to observation size
        scale_x = self.image_size / 42  # 42 tiles per row
        scale_y = self.image_size / 23  # 23 tiles per column

        # Process tiles
        for (x, y), tile_id in tile_data.items():
            if tile_id == 1:  # Solid tile
                # Convert tile coordinates to pixel coordinates in observation space
                x_start = int(x * scale_x)
                y_start = int(y * scale_y)
                x_end = int((x + 1) * scale_x)
                y_end = int((y + 1) * scale_y)

                # Mark tile area as obstacle
                obstacle_map[y_start:y_end, x_start:x_end] = 1.0

        # Process segments
        for (x, y), segments in segment_data.items():
            if segments:  # If there are any segments in this cell
                # Convert cell coordinates to pixel coordinates
                x_start = int(x * scale_x)
                y_start = int(y * scale_y)
                x_end = int((x + 1) * scale_x)
                y_end = int((y + 1) * scale_y)

                # Mark segment area as obstacle
                # Note: This is a simplification - in reality segments could be drawn more precisely
                obstacle_map[y_start:y_end, x_start:x_end] = 1.0

        return obstacle_map
