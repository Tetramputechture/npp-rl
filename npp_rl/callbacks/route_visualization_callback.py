"""Route visualization callback for tracking successful agent paths.

This callback records the route (player positions) taken by the agent through
each level upon successful completion. It saves these routes as images to help
visualize learning progress and agent behavior.

The implementation is designed to be performant and avoid memory bloat by:
- Only recording routes for successful completions
- Using efficient numpy arrays for position tracking
- Saving images asynchronously (optional)
- Rate limiting visualization frequency
- Cleaning up old visualizations
"""

import logging
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from npp_rl.rendering import render_tiles_to_axis, render_mines_to_axis

logger = logging.getLogger(__name__)


class RouteVisualizationCallback(BaseCallback):
    """Callback for visualizing successful agent routes through levels.

    This callback tracks the agent's position throughout an episode and
    saves a visualization when the agent successfully completes a level.

    Features:
    - Only records successful completions to reduce overhead
    - Efficient position tracking with fixed-size buffers
    - Configurable visualization frequency to avoid slowdown
    - Asynchronous image saving (optional)
    - Automatic cleanup of old visualizations
    - TensorBoard integration for route images

    Visualization Elements:
    - Agent path: Gradient-colored line from start to end
    - Start position: Blue circle (agent's initial position)
    - End position: Green circle (where agent reached exit)
    - Exit switch: Red star (objective to activate)
    - Exit door: Purple diamond (exit after switch activation)
    - Title shows: Episode reward (cumulative), length, level ID
    """

    def __init__(
        self,
        save_dir: str,
        max_routes_per_checkpoint: int = 10,
        visualization_freq: int = 50000,
        max_stored_routes: int = 100,
        async_save: bool = True,
        image_size: Tuple[int, int] = (800, 600),
        verbose: int = 0,
    ):
        """Initialize route visualization callback.

        Args:
            save_dir: Directory to save route visualizations
            max_routes_per_checkpoint: Maximum routes to save per checkpoint interval
            visualization_freq: How often (in timesteps) to save route visualizations
            max_stored_routes: Maximum number of route images to keep (oldest deleted)
            async_save: If True, save images asynchronously to avoid blocking training
            image_size: Size of output images (width, height)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.verbose = verbose
        self.save_dir = Path(save_dir)
        self.max_routes_per_checkpoint = max_routes_per_checkpoint
        self.visualization_freq = visualization_freq
        self.max_stored_routes = max_stored_routes
        self.async_save = async_save
        self.image_size = image_size

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Route tracking per environment
        self.env_routes = defaultdict(
            lambda: {
                "positions": [],
                "level_id": None,
                "start_time": 0,
                "tiles": None,
                "mines": None,
                "locked_doors": None,
                "is_success": None,
            }
        )

        # Statistics
        self.routes_saved_this_checkpoint = 0
        self.last_visualization_step = 0
        self.total_routes_saved = 0

        # Saved file tracking for cleanup
        self.saved_files = deque(maxlen=max_stored_routes)

        # Threading for async saves
        self.save_queue = []
        self.save_thread = None

        # TensorBoard writer
        self.tb_writer = None

        # Import matplotlib only when needed (lazy import)
        self._matplotlib_imported = False

    def _import_matplotlib(self):
        """Lazy import matplotlib to avoid startup overhead."""
        if not self._matplotlib_imported:
            try:
                import matplotlib

                # Only set backend if not already set or if it's incompatible with headless mode
                # This prevents issues in multi-process environments or when matplotlib is already imported
                try:
                    current_backend = matplotlib.get_backend()
                    if current_backend not in ["Agg", "Cairo", "PDF", "PS", "SVG"]:
                        matplotlib.use("Agg", force=True)
                except Exception:
                    # If we can't check or set the backend, try anyway but don't fail
                    try:
                        matplotlib.use("Agg")
                    except Exception:
                        logger.warning(
                            "Could not set matplotlib backend to Agg - visualization may fail in headless mode"
                        )

                import matplotlib.pyplot as plt

                self.plt = plt
                self._matplotlib_imported = True
            except ImportError:
                logger.warning(
                    "matplotlib not available - route visualization disabled"
                )
                self._matplotlib_imported = False
        return self._matplotlib_imported

    def _init_callback(self) -> None:
        """Initialize callback after model setup."""
        # Find TensorBoard writer
        from stable_baselines3.common.logger import TensorBoardOutputFormat

        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break

        # Add warning about position tracking requirement
        logger.info(
            f"Route visualization callback initialized (saving to {self.save_dir})"
        )
        logger.info(
            f"Will visualize up to {self.max_routes_per_checkpoint} routes every {self.visualization_freq} steps"
        )
        logger.info(
            "⚠️  Route visualization requires PositionTrackingWrapper to be applied to environments"
        )
        logger.info(
            "   If routes are not being captured, ensure the wrapper is in the environment pipeline"
        )

    def _on_step(self) -> bool:
        """Called after each environment step.

        Returns:
            bool: If False, training will be stopped
        """
        # Track positions for all environments
        if "dones" in self.locals and "infos" in self.locals:
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            # Try to get positions from the environment directly
            # This is more reliable than trying to parse observations
            for env_idx, (done, info) in enumerate(zip(dones, infos)):
                # Track position for this environment
                self._track_position_from_env(env_idx, info)

                # Check for episode completion
                if done:
                    self._handle_episode_end(env_idx, info)

        # Check if it's time to save visualizations
        if self.num_timesteps - self.last_visualization_step >= self.visualization_freq:
            self._process_save_queue()
            self.routes_saved_this_checkpoint = 0
            self.last_visualization_step = self.num_timesteps

        return True

    def _track_position_from_env(self, env_idx: int, info: Dict[str, Any]) -> None:
        """Track agent position for an environment from info dict.

        Args:
            env_idx: Environment index
            info: Info dictionary from environment step
        """
        try:
            # Position should be provided by PositionTrackingWrapper in info dict
            pos = None

            # Check if position is in info dict (added by PositionTrackingWrapper)
            if "player_position" in info:
                pos = info["player_position"]
            elif "ninja_position" in info:
                pos = info["ninja_position"]
            elif "position" in info:
                pos = info["position"]

            # If we got a position, store it
            if pos is not None:
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    pos_x, pos_y = float(pos[0]), float(pos[1])

                    # If this is the first position of a new episode, capture level data
                    if len(self.env_routes[env_idx]["positions"]) == 0:
                        self.env_routes[env_idx]["tiles"] = self._get_tile_data(env_idx)
                        self.env_routes[env_idx]["mines"] = self._get_mine_data(env_idx)
                        self.env_routes[env_idx]["locked_doors"] = (
                            self._get_locked_door_data(env_idx)
                        )

                    self.env_routes[env_idx]["positions"].append((pos_x, pos_y))
        except Exception as e:
            logger.info(f"Could not track position for env {env_idx}: {e}")

    def _handle_episode_end(self, env_idx: int, info: Dict[str, Any]) -> None:
        """Handle episode completion and potentially save route.

        Args:
            env_idx: Environment index
            info: Episode info dictionary
        """
        route_positions = info["episode_route"]

        # Only save routes for successful completions with valid position data
        if route_positions and len(route_positions) > 0:
            # Check if we should save this route
            if self.routes_saved_this_checkpoint < self.max_routes_per_checkpoint:
                self._queue_route_save(env_idx, info, route_positions)
                self.routes_saved_this_checkpoint += 1

        # Clear route data for this environment
        self.env_routes[env_idx] = {
            "positions": [],
            "level_id": info.get("level_id", None),
            "start_time": self.num_timesteps,
            "tiles": None,
            "mines": None,
            "locked_doors": None,
        }

    def _queue_route_save(
        self,
        env_idx: int,
        info: Dict[str, Any],
        route_positions: List[Tuple[float, float]],
    ) -> None:
        """Queue a route for saving.

        Args:
            env_idx: Environment index
            info: Episode info dictionary
            route_positions: List of (x, y) position tuples for the route
        """
        exit_switch_pos = info.get("exit_switch_pos", None)
        exit_door_pos = info.get("exit_door_pos", None)
        is_success = info.get("is_success", False)

        # Try to get episode reward from various possible locations
        episode_reward = 0.0

        episode_reward = float(info["r"])

        # Get curriculum stage and generator type (more meaningful than level ID for display)
        curriculum_stage = info.get("curriculum_stage", "unknown")
        curriculum_generator = info.get("curriculum_generator", "unknown")
        level_id = info.get("level_id", f"env_{env_idx}")

        # Get episode length
        episode_length = info.get("l", len(route_positions))

        # Use stored tile, mine, and locked door data (captured at episode start, not end)
        # This ensures we get the correct level data before auto-reset
        tiles = self.env_routes[env_idx].get("tiles", {})
        mines = self.env_routes[env_idx].get("mines", [])
        locked_doors = self.env_routes[env_idx].get("locked_doors", [])

        route_data = {
            "positions": list(route_positions),
            "level_id": level_id,
            "curriculum_stage": curriculum_stage,
            "curriculum_generator": curriculum_generator,
            "timestep": self.num_timesteps,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "exit_switch_pos": exit_switch_pos,
            "exit_door_pos": exit_door_pos,
            "tiles": tiles,
            "mines": mines,
            "locked_doors": locked_doors,
            "is_success": is_success,
        }

        self.save_queue.append(route_data)

        if self.verbose >= 1:
            logger.info(
                f"Queued route visualization for env {env_idx} - "
                f"Stage: {curriculum_stage}, Level: {level_id}, "
            )

    def _process_save_queue(self) -> None:
        """Process queued routes and save visualizations."""
        if not self.save_queue:
            return

        if not self._import_matplotlib():
            logger.warning(
                "Cannot save route visualizations - matplotlib not available"
            )
            self.save_queue.clear()
            return

        if (
            self.async_save
            and self.save_thread is None
            or not self.save_thread.is_alive()
        ):
            # Start async save thread
            self.save_thread = threading.Thread(
                target=self._save_routes_async,
                args=(list(self.save_queue),),
                daemon=True,
            )
            self.save_thread.start()
            self.save_queue.clear()
        else:
            # Save synchronously
            for route_data in self.save_queue:
                self._save_route_visualization(route_data)
            self.save_queue.clear()

    def _save_routes_async(self, routes: List[Dict[str, Any]]) -> None:
        """Save routes asynchronously in background thread.

        Args:
            routes: List of route data dictionaries
        """
        for route_data in routes:
            try:
                self._save_route_visualization(route_data)
            except Exception as e:
                logger.error(f"Error saving route visualization: {e}")

    def _save_route_visualization(self, route_data: Dict[str, Any]) -> None:
        """Save a single route visualization.

        Args:
            route_data: Route data dictionary
        """
        positions = np.array(route_data["positions"])
        if len(positions) == 0:
            return

        # Create figure
        fig, ax = self.plt.subplots(
            figsize=(self.image_size[0] / 100, self.image_size[1] / 100), dpi=100
        )

        # Set background color to light green for successful routes
        is_success = route_data.get("is_success", False)
        if is_success:
            light_green = "#90EE90"  # Light green color
            fig.patch.set_facecolor(light_green)
            ax.set_facecolor(light_green)

        # Set aspect ratio to equal for proper level visualization
        ax.set_aspect("equal")

        # Calculate bounds from route positions with padding
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]

        # Add padding around the route (in pixels)
        padding = 100  # 100 pixels padding on each side
        x_min = np.min(x_coords) - padding
        x_max = np.max(x_coords) + padding
        y_min = np.min(y_coords) - padding
        y_max = np.max(y_coords) + padding

        # Include exit switch and door in bounds if available
        if route_data.get("exit_switch_pos"):
            switch_x, switch_y = route_data["exit_switch_pos"]
            x_min = min(x_min, switch_x - padding)
            x_max = max(x_max, switch_x + padding)
            y_min = min(y_min, switch_y - padding)
            y_max = max(y_max, switch_y + padding)

        if route_data.get("exit_door_pos"):
            door_x, door_y = route_data["exit_door_pos"]
            x_min = min(x_min, door_x - padding)
            x_max = max(x_max, door_x + padding)
            y_min = min(y_min, door_y - padding)
            y_max = max(y_max, door_y + padding)

        # Include locked doors in bounds if available
        if route_data.get("locked_doors"):
            for door in route_data["locked_doors"]:
                # Include switch position
                switch_x = door.get("switch_x", 0)
                switch_y = door.get("switch_y", 0)
                x_min = min(x_min, switch_x - padding)
                x_max = max(x_max, switch_x + padding)
                y_min = min(y_min, switch_y - padding)
                y_max = max(y_max, switch_y + padding)
                # Include door segment position
                door_x = door.get("door_x", 0)
                door_y = door.get("door_y", 0)
                x_min = min(x_min, door_x - padding)
                x_max = max(x_max, door_x + padding)
                y_min = min(y_min, door_y - padding)
                y_max = max(y_max, door_y + padding)

        # Render tiles if available (must be before route so tiles are behind)
        # Only render tiles within the visible bounds for performance
        if route_data.get("tiles"):
            # Filter tiles to only those in visible area
            # Tile coordinates are in grid units, convert bounds to grid units
            tile_x_min = int((x_min / 24.0) - 1)
            tile_x_max = int((x_max / 24.0) + 2)
            tile_y_min = int((y_min / 24.0) - 1)
            tile_y_max = int((y_max / 24.0) + 2)

            visible_tiles = {
                coords: tile_type
                for coords, tile_type in route_data["tiles"].items()
                if tile_x_min <= coords[0] <= tile_x_max
                and tile_y_min <= coords[1] <= tile_y_max
            }

            render_tiles_to_axis(
                ax,
                visible_tiles,
                tile_size=24.0,
                tile_color="#606060",  # Dark gray for tiles
                alpha=1.0,  # Solid fill
            )
        # Render mines if available (after tiles, before route)
        # Mines already have visibility culling in the render function
        if route_data.get("mines"):
            render_mines_to_axis(
                ax,
                route_data["mines"],
                tile_color="#FF0000",  # Red for dangerous mines
                safe_color="#44FFFF",  # Cyan for safe mines
                alpha=0.8,
            )

        # Render locked doors if available (after mines, before route)
        if route_data.get("locked_doors"):
            for door in route_data["locked_doors"]:
                switch_x = door.get("switch_x", 0)
                switch_y = door.get("switch_y", 0)
                door_x = door.get("door_x", 0)
                door_y = door.get("door_y", 0)
                is_closed = door.get("closed", True)
                segment_x1 = door.get("segment_x1", door_x)
                segment_y1 = door.get("segment_y1", door_y)
                segment_x2 = door.get("segment_x2", door_x)
                segment_y2 = door.get("segment_y2", door_y)

                # Draw door segment (only if closed)
                if is_closed:
                    ax.plot(
                        [segment_x1, segment_x2],
                        [segment_y1, segment_y2],
                        color="#000000",  # Black for closed door segments
                        linewidth=3,
                        alpha=0.9,
                        zorder=3,  # Above tiles but below route
                        label="Locked Door"
                        if door == route_data["locked_doors"][0]
                        else "",
                    )
                else:
                    # Draw open door segment (dashed, lighter) for debugging
                    ax.plot(
                        [segment_x1, segment_x2],
                        [segment_y1, segment_y2],
                        color="#000000",
                        linewidth=2,
                        linestyle="--",
                        alpha=0.3,
                        zorder=1,  # Behind everything
                    )

                # Draw switch position
                switch_color = (
                    "#00AA00" if not is_closed else "#FF6600"
                )  # Green if collected, orange if not
                ax.scatter(
                    switch_x,
                    switch_y,
                    c=switch_color,
                    s=80,
                    marker="s",  # Square marker for switches
                    zorder=6,
                    edgecolors="white",
                    linewidths=1.5,
                    label="Door Switch"
                    if door == route_data["locked_doors"][0]
                    else "",
                )

                # Draw switch radius circle
                switch_radius = door.get("switch_radius", 5)
                switch_circle = self.plt.Circle(
                    (switch_x, switch_y),
                    switch_radius,
                    color=switch_color,
                    fill=False,
                    linestyle=":",
                    linewidth=1,
                    alpha=0.5,
                    zorder=2,
                )
                ax.add_patch(switch_circle)

        # Plot route with color gradient (blue=start, green=end)
        num_points = len(positions)
        colors = self.plt.cm.viridis(np.linspace(0, 1, num_points))

        # Plot positions as a scatter plot with gradient
        for i in range(num_points - 1):
            ax.plot(
                positions[i : i + 2, 0],
                positions[i : i + 2, 1],
                color=colors[i],
                linewidth=2,
                alpha=0.7,
                zorder=2,  # Above tiles (1) but below mines (3) and markers (5+)
            )

        # Mark start position
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            c="blue",
            s=200,
            marker="o",
            label="Start",
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )

        # Mark agent's final position (where they actually ended)
        agent_end_x, agent_end_y = positions[-1, 0], positions[-1, 1]
        ax.scatter(
            agent_end_x,
            agent_end_y,
            c="green",
            s=200,
            marker="o",
            label="Agent End",
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )

        # Draw agent radius circle at end position to show overlap zone
        agent_radius = 10  # N++ agent radius
        agent_circle = self.plt.Circle(
            (agent_end_x, agent_end_y),
            agent_radius,
            color="green",
            fill=False,
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            zorder=4,
        )
        ax.add_patch(agent_circle)
        # Mark exit switch position (if available)
        if route_data.get("exit_switch_pos") is not None:
            exit_x, exit_y = route_data["exit_switch_pos"]
            ax.scatter(
                exit_x,
                exit_y,
                c="red",
                s=100,
                marker="D",
                label="Exit Switch",
                zorder=6,
                edgecolors="orange",
                linewidths=2,
            )

            # Draw switch radius circle to show trigger zone
            switch_radius = 6  # N++ exit switch radius
            switch_circle = self.plt.Circle(
                (exit_x, exit_y),
                switch_radius,
                color="orange",
                fill=False,
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                zorder=3,
            )
            ax.add_patch(switch_circle)

        # Mark exit door position (if available)
        if route_data.get("exit_door_pos") is not None:
            door_x, door_y = route_data["exit_door_pos"]
            ax.scatter(
                door_x,
                door_y,
                c="purple",
                s=200,
                marker="*",
                label="Exit Door",
                zorder=6,
                edgecolors="white",
                linewidths=2,
            )

            # Draw door radius circle to show exit zone
            door_radius = 12  # N++ exit door radius
            door_circle = self.plt.Circle(
                (door_x, door_y),
                door_radius,
                color="purple",
                fill=False,
                linestyle=":",
                linewidth=1,
                alpha=0.4,
                zorder=3,
            )
            ax.add_patch(door_circle)

        # Add labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Build title with curriculum stage and generator prominently displayed
        is_success = route_data.get("is_success", False)
        title = "Successful Route" if is_success else "Failed Route"
        title_parts = [f"{title} - Step {route_data['timestep']}"]

        # Show generator type if available
        if (
            route_data.get("curriculum_generator")
            and route_data["curriculum_generator"] != "unknown"
        ):
            title_parts.append(f"{route_data['curriculum_generator']}")

        # Show episode stats
        title_parts.append(f"{route_data['episode_length']}")
        title_parts.append(f"Reward: {route_data['episode_reward']:.2f}")

        # Combine into multi-line title
        ax.set_title(title_parts[0] + "\n" + " | ".join(title_parts[1:]))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Set axis limits to the calculated bounds (zoomed to route area)
        # Ensure minimum range to prevent overly thin visualizations
        min_range = 100  # Minimum range in pixels
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Expand bounds if too narrow (vertical/horizontal corridors)
        if x_range < min_range:
            x_center = (x_min + x_max) / 2
            x_min = x_center - min_range / 2
            x_max = x_center + min_range / 2

        if y_range < min_range:
            y_center = (y_min + y_max) / 2
            y_min = y_center - min_range / 2
            y_max = y_center + min_range / 2

        # Apply the calculated limits
        # Y-axis inverted: set_ylim(y_max, y_min) ensures Y=0 at top (N++ coords)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Inverted for N++ coordinate system

        # Save figure with curriculum stage in filename
        stage = route_data.get("curriculum_stage", "unknown")
        level_id = route_data["level_id"]
        # Sanitize stage name for filename (replace spaces and special chars)
        stage_clean = stage.replace(" ", "_").replace("/", "-")
        filename = (
            f"route_step{route_data['timestep']:09d}_{stage_clean}_{level_id}.png"
        )
        filepath = self.save_dir / filename

        # Set x-axis ticks to appear at the top for a "top-left origin" feel
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        fig.savefig(filepath, dpi=100, bbox_inches="tight")
        self.plt.close(fig)

        # Track saved file
        self.saved_files.append(filepath)
        self.total_routes_saved += 1

        # Log to TensorBoard if available
        if self.tb_writer is not None:
            try:
                from PIL import Image
                import torch

                # Load image and convert to tensor
                img = Image.open(filepath)
                img_array = np.array(img)

                # Convert to format expected by TensorBoard (CHW)
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    self.tb_writer.add_image(
                        f"routes/level_{route_data['level_id']}",
                        img_tensor,
                        route_data["timestep"],
                    )
            except Exception as e:
                logger.info(f"Could not log route to TensorBoard: {e}")

        # Cleanup old files if we exceed limit
        self._cleanup_old_files()

        if self.verbose >= 1:
            logger.info(
                f"Saved route visualization: {filename} "
                f"(total saved: {self.total_routes_saved})"
            )

    def _cleanup_old_files(self) -> None:
        """Remove old visualization files to prevent disk bloat."""
        while len(self.saved_files) > self.max_stored_routes:
            old_file = self.saved_files.popleft()
            try:
                if old_file.exists():
                    old_file.unlink()
                    if self.verbose >= 2:
                        logger.info(f"Removed old route visualization: {old_file.name}")
            except Exception as e:
                logger.warning(f"Could not remove old file {old_file}: {e}")

    def _get_tile_data(self, env_idx: int = 0) -> Dict[Tuple[int, int], int]:
        """Extract tile data from the environment.

        Args:
            env_idx: Environment index (default 0)

        Returns:
            Dictionary mapping (x, y) grid coordinates to tile type values,
            or empty dict if tiles are not accessible
        """
        try:
            # Start with training_env - it might be VecNormalize wrapping VecEnv
            venv = self.training_env

            # Unwrap VecNormalize wrapper if present
            if hasattr(venv, "venv"):
                venv = venv.venv

            # Now venv should be the actual VecEnv (SubprocVecEnv or DummyVecEnv)
            # For SubprocVecEnv, use env_method to call remote method
            if hasattr(venv, "env_method") and not hasattr(venv, "envs"):
                # SubprocVecEnv - call remote method
                try:
                    results = venv.env_method(
                        "get_route_visualization_tile_data", indices=[env_idx]
                    )
                    if results and len(results) > 0 and results[0] is not None:
                        return dict(results[0])
                    else:
                        if self.verbose >= 1:
                            logger.warning(
                                f"env_method returned empty result for tile data (env {env_idx})"
                            )
                        return {}
                except Exception as e:
                    if self.verbose >= 1:
                        logger.warning(
                            f"env_method failed for tile data (env {env_idx}): {e}. "
                            "Falling back to direct access."
                        )
                    # Fall through to direct access fallback

            # Direct access for DummyVecEnv or fallback from SubprocVecEnv failure
            if hasattr(venv, "envs"):
                # DummyVecEnv - direct access
                env = venv.envs[env_idx]
            else:
                # Can't access - return empty
                if self.verbose >= 1:
                    logger.warning(
                        f"Cannot access tile data for env {env_idx}: "
                        "No accessible environment found."
                    )
                return {}

            # Unwrap wrappers to get to base environment
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env

            if hasattr(env, "nplay_headless"):
                tile_dic = env.nplay_headless.get_tile_data()
                return dict(tile_dic)  # Make a copy
            else:
                if self.verbose >= 1:
                    logger.warning(
                        f"Environment {env_idx} does not have nplay_headless attribute. "
                        "Cannot extract tile data."
                    )
                return {}
        except Exception as e:
            if self.verbose >= 1:
                logger.warning(f"Could not extract tile data for env {env_idx}: {e}")
            return {}

    def _get_mine_data(self, env_idx: int = 0) -> List[Dict[str, Any]]:
        """Extract mine data from the environment.

        Args:
            env_idx: Environment index (default 0)

        Returns:
            List of mine dictionaries with keys: x, y, state, radius
            Returns empty list if mines are not accessible
        """
        try:
            # Start with training_env - it might be VecNormalize wrapping VecEnv
            venv = self.training_env

            # Unwrap VecNormalize wrapper if present
            if hasattr(venv, "venv"):
                venv = venv.venv

            # Now venv should be the actual VecEnv (SubprocVecEnv or DummyVecEnv)
            # For SubprocVecEnv, use env_method to call remote method
            if hasattr(venv, "env_method") and not hasattr(venv, "envs"):
                # SubprocVecEnv - call remote method
                try:
                    results = venv.env_method(
                        "get_route_visualization_mine_data", indices=[env_idx]
                    )
                    if results and len(results) > 0 and results[0] is not None:
                        return results[0]
                    else:
                        if self.verbose >= 1:
                            logger.warning(
                                f"env_method returned empty result for mine data (env {env_idx})"
                            )
                        return []
                except Exception as e:
                    if self.verbose >= 1:
                        logger.warning(
                            f"env_method failed for mine data (env {env_idx}): {e}. "
                            "Falling back to direct access."
                        )
                    # Fall through to direct access fallback

            # Direct access for DummyVecEnv or fallback from SubprocVecEnv failure
            if hasattr(venv, "envs"):
                # DummyVecEnv - direct access
                env = venv.envs[env_idx]
            else:
                # Can't access - return empty
                if self.verbose >= 1:
                    logger.warning(
                        f"Cannot access mine data for env {env_idx}: "
                        "No accessible environment found."
                    )
                return []

            # Unwrap wrappers to get to base environment
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env

            if hasattr(env, "nplay_headless"):
                # Use unified method that handles both entity types 1 and 21
                return env.nplay_headless.get_all_mine_data_for_visualization()
            else:
                if self.verbose >= 1:
                    logger.warning(
                        f"Environment {env_idx} does not have nplay_headless attribute. "
                        "Cannot extract mine data."
                    )
                return []
        except Exception as e:
            if self.verbose >= 1:
                logger.warning(f"Could not extract mine data for env {env_idx}: {e}")
            return []

    def _get_locked_door_data(self, env_idx: int = 0) -> List[Dict[str, Any]]:
        """Extract locked door data from the environment.

        Args:
            env_idx: Environment index (default 0)

        Returns:
            List of locked door dictionaries with keys:
            - switch_x, switch_y: Switch position
            - door_x, door_y: Door segment center position
            - segment_x1, segment_y1, segment_x2, segment_y2: Door segment endpoints
            - closed: Whether door is closed (True) or open (False)
            - active: Whether switch is still collectible (True = not collected, door closed)
            - switch_radius: Switch radius
            Returns empty list if locked doors are not accessible
        """
        try:
            # Start with training_env - it might be VecNormalize wrapping VecEnv
            venv = self.training_env

            # Unwrap VecNormalize wrapper if present
            if hasattr(venv, "venv"):
                venv = venv.venv

            # Now venv should be the actual VecEnv (SubprocVecEnv or DummyVecEnv)
            # For SubprocVecEnv, use env_method to call remote method
            if hasattr(venv, "env_method") and not hasattr(venv, "envs"):
                # SubprocVecEnv - call remote method
                try:
                    results = venv.env_method(
                        "get_route_visualization_locked_door_data", indices=[env_idx]
                    )
                    if results and len(results) > 0 and results[0] is not None:
                        return results[0]
                    else:
                        if self.verbose >= 1:
                            logger.warning(
                                f"env_method returned empty result for locked door data (env {env_idx})"
                            )
                        return []
                except Exception as e:
                    if self.verbose >= 1:
                        logger.warning(
                            f"env_method failed for locked door data (env {env_idx}): {e}. "
                            "Falling back to direct access."
                        )
                    # Fall through to direct access fallback

            # Direct access for DummyVecEnv or fallback from SubprocVecEnv failure
            if hasattr(venv, "envs"):
                # DummyVecEnv - direct access
                env = venv.envs[env_idx]
            else:
                # Can't access - return empty
                if self.verbose >= 1:
                    logger.warning(
                        f"Cannot access locked door data for env {env_idx}: "
                        "No accessible environment found."
                    )
                return []

            # Unwrap wrappers to get to base environment
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env

            if hasattr(env, "nplay_headless"):
                locked_doors = []
                locked_door_entities = env.nplay_headless.locked_doors()

                # Import door constants
                try:
                    from nclone.entity_classes.entity_door_locked import (
                        EntityDoorLocked,
                    )
                except ImportError:
                    # Fallback to default values if import fails
                    EntityDoorLocked = type("EntityDoorLocked", (), {"RADIUS": 5})

                for door_entity in locked_door_entities:
                    # Get switch position (entity position)
                    switch_x = float(getattr(door_entity, "xpos", 0.0))
                    switch_y = float(getattr(door_entity, "ypos", 0.0))

                    # Get door segment
                    segment = getattr(door_entity, "segment", None)
                    if segment:
                        segment_x1 = float(getattr(segment, "x1", 0.0))
                        segment_y1 = float(getattr(segment, "y1", 0.0))
                        segment_x2 = float(getattr(segment, "x2", 0.0))
                        segment_y2 = float(getattr(segment, "y2", 0.0))
                        door_x = (segment_x1 + segment_x2) * 0.5
                        door_y = (segment_y1 + segment_y2) * 0.5
                    else:
                        # Fallback if segment not available
                        segment_x1 = segment_y1 = segment_x2 = segment_y2 = 0.0
                        door_x = door_y = 0.0

                    # Get door state
                    closed = bool(getattr(door_entity, "closed", True))
                    active = bool(getattr(door_entity, "active", True))

                    locked_doors.append(
                        {
                            "switch_x": switch_x,
                            "switch_y": switch_y,
                            "door_x": door_x,
                            "door_y": door_y,
                            "segment_x1": segment_x1,
                            "segment_y1": segment_y1,
                            "segment_x2": segment_x2,
                            "segment_y2": segment_y2,
                            "closed": closed,
                            "active": active,
                            "switch_radius": float(EntityDoorLocked.RADIUS),
                        }
                    )

                return locked_doors
            else:
                if self.verbose >= 1:
                    logger.warning(
                        f"Environment {env_idx} does not have nplay_headless attribute. "
                        "Cannot extract locked door data."
                    )
                return []
        except Exception as e:
            if self.verbose >= 1:
                logger.warning(
                    f"Could not extract locked door data for env {env_idx}: {e}"
                )
            return []

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Process any remaining queued saves
        self._process_save_queue()

        # Wait for async saves to complete
        if self.save_thread is not None and self.save_thread.is_alive():
            self.save_thread.join(timeout=10)

        logger.info(
            f"Route visualization completed. Total routes saved: {self.total_routes_saved}"
        )
