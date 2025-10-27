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
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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
            lambda: {"positions": [], "level_id": None, "start_time": 0}
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
                    self.env_routes[env_idx]["positions"].append((pos_x, pos_y))
        except Exception as e:
            self.logger.debug(f"Could not track position for env {env_idx}: {e}")

    def _handle_episode_end(self, env_idx: int, info: Dict[str, Any]) -> None:
        """Handle episode completion and potentially save route.

        Args:
            env_idx: Environment index
            info: Episode info dictionary
        """
        # DEBUG: Log all keys in info dict to understand what data is available
        print(self.verbose)
        if self.verbose >= 2:
            self.logger.debug(
                f"Episode end info keys for env {env_idx}: {list(info.keys())}"
            )
            if "episode" in info:
                self.logger.debug(
                    f"  episode dict keys: {list(info['episode'].keys())}"
                )
                self.logger.debug(f"  episode dict: {info['episode']}")

        # Check if episode was successful
        is_success = False
        if "success" in info:
            is_success = info["success"]
        elif "is_success" in info:
            is_success = info["is_success"]
        # Get route from PositionTrackingWrapper if available (more reliable than step-by-step tracking)
        route_positions = None
        route_source = "none"

        if "episode_route" in info:
            route_positions = info["episode_route"]
            route_source = f"episode_route ({len(route_positions)} positions)"
        elif self.env_routes[env_idx]["positions"]:
            route_positions = self.env_routes[env_idx]["positions"]
            route_source = f"step-by-step tracking ({len(route_positions)} positions)"

        if self.verbose >= 1:
            logger.info(
                f"Episode end env {env_idx}: success={is_success}, "
                f"route_source={route_source}, "
                f"step_tracked_positions={len(self.env_routes[env_idx]['positions'])}"
            )

        # Only save routes for successful completions with valid position data
        if is_success and route_positions and len(route_positions) > 0:
            # Check if we should save this route
            if self.routes_saved_this_checkpoint < self.max_routes_per_checkpoint:
                self._queue_route_save(env_idx, info, route_positions)
                self.routes_saved_this_checkpoint += 1

        # Clear route data for this environment
        self.env_routes[env_idx] = {
            "positions": [],
            "level_id": info.get("level_id", None),
            "start_time": self.num_timesteps,
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
        # Try to get exit switch and door positions from environment
        exit_switch_pos = None
        exit_door_pos = None
        try:
            # Access the base environment
            env = self.training_env.envs[env_idx]
            while hasattr(env, "env"):
                env = env.env

            # Get exit switch and door positions from nplay_headless
            if hasattr(env, "nplay_headless"):
                if hasattr(env.nplay_headless, "exit_switch_position"):
                    exit_switch_pos = env.nplay_headless.exit_switch_position()
                if hasattr(env.nplay_headless, "exit_door_position"):
                    exit_door_pos = env.nplay_headless.exit_door_position()
        except Exception as e:
            self.logger.debug(f"Could not get exit switch/door positions: {e}")

        # Try to get episode reward from various possible locations
        episode_reward = 0.0

        episode_reward = float(info["r"])

        # Get curriculum stage (more meaningful than level ID for display)
        curriculum_stage = info.get("curriculum_stage", "unknown")
        level_id = info.get("level_id", f"env_{env_idx}")

        # Get episode length
        episode_length = info.get("l", len(route_positions))

        route_data = {
            "positions": list(route_positions),
            "level_id": level_id,
            "curriculum_stage": curriculum_stage,
            "timestep": self.num_timesteps,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "exit_switch_pos": exit_switch_pos,
            "exit_door_pos": exit_door_pos,
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
            )

        # Mark start position
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            c="blue",
            s=150,
            marker="o",
            label="Start",
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )

        # Mark agent's final position (where they actually ended)
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            c="green",
            s=150,
            marker="o",
            label="Agent End",
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )

        # Mark exit switch position (if available)
        if route_data.get("exit_switch_pos") is not None:
            exit_x, exit_y = route_data["exit_switch_pos"]
            ax.scatter(
                exit_x,
                exit_y,
                c="red",
                s=200,
                marker="*",
                label="Exit Switch",
                zorder=6,
                edgecolors="yellow",
                linewidths=2,
            )

        # Mark exit door position (if available)
        if route_data.get("exit_door_pos") is not None:
            door_x, door_y = route_data["exit_door_pos"]
            ax.scatter(
                door_x,
                door_y,
                c="purple",
                s=200,
                marker="D",
                label="Exit Door",
                zorder=6,
                edgecolors="white",
                linewidths=2,
            )

        # Add labels and title
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position (0 = Top)")

        # Build title with curriculum stage prominently displayed
        title_parts = [f"Successful Route - Step {route_data['timestep']}"]

        # Show curriculum stage if available
        if (
            route_data.get("curriculum_stage")
            and route_data["curriculum_stage"] != "unknown"
        ):
            title_parts.append(f"Stage: {route_data['curriculum_stage']}")

        # Show level ID
        title_parts.append(f"Level: {route_data['level_id']}")

        # Show episode stats
        title_parts.append(f"Length: {route_data['episode_length']}")
        title_parts.append(f"Reward: {route_data['episode_reward']:.2f}")

        # Combine into multi-line title
        ax.set_title(title_parts[0] + "\n" + " | ".join(title_parts[1:]))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # IMPORTANT: Invert Y-axis so Y=0 is at top (matches level coordinate system)
        ax.invert_yaxis()

        # Set aspect ratio to equal for proper level visualization
        ax.set_aspect("equal")

        # Save figure with curriculum stage in filename
        stage = route_data.get("curriculum_stage", "unknown")
        level_id = route_data["level_id"]
        # Sanitize stage name for filename (replace spaces and special chars)
        stage_clean = stage.replace(" ", "_").replace("/", "-")
        filename = (
            f"route_step{route_data['timestep']:09d}_{stage_clean}_{level_id}.png"
        )
        filepath = self.save_dir / filename

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
                self.logger.debug(f"Could not log route to TensorBoard: {e}")

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
                        self.logger.debug(
                            f"Removed old route visualization: {old_file.name}"
                        )
            except Exception as e:
                logger.warning(f"Could not remove old file {old_file}: {e}")

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
