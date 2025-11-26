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
import random
import threading
from collections import deque
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
        episode_sampling_rate: float = 1.0,
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
            episode_sampling_rate: Fraction of episodes to track (0.0 to 1.0).
                Reduces overhead by only tracking a percentage of episodes.
                Default 1.0 tracks all episodes. Use 0.1 to track 10% of episodes.
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
        self.episode_sampling_rate = max(
            0.0, min(1.0, episode_sampling_rate)
        )  # Clamp to [0, 1]

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track which environments are currently being sampled for this episode
        # Maps env_idx -> bool indicating if we're tracking this episode
        self.tracking_episode = {}

        # Episode counter per environment for consistent sampling
        self.episode_counters = {}

        # Contamination detection: track episode hashes to detect duplicates
        # Maps episode_hash -> (env_idx, timestep) for recent episodes
        self.recent_episode_hashes = {}

        # Per-environment rate limiting: track last save timestep per env
        # Prevents saving too many near-identical routes from undertrained agents
        # Maps env_idx -> last_timestep_saved
        self.last_save_timestep_per_env = {}

        # Minimum timesteps between saves from same environment
        # With N parallel envs, consecutive episodes from same env are N steps apart
        # Set to 50 to require ~10 episodes between saves (assuming 5 envs)
        self.min_timesteps_between_saves = 50

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
                        print(
                            "Could not set matplotlib backend to Agg - visualization may fail in headless mode"
                        )

                import matplotlib.pyplot as plt

                self.plt = plt
                self._matplotlib_imported = True
            except ImportError:
                print("matplotlib not available - route visualization disabled")
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
            "⚠️  Route visualization requires position tracking (integrated into BaseNppEnvironment)"
        )
        logger.info(
            "   If routes are not being captured, ensure the wrapper is in the environment pipeline"
        )

    def _on_step(self) -> bool:
        """Called after each environment step.

        Note: With frame skipping, this is called once per ACTION (not per frame).
        Position tracking is integrated into BaseNppEnvironment automatically.

        Returns:
            bool: If False, training will be stopped
        """
        if "dones" in self.locals and "infos" in self.locals:
            dones = self.locals["dones"]
            infos = self.locals["infos"]

            for env_idx, (done, info) in enumerate(zip(dones, infos)):
                # Decide whether to sample this episode (only once at episode start)
                # Detected by checking if we haven't made a decision yet for this env
                if (
                    env_idx not in self.tracking_episode
                    or not self.tracking_episode[env_idx]
                ):
                    if env_idx not in self.episode_counters:
                        self.episode_counters[env_idx] = 0

                    # Only decide on sampling if we're not currently tracking
                    # (meaning this is a new episode)
                    if not self.tracking_episode.get(env_idx, False):
                        self.episode_counters[env_idx] += 1
                        should_track = random.random() < self.episode_sampling_rate
                        self.tracking_episode[env_idx] = should_track

                        if self.verbose >= 2 and should_track:
                            logger.debug(
                                f"Will track episode for env {env_idx} "
                                f"(episode #{self.episode_counters[env_idx]})"
                            )

                # Handle episode completion
                # BaseNppEnvironment adds info["episode_route"] automatically
                # CRITICAL: Only handle episode end ONCE per episode
                # VecEnv may report done=True for multiple steps before reset
                if done and self.tracking_episode.get(env_idx, False):
                    # Process episode end immediately and clear tracking flag
                    self._handle_episode_end(env_idx, info)
                    # Clear tracking flag to prevent duplicate processing
                    self.tracking_episode[env_idx] = False

        # Check if it's time to save visualizations
        if self.num_timesteps - self.last_visualization_step >= self.visualization_freq:
            self._process_save_queue()
            self.routes_saved_this_checkpoint = 0
            self.last_visualization_step = self.num_timesteps
            # Clear contamination detection hashes at each checkpoint to prevent unbounded growth
            self.recent_episode_hashes.clear()
            # Clear per-env rate limiting at each checkpoint (start fresh)
            self.last_save_timestep_per_env.clear()

        # MEMORY OPTIMIZATION: Periodically clean up stale tracking data
        # to prevent unbounded growth of defaultdicts in long-running training
        # Check every 10000 timesteps
        if self.num_timesteps % 10000 == 0:
            self._cleanup_stale_tracking_data()

        return True

    def _handle_episode_end(self, env_idx: int, info: Dict[str, Any]) -> None:
        """Handle episode completion and potentially save route.

        This is called once per episode end (after frame skipping completes).
        BaseNppEnvironment provides info["episode_route"] with all positions.

        Args:
            env_idx: Environment index
            info: Episode info dictionary (includes episode_route from BaseNppEnvironment)
        """
        # Check if we were tracking this episode
        should_save = self.tracking_episode.get(env_idx, False)

        # Always reset tracking flag for next episode
        # NOTE: Caller (_on_step) already set this to False to prevent duplicate calls
        if env_idx in self.tracking_episode:
            del self.tracking_episode[env_idx]

        # Early exit if not tracking this episode
        if not should_save:
            return

        # Get route from BaseNppEnvironment (added to info at episode end)
        route_positions = info.get("episode_route", None)

        # DEBUG: Log route details to help diagnose duplicate routes with different lengths
        if self.verbose >= 2:
            logger.debug(
                f"Env {env_idx} episode end: {len(route_positions)} positions, "
                f"first 3: {route_positions[:3]}, last 3: {route_positions[-3:]}"
            )

        # Only save if we haven't hit the checkpoint limit
        if self.routes_saved_this_checkpoint < self.max_routes_per_checkpoint:
            # Capture level data at episode end (while we still have access before next reset)
            # Store temporarily for _queue_route_save to use
            tiles = self._get_tile_data(env_idx)
            mines = self._get_mine_data(env_idx)
            locked_doors = self._get_locked_door_data(env_idx)

            # Add level data to info for queue_route_save to use
            info["_route_viz_tiles"] = tiles
            info["_route_viz_mines"] = mines
            info["_route_viz_locked_doors"] = locked_doors

            was_queued = self._queue_route_save(env_idx, info, route_positions)
            # Only increment counter if route was actually queued (not a duplicate)
            if was_queued:
                self.routes_saved_this_checkpoint += 1

    def _queue_route_save(
        self,
        env_idx: int,
        info: Dict[str, Any],
        route_positions: List[Tuple[float, float]],
    ) -> bool:
        """Queue a route for saving.

        Args:
            env_idx: Environment index
            info: Episode info dictionary
            route_positions: List of (x, y) position tuples for the route

        Returns:
            True if route was queued, False if skipped (duplicate)
        """
        exit_switch_pos = info.get("exit_switch_pos", None)
        exit_door_pos = info.get("exit_door_pos", None)
        is_success = info.get("is_success", False)

        # Try to get episode reward from various possible locations
        # SB3 stores episode info in info["episode"] dict or directly in info
        if "episode" in info and isinstance(info["episode"], dict):
            episode_reward = float(info["episode"].get("r", 0.0))
            episode_length = int(info["episode"].get("l", len(route_positions)))
        else:
            episode_reward = float(info.get("r", 0.0))
            episode_length = int(info.get("l", len(route_positions)))

            # Debug logging if reward is missing or zero (potential issue)
            if self.verbose >= 1 and episode_reward == 0.0:
                logger.warning(
                    f"Episode reward is 0.0 for env {env_idx}. "
                    f"is_success={is_success}, "
                    f"episode_length={episode_length}, "
                    f"'episode' in info: {'episode' in info}, "
                    f"'r' in info: {'r' in info}, "
                    f"info['episode'] if present: {info.get('episode', 'N/A')}, "
                    f"info['r'] if present: {info.get('r', 'N/A')}, "
                    f"Available keys: {list(info.keys())[:20]}"
                )

        # Get curriculum stage and generator type (more meaningful than level ID for display)
        curriculum_stage = info.get("curriculum_stage", "unknown")
        curriculum_generator = info.get("curriculum_generator", "unknown")
        level_id = info.get("level_id", f"env_{env_idx}")

        # Get terminal impact (boolean indicating if ninja died from terminal impact)
        terminal_impact = info.get("terminal_impact", False)

        # Get worker PID if available (for multiprocessing debugging)
        worker_pid = info.get("worker_pid", None)

        # Get frame skip info if available
        frame_skip_stats = info.get("frame_skip_stats", {})
        skip_value = frame_skip_stats.get("skip_value", None)

        # Use level data captured at episode end (added by _handle_episode_end)
        # This ensures we get the correct level data before auto-reset
        tiles = info.get("_route_viz_tiles", {})
        mines = info.get("_route_viz_mines", [])
        locked_doors = info.get("_route_viz_locked_doors", [])

        # Calculate episode hash for contamination detection
        # Use first 5 positions to create a unique identifier
        episode_hash = self._calculate_episode_hash(route_positions)

        # DEBUG: Check for near-duplicate routes (same env, close in time, similar routes)
        # This helps diagnose why visually identical routes have different hashes
        last_save = self.last_save_timestep_per_env.get(env_idx, 0)
        timesteps_since_last = self.num_timesteps - last_save

        if timesteps_since_last > 0 and timesteps_since_last < 100:
            # Recent save from same env - check for position tracking issues
            if self.verbose >= 1:
                logger.warning(
                    f"⚠️  Env {env_idx}: New route only {timesteps_since_last} steps after last save. "
                    f"Route length: {len(route_positions)}, Hash: {episode_hash}. "
                    f"First 3 pos: {route_positions[:3]}, Last 3 pos: {route_positions[-3:]}. "
                    f"This may indicate position tracking bugs or undertrained agent repetition."
                )

        # Check 1: Per-environment rate limiting
        # Prevent saving too many near-identical routes from undertrained agents
        if timesteps_since_last < self.min_timesteps_between_saves:
            if self.verbose >= 1:
                logger.info(
                    f"Skipping route from env {env_idx} at step {self.num_timesteps} "
                    f"(only {timesteps_since_last} steps since last save, "
                    f"minimum is {self.min_timesteps_between_saves}). "
                    f"Hash: {episode_hash}"
                )
            return False  # Route not queued due to rate limit

        # Check 2: Exact duplicate detection (same hash)
        should_skip_duplicate = False
        if episode_hash in self.recent_episode_hashes:
            prev_env_idx, prev_timestep = self.recent_episode_hashes[episode_hash]
            if prev_env_idx != env_idx:
                # Different environment produced same route - critical bug!
                logger.warning(
                    f"⚠️  ROUTE CONTAMINATION DETECTED: "
                    f"Env {env_idx} at step {self.num_timesteps} has IDENTICAL route hash ({episode_hash}) "
                    f"as env {prev_env_idx} at step {prev_timestep}. "
                    f"This suggests data sharing or env_idx confusion!"
                )
            else:
                # Same environment produced duplicate route - skip to avoid redundant saves
                if self.verbose >= 1:
                    logger.info(
                        f"Skipping exact duplicate route from env {env_idx} at step {self.num_timesteps} "
                        f"(hash: {episode_hash}, previous at step {prev_timestep}). "
                        f"Agent is replaying identical trajectory."
                    )
                should_skip_duplicate = True

        # Only save if not a duplicate from same environment
        if should_skip_duplicate:
            return False  # Route not queued due to duplicate

        # Store hash for future contamination checks
        self.recent_episode_hashes[episode_hash] = (env_idx, self.num_timesteps)

        # Update last save timestep for this environment
        self.last_save_timestep_per_env[env_idx] = self.num_timesteps

        # DEBUG: Check route length before creating route_data
        route_len_before = len(route_positions)
        route_list = list(route_positions)
        route_len_after = len(route_list)

        if route_len_before != route_len_after:
            logger.error(
                f"[ROUTE_BUG] Route length changed during list() conversion! "
                f"Before: {route_len_before}, After: {route_len_after}"
            )

        if route_len_after < 10:
            logger.warning(
                f"[ROUTE_BUG] Very short route being saved: {route_len_after} positions. "
                f"Positions: {route_list}"
            )

        route_data = {
            "env_idx": env_idx,
            "positions": route_list,
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
            "terminal_impact": terminal_impact,
            "truncated": info.get("truncated", False),
            "episode_hash": episode_hash,
            "worker_pid": worker_pid,
            "skip_value": skip_value,
        }

        self.save_queue.append(route_data)

        if self.verbose >= 1:
            logger.info(
                f"Queued route visualization for env {env_idx} - "
                f"Stage: {curriculum_stage}, Level: {level_id}, Hash: {episode_hash}"
            )

        return True  # Route successfully queued

    def _calculate_episode_hash(self, positions: List[Tuple[float, float]]) -> str:
        """Calculate a hash for an episode based on its route positions.

        Uses first 10 positions + last 5 positions + total length to create
        a compact identifier that can detect identical routes between different
        environments while minimizing false positives.

        Args:
            positions: List of (x, y) position tuples

        Returns:
            Short hex hash string (5 characters)
        """
        import hashlib

        # Sample from start and end of route for better uniqueness
        route_length = len(positions)
        if route_length <= 15:
            # Short routes: use all positions
            sample_positions = positions
        else:
            # Longer routes: use first 10 and last 5 positions
            sample_positions = positions[:10] + positions[-5:]

        # Include route length to differentiate routes with same start/end
        hash_input = f"{route_length}:{sample_positions}"

        # Convert to bytes for hashing
        position_bytes = hash_input.encode("utf-8")
        hash_obj = hashlib.md5(position_bytes)

        # Return first 5 hex characters for compact display
        return hash_obj.hexdigest()[:5]

    def _process_save_queue(self) -> None:
        """Process queued routes and save visualizations."""
        if not self.save_queue:
            return

        if not self._import_matplotlib():
            print("Cannot save route visualizations - matplotlib not available")
            self.save_queue.clear()
            return

        # MEMORY PROTECTION: Limit queue size to prevent unbounded growth
        # If saves are slower than episode completions, drop oldest routes
        MAX_QUEUE_SIZE = 1000
        if len(self.save_queue) > MAX_QUEUE_SIZE:
            logger.warning(
                f"Route save queue exceeded {MAX_QUEUE_SIZE} items, "
                f"dropping {len(self.save_queue) - MAX_QUEUE_SIZE} oldest routes"
            )
            self.save_queue = self.save_queue[-MAX_QUEUE_SIZE:]

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
                print(f"Error saving route visualization: {e}")

    def _save_route_visualization(self, route_data: Dict[str, Any]) -> None:
        """Save a single route visualization.

        Args:
            route_data: Route data dictionary
        """
        # DEBUG: Log route length at visualization time
        raw_positions = route_data["positions"]

        positions = np.array(raw_positions)
        if len(positions) == 0:
            return

        if len(positions) < 10:
            logger.warning(
                f"[ROUTE_VIZ] Short route ({len(positions)} positions) being visualized! "
                f"First/last: {positions[0] if len(positions) > 0 else 'N/A'} / "
                f"{positions[-1] if len(positions) > 0 else 'N/A'}"
            )

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

        # Render mines if available
        # Mines already have visibility culling in the render function
        if route_data.get("mines"):
            render_mines_to_axis(
                ax,
                route_data["mines"],
                tile_color="#FF0000",  # Red for dangerous mines
                safe_color="#44FFFF",  # Cyan for safe mines
                alpha=0.8,
            )

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
                    alpha=0.8,
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
                linewidth=1,
                alpha=0.8,
                zorder=2,  # Above tiles (1) but below mines (3) and markers (5+)
            )

        # Mark start position
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            c="blue",
            s=100,
            marker="o",
            label="Start",
            zorder=5,
            edgecolors="white",
            linewidths=1,
        )

        # Mark agent's final position (where they actually ended)
        agent_end_x, agent_end_y = positions[-1, 0], positions[-1, 1]
        ax.scatter(
            agent_end_x,
            agent_end_y,
            c="green",
            s=100,
            marker="o",
            label="Agent End",
            zorder=5,
            edgecolors="white",
            linewidths=1,
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
            alpha=0.8,
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
                alpha=0.8,
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
                s=100,
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
                alpha=0.8,
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

        # Show episode stats (clarify frames vs actions)
        episode_length = route_data["episode_length"]
        # Try to determine actions from route length or frame skip stats
        route_positions = len(route_data.get("positions", []))
        skip_value = route_data.get("skip_value", None)

        if route_positions > 0 and route_positions < episode_length:
            # Route length suggests frame skip is being used
            # Show both actions (route length) and frames (episode_length)
            if skip_value:
                title_parts.append(
                    f"{route_positions} acts, {episode_length} frms (skip={skip_value})"
                )
            else:
                title_parts.append(
                    f"{route_positions} actions ({episode_length} frames)"
                )
        else:
            # No frame skip or unknown - just show frames
            title_parts.append(f"{episode_length} frames")

        title_parts.append(f"Reward: {route_data['episode_reward']:.2f}")

        # Show terminal impact info for failed routes only
        if not is_success:
            terminal_impact = route_data.get("terminal_impact", False)
            was_truncated = route_data.get("truncated", False)

            if was_truncated:
                title_parts.append("T: timeout")
            elif not terminal_impact:
                title_parts.append("T: mines")
            else:
                title_parts.append("T: terminal_impact")

        # DEBUG INFO: Add environment index, episode hash, and worker PID
        debug_parts = []
        env_idx = route_data.get("env_idx", "?")
        debug_parts.append(f"Env {env_idx}")

        episode_hash = route_data.get("episode_hash", "?????")
        debug_parts.append(f"Hash: {episode_hash}")

        worker_pid = route_data.get("worker_pid", None)
        if worker_pid is not None:
            debug_parts.append(f"PID: {worker_pid}")

        # Combine into multi-line title with debug info on third line
        ax.set_title(
            title_parts[0]
            + "\n"
            + " | ".join(title_parts[1:])
            + "\n"
            + " | ".join(debug_parts)
        )
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
        # CRITICAL: Include env_idx to prevent filename collisions when multiple
        # environments finish at same timestep on same level
        env_idx = route_data.get("env_idx", 0)
        stage = route_data.get("curriculum_stage", "unknown")
        level_id = route_data["level_id"]
        # Sanitize stage name for filename (replace spaces and special chars)
        stage_clean = stage.replace(" ", "_").replace("/", "-")
        filename = f"route_env{env_idx:02d}_step{route_data['timestep']:09d}_{stage_clean}_{level_id}.png"
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
                print(f"Could not remove old file {old_file}: {e}")

    def _cleanup_stale_tracking_data(self) -> None:
        """Clean up stale tracking data to prevent memory leaks in long-running training.

        Removes entries from tracking dictionaries that are no longer actively being used.
        This prevents dicts from accumulating unbounded entries.
        """
        # Get currently tracked environment indices
        active_envs = set(self.tracking_episode.keys())

        # Clean up episode counters for environments that haven't been seen recently
        # Keep only active environments + recent ones (last 128 env indices)
        if len(self.episode_counters) > 128:
            # Find stale entries (not in active set and beyond recent range)
            max_env_idx = (
                max(self.episode_counters.keys()) if self.episode_counters else 0
            )
            recent_threshold = max_env_idx - 128

            stale_keys = [
                env_idx
                for env_idx in self.episode_counters.keys()
                if env_idx not in active_envs and env_idx < recent_threshold
            ]

            for key in stale_keys:
                del self.episode_counters[key]

            if stale_keys and self.verbose >= 2:
                logger.info(
                    f"Cleaned up {len(stale_keys)} stale episode counter entries "
                    f"(kept {len(self.episode_counters)} active entries)"
                )

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
                    if self.verbose >= 2:
                        logger.debug(f"Fetching tile data for env_idx={env_idx}")
                    results = venv.env_method(
                        "get_route_visualization_tile_data", indices=[env_idx]
                    )
                    if results and len(results) > 0 and results[0] is not None:
                        if self.verbose >= 2:
                            logger.debug(
                                f"Successfully fetched tile data for env_idx={env_idx}"
                            )
                        return dict(results[0])
                    else:
                        if self.verbose >= 1:
                            print(
                                f"env_method returned empty result for tile data (env {env_idx})"
                            )
                        return {}
                except Exception as e:
                    if self.verbose >= 1:
                        print(
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
                    print(
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
                    print(
                        f"Environment {env_idx} does not have nplay_headless attribute. "
                        "Cannot extract tile data."
                    )
                return {}
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not extract tile data for env {env_idx}: {e}")
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
                    if self.verbose >= 2:
                        logger.debug(f"Fetching mine data for env_idx={env_idx}")
                    results = venv.env_method(
                        "get_route_visualization_mine_data", indices=[env_idx]
                    )
                    if results and len(results) > 0 and results[0] is not None:
                        if self.verbose >= 2:
                            logger.debug(
                                f"Successfully fetched mine data for env_idx={env_idx}"
                            )
                        return results[0]
                    else:
                        if self.verbose >= 1:
                            print(
                                f"env_method returned empty result for mine data (env {env_idx})"
                            )
                        return []
                except Exception as e:
                    if self.verbose >= 1:
                        print(
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
                    print(
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
                    print(
                        f"Environment {env_idx} does not have nplay_headless attribute. "
                        "Cannot extract mine data."
                    )
                return []
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not extract mine data for env {env_idx}: {e}")
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
                    if self.verbose >= 2:
                        logger.debug(f"Fetching locked door data for env_idx={env_idx}")
                    results = venv.env_method(
                        "get_route_visualization_locked_door_data", indices=[env_idx]
                    )
                    if results and len(results) > 0 and results[0] is not None:
                        if self.verbose >= 2:
                            logger.debug(
                                f"Successfully fetched locked door data for env_idx={env_idx}"
                            )
                        return results[0]
                    else:
                        if self.verbose >= 1:
                            print(
                                f"env_method returned empty result for locked door data (env {env_idx})"
                            )
                        return []
                except Exception as e:
                    if self.verbose >= 1:
                        print(
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
                    print(
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
                    print(
                        f"Environment {env_idx} does not have nplay_headless attribute. "
                        "Cannot extract locked door data."
                    )
                return []
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not extract locked door data for env {env_idx}: {e}")
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
