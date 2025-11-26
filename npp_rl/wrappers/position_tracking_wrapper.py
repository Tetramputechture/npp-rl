"""Position tracking wrapper for recording agent routes.

This wrapper tracks the agent's position throughout episodes and adds
position data to the info dictionary for use by visualization callbacks.
"""

import logging
from typing import Optional, Tuple

import gymnasium as gym

logger = logging.getLogger(__name__)


class PositionTrackingWrapper(gym.Wrapper):
    """Wrapper that tracks agent position and adds it to info dict.

    This wrapper extracts the player/ninja position at each step and
    adds it to the info dictionary, making it available for route
    visualization callbacks.
    """

    def __init__(self, env: gym.Env):
        """Initialize position tracking wrapper.

        Args:
            env: Environment to wrap
        """
        super().__init__(env)
        self.current_route = []
        self._warned_about_position = False
        self._instance_id = id(self)  # Track instance identity for debugging
        logger.info(
            f"PositionTrackingWrapper created with instance ID: {self._instance_id}"
        )

    def step(self, action):
        """Execute action and track position.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # DIAGNOSTIC: Track step() calls
        route_len_before = len(self.current_route)
        logger.info(
            f"[DIAGNOSTIC] PositionTrackingWrapper.step() called on instance {self._instance_id}, "
            f"route length before: {route_len_before}"
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if episode is ending
        done = terminated or truncated

        # Extract and track position
        position = self._get_position()
        logger.info(f"[DIAGNOSTIC] _get_position() returned: {position}, done={done}")

        if position is not None:
            self.current_route.append(position)
            route_len_after = len(self.current_route)
            logger.info(
                f"[DIAGNOSTIC] Position appended, route length: {route_len_before} -> {route_len_after}"
            )
            info["player_position"] = position
        elif done and len(self.current_route) > 0:
            # Episode ended but position is invalid (ninja dead/despawned)
            # Duplicate last valid position to ensure route has at least 2 points
            logger.warning(
                f"[DIAGNOSTIC] Position unavailable at episode end (instance {self._instance_id}), "
                f"duplicating last position for visualization"
            )
            self.current_route.append(self.current_route[-1])
            if not self._warned_about_position:
                logger.warning(
                    "Position unavailable at episode end, duplicating last position for visualization"
                )
                self._warned_about_position = True

        # Add complete route to info on episode end
        if done and self.current_route:
            # Make a copy to prevent any mutation issues
            info["episode_route"] = list(self.current_route)
            info["route_length"] = len(self.current_route)

            logger.warning(
                f"[DIAGNOSTIC] Episode ending (instance {self._instance_id}): "
                f"route_length={len(self.current_route)}, "
                f"first_pos={self.current_route[0] if self.current_route else None}, "
                f"last_pos={self.current_route[-1] if self.current_route else None}"
            )

            # Include exit switch/door positions from THIS level (before auto-reset)
            if self.current_exit_switch_pos is not None:
                info["exit_switch_pos"] = self.current_exit_switch_pos
            if self.current_exit_door_pos is not None:
                info["exit_door_pos"] = self.current_exit_door_pos

            # CRITICAL: Clear route immediately after copying to info
            # This prevents any potential contamination if reset() isn't called promptly
            # In VecEnv, reset may be deferred or handled by training loop
            self.current_route = []

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and clear route.

        Args:
            **kwargs: Additional reset arguments

        Returns:
            Initial observation and info
        """
        logger.info(
            f"[DIAGNOSTIC] PositionTrackingWrapper.reset() called on instance {self._instance_id}"
        )

        obs, info = self.env.reset(**kwargs)

        # Clear route tracking (defensive: should already be cleared at episode end)
        self.current_route = []

        # Track initial position
        position = self._get_position()
        if position is not None:
            self.current_route.append(position)
            info["player_position"] = position
            logger.info(
                f"[DIAGNOSTIC] Reset: initial position {position} added, route length: {len(self.current_route)}"
            )
        else:
            logger.error("[DIAGNOSTIC] Reset: Could not get initial position!")

        # Store them so we can include them at episode end (before auto-reset changes them)
        self.current_exit_switch_pos = self._get_exit_switch_position()
        self.current_exit_door_pos = self._get_exit_door_position()

        return obs, info

    def _get_position(self) -> Optional[Tuple[float, float]]:
        """Get current player position from environment.

        Returns:
            Tuple of (x, y) position, or None if position unavailable
        """
        try:
            # Method 1: Try to get from nplay_headless (most direct)
            env = self.env
            unwrap_count = 0
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env
                unwrap_count += 1
                if unwrap_count > 10:  # Safety check
                    logger.error(
                        f"[DIAGNOSTIC] Unwrapped {unwrap_count} times without finding nplay_headless!"
                    )
                    break

            if not hasattr(env, "nplay_headless"):
                logger.error(
                    f"[DIAGNOSTIC] Could not find nplay_headless in wrapper chain after {unwrap_count} unwraps!"
                )
                return None

            pos = env.nplay_headless.ninja_position()
            # Use info level for position success (very frequent, but important for diagnosis)
            logger.info(f"[DIAGNOSTIC] _get_position() SUCCESS: {pos}")
            return (float(pos[0]), float(pos[1]))

        except Exception as e:
            logger.error(
                f"[DIAGNOSTIC] _get_position() FAILED with exception: {e}",
                exc_info=True,
            )
            if not self._warned_about_position:
                print(f"Could not get player position: {e}")
                print("Route visualization may not work correctly")
                self._warned_about_position = True
            print(f"Could not get position: {e}")

        return None

    def __getattr__(self, name):
        """Forward unknown attributes to wrapped environment.

        This allows methods like get_route_visualization_tile_data to be
        called through the wrapper chain when using env_method.

        Note: __getattr__ is only called for attributes that don't exist,
        so we don't need to explicitly check for instance variables.

        Args:
            name: Attribute name

        Returns:
            Attribute from wrapped environment
        """
        # Forward to wrapped environment
        return getattr(self.env, name)

    def _get_exit_switch_position(self) -> Optional[Tuple[float, float]]:
        """Get exit switch position from environment.

        Returns:
            Tuple of (x, y) position, or None if unavailable
        """
        try:
            env = self.env
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env

            pos = env.nplay_headless.exit_switch_position()
            return (float(pos[0]), float(pos[1]))
        except Exception as e:
            logger.debug(f"Could not get exit switch position: {e}")

        return None

    def _get_exit_door_position(self) -> Optional[Tuple[float, float]]:
        """Get exit door position from environment.

        Returns:
            Tuple of (x, y) position, or None if unavailable
        """
        try:
            env = self.env
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env

            pos = env.nplay_headless.exit_door_position()
            return (float(pos[0]), float(pos[1]))
        except Exception as e:
            logger.debug(f"Could not get exit door position: {e}")

        return None
