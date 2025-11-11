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

    def step(self, action):
        """Execute action and track position.

        Args:
            action: Action to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract and track position
        position = self._get_position()
        if position is not None:
            self.current_route.append(position)
            info["player_position"] = position

        # Add complete route to info on episode end
        done = terminated or truncated
        if done and self.current_route:
            info["episode_route"] = list(self.current_route)
            info["route_length"] = len(self.current_route)

            # Include exit switch/door positions from THIS level (before auto-reset)
            if self.current_exit_switch_pos is not None:
                info["exit_switch_pos"] = self.current_exit_switch_pos
            if self.current_exit_door_pos is not None:
                info["exit_door_pos"] = self.current_exit_door_pos

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and clear route.

        Args:
            **kwargs: Additional reset arguments

        Returns:
            Initial observation and info
        """
        obs, info = self.env.reset(**kwargs)

        # Clear route tracking
        self.current_route = []

        # Track initial position
        position = self._get_position()
        self.current_route.append(position)
        info["player_position"] = position

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
            while hasattr(env, "env") and not hasattr(env, "nplay_headless"):
                env = env.env

            pos = env.nplay_headless.ninja_position()
            return (float(pos[0]), float(pos[1]))

        except Exception as e:
            if not self._warned_about_position:
                print(f"Could not get player position: {e}")
                print("Route visualization may not work correctly")
                self._warned_about_position = True
            print(f"Could not get position: {e}")

        return None

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
