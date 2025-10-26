"""Position tracking wrapper for recording agent routes.

This wrapper tracks the agent's position throughout episodes and adds
position data to the info dictionary for use by visualization callbacks.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

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
        self._warned_about_position = False  # Only warn once about position unavailability
        
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
            # Add current position to info
            info['player_position'] = position
        
        # Add complete route to info on episode end
        done = terminated or truncated
        if done and self.current_route:
            info['episode_route'] = list(self.current_route)
            info['route_length'] = len(self.current_route)
        
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
        if position is not None:
            self.current_route.append(position)
            info['player_position'] = position
        
        return obs, info
    
    def _get_position(self) -> Optional[Tuple[float, float]]:
        """Get current player position from environment.
        
        Returns:
            Tuple of (x, y) position, or None if position unavailable
        """
        try:
            # Method 1: Try to get from nplay_headless (most direct)
            env = self.env
            while hasattr(env, 'env') and not hasattr(env, 'nplay_headless'):
                env = env.env
            
            if hasattr(env, 'nplay_headless') and hasattr(env.nplay_headless, 'ninja_position'):
                pos = env.nplay_headless.ninja_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    return (float(pos[0]), float(pos[1]))
            
            # Method 2: Try to extract from game state if available
            if hasattr(env, 'game_state'):
                # Assuming game_state has position info
                # This is environment-specific and may need adjustment
                pass
            
        except Exception as e:
            if not self._warned_about_position:
                logger.warning(f"Could not get player position: {e}")
                logger.warning("Route visualization may not work correctly")
                self._warned_about_position = True
            logger.debug(f"Could not get position: {e}")
        
        return None
