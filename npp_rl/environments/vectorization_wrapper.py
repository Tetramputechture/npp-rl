"""
Vectorization wrapper for NppEnvironment environment.
Handles proper initialization for SubprocVecEnv compatibility.
"""

import gymnasium as gym
from nclone.gym_environment.npp_environment import (
    NppEnvironment,
)


class VectorizationWrapper(gym.Wrapper):
    """
    Wrapper to make NppEnvironment work better with SubprocVecEnv.
    Handles proper initialization and cleanup.
    """

    def __init__(self, env_kwargs=None):
        """
        Initialize the wrapper.

        Args:
            env_kwargs: Dictionary of environment configuration parameters
        """
        if env_kwargs is None:
            env_kwargs = {
                "render_mode": "rgb_array",
                "enable_frame_stack": True,
                "observation_profile": "rich",
                "enable_pbrs": True,
                "pbrs_weights": {
                    "objective_weight": 1.0,
                    "hazard_weight": 0.5,
                    "impact_weight": 0.3,
                    "exploration_weight": 0.2,
                },
                "pbrs_gamma": 0.99,
                "enable_reachability_features": False,  # Can be overridden
            }

        self.env_kwargs = env_kwargs
        env = NppEnvironment(**env_kwargs)
        super().__init__(env)

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step the environment."""
        return self.env.step(action)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        if hasattr(self.env, "close"):
            self.env.close()
        super().close()

    def __getstate__(self):
        """Custom pickle method."""
        # Only store the configuration, not the environment itself
        return {"env_kwargs": self.env_kwargs}

    def __setstate__(self, state):
        """Custom unpickle method."""
        self.env_kwargs = state["env_kwargs"]
        # Recreate the environment
        env = NppEnvironment(**self.env_kwargs)
        self.env = env
        # Set up wrapper attributes
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.metadata = getattr(env, "metadata", {})


def make_vectorizable_env(env_kwargs=None):
    """
    Factory function to create a vectorizable environment.

    Args:
        env_kwargs: Dictionary of environment configuration parameters

    Returns:
        Callable that returns VectorizationWrapper instance
    """

    def _make_env():
        return VectorizationWrapper(env_kwargs)

    return _make_env
