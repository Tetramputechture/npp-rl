"""Reward diagnostic wrapper to identify NaN rewards before VecNormalize.

This wrapper intercepts rewards from each environment and logs which
environment produces NaN values, helping debug reward calculation issues.
"""

import logging
from typing import Any, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

logger = logging.getLogger(__name__)


class RewardDiagnosticWrapper(VecEnvWrapper):
    """Wrapper that logs reward diagnostics before VecNormalize.

    This wrapper checks rewards from each environment and logs detailed
    information when NaN or Inf values are detected, including:
    - Which environment index produced the invalid reward
    - The action that was taken
    - The reward value (NaN/Inf)
    - The observation state when the invalid reward occurred
    """

    def __init__(self, venv):
        """Initialize reward diagnostic wrapper.

        Args:
            venv: Vectorized environment to wrap
        """
        super().__init__(venv)
        self.step_count = 0
        self.last_actions = None  # Store last actions to correlate with rewards
        logger.info("RewardDiagnosticWrapper initialized - will log NaN/Inf rewards")

    def reset(self) -> Any:
        """Reset environment and clear diagnostic state.

        Returns:
            Initial observation
        """
        # Reset step count and clear actions on reset
        self.step_count = 0
        self.last_actions = None
        return self.venv.reset()

    def step(self, actions):
        """Step environment and store actions for diagnostic logging.

        Args:
            actions: Actions to take in each environment

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        # Store actions to correlate with rewards in step_wait()
        self.last_actions = actions
        return super().step(actions)

    def step_wait(self) -> Tuple[Any, np.ndarray, np.ndarray, list]:
        """Wait for step to complete and check rewards for NaN/Inf.

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.step_count += 1

        # Check each environment's reward for NaN or Inf
        for env_idx in range(self.num_envs):
            reward = rewards[env_idx]

            # Get action for this environment
            action = None
            if self.last_actions is not None:
                if isinstance(self.last_actions, np.ndarray):
                    action = (
                        self.last_actions[env_idx]
                        if env_idx < len(self.last_actions)
                        else None
                    )
                elif isinstance(self.last_actions, (list, tuple)):
                    action = (
                        self.last_actions[env_idx]
                        if env_idx < len(self.last_actions)
                        else None
                    )

            if np.isnan(reward):
                # Extract observation details if available
                obs_details = {}
                if isinstance(obs, dict):
                    # Try to get player position from observation
                    for key in [
                        "player_x",
                        "player_y",
                        "switch_x",
                        "switch_y",
                        "exit_door_x",
                        "exit_door_y",
                    ]:
                        if key in obs:
                            val = obs[key]
                            if isinstance(val, np.ndarray) and env_idx < len(val):
                                obs_details[key] = val[env_idx]
                            elif not isinstance(val, np.ndarray):
                                obs_details[key] = val

                logger.warning(
                    f"[REWARD_CHECK] STEP {self.step_count} - Env {env_idx}: "
                    f"reward=NaN, action={action}, obs_details={obs_details}"
                )
            elif np.isinf(reward):
                logger.warning(
                    f"[REWARD_CHECK] STEP {self.step_count} - Env {env_idx}: "
                    f"reward=Inf ({reward}), action={action}"
                )

        return obs, rewards, dones, infos
