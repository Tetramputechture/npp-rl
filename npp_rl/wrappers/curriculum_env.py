"""Curriculum learning environment wrapper.

Wraps the N++ environment to sample levels from the curriculum manager
based on current difficulty stage.
"""

import logging
from typing import Dict, Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

logger = logging.getLogger(__name__)


class CurriculumEnv(gym.Wrapper):
    """Environment wrapper that samples levels from curriculum manager.

    This wrapper integrates with the curriculum manager to:
    1. Sample levels from appropriate difficulty stage
    2. Track episode success for curriculum advancement
    3. Automatically progress through difficulty levels
    """

    def __init__(
        self, env: gym.Env, curriculum_manager, check_advancement_freq: int = 10
    ):
        """Initialize curriculum environment wrapper.

        Args:
            env: Base N++ environment
            curriculum_manager: CurriculumManager instance
            check_advancement_freq: Check for advancement every N episodes
        """
        super().__init__(env)
        self.curriculum_manager = curriculum_manager
        self.check_advancement_freq = check_advancement_freq

        # Episode tracking
        self.episode_count = 0
        self.current_level_stage = None
        self.current_level_data = None

        # For subprocess sync: track what stage we're sampling from
        self._last_known_stage = curriculum_manager.get_current_stage()

        logger.info("Curriculum environment initialized")
        logger.info(f"Starting stage: {self._last_known_stage}")

    def set_curriculum_stage(self, stage: str):
        """Update the curriculum stage for this environment.

        This is used by VecEnvWrapper to sync stage changes to subprocesses.

        Args:
            stage: New curriculum stage to sample from
        """
        self._last_known_stage = stage
        # Force curriculum_manager to use this stage
        if hasattr(self.curriculum_manager, "current_stage_idx"):
            # Find stage index
            try:
                stage_idx = self.curriculum_manager.CURRICULUM_ORDER.index(stage)
                self.curriculum_manager.current_stage_idx = stage_idx
                logger.info(f"Curriculum stage updated to: {stage}")
            except ValueError:
                logger.warning(f"Unknown curriculum stage: {stage}")

    def reset(self, **kwargs):
        """Reset environment with level from curriculum.

        Returns:
            observation, info dict
        """
        # Sample level from curriculum
        level_data = self.curriculum_manager.sample_level()

        if level_data is None:
            # Fallback to default reset if no curriculum level available
            logger.warning("No curriculum level available, using default reset")
            return self.env.reset(**kwargs)

        # Store current level info
        self.current_level_data = level_data
        self.current_level_stage = level_data.get(
            "category", level_data.get("metadata", {}).get("category", "unknown")
        )

        # Load the specific map from level data
        if "map_data" in level_data:
            # Access the unwrapped environment to load the map
            base_env = self.env.unwrapped
            if hasattr(base_env, "nplay_headless"):
                logger.debug(
                    f"Loading curriculum level: {level_data.get('level_id', 'unknown')} "
                    f"from stage: {self.current_level_stage}"
                )
                base_env.nplay_headless.load_map_from_map_data(level_data["map_data"])
                
                # Pass skip_map_load=True to prevent overwriting curriculum map
                # Merge with existing options if provided
                reset_options = kwargs.get("options", {})
                if reset_options is None:
                    reset_options = {}
                reset_options["skip_map_load"] = True
                kwargs["options"] = reset_options
            else:
                logger.warning(
                    "Environment does not have nplay_headless attribute, "
                    "cannot load map data"
                )
        else:
            logger.warning(
                f"Level data missing 'map_data' key for level: "
                f"{level_data.get('level_id', 'unknown')}"
            )

        # Reset environment after loading the map
        # Note: skip_map_load=True option prevents env.reset() from loading a new map
        obs, info = self.env.reset(**kwargs)

        # Add curriculum info
        info["curriculum_stage"] = self.current_level_stage
        info["curriculum_level_id"] = level_data.get("level_id", "unknown")

        return obs, info

    def step(self, action):
        """Execute action in environment.

        Args:
            action: Action to execute

        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add curriculum info
        info["curriculum_stage"] = self.current_level_stage

        # Track episode completion
        if terminated or truncated:
            self._on_episode_end(info)

        return obs, reward, terminated, truncated, info

    def _on_episode_end(self, info: Dict[str, Any]) -> None:
        """Handle episode completion.

        Args:
            info: Episode info dict
        """
        self.episode_count += 1

        # Record episode result in curriculum
        # NppEnvironment returns "is_success", but we check both for compatibility
        success = info.get("is_success", info.get("success", False))

        if self.current_level_stage:
            self.curriculum_manager.record_episode(self.current_level_stage, success)

        # Periodically check for curriculum advancement
        if self.episode_count % self.check_advancement_freq == 0:
            advanced = self.curriculum_manager.check_advancement()

            if advanced:
                logger.info(
                    f"Curriculum advanced to: "
                    f"{self.curriculum_manager.get_current_stage()}"
                )

    def get_curriculum_progress(self) -> str:
        """Get curriculum progress summary.

        Returns:
            Progress summary string
        """
        return self.curriculum_manager.get_progress_summary()


def make_curriculum_env(
    base_env_fn, curriculum_manager, check_advancement_freq: int = 10
):
    """Create curriculum environment from base environment function.

    Args:
        base_env_fn: Function that creates base environment
        curriculum_manager: CurriculumManager instance
        check_advancement_freq: Advancement check frequency

    Returns:
        Wrapped curriculum environment
    """
    base_env = base_env_fn()
    return CurriculumEnv(base_env, curriculum_manager, check_advancement_freq)


class CurriculumVecEnvWrapper(VecEnvWrapper):
    """Wrapper for vectorized environments with curriculum learning.

    This wraps a VecEnv to add curriculum tracking across all parallel environments.
    """

    def __init__(self, venv, curriculum_manager, check_advancement_freq: int = 10):
        """Initialize vectorized curriculum wrapper.

        Args:
            venv: Vectorized environment
            curriculum_manager: CurriculumManager instance
            check_advancement_freq: Check advancement every N episodes
        """
        super().__init__(venv)
        self.curriculum_manager = curriculum_manager
        self.check_advancement_freq = check_advancement_freq

        # Track episodes per environment
        self.env_episode_counts = np.zeros(self.num_envs, dtype=int)
        self.total_episodes = 0

        logger.info(f"Curriculum VecEnv wrapper initialized for {self.num_envs} envs")

        # Sync initial curriculum stage to all environments
        initial_stage = curriculum_manager.get_current_stage()
        logger.info(
            f"Syncing initial curriculum stage '{initial_stage}' to all environments"
        )
        self._sync_curriculum_stage(initial_stage)

    def step_wait(self):
        """Wait for step to complete and track curriculum progress."""
        obs, rewards, dones, infos = self.venv.step_wait()

        # Track episode completions
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.env_episode_counts[i] += 1
                self.total_episodes += 1

                # Record in curriculum
                # NppEnvironment returns "is_success", but we check both for compatibility
                success = info.get("is_success", info.get("success", False))
                stage = info.get("curriculum_stage", "unknown")

                if stage != "unknown":
                    self.curriculum_manager.record_episode(stage, success)

                # Check advancement
                if self.total_episodes % self.check_advancement_freq == 0:
                    advanced = self.curriculum_manager.check_advancement()

                    if advanced:
                        new_stage = self.curriculum_manager.get_current_stage()
                        logger.info(f"Curriculum advanced to: {new_stage}")

                        # Sync stage to all subprocess environments
                        # This ensures they sample from the new stage
                        self._sync_curriculum_stage(new_stage)

        return obs, rewards, dones, infos

    def _sync_curriculum_stage(self, stage: str):
        """Synchronize curriculum stage to all subprocess environments.

        Args:
            stage: New curriculum stage to set
        """
        try:
            # Use env_method to call set_curriculum_stage on all envs
            # This works for both SubprocVecEnv and DummyVecEnv
            if hasattr(self.venv, "env_method"):
                self.venv.env_method("set_curriculum_stage", stage)
                logger.debug(
                    f"Synced curriculum stage '{stage}' to all {self.num_envs} environments"
                )
            else:
                logger.warning(
                    "VecEnv does not support env_method, cannot sync curriculum stage"
                )
        except Exception as e:
            logger.error(f"Failed to sync curriculum stage: {e}")

    def reset(self):
        """Reset all environments."""
        # Note: Actual curriculum level sampling needs to be implemented
        # in the base environment factory function
        return self.venv.reset()

    def get_curriculum_progress(self) -> str:
        """Get curriculum progress summary."""
        return self.curriculum_manager.get_progress_summary()
