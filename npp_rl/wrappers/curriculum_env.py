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
    
    Note: When used with vectorized environments (n_envs > 1), curriculum progression
    should be managed by CurriculumVecEnvWrapper in the main process, not by individual
    environment instances. Set enable_local_tracking=False in this case.
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_manager,
        check_advancement_freq: int = 10,
        enable_local_tracking: bool = True,
    ):
        """Initialize curriculum environment wrapper.

        Args:
            env: Base N++ environment
            curriculum_manager: CurriculumManager instance (shared reference)
            check_advancement_freq: Check for advancement every N episodes
            enable_local_tracking: If False, disables local episode recording and
                advancement checking. Use False when wrapped by CurriculumVecEnvWrapper
                to avoid duplicate tracking in subprocess environments.
        """
        super().__init__(env)
        self.curriculum_manager = curriculum_manager
        self.check_advancement_freq = check_advancement_freq
        self.enable_local_tracking = enable_local_tracking

        # Episode tracking (only used if enable_local_tracking=True)
        self.episode_count = 0
        self.current_level_stage = None
        self.current_level_data = None

        # For subprocess sync: track what stage we're sampling from
        self._last_known_stage = curriculum_manager.get_current_stage()

        logger.info("Curriculum environment initialized")
        logger.info(f"Starting stage: {self._last_known_stage}")
        logger.info(f"Local tracking: {'enabled' if enable_local_tracking else 'disabled (managed by VecEnvWrapper)'}")

    def set_curriculum_stage(self, stage: str):
        """Update the curriculum stage for this environment.

        This is used by VecEnvWrapper to sync stage changes to subprocesses.

        Args:
            stage: New curriculum stage to sample from
        """
        self._last_known_stage = stage
        # Force curriculum_manager to use this stage
        # CRITICAL: Must update BOTH current_stage and current_stage_idx for consistency
        if hasattr(self.curriculum_manager, "current_stage_idx"):
            # Find stage index
            try:
                stage_idx = self.curriculum_manager.CURRICULUM_ORDER.index(stage)
                self.curriculum_manager.current_stage_idx = stage_idx
                self.curriculum_manager.current_stage = stage  # CRITICAL: Also update current_stage
                logger.info(f"Curriculum stage updated to: {stage} (index: {stage_idx})")
            except ValueError:
                logger.warning(f"Unknown curriculum stage: {stage}")

    def set_adaptive_mixing_ratio(self, stage: str, ratio: float):
        """Set the adaptive mixing ratio for a stage.

        This is used by VecEnvWrapper to sync mixing ratios from main process
        to subprocesses. Subprocesses use the synced ratio instead of calculating
        from stale performance data.

        Args:
            stage: Stage name
            ratio: Mixing ratio (0.0 to 1.0)
        """
        if hasattr(self.curriculum_manager, 'stage_mixing_ratios'):
            self.curriculum_manager.stage_mixing_ratios[stage] = ratio
            logger.debug(f"Mixing ratio for stage '{stage}' set to {ratio:.1%}")

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
        
        # Defensive: safely extract stage from level data
        # Try multiple possible locations for category/stage info
        if "category" in level_data:
            self.current_level_stage = level_data["category"]
        elif "metadata" in level_data and isinstance(level_data["metadata"], dict):
            self.current_level_stage = level_data["metadata"].get("category", "unknown")
        else:
            self.current_level_stage = "unknown"
            logger.warning(
                f"Could not determine stage for level: {level_data.get('level_id', 'unknown')}"
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

        # Add curriculum info (defensive: ensure current_level_stage is set)
        if hasattr(self, 'current_level_stage') and self.current_level_stage:
            info["curriculum_stage"] = self.current_level_stage
        else:
            info["curriculum_stage"] = "unknown"
            logger.warning("current_level_stage not set, using 'unknown'")

        # Track episode completion
        if terminated or truncated:
            self._on_episode_end(info)

        return obs, reward, terminated, truncated, info

    def _on_episode_end(self, info: Dict[str, Any]) -> None:
        """Handle episode completion.

        Args:
            info: Episode info dict
        """
        # Only track locally if enabled (disabled when used with VecEnvWrapper)
        if not self.enable_local_tracking:
            return
            
        self.episode_count += 1

        # Record episode result in curriculum
        # NppEnvironment returns "is_success", but we check both for compatibility
        success = info.get("is_success", info.get("success", False))

        # Defensive: ensure we have a valid stage before recording
        stage = getattr(self, 'current_level_stage', None)
        if stage and stage != "unknown":
            self.curriculum_manager.record_episode(stage, success)
        else:
            logger.debug("Skipping episode recording - no valid curriculum stage")

        # Periodically check for curriculum advancement or regression
        if self.episode_count % self.check_advancement_freq == 0:
            # First check for regression (higher priority - prevent catastrophic forgetting)
            regressed = self.curriculum_manager.check_regression()
            if regressed:
                logger.warning(
                    f"Curriculum regressed to: "
                    f"{self.curriculum_manager.get_current_stage()}"
                )
            else:
                # If not regressed, check for advancement
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
    base_env_fn,
    curriculum_manager,
    check_advancement_freq: int = 10,
    enable_local_tracking: bool = True,
):
    """Create curriculum environment from base environment function.

    Args:
        base_env_fn: Function that creates base environment
        curriculum_manager: CurriculumManager instance
        check_advancement_freq: Advancement check frequency
        enable_local_tracking: If False, disables local tracking (for use with VecEnv)

    Returns:
        Wrapped curriculum environment
    """
    base_env = base_env_fn()
    return CurriculumEnv(
        base_env, curriculum_manager, check_advancement_freq, enable_local_tracking
    )


class CurriculumVecEnvWrapper(VecEnvWrapper):
    """Wrapper for vectorized environments with curriculum learning.

    This wraps a VecEnv to add curriculum tracking across all parallel environments.
    
    IMPORTANT: This wrapper is the single source of truth for curriculum progression
    when using multiple environments (n_envs > 1). It:
    1. Tracks all episode completions across all environments globally
    2. Records performance in the shared curriculum manager
    3. Checks for curriculum advancement centrally
    4. Synchronizes stage changes to all subprocess environments
    
    Individual CurriculumEnv wrappers in subprocesses should have local tracking
    disabled to avoid duplicate tracking.
    """

    def __init__(self, venv, curriculum_manager, check_advancement_freq: int = 10):
        """Initialize vectorized curriculum wrapper.

        Args:
            venv: Vectorized environment (SubprocVecEnv or DummyVecEnv)
            curriculum_manager: CurriculumManager instance (shared in main process)
            check_advancement_freq: Check advancement every N total episodes across all envs
        """
        super().__init__(venv)
        self.curriculum_manager = curriculum_manager
        self.check_advancement_freq = check_advancement_freq

        # Track episodes per environment and globally
        self.env_episode_counts = np.zeros(self.num_envs, dtype=int)
        self.total_episodes = 0
        
        # Track last advancement check to avoid redundant checks
        self.last_advancement_check = 0

        logger.info(f"Curriculum VecEnv wrapper initialized for {self.num_envs} envs")
        logger.info(f"Global episode tracking enabled - checking advancement every {check_advancement_freq} episodes")

        # Sync initial curriculum stage to all environments
        initial_stage = curriculum_manager.get_current_stage()
        logger.info(
            f"Syncing initial curriculum stage '{initial_stage}' to all {self.num_envs} environments"
        )
        self._sync_curriculum_stage(initial_stage)

    def step_wait(self):
        """Wait for step to complete and track curriculum progress globally.
        
        This method is the central point for curriculum tracking across all environments.
        It records all episodes and checks for advancement in the main process.
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        # Track episode completions across all environments
        episodes_completed_this_step = 0
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.env_episode_counts[i] += 1
                self.total_episodes += 1
                episodes_completed_this_step += 1

                # Record episode result in curriculum manager (main process tracking)
                # NppEnvironment returns "is_success", but we check both for compatibility
                success = info.get("is_success", info.get("success", False))
                stage = info.get("curriculum_stage", "unknown")

                # Defensive: only record if we have a valid stage
                if stage and stage != "unknown":
                    try:
                        self.curriculum_manager.record_episode(stage, success)
                        logger.debug(
                            f"[VecEnv] Env {i} completed episode {self.env_episode_counts[i]}: "
                            f"stage={stage}, success={success}, total_episodes={self.total_episodes}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[VecEnv] Error recording episode for env {i}: {e}",
                            exc_info=True
                        )
                else:
                    logger.warning(
                        f"[VecEnv] Env {i} completed episode without curriculum stage info"
                    )

        # Check for advancement after processing all completed episodes
        # Only check if we've reached the check frequency
        if (
            episodes_completed_this_step > 0
            and self.total_episodes >= self.last_advancement_check + self.check_advancement_freq
        ):
            self.last_advancement_check = self.total_episodes
            
            try:
                current_stage = self.curriculum_manager.get_current_stage()
                stage_perf = self.curriculum_manager.get_stage_performance(current_stage)
                
                # Defensive: ensure all required keys exist in stage_perf
                success_rate = stage_perf.get('success_rate', 0.0)
                episodes = stage_perf.get('episodes', 0)
                can_advance = stage_perf.get('can_advance', False)
                
                logger.info(
                    f"[VecEnv] Advancement check at {self.total_episodes} episodes: "
                    f"stage={current_stage}, success_rate={success_rate:.2%}, "
                    f"episodes={episodes}, can_advance={can_advance}"
                )
                
                # FIXED: Check regression first (higher priority)
                regressed = self.curriculum_manager.check_regression()
                if regressed:
                    new_stage = self.curriculum_manager.get_current_stage()
                    logger.warning(
                        f"[VecEnv] ⚠️ Curriculum regressed to: {new_stage} "
                        f"(syncing to all {self.num_envs} environments)"
                    )
                    self._sync_curriculum_stage(new_stage)
                else:
                    # If not regressed, check for advancement
                    advanced = self.curriculum_manager.check_advancement()

                    if advanced:
                        new_stage = self.curriculum_manager.get_current_stage()
                        logger.info(
                            f"[VecEnv] ✨ Curriculum advanced to: {new_stage} "
                            f"(syncing to all {self.num_envs} environments)"
                        )

                        # Sync stage to all subprocess environments
                        # This ensures ALL environments sample from the new stage
                        self._sync_curriculum_stage(new_stage)
                    else:
                        # Even if no advancement, periodically sync mixing ratios
                        # Mixing ratios adapt as performance changes within a stage
                        if self.total_episodes % 50 == 0:
                            self._sync_mixing_ratios()
            except Exception as e:
                logger.error(
                    f"[VecEnv] Error during advancement check at episode {self.total_episodes}: {e}",
                    exc_info=True
                )

        return obs, rewards, dones, infos

    def _sync_curriculum_stage(self, stage: str):
        """Synchronize curriculum stage and adaptive mixing ratio to all subprocess environments.
        
        This is critical for ensuring all environments sample from the same stage
        after advancement. Also syncs the adaptive mixing ratio to ensure subprocesses
        use correct mixing based on current performance (not stale data).
        
        Works with both SubprocVecEnv and DummyVecEnv.

        Args:
            stage: New curriculum stage to set across all environments
        """
        try:
            # Use env_method to call set_curriculum_stage on all envs
            # This works for both SubprocVecEnv and DummyVecEnv
            if hasattr(self.venv, "env_method"):
                self.venv.env_method("set_curriculum_stage", stage)
                
                # Also sync adaptive mixing ratio from main process
                # This ensures subprocesses use up-to-date mixing based on current performance
                if self.curriculum_manager.enable_adaptive_mixing:
                    mixing_ratio = self.curriculum_manager._get_adaptive_mixing_ratio(stage)
                    self.venv.env_method("set_adaptive_mixing_ratio", stage, mixing_ratio)
                    logger.info(
                        f"[VecEnv] Synced stage '{stage}' (mixing: {mixing_ratio:.1%}) "
                        f"to all {self.num_envs} environments"
                    )
                else:
                    logger.info(
                        f"[VecEnv] Synced stage '{stage}' to all {self.num_envs} environments"
                    )
            else:
                logger.warning(
                    "[VecEnv] VecEnv does not support env_method, cannot sync curriculum stage. "
                    "Stage advancement may not work correctly!"
                )
        except Exception as e:
            logger.error(f"[VecEnv] Failed to sync curriculum: {e}", exc_info=True)
    
    def _sync_mixing_ratios(self):
        """Sync current adaptive mixing ratios to all subprocesses.
        
        This should be called periodically during training (not just at stage changes)
        to ensure subprocesses have current adaptive ratios as performance changes.
        """
        if not self.curriculum_manager.enable_adaptive_mixing:
            return
        
        current_stage = self.curriculum_manager.get_current_stage()
        
        try:
            mixing_ratio = self.curriculum_manager._get_adaptive_mixing_ratio(current_stage)
            if hasattr(self.venv, "env_method"):
                self.venv.env_method("set_adaptive_mixing_ratio", current_stage, mixing_ratio)
                logger.debug(
                    f"[VecEnv] Synced mixing ratio for '{current_stage}': {mixing_ratio:.1%}"
                )
        except Exception as e:
            logger.error(f"[VecEnv] Failed to sync mixing ratios: {e}", exc_info=True)

    def reset(self):
        """Reset all environments."""
        # Note: Actual curriculum level sampling needs to be implemented
        # in the base environment factory function
        return self.venv.reset()

    def get_curriculum_progress(self) -> str:
        """Get curriculum progress summary."""
        return self.curriculum_manager.get_progress_summary()
