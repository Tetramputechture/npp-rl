"""Curriculum learning environment wrappers.

High-performance wrappers that minimize overhead in the hot paths (reset/step)
while seamlessly integrating curriculum learning with N++ environment training.
"""

import logging
from typing import Dict, Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from npp_rl.training.curriculum_components import CURRICULUM_ORDER

logger = logging.getLogger(__name__)


class CurriculumEnv(gym.Wrapper):
    """Curriculum environment wrapper.

    High-performance wrapper that integrates with the curriculum manager to:
    - Sample levels from appropriate difficulty stage
    - Track episode success for curriculum advancement
    - Automatically progress through difficulty levels
    - Minimize overhead in reset/step methods for maximum throughput
    """

    def __init__(
        self,
        env: gym.Env,
        curriculum_manager,
        check_advancement_freq: int = 10,
        enable_local_tracking: bool = True,
    ):
        """Initialize optimized curriculum environment wrapper.

        Args:
            env: Base N++ environment
            curriculum_manager: CurriculumManager instance (optimized or original)
            check_advancement_freq: Check for advancement every N episodes
            enable_local_tracking: If False, disables local tracking for VecEnv usage
        """
        super().__init__(env)
        self.curriculum_manager = curriculum_manager
        self.check_advancement_freq = check_advancement_freq
        self.enable_local_tracking = enable_local_tracking

        # Episode tracking (only used if enable_local_tracking=True)
        self.episode_count = 0

        # Cached curriculum state to avoid repeated lookups
        # Defer this call until first use to avoid triggering curriculum manager initialization
        self._cached_stage = None
        self._stage_cache_valid = False

        # Pre-allocate commonly used strings to avoid repeated allocation
        self._str_unknown = "unknown"
        self._str_sampled_stage = "sampled_stage"
        self._str_sampled_generator = "sampled_generator"
        self._str_category = "category"
        self._str_metadata = "metadata"
        self._str_generator = "generator"
        self._str_map_data = "map_data"
        self._str_level_id = "level_id"
        self._str_action_mask = "action_mask"

        # Cache for episode info to reduce allocations
        self._episode_info_cache = {}

        # Current level state (cached to avoid repeated dict lookups)
        self.current_level_stage = None
        self.current_generator_type = None
        self._current_level_id = None

        # Initialize debug flag from environment (cached)
        self.debug = False
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "nplay_headless"):
            self.debug = env.unwrapped.nplay_headless.sim.sim_config.debug

        logger.info(
            f"Optimized Curriculum environment initialized (debug: {self.debug})"
        )
        logger.info(
            f"Starting stage: {'deferred' if self._cached_stage is None else self._cached_stage}"
        )
        logger.info(
            f"Local tracking: {'enabled' if enable_local_tracking else 'disabled'}"
        )

    def _ensure_cached_stage(self):
        """Ensure cached stage is loaded (lazy initialization)."""
        if self._cached_stage is None:
            self._cached_stage = self.curriculum_manager.get_current_stage()
            self._stage_cache_valid = True

    def set_curriculum_stage(self, stage: str):
        """Update curriculum stage (optimized for VecEnv synchronization).

        Args:
            stage: New curriculum stage to sample from
        """
        self._cached_stage = stage
        self._stage_cache_valid = True

        # Update curriculum manager state efficiently
        if hasattr(self.curriculum_manager, "current_stage_idx"):
            try:
                stage_idx = CURRICULUM_ORDER.index(stage)
                self.curriculum_manager.current_stage_idx = stage_idx
                self.curriculum_manager.current_stage = stage
                logger.debug(
                    f"Curriculum stage updated to: {stage} (index: {stage_idx})"
                )
            except (ValueError, AttributeError):
                logger.warning(f"Unknown curriculum stage: {stage}")

    def set_adaptive_mixing_ratio(self, stage: str, ratio: float):
        """Set adaptive mixing ratio (optimized for VecEnv synchronization).

        Args:
            stage: Stage name
            ratio: Mixing ratio (0.0 to 1.0)
        """
        # Handle both optimized and original curriculum manager types
        if hasattr(self.curriculum_manager, "_cached_mixing_ratios"):
            # Optimized manager
            self.curriculum_manager._cached_mixing_ratios[stage] = ratio
            self.curriculum_manager._mixing_cache_valid = True
        elif hasattr(self.curriculum_manager, "stage_mixing_ratios"):
            # Original manager
            self.curriculum_manager.stage_mixing_ratios[stage] = ratio

        logger.debug(f"Mixing ratio for stage '{stage}' set to {ratio:.1%}")

    def reset(self, **kwargs):
        """Optimized reset with minimal overhead.

        Returns:
            observation, info dict
        """
        # Fast curriculum level sampling
        level_data = self.curriculum_manager.sample_level()

        if level_data is None:
            logger.warning("No curriculum level available, using default reset")
            return self.env.reset(**kwargs)

        # Extract level metadata efficiently (minimize dict lookups)
        self._extract_level_metadata(level_data)

        # Load map efficiently
        map_data = level_data.get(self._str_map_data)
        if map_data is not None:
            # Direct access to unwrapped environment for speed
            base_env = self.env.unwrapped
            if self.debug:
                logger.debug(
                    f"Loading curriculum level: {self._current_level_id} from stage: {self.current_level_stage}"
                )

            base_env.nplay_headless.load_map_from_map_data(map_data)

            # Efficient options handling
            reset_options = kwargs.get("options", {})
            if reset_options is None:
                reset_options = {}
            reset_options["skip_map_load"] = True
            kwargs["options"] = reset_options
        else:
            raise ValueError(
                f"Level data missing 'map_data' key for level: {self._current_level_id}"
            )

        # Reset environment
        obs, info = self.env.reset(**kwargs)

        # Add curriculum info efficiently (reuse cached strings)
        info["curriculum_stage"] = self.current_level_stage
        info["curriculum_level_id"] = self._current_level_id
        info["curriculum_generator"] = self.current_generator_type

        return obs, info

    def _extract_level_metadata(self, level_data: Dict[str, Any]) -> None:
        """Efficiently extract and cache level metadata.

        Args:
            level_data: Level dictionary from curriculum manager
        """
        # Priority-based stage extraction (optimized lookups)
        if self._str_sampled_stage in level_data:
            self.current_level_stage = level_data[self._str_sampled_stage]
        elif self._str_category in level_data:
            self.current_level_stage = level_data[self._str_category]
        else:
            # Fallback to metadata (more expensive)
            metadata = level_data.get(self._str_metadata)
            if isinstance(metadata, dict):
                self.current_level_stage = metadata.get(
                    self._str_category, self._str_unknown
                )
            else:
                raise ValueError(
                    f"NO STAGE INFO for level {level_data.get(self._str_level_id, self._str_unknown)}"
                )

        # Extract generator type efficiently
        self.current_generator_type = level_data.get(self._str_sampled_generator)
        if self.current_generator_type is None:
            metadata = level_data.get(self._str_metadata)
            if isinstance(metadata, dict):
                self.current_generator_type = metadata.get(self._str_generator)

        # Cache level ID
        self._current_level_id = level_data.get(self._str_level_id, self._str_unknown)

    def step(self, action):
        """Optimized step with minimal overhead.

        Args:
            action: Action to execute

        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Optimized action mask handling (only if needed)
        if isinstance(obs, dict) and self._str_action_mask in obs:
            # Fast path: only copy if debug logging is needed or wrapper issues detected
            action_mask = obs[self._str_action_mask]
            if self.debug or not action_mask.flags["OWNDATA"]:
                mask = np.array(action_mask, copy=True)
                if not mask.flags["C_CONTIGUOUS"]:
                    mask = np.ascontiguousarray(mask)
                obs[self._str_action_mask] = mask

        # Add curriculum info efficiently (use cached values)
        if self.current_level_stage:
            info["curriculum_stage"] = self.current_level_stage
        else:
            info["curriculum_stage"] = self._str_unknown

        if self.current_generator_type:
            info["curriculum_generator"] = self.current_generator_type

        # Ensure has_won consistency (fast check)
        player_won = info.get("player_won")
        if player_won is not None and "has_won" not in info:
            info["has_won"] = player_won

        # Handle episode completion efficiently
        if terminated or truncated:
            self._on_episode_end_optimized(info)

        return obs, reward, terminated, truncated, info

    def _on_episode_end_optimized(self, info: Dict[str, Any]) -> None:
        """Optimized episode completion handling.

        Args:
            info: Episode info dict
        """
        # Early exit if local tracking disabled
        if not self.enable_local_tracking:
            return

        self.episode_count += 1

        # Fast success determination (avoid multiple dict lookups)
        success = (
            info.get("has_won")
            or info.get("player_won")
            or info.get("is_success")
            or info.get("success", False)
        )

        # Only record if we have valid stage (use cached value)
        if self.current_level_stage and self.current_level_stage != self._str_unknown:
            frames = info.get("l")  # Frame count

            try:
                self.curriculum_manager.record_episode(
                    self.current_level_stage,
                    success,
                    self.current_generator_type,
                    frames,
                )
            except Exception as e:
                logger.debug(f"Error recording episode: {e}")

        # Efficient advancement checking (reduce frequency for performance)
        if self.episode_count % self.check_advancement_freq == 0:
            try:
                # Check regression first (higher priority)
                if not self.curriculum_manager.check_regression():
                    # Check advancement if no regression
                    if self.curriculum_manager.check_advancement():
                        # Invalidate stage cache on advancement
                        self._stage_cache_valid = False
                        self._cached_stage = self.curriculum_manager.get_current_stage()
                        self._stage_cache_valid = True
            except Exception as e:
                logger.debug(f"Error during advancement check: {e}")

    def get_curriculum_progress(self) -> str:
        """Get curriculum progress summary."""
        return self.curriculum_manager.get_progress_summary()


class CurriculumVecEnvWrapper(VecEnvWrapper):
    """Wrapper for vectorized environments with curriculum learning.

    High-performance implementation with optimizations:
    - Batched state synchronization to minimize env_method calls
    - Efficient episode tracking with reduced overhead per environment
    - Cached performance calculations to avoid redundant work
    - Streamlined advancement checking with adaptive frequency
    """

    def __init__(
        self,
        venv,
        curriculum_manager,
        check_advancement_freq: int = 10,
        batch_sync_freq: int = 50,
    ):
        """Initialize optimized vectorized curriculum wrapper.

        Args:
            venv: Vectorized environment (SubprocVecEnv or DummyVecEnv)
            curriculum_manager: CurriculumManager instance (shared in main process)
            check_advancement_freq: Check advancement every N total episodes across all envs
            batch_sync_freq: Sync mixing ratios every N episodes for efficiency
        """
        super().__init__(venv)
        self.curriculum_manager = curriculum_manager
        self.check_advancement_freq = check_advancement_freq
        self.batch_sync_freq = batch_sync_freq

        # Efficient episode tracking
        self.env_episode_counts = np.zeros(self.num_envs, dtype=np.int32)
        self.total_episodes = 0
        self.last_advancement_check = 0
        self.last_batch_sync = 0

        # Cache for performance metrics to avoid repeated calculations
        self._cached_stage_perf = {}
        self._perf_cache_valid = False
        self._last_cached_stage = None

        # Pre-allocated strings for efficiency
        self._str_has_won = "has_won"
        self._str_player_won = "player_won"
        self._str_is_success = "is_success"
        self._str_success = "success"
        self._str_curriculum_stage = "curriculum_stage"
        self._str_curriculum_generator = "curriculum_generator"
        self._str_unknown = "unknown"
        self._str_action_mask = "action_mask"

        # Initialize debug flag (cached)
        self.debug = False
        try:
            if hasattr(venv, "envs") and len(venv.envs) > 0:
                first_env = venv.envs[0]
                if hasattr(first_env, "unwrapped") and hasattr(
                    first_env.unwrapped, "nplay_headless"
                ):
                    self.debug = first_env.unwrapped.nplay_headless.sim.sim_config.debug
        except Exception:
            pass

        logger.info(
            f"Optimized Curriculum VecEnv wrapper initialized for {self.num_envs} envs"
        )
        logger.info(
            f"Check frequency: {check_advancement_freq}, Batch sync frequency: {batch_sync_freq}"
        )

        # Initial stage synchronization
        initial_stage = curriculum_manager.get_current_stage()
        self._sync_curriculum_stage_optimized(initial_stage)

    def step_wait(self):
        """Optimized step_wait with minimal per-step overhead.

        This method is the hot path for vectorized training and has been heavily
        optimized for maximum throughput.
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        # Efficient action mask handling for batch observations
        if isinstance(obs, dict) and self._str_action_mask in obs:
            # Fast copy only if needed (ownership check)
            action_mask = obs[self._str_action_mask]
            if not action_mask.flags["OWNDATA"]:
                obs[self._str_action_mask] = action_mask.copy()

        # Fast episode completion tracking
        episodes_completed = 0
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.env_episode_counts[i] += 1
                self.total_episodes += 1
                episodes_completed += 1

                # Fast success determination (optimized logic)
                success = (
                    info.get(self._str_has_won)
                    or info.get(self._str_player_won)
                    or info.get(self._str_is_success)
                    or info.get(self._str_success, False)
                )

                stage = info.get(self._str_curriculum_stage)
                generator_type = info.get(self._str_curriculum_generator)
                frames = info.get("l")

                # Record episode efficiently (only if valid stage)
                if stage and stage != self._str_unknown:
                    try:
                        self.curriculum_manager.record_episode(
                            stage, success, generator_type, frames
                        )

                        if self.debug:
                            logger.debug(
                                f"[VecEnv] Env {i}: stage={stage}, success={success}, total={self.total_episodes}"
                            )
                    except Exception as e:
                        if self.debug:
                            logger.warning(
                                f"[VecEnv] Error recording episode for env {i}: {e}"
                            )

        # Optimized advancement checking (adaptive frequency)
        if (
            episodes_completed > 0
            and self.total_episodes
            >= self.last_advancement_check + self.check_advancement_freq
        ):
            self._check_advancement_optimized()

        # Periodic batch synchronization of mixing ratios
        elif (
            episodes_completed > 0
            and self.total_episodes >= self.last_batch_sync + self.batch_sync_freq
        ):
            self._batch_sync_mixing_ratios()

        return obs, rewards, dones, infos

    def _check_advancement_optimized(self) -> None:
        """Optimized advancement checking with caching."""
        self.last_advancement_check = self.total_episodes

        try:
            current_stage = self.curriculum_manager.get_current_stage()

            # Use cached performance if available and valid
            if (
                self._perf_cache_valid
                and self._last_cached_stage == current_stage
                and current_stage in self._cached_stage_perf
            ):
                stage_perf = self._cached_stage_perf[current_stage]
            else:
                # Calculate and cache performance
                stage_perf = self.curriculum_manager.get_stage_performance(
                    current_stage
                )
                self._cached_stage_perf[current_stage] = stage_perf
                self._perf_cache_valid = True
                self._last_cached_stage = current_stage

            success_rate = stage_perf.get("success_rate", 0.0)
            episodes = stage_perf.get("episodes", 0)
            can_advance = stage_perf.get("can_advance", False)

            if self.debug:
                logger.info(
                    f"[VecEnv] Check at {self.total_episodes} episodes: "
                    f"stage={current_stage}, success={success_rate:.2%}, episodes={episodes}"
                )

            # Check regression first
            if self.curriculum_manager.check_regression():
                new_stage = self.curriculum_manager.get_current_stage()
                logger.info(f"[VecEnv] Regression to {new_stage}")
                self._sync_curriculum_stage_optimized(new_stage)
                self._invalidate_cache()
            # Check advancement
            elif self.curriculum_manager.check_advancement():
                new_stage = self.curriculum_manager.get_current_stage()
                logger.info(f"[VecEnv] Advancement to {new_stage}")
                self._sync_curriculum_stage_optimized(new_stage)
                self._invalidate_cache()

        except Exception as e:
            if self.debug:
                logger.warning(f"[VecEnv] Error during advancement check: {e}")

    def _batch_sync_mixing_ratios(self) -> None:
        """Efficiently sync mixing ratios in batch."""
        self.last_batch_sync = self.total_episodes

        try:
            current_stage = self.curriculum_manager.get_current_stage()

            # Get fresh mixing ratio (may trigger calculation)
            if hasattr(self.curriculum_manager, "_get_cached_mixing_ratio"):
                # Optimized manager
                mixing_ratio = self.curriculum_manager._get_cached_mixing_ratio(
                    current_stage
                )
            elif hasattr(self.curriculum_manager, "_get_adaptive_mixing_ratio"):
                # Original manager
                mixing_ratio = self.curriculum_manager._get_adaptive_mixing_ratio(
                    current_stage
                )
            else:
                return

            # Batch sync to all environments
            if hasattr(self.venv, "env_method"):
                self.venv.env_method(
                    "set_adaptive_mixing_ratio", current_stage, mixing_ratio
                )

                if self.debug:
                    logger.debug(
                        f"[VecEnv] Batch synced mixing ratio for '{current_stage}': {mixing_ratio:.1%}"
                    )

        except Exception as e:
            if self.debug:
                logger.warning(f"[VecEnv] Error during batch mixing ratio sync: {e}")

    def _sync_curriculum_stage_optimized(self, stage: str) -> None:
        """Optimized curriculum stage synchronization.

        Args:
            stage: New curriculum stage to set across all environments
        """
        try:
            if hasattr(self.venv, "env_method"):
                # Sync stage efficiently
                stage_results = self.venv.env_method("set_curriculum_stage", stage)

                if stage_results is not None and len(stage_results) == self.num_envs:
                    logger.info(
                        f"[VecEnv] Synced stage '{stage}' to {len(stage_results)} environments"
                    )
                else:
                    logger.warning(
                        f"[VecEnv] Stage sync may have failed - got {len(stage_results) if stage_results else 0} results"
                    )

                # Also sync adaptive mixing ratio efficiently
                if hasattr(self.curriculum_manager, "_get_cached_mixing_ratio"):
                    # Optimized manager
                    mixing_ratio = self.curriculum_manager._get_cached_mixing_ratio(
                        stage
                    )
                elif hasattr(self.curriculum_manager, "_get_adaptive_mixing_ratio"):
                    # Original manager
                    mixing_ratio = self.curriculum_manager._get_adaptive_mixing_ratio(
                        stage
                    )
                else:
                    return

                mixing_results = self.venv.env_method(
                    "set_adaptive_mixing_ratio", stage, mixing_ratio
                )

                if mixing_results is not None and len(mixing_results) == self.num_envs:
                    logger.info(
                        f"[VecEnv] Synced mixing ratio {mixing_ratio:.1%} for '{stage}'"
                    )
                else:
                    logger.warning("[VecEnv] Mixing sync may have failed")
            else:
                logger.error(
                    "[VecEnv] VecEnv does not support env_method - cannot sync curriculum!"
                )

        except Exception as e:
            logger.error(f"[VecEnv] Failed to sync curriculum: {e}")

    def _invalidate_cache(self) -> None:
        """Invalidate performance cache after stage changes."""
        self._perf_cache_valid = False
        self._cached_stage_perf.clear()

    def reset(self):
        """Reset all environments."""
        return self.venv.reset()

    def get_curriculum_progress(self) -> str:
        """Get curriculum progress summary."""
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
