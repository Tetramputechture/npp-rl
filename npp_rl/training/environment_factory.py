"""Environment factory for creating training and evaluation environments."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecCheckNan,
)

from nclone.gym_environment.config import EnvironmentConfig, PBRSConfig
from nclone.gym_environment.frame_stack_wrapper import FrameStackWrapper

# FrameSkipWrapper removed - frame skip now integrated into BaseNppEnvironment
from nclone.gym_environment.npp_environment import NppEnvironment

from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper, CurriculumEnv
from npp_rl.wrappers.gpu_observation_wrapper import GPUObservationWrapper
from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.vectorization.shared_memory_vecenv import (
    SharedMemorySubprocVecEnv,
    SharedMemoryObservationWrapper,
)

logger = logging.getLogger(__name__)


class EnvironmentFactory:
    """Creates vectorized training and evaluation environments."""

    def __init__(
        self,
        use_curriculum: bool = False,
        curriculum_manager=None,
        frame_stack_config: Optional[Dict[str, Any]] = None,
        frame_skip_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = 1.0,
        output_dir: Optional[Path] = None,
        pretrained_checkpoint: Optional[str] = None,
        test_dataset_path: Optional[str] = None,
        architecture_config: Optional[ArchitectureConfig] = None,
        reward_config=None,  # RewardConfig for curriculum-aware reward system
        custom_map_path: Optional[
            str
        ] = None,  # Single level file path (overrides dataset)
        enable_profiling: bool = False,  # Enable environment-level profiling
        use_shared_memory: bool = True,  # Use shared memory for efficient scaling
    ):
        """Initialize environment factory.

        NOTE: PBRS is always enabled in base environment. No enable_pbrs flag.
        Graph building is automatically configured for PBRS requirements.

        Args:
            use_curriculum: Enable curriculum learning
            curriculum_manager: Curriculum manager instance
            frame_stack_config: Frame stacking configuration dict
            frame_skip_config: Frame skip configuration dict with keys:
                - enable: bool (default: False)
                - skip: int (default: 4, recommended for N++ based on input buffers)
                - accumulate_rewards: bool (default: True)
            pbrs_gamma: Discount factor for PBRS (base environment)
            output_dir: Output directory for BC normalization stats
            pretrained_checkpoint: Path to pretrained BC checkpoint (for normalization)
            test_dataset_path: Path to test dataset (for evaluation environments)
            architecture_config: Architecture configuration (used to disable rendering if visual modalities not used)
            reward_config: RewardConfig instance for curriculum-aware reward system
            custom_map_path: Path to single level file (overrides dataset selection when specified)
        """
        self.use_curriculum = use_curriculum
        self.test_dataset_path = test_dataset_path
        self.curriculum_manager = curriculum_manager
        self.frame_stack_config = frame_stack_config or {}
        self.frame_skip_config = frame_skip_config or {}
        self.pbrs_gamma = pbrs_gamma
        self.output_dir = output_dir
        self.pretrained_checkpoint = pretrained_checkpoint
        self.bc_normalization_applied = False
        self.vec_normalize_wrapper = None
        self.architecture_config = architecture_config
        self.reward_config = reward_config
        self.custom_map_path = custom_map_path
        self.enable_profiling = enable_profiling
        self.use_shared_memory = use_shared_memory

    def create_training_env(self, num_envs: int, gamma: float = 0.99) -> VecNormalize:
        """Create vectorized training environment.

        Args:
            num_envs: Number of parallel environments
            gamma: Discount factor for VecNormalize
            enable_visualization: Enable human rendering for one environment
            vis_env_idx: Index of environment to visualize

        Returns:
            Vectorized training environment with VecNormalize wrapper
        """
        import time

        _env_setup_start = time.perf_counter()
        logger.info(f"Setting up {num_envs} training environments...")

        # Create environment factory function
        def make_env(rank: int, visualize: bool = False):
            return self._make_single_env(
                rank=rank,
                include_curriculum=self.use_curriculum,
                visualize=visualize,
            )

        # Create vectorized environment
        # Strategy:
        # - DummyVecEnv: 1-4 envs (IPC overhead not worth it)
        # - SharedMemorySubprocVecEnv: 5+ envs (zero-copy, scales to 128+)
        # - Fallback SubprocVecEnv: if shared memory disabled

        import os

        force_dummy = os.environ.get("FORCE_DUMMY_VEC_ENV", "0") == "1"

        # Determine vectorization strategy
        use_dummy = num_envs <= 4 or force_dummy
        use_shared_memory_impl = (
            self.use_shared_memory and num_envs > 4 and not force_dummy
        )

        if force_dummy and num_envs > 4:
            logger.warning(
                f"⚠️  FORCE_DUMMY_VEC_ENV=1: Using DummyVecEnv for {num_envs} environments"
            )
        elif use_dummy:
            logger.info(
                f"Using DummyVecEnv for {num_envs} environments (sequential execution)"
            )
        elif use_shared_memory_impl:
            logger.info(
                f"Using SharedMemorySubprocVecEnv for {num_envs} environments "
                f"(zero-copy observations, scales to 128+)"
            )
        else:
            logger.info(
                f"Using SubprocVecEnv for {num_envs} environments (standard multiprocessing)"
            )

        env_fns = [make_env(i, visualize=False) for i in range(num_envs)]

        if use_dummy:
            # DummyVecEnv: single process, sequential execution
            env = DummyVecEnv(env_fns)
            logger.info("✓ DummyVecEnv initialized")
        elif use_shared_memory_impl:
            # SharedMemorySubprocVecEnv: zero-copy via shared memory
            try:
                # Create dummy env to get observation space
                dummy_env = env_fns[0]()
                obs_space = dummy_env.observation_space
                dummy_env.close()

                # Pre-allocate shared memory
                shared_obs = SharedMemoryObservationWrapper(obs_space, num_envs)

                # Create vectorized env with shared memory
                # Let SharedMemorySubprocVecEnv auto-select best start method
                env = SharedMemorySubprocVecEnv(env_fns, shared_memory=shared_obs)
                logger.info(
                    f"✓ SharedMemorySubprocVecEnv initialized with {num_envs} workers"
                )
            except Exception as e:
                logger.error(f"Failed to create SharedMemorySubprocVecEnv: {e}")
                logger.warning("Falling back to standard SubprocVecEnv")
                # Use fork on Linux for fast startup, spawn elsewhere
                import sys

                start_method = "fork" if sys.platform == "linux" else "spawn"
                env = SubprocVecEnv(env_fns, start_method=start_method)
                logger.info("✓ SubprocVecEnv initialized (fallback)")
        else:
            # Standard SubprocVecEnv: pickle serialization
            # Use fork on Linux for fast startup, spawn elsewhere
            import sys

            start_method = "fork" if sys.platform == "linux" else "spawn"
            env = SubprocVecEnv(env_fns, start_method=start_method)
            logger.info("✓ SubprocVecEnv initialized")

        # Apply VecNormalize for observation wrapper compatibility only
        # Reward normalization DISABLED - PBRS shaping already provides proper scaling
        # Normalizing PBRS rewards would compress gradients and violate policy invariance
        logger.info("Applying VecNormalize wrapper (observation wrapper only)...")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=False,  # Keep False - BC handles this
            norm_reward=False,  # DISABLED: PBRS provides proper scaling
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            epsilon=1e-8,
        )
        self.vec_normalize_wrapper = env
        logger.info("✓ VecNormalize wrapper applied (reward normalization DISABLED)")

        # Log PBRS configuration for verification
        logger.info("PBRS Configuration:")
        logger.info(f"  pbrs_gamma: {self.pbrs_gamma}")
        logger.info(
            "  VecNormalize reward normalization: DISABLED (PBRS handles scaling)"
        )
        logger.info(
            "  Expected PBRS reward range: [-5.0, +5.0] in early training phase"
        )

        # Add NaN/Inf checking wrapper for early detection and detailed diagnostics
        # Reference: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_check_nan.html
        env = VecCheckNan(
            env,
            raise_exception=True,  # Fail fast on NaN/inf detection
            check_inf=True,  # Check for infinity values too
            warn_once=False,  # Log every occurrence for complete diagnostics
        )
        logger.info("✓ VecCheckNan wrapper added for NaN/inf detection")

        # Apply BC observation normalization if pretrained checkpoint is used
        if self.pretrained_checkpoint and self.output_dir:
            logger.info("Attempting to apply BC observation normalization...")
            self._apply_bc_normalization(env, gamma)
        else:
            logger.info("No BC checkpoint - skipping BC observation normalization")

        # Wrap with GPU observation transfer if GPU is available
        if torch.cuda.is_available():
            logger.info(
                "GPU available - applying GPUObservationWrapper for memory optimization..."
            )
            env = GPUObservationWrapper(env, use_pinned_memory=True)
            logger.info("✓ GPUObservationWrapper applied (observations will be on GPU)")

        # Wrap with curriculum tracking if enabled
        if self.use_curriculum and self.curriculum_manager:
            logger.info("Wrapping environments with global curriculum tracking...")
            env = CurriculumVecEnvWrapper(
                env, self.curriculum_manager, check_advancement_freq=10
            )

        logger.info(f"✓ Environments created: {num_envs} training")

        if self.use_curriculum:
            logger.info("Curriculum tracking enabled across all environments")

        _env_setup_time = time.perf_counter() - _env_setup_start
        logger.info(
            f"[TIMING] create_training_env ({num_envs} envs): {_env_setup_time:.3f}s"
        )

        return env

    def create_eval_env(self) -> DummyVecEnv:
        """Create single evaluation environment.

        Returns:
            Evaluation environment
        """
        logger.info("Creating evaluation environment...")

        def make_eval_env():
            # Frame skip is now integrated into the environment
            frame_skip = 4  # Default
            if self.frame_skip_config.get("enable", False):
                frame_skip = self.frame_skip_config.get("skip", 4)

            env_config = EnvironmentConfig.for_training(
                test_dataset_path=self.test_dataset_path,
                custom_map_path=self.custom_map_path,
                pbrs=PBRSConfig(pbrs_gamma=self.pbrs_gamma),
                enable_profiling=self.enable_profiling,
            )
            env_config.frame_skip = frame_skip  # Set frame skip in config

            # Disable visual observations if architecture doesn't use visual modalities
            if self.architecture_config is not None:
                modalities = self.architecture_config.modalities
                if not modalities.use_player_frame and not modalities.use_global_view:
                    env_config.enable_visual_observations = False
                    logger.info(
                        f"Architecture '{self.architecture_config.name}' doesn't use visual modalities - "
                        "disabling rendering for maximum performance"
                    )

            env = NppEnvironment(config=env_config)

            if frame_skip > 1:
                logger.info(
                    f"✓ Frame skip integrated in environment: skip={frame_skip} frames"
                )

            # Apply FrameStackWrapper if frame stacking is enabled
            if self.frame_stack_config and (
                self.frame_stack_config.get("enable_visual_frame_stacking", False)
                or self.frame_stack_config.get("enable_state_stacking", False)
            ):
                env = FrameStackWrapper(
                    env,
                    visual_stack_size=self.frame_stack_config.get(
                        "visual_stack_size", 4
                    ),
                    state_stack_size=self.frame_stack_config.get("state_stack_size", 4),
                    enable_visual_stacking=self.frame_stack_config.get(
                        "enable_visual_frame_stacking", False
                    ),
                    enable_state_stacking=self.frame_stack_config.get(
                        "enable_state_stacking", False
                    ),
                    padding_type=self.frame_stack_config.get("padding_type", "zero"),
                )

            return env

        return DummyVecEnv([make_eval_env])

    def _make_single_env(
        self, rank: int, include_curriculum: bool, visualize: bool = False
    ) -> Callable:
        """Create a single environment factory function.

        Args:
            rank: Environment index/rank
            include_curriculum: Whether to wrap with curriculum
            visualize: Enable human rendering

        Returns:
            Environment factory function
        """

        def _init():
            # Only log for first environment to reduce noise
            if rank == 0:
                logger.info("[Env 0] Creating NppEnvironment...")

            # Frame skip is now integrated into the environment
            frame_skip = 4  # Default
            if self.frame_skip_config.get("enable", False):
                frame_skip = self.frame_skip_config.get("skip", 4)

            env_config = EnvironmentConfig.for_training(
                test_dataset_path=self.test_dataset_path,
                custom_map_path=self.custom_map_path,
                pbrs=PBRSConfig(pbrs_gamma=self.pbrs_gamma),
            )
            env_config.frame_skip = frame_skip  # Set frame skip in config

            # Pass reward_config if available (for curriculum-aware reward system)
            if self.reward_config is not None:
                env_config.reward_config = self.reward_config

            # Disable visual observations if architecture doesn't use visual modalities
            # (unless visualization is explicitly requested)
            if not visualize and self.architecture_config is not None:
                modalities = self.architecture_config.modalities
                if not modalities.use_player_frame and not modalities.use_global_view:
                    env_config.enable_visual_observations = False
                    if rank == 0:
                        logger.info(
                            f"Architecture '{self.architecture_config.name}' doesn't use visual modalities - "
                            "disabling rendering for maximum performance"
                        )

            # Enable human rendering for visualization (overrides disable above)
            if visualize:
                env_config.render.render_mode = "human"
                env_config.enable_visual_observations = (
                    True  # Re-enable for visualization
                )
                logger.info(f"[Env {rank}] Rendering enabled for visualization")

            env = NppEnvironment(config=env_config)

            # Frame skip is now integrated into BaseNppEnvironment
            if rank == 0 and frame_skip > 1:
                logger.info(
                    f"✓ Frame skip integrated in environment: skip={frame_skip} frames"
                )
                logger.info(
                    f"  → Agent decisions reduced by {(1 - 1 / frame_skip) * 100:.0f}% "
                    f"(4-frame skip is within all N++ input buffers)"
                )

            # Apply FrameStackWrapper if frame stacking is enabled
            if self.frame_stack_config and (
                self.frame_stack_config.get("enable_visual_frame_stacking", False)
                or self.frame_stack_config.get("enable_state_stacking", False)
            ):
                env = FrameStackWrapper(
                    env,
                    visual_stack_size=self.frame_stack_config.get(
                        "visual_stack_size", 4
                    ),
                    state_stack_size=self.frame_stack_config.get("state_stack_size", 4),
                    enable_visual_stacking=self.frame_stack_config.get(
                        "enable_visual_frame_stacking", False
                    ),
                    enable_state_stacking=self.frame_stack_config.get(
                        "enable_state_stacking", False
                    ),
                    padding_type=self.frame_stack_config.get("padding_type", "zero"),
                )
                if rank == 0:
                    logger.info("✓ FrameStackWrapper applied to environment")

            # Position tracking is now integrated into BaseNppEnvironment (no wrapper needed)

            # Wrap with curriculum if enabled
            if include_curriculum and self.curriculum_manager:
                env = CurriculumEnv(
                    env,
                    self.curriculum_manager,
                    check_advancement_freq=10,
                    enable_local_tracking=False,  # Disabled for VecEnv
                )

            return env

        return _init

    def _apply_bc_normalization(self, env: VecNormalize, gamma: float) -> None:
        """Apply BC observation normalization to existing VecNormalize wrapper.

        Args:
            env: VecNormalize environment to modify
            gamma: Discount factor
        """
        bc_norm_stats_path = (
            self.output_dir / "pretrain" / "cache" / "normalization_stats.npz"
        )

        if not bc_norm_stats_path.exists():
            print(f"BC normalization stats not found at {bc_norm_stats_path}")
            print(
                "Continuing without BC observation normalization - transfer learning may be degraded"
            )
            self.bc_normalization_applied = False
            return

        logger.info("Loading BC observation normalization...")

        try:
            # Load BC normalization statistics
            bc_stats = np.load(bc_norm_stats_path)

            # Initialize obs_rms with BC statistics
            from stable_baselines3.common.running_mean_std import RunningMeanStd

            # Extract unique observation keys from BC stats
            keys = set(
                k.rsplit("_", 1)[0]
                for k in bc_stats.keys()
                if "_mean" in k or "_std" in k
            )

            if not hasattr(env, "obs_rms"):
                env.obs_rms = {}

            initialized_count = 0
            initialized_keys = []

            # Get actual observation space from environment to validate shapes
            test_obs = env.reset()

            for key in keys:
                mean_key = f"{key}_mean"
                std_key = f"{key}_std"

                if mean_key in bc_stats and std_key in bc_stats:
                    mean = bc_stats[mean_key]
                    std = bc_stats[std_key]

                    # Validate and reshape to match current observation space
                    if key in test_obs:
                        # Get expected shape (single env observation)
                        expected_shape = (
                            test_obs[key][0].shape
                            if test_obs[key].ndim > 1
                            else test_obs[key].shape
                        )

                        # Handle shape mismatches due to frame stacking differences
                        # BC dataset uses np.concatenate() which flattens stacked states: (stack_size * state_dim,)
                        # Environment uses FrameStackWrapper which stacks states: (stack_size, state_dim)
                        if mean.shape != expected_shape:
                            reshaped = False

                            # Case 1: BC stats are flattened (1D) but environment expects stacked (2D)
                            # This is the common case when BC was trained with frame stacking
                            if (
                                mean.ndim == 1
                                and len(expected_shape) == 2
                                and mean.size == np.prod(expected_shape)
                            ):
                                # Reshape from flattened (stack_size * state_dim,) to stacked (stack_size, state_dim)
                                # The reshape preserves element order: frame 0's elements become row 0, etc.
                                mean = mean.reshape(expected_shape)
                                std = std.reshape(expected_shape)
                                reshaped = True
                                logger.info(
                                    f"  ↻ Reshaped '{key}' BC stats from {bc_stats[mean_key].shape} "
                                    f"to {expected_shape} to match stacked observation format"
                                )

                            # Case 2: BC stats are stacked (2D) but environment expects flattened (1D)
                            # This is less common but could happen if BC used a different stacking method
                            elif (
                                mean.ndim == 2
                                and len(expected_shape) == 1
                                and mean.size == expected_shape[0]
                            ):
                                # Reshape from stacked (stack_size, state_dim) to flattened (stack_size * state_dim,)
                                mean = mean.flatten()
                                std = std.flatten()
                                reshaped = True
                                logger.info(
                                    f"  ↻ Reshaped '{key}' BC stats from {bc_stats[mean_key].shape} "
                                    f"to {expected_shape} to match flattened observation format"
                                )

                            # Case 3: BC stats are single state (1D) but environment expects stacked (2D)
                            # This happens when BC was trained without state stacking but RL uses state stacking
                            # (e.g., AttentiveStateMLP architectures disable state stacking in BC)
                            elif (
                                mean.ndim == 1
                                and len(expected_shape) == 2
                                and expected_shape[0] > 1  # Stack size > 1
                                and mean.size
                                == expected_shape[1]  # State dimension matches
                            ):
                                # Replicate single-state stats across all frames in the stack
                                stack_size = expected_shape[0]
                                mean = np.tile(mean, (stack_size, 1))
                                std = np.tile(std, (stack_size, 1))
                                reshaped = True
                                logger.info(
                                    f"  ↻ Replicated '{key}' BC stats from single state {bc_stats[mean_key].shape} "
                                    f"to stacked {expected_shape} (BC trained without state stacking, "
                                    f"RL uses state stacking)"
                                )

                            # If reshaping didn't work (incompatible sizes), skip this key
                            if not reshaped:
                                print(
                                    f"  ⚠️  Skipping '{key}': BC stats shape {bc_stats[mean_key].shape} "
                                    f"doesn't match current observation shape {expected_shape} "
                                    f"(total size mismatch: {mean.size} vs {np.prod(expected_shape)}). "
                                    f"This may indicate incompatible frame stacking configurations."
                                )
                                continue

                    # Validate BC stats for inf/NaN before using
                    if np.any(np.isinf(mean)) or np.any(np.isnan(mean)):
                        print(
                            f"  ⚠️  BC stats for '{key}' contain inf/NaN in mean, skipping"
                        )
                        continue

                    if np.any(np.isinf(std)) or np.any(np.isnan(std)):
                        print(
                            f"  ⚠️  BC stats for '{key}' contain inf/NaN in std, skipping"
                        )
                        continue

                    # Additional check: ensure std is not zero (would cause division by zero)
                    if np.any(std == 0.0):
                        print(
                            f"  ⚠️  BC stats for '{key}' contain zero std, replacing with 1.0"
                        )
                        std = np.where(std == 0.0, 1.0, std)

                    # Initialize RunningMeanStd for this observation key
                    rms = RunningMeanStd(shape=mean.shape)
                    rms.mean = mean.copy()
                    rms.var = (std**2).copy()
                    rms.count = 1e6  # High count = stable statistics

                    env.obs_rms[key] = rms
                    initialized_keys.append(key)
                    initialized_count += 1
                    logger.info(
                        f"  ✓ Initialized normalization for '{key}': "
                        f"mean={mean.mean():.3f}, std={std.mean():.3f}"
                    )

            # Set norm_obs_keys BEFORE enabling norm_obs
            # VecNormalize requires this to be set when norm_obs=True
            if initialized_count > 0:
                env.norm_obs_keys = initialized_keys
                env.norm_obs = True
                logger.info(
                    f"  ✓ Enabled observation normalization for keys: {initialized_keys}"
                )

            if initialized_count == 0:
                print(
                    "  WARNING: No observation normalization statistics were initialized! "
                    "Check BC stats file format."
                )
                self.bc_normalization_applied = False
            else:
                logger.info(
                    f"✓ BC observation normalization applied "
                    f"({initialized_count} observation keys)"
                )
                self.bc_normalization_applied = True

            logger.info("=" * 60)

        except Exception as e:
            print("!" * 60)
            print("Failed to apply BC observation normalization!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"BC stats path: {bc_norm_stats_path}")
            # print("Full traceback:", exc_info=True)
            print("!" * 60)
            print(
                "⚠️  Continuing without BC observation normalization - transfer learning may be degraded"
            )
            self.bc_normalization_applied = False
