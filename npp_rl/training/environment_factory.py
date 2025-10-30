"""Environment factory for creating training and evaluation environments."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig
from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper
from npp_rl.wrappers.gpu_observation_wrapper import GPUObservationWrapper

logger = logging.getLogger(__name__)


class EnvironmentFactory:
    """Creates vectorized training and evaluation environments."""

    def __init__(
        self,
        use_curriculum: bool = False,
        curriculum_manager=None,
        frame_stack_config: Optional[Dict[str, Any]] = None,
        enable_pbrs: bool = False,
        pbrs_gamma: float = 0.99,
        enable_mine_avoidance_reward: bool = True,
        output_dir: Optional[Path] = None,
        pretrained_checkpoint: Optional[str] = None,
        enable_icm: bool = False,
        icm_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize environment factory.

        Args:
            use_curriculum: Enable curriculum learning
            curriculum_manager: Curriculum manager instance
            frame_stack_config: Frame stacking configuration dict
            enable_pbrs: Enable Potential-Based Reward Shaping
            pbrs_gamma: Discount factor for PBRS
            enable_mine_avoidance_reward: Enable mine avoidance component in PBRS
            output_dir: Output directory for BC normalization stats
            pretrained_checkpoint: Path to pretrained BC checkpoint (for normalization)
            enable_icm: Enable Intrinsic Curiosity Module (ICM)
            icm_config: ICM configuration dict (eta, alpha, etc.)
        """
        self.use_curriculum = use_curriculum
        self.curriculum_manager = curriculum_manager
        self.frame_stack_config = frame_stack_config or {}
        self.enable_pbrs = enable_pbrs
        self.pbrs_gamma = pbrs_gamma
        self.enable_mine_avoidance_reward = enable_mine_avoidance_reward
        self.output_dir = output_dir
        self.pretrained_checkpoint = pretrained_checkpoint
        self.bc_normalization_applied = False
        self.vec_normalize_wrapper = None
        
        # ICM integration
        self.enable_icm = enable_icm
        self.icm_config = icm_config or {}
        self.icm_integration = None

    def create_training_env(
        self,
        num_envs: int,
        gamma: float = 0.99,
        enable_visualization: bool = False,
        vis_env_idx: int = 0,
    ) -> VecNormalize:
        """Create vectorized training environment.

        Args:
            num_envs: Number of parallel environments
            gamma: Discount factor for VecNormalize
            enable_visualization: Enable human rendering for one environment
            vis_env_idx: Index of environment to visualize

        Returns:
            Vectorized training environment with VecNormalize wrapper
        """
        logger.info(f"Setting up {num_envs} training environments...")

        # Create environment factory function
        def make_env(rank: int, visualize: bool = False):
            return self._make_single_env(
                rank=rank,
                include_curriculum=self.use_curriculum,
                visualize=visualize,
            )

        # Create vectorized environment
        # Force DummyVecEnv when visualization is enabled (pygame doesn't work across processes)
        use_subproc = num_envs > 4 and not enable_visualization

        if num_envs > 4 and enable_visualization:
            logger.warning(
                "Visualization enabled with >4 environments. "
                "Forcing DummyVecEnv (single-process) for reliable pygame rendering. "
                "This may be slower than SubprocVecEnv. "
                "Consider using --num-envs 4 or less for best performance."
            )

        if use_subproc:
            logger.info(
                f"Initializing SubprocVecEnv with {num_envs} worker processes..."
            )
            env_fns = [
                make_env(i, visualize=(enable_visualization and i == vis_env_idx))
                for i in range(num_envs)
            ]
            env = SubprocVecEnv(env_fns)
            logger.info("SubprocVecEnv initialization complete")
        else:
            logger.info(
                f"Initializing DummyVecEnv with {num_envs} environments (single process)..."
            )
            env_fns = [
                make_env(i, visualize=(enable_visualization and i == vis_env_idx))
                for i in range(num_envs)
            ]
            env = DummyVecEnv(env_fns)
            logger.info("DummyVecEnv initialization complete")

        # Apply VecNormalize for reward normalization only
        # Observation normalization handled separately via BC stats
        logger.info("Applying VecNormalize wrapper for reward normalization...")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=False,  # Keep False - BC handles this
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            epsilon=1e-8,
        )
        self.vec_normalize_wrapper = env
        logger.info("✓ VecNormalize wrapper applied (reward normalization only)")

        # Apply BC observation normalization if pretrained checkpoint is used
        if self.pretrained_checkpoint and self.output_dir:
            logger.info("Attempting to apply BC observation normalization...")
            self._apply_bc_normalization(env, gamma)
        else:
            logger.info("No BC checkpoint - skipping BC observation normalization")

        # Wrap with GPU observation transfer if GPU is available
        if torch.cuda.is_available():
            logger.info("GPU available - applying GPUObservationWrapper for memory optimization...")
            env = GPUObservationWrapper(env, use_pinned_memory=True)
            logger.info("✓ GPUObservationWrapper applied (observations will be on GPU)")

        # Wrap with ICM intrinsic rewards if enabled
        if self.enable_icm:
            logger.info("ICM enabled - creating intrinsic curiosity module...")
            from npp_rl.training.icm_integration import create_icm_integration
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.icm_integration = create_icm_integration(
                enable_icm=True,
                icm_config=self.icm_config,
                device=device,
            )
            
            if self.icm_integration is not None:
                env = self.icm_integration.wrap_environment(env, policy=None)
                logger.info("✓ ICM wrapper applied - intrinsic rewards enabled")
            else:
                logger.warning("⚠ ICM integration failed - continuing without ICM")

        # Wrap with curriculum tracking if enabled
        if self.use_curriculum and self.curriculum_manager:
            logger.info("Wrapping environments with global curriculum tracking...")
            env = CurriculumVecEnvWrapper(
                env, self.curriculum_manager, check_advancement_freq=10
            )

        logger.info(f"✓ Environments created: {num_envs} training")
        logger.info(f"✓ Using {'DummyVecEnv' if num_envs <= 4 else 'SubprocVecEnv'}")

        if self.use_curriculum:
            logger.info("Curriculum tracking enabled across all environments")

        return env

    def create_eval_env(self) -> DummyVecEnv:
        """Create single evaluation environment.

        Returns:
            Evaluation environment
        """
        logger.info("Creating evaluation environment...")

        def make_eval_env():
            env_config = EnvironmentConfig.for_training()
            env = NppEnvironment(config=env_config)

            # Apply FrameStackWrapper if frame stacking is enabled
            if self.frame_stack_config and (
                self.frame_stack_config.get("enable_visual_frame_stacking", False)
                or self.frame_stack_config.get("enable_state_stacking", False)
            ):
                from nclone.gym_environment.frame_stack_wrapper import FrameStackWrapper

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

            env_config = EnvironmentConfig.for_training()

            # Enable human rendering for visualization
            if visualize:
                env_config.render.render_mode = "human"
                logger.info(f"[Env {rank}] Rendering enabled for visualization")

            env = NppEnvironment(config=env_config)

            # Apply FrameStackWrapper if frame stacking is enabled
            if self.frame_stack_config and (
                self.frame_stack_config.get("enable_visual_frame_stacking", False)
                or self.frame_stack_config.get("enable_state_stacking", False)
            ):
                from nclone.gym_environment.frame_stack_wrapper import FrameStackWrapper

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

            # Wrap with position tracking for route visualization
            from npp_rl.wrappers import PositionTrackingWrapper

            env = PositionTrackingWrapper(env)

            # Wrap with PBRS if enabled
            if self.enable_pbrs:
                from npp_rl.wrappers import HierarchicalRewardWrapper

                env = HierarchicalRewardWrapper(
                    env,
                    enable_pbrs=True,
                    pbrs_gamma=self.pbrs_gamma,
                    enable_mine_avoidance=self.enable_mine_avoidance_reward,
                    log_reward_components=True,
                )
                if rank == 0:
                    logger.info(
                        f"✓ PBRS enabled (gamma={self.pbrs_gamma}, mine_avoidance={self.enable_mine_avoidance_reward})"
                    )

            # Wrap with curriculum if enabled
            if include_curriculum and self.curriculum_manager:
                from npp_rl.wrappers.curriculum_env import CurriculumEnv

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
            logger.warning(f"BC normalization stats not found at {bc_norm_stats_path}")
            logger.warning(
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
            for key in keys:
                mean_key = f"{key}_mean"
                std_key = f"{key}_std"

                if mean_key in bc_stats and std_key in bc_stats:
                    mean = bc_stats[mean_key]
                    std = bc_stats[std_key]

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
                logger.warning(
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
            logger.error("!" * 60)
            logger.error("Failed to apply BC observation normalization!")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"BC stats path: {bc_norm_stats_path}")
            logger.error("Full traceback:", exc_info=True)
            logger.error("!" * 60)
            logger.warning(
                "⚠️  Continuing without BC observation normalization - transfer learning may be degraded"
            )
            self.bc_normalization_applied = False
