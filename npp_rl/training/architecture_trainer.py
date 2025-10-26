"""Architecture trainer for single architecture training runs.

Handles training for a specific architecture configuration including
setup, training loop, evaluation, and checkpointing.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig
from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from npp_rl.training.curriculum_manager import create_curriculum_manager
from npp_rl.wrappers.curriculum_env import CurriculumEnv, CurriculumVecEnvWrapper

logger = logging.getLogger(__name__)


class VerboseTrainingCallback(BaseCallback):
    """Callback for verbose logging during training to help debug hangs."""

    def __init__(self, log_freq: int = 1, verbose: int = 1):
        """
        Args:
            log_freq: How often to log (in number of rollouts/updates)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.update_count = 0
        self.start_time = None
        self.last_log_time = None

    def _on_training_start(self) -> None:
        """Called before the first rollout."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("=" * 60)
        logger.info("VerboseTrainingCallback: Training started")
        logger.info("Beginning first environment reset and rollout collection...")
        logger.info("=" * 60)

    def _on_rollout_start(self) -> None:
        """Called before collecting a new rollout."""
        if self.update_count % self.log_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            since_last = current_time - self.last_log_time
            logger.info(
                f"[Update {self.update_count}] Starting rollout collection "
                f"(elapsed: {elapsed:.1f}s, since last: {since_last:.1f}s)"
            )
            self.last_log_time = current_time

    def _on_step(self) -> bool:
        """Called after each environment step during rollout."""
        return True

    def _on_rollout_end(self) -> None:
        """Called after rollout is collected."""
        if self.update_count % self.log_freq == 0:
            logger.info(
                f"[Update {self.update_count}] Rollout complete - "
                f"timesteps: {self.num_timesteps}, starting gradient update..."
            )
        self.update_count += 1

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"VerboseTrainingCallback: Training ended after {total_time:.1f}s")
        logger.info("=" * 60)


class ArchitectureTrainer:
    """Manages training for a specific architecture configuration."""

    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        train_dataset_path: str,
        test_dataset_path: str,
        output_dir: Path,
        device_id: int = 0,
        world_size: int = 1,
        tensorboard_writer: Optional[SummaryWriter] = None,
        use_mixed_precision: bool = False,
        use_hierarchical_ppo: bool = False,
        use_curriculum: bool = False,
        curriculum_kwargs: Optional[Dict[str, Any]] = None,
        use_distributed: bool = False,
        frame_stack_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize architecture trainer.

        Args:
            architecture_config: Architecture configuration
            train_dataset_path: Path to training dataset
            test_dataset_path: Path to test dataset
            output_dir: Output directory for checkpoints/logs
            device_id: GPU device ID
            world_size: Number of GPUs (for distributed training)
            tensorboard_writer: DEPRECATED - Not used. SB3's built-in tensorboard
                logging is used instead. Kept for backward compatibility.
            use_mixed_precision: Enable mixed precision training
            use_hierarchical_ppo: Use hierarchical PPO instead of standard PPO
            use_curriculum: Enable curriculum learning
            curriculum_kwargs: Curriculum manager configuration
            use_distributed: Enable DistributedDataParallel mode for multi-GPU
            frame_stack_config: Frame stacking configuration dict with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str
        """
        self.architecture_config = architecture_config
        self.train_dataset_path = Path(train_dataset_path)
        self.test_dataset_path = Path(test_dataset_path)
        self.output_dir = Path(output_dir)
        self.device_id = device_id
        self.world_size = world_size
        self.tensorboard_writer = tensorboard_writer
        self.use_mixed_precision = use_mixed_precision
        self.use_hierarchical_ppo = use_hierarchical_ppo
        self.use_curriculum = use_curriculum
        self.curriculum_kwargs = curriculum_kwargs or {}
        self.use_distributed = use_distributed
        self.frame_stack_config = frame_stack_config or {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.model = None
        self.env = None
        self.eval_env = None
        self.curriculum_manager = None
        self.bc_normalization_applied = False
        self.pretrained_checkpoint = None
        self.bc_pretrain_enabled = False

        logger.info(f"Initialized trainer for architecture: {architecture_config.name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: cuda:{device_id}")
        logger.info(f"Hierarchical PPO: {use_hierarchical_ppo}")
        logger.info(f"Curriculum learning: {use_curriculum}")
        if frame_stack_config:
            logger.info(f"Frame stacking: {frame_stack_config}")

    def _load_bc_pretrained_weights(self, checkpoint_path: str):
        """Load BC pretrained weights into PPO policy.

        Maps BC checkpoint structure to PPO policy structure:
        - BC: feature_extractor.* → PPO feature extractor (depends on policy type)
        - BC policy_head is ignored (PPO trains its own action/value heads)

        PPO policies can have different feature extractor structures:
        1. Shared: features_extractor.* (share_features_extractor=True, default)
        2. Separate: pi_features_extractor.* and vf_features_extractor.* (share_features_extractor=False)
        3. Hierarchical: mlp_extractor.features_extractor.* (HierarchicalActorCriticPolicy)

        The code automatically detects the structure and maps BC weights accordingly.

        Args:
            checkpoint_path: Path to BC checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.model.device, weights_only=False
        )

        if "policy_state_dict" not in checkpoint:
            logger.warning(
                f"Checkpoint does not contain 'policy_state_dict'. "
                f"Found keys: {list(checkpoint.keys())}"
            )
            return

        bc_state_dict = checkpoint["policy_state_dict"]

        # Detect if PPO model uses shared, separate, or hierarchical (nested) feature extractors
        # Priority: hierarchical > separate > shared (to avoid duplicate mappings)
        policy_keys = list(self.model.policy.state_dict().keys())

        # Check for hierarchical first (HierarchicalActorCriticPolicy)
        uses_hierarchical_extractor = any(
            "mlp_extractor.features_extractor." in k for k in policy_keys
        )

        # Check for separate extractors (share_features_extractor=False)
        uses_separate_extractors = any(
            k.startswith("pi_features_extractor.") for k in policy_keys
        ) or any(k.startswith("vf_features_extractor.") for k in policy_keys)

        # Check for shared extractor (share_features_extractor=True, default)
        uses_shared_extractor = any(
            k.startswith("features_extractor.") for k in policy_keys
        )

        if (
            not uses_shared_extractor
            and not uses_separate_extractors
            and not uses_hierarchical_extractor
        ):
            logger.warning(
                "PPO model has no recognizable feature extractor keys! "
                "Cannot load BC weights."
            )
            return

        # Determine which mappings to use
        # Important: Hierarchical policies may have BOTH hierarchical and shared/separate
        # extractors in the state_dict. We need to check if they're references or separate.
        #
        # Strategy:
        # 1. If hierarchical extractor exists, always map to it
        # 2. Also check if shared/separate extractors are the SAME object as hierarchical
        # 3. If they're references (same object), mapping to hierarchical is enough
        # 4. If they're separate objects, we need to map to ALL of them

        # Check if extractors are references or separate objects
        map_hierarchical = uses_hierarchical_extractor
        map_shared = False
        map_separate = False

        if uses_hierarchical_extractor:
            # Hierarchical always needs mapping
            # Check if shared/separate are references or separate objects
            if (
                hasattr(self.model.policy, "features_extractor")
                and hasattr(self.model.policy, "mlp_extractor")
                and hasattr(self.model.policy.mlp_extractor, "features_extractor")
            ):
                # Check if they're the same object
                is_same_object = (
                    self.model.policy.features_extractor
                    is self.model.policy.mlp_extractor.features_extractor
                )
                if not is_same_object and uses_shared_extractor:
                    # They're separate objects - need to map to both
                    map_shared = True
                    logger.info(
                        "  Note: Model has separate features_extractor and mlp_extractor.features_extractor"
                    )

            if (
                hasattr(self.model.policy, "pi_features_extractor")
                and hasattr(self.model.policy, "mlp_extractor")
                and hasattr(self.model.policy.mlp_extractor, "features_extractor")
            ):
                # Check if pi_features_extractor is the same object
                is_same_pi = (
                    self.model.policy.pi_features_extractor
                    is self.model.policy.mlp_extractor.features_extractor
                )
                if not is_same_pi and uses_separate_extractors:
                    # They're separate objects - need to map to both
                    map_separate = True
                    logger.info(
                        "  Note: Model has separate pi/vf_features_extractor and mlp_extractor.features_extractor"
                    )

        elif uses_separate_extractors:
            # No hierarchical, just separate
            map_separate = True

        elif uses_shared_extractor:
            # No hierarchical, just shared
            map_shared = True

        else:
            logger.warning("Cannot determine feature extractor type!")
            return

        # Determine extractor type for logging
        extractor_types = []
        if map_hierarchical:
            extractor_types.append("hierarchical")
        if map_shared:
            extractor_types.append("shared")
        if map_separate:
            extractor_types.append("separate")
        extractor_type = " + ".join(extractor_types)

        # Map BC feature_extractor weights to appropriate PPO structure
        # BC saves: feature_extractor.*
        # PPO expects:
        #   - features_extractor.* (shared)
        #   - pi_features_extractor.* + vf_features_extractor.* (separate)
        #   - mlp_extractor.features_extractor.* (hierarchical)
        mapped_state_dict = {}

        for key, value in bc_state_dict.items():
            if key.startswith("feature_extractor."):
                # Remove "feature_extractor." prefix to get the sub-key
                sub_key = key[len("feature_extractor.") :]

                # Map to appropriate target based on determined extractor type
                if map_hierarchical:
                    # Map to hierarchical extractor (nested in mlp_extractor)
                    hierarchical_key = f"mlp_extractor.features_extractor.{sub_key}"
                    mapped_state_dict[hierarchical_key] = value
                    logger.debug(f"Mapped {key} → {hierarchical_key}")

                if map_shared:
                    # Map to shared features_extractor
                    shared_key = f"features_extractor.{sub_key}"
                    mapped_state_dict[shared_key] = value
                    logger.debug(f"Mapped {key} → {shared_key}")

                if map_separate:
                    # Map to both pi_features_extractor and vf_features_extractor
                    pi_key = f"pi_features_extractor.{sub_key}"
                    vf_key = f"vf_features_extractor.{sub_key}"

                    mapped_state_dict[pi_key] = value
                    mapped_state_dict[vf_key] = (
                        value.clone()
                    )  # Clone to avoid shared references

                    logger.debug(f"Mapped {key} → {pi_key} and {vf_key}")

            elif key.startswith("policy_head."):
                # Skip policy head weights (PPO will train its own)
                logger.debug(f"Skipping {key} (policy head not used in PPO)")
            else:
                logger.debug(f"Skipping unknown key: {key}")

        if not mapped_state_dict:
            logger.warning("No feature extractor weights found in BC checkpoint")
            return

        # Load only the feature extractor weights with strict=False
        # This allows loading partial weights without errors
        try:
            missing_keys, unexpected_keys = self.model.policy.load_state_dict(
                mapped_state_dict, strict=False
            )

            # Log summary (extractor_type already determined above)
            logger.info("✓ Loaded BC pretrained feature extractor weights")
            logger.info(
                f"  Loaded {len(mapped_state_dict)} weight tensors (BC → {extractor_type})"
            )

            if missing_keys:
                logger.info(
                    f"  Missing keys (will use random init): {len(missing_keys)}"
                )
                logger.info(f"    Examples: {missing_keys[:5]}")

                # Categorize missing keys to understand what's not loaded
                feature_ext_missing = [
                    k for k in missing_keys if "features_extractor." in k
                ]
                hierarchical_missing = [
                    k
                    for k in missing_keys
                    if "mlp_extractor." in k and "features_extractor" not in k
                ]
                action_value_missing = [
                    k for k in missing_keys if "action_net." in k or "value_net." in k
                ]
                other_missing = [
                    k
                    for k in missing_keys
                    if k
                    not in feature_ext_missing
                    + hierarchical_missing
                    + action_value_missing
                ]

                if feature_ext_missing:
                    logger.info(
                        f"    Features extractor keys missing: {len(feature_ext_missing)}"
                    )
                    if (
                        map_hierarchical
                        and "features_extractor." in feature_ext_missing[0]
                    ):
                        logger.info(
                            "      (These may be references to mlp_extractor.features_extractor - OK)"
                        )
                if hierarchical_missing:
                    logger.info(
                        f"    Hierarchical policy keys missing: {len(hierarchical_missing)} (expected)"
                    )
                if action_value_missing:
                    logger.info(
                        f"    Action/value head keys missing: {len(action_value_missing)} (expected)"
                    )
                if other_missing:
                    logger.info(f"    Other keys missing: {len(other_missing)}")

            if unexpected_keys:
                logger.warning(
                    f"  Unexpected keys in checkpoint: {len(unexpected_keys)}"
                )
                logger.info(f"    Examples: {unexpected_keys[:5]}")

            # Log what was actually loaded based on extractor type
            logger.info("  ✓ Feature extractor weights loaded successfully")

            if "hierarchical" in extractor_type:
                logger.info(
                    "  ✓ Using hierarchical feature extractor (nested in mlp_extractor)"
                )
                if map_shared or map_separate:
                    logger.info(
                        f"  ✓ Also mapped to {extractor_type.replace('hierarchical + ', '')}"
                    )
                logger.info(
                    "  → High-level and low-level policy heads will be trained from scratch"
                )
            elif extractor_type == "shared":
                logger.info("  ✓ Using shared feature extractor for policy and value")
                logger.info("  → Policy and value heads will be trained from scratch")
            elif extractor_type == "separate":
                logger.info("  ✓ Loaded into both policy and value feature extractors")
                logger.info("  → Policy and value heads will be trained from scratch")
            else:
                logger.warning(f"  ⚠ Unknown extractor type: {extractor_type}")

        except Exception as e:
            logger.error(f"Failed to load mapped weights: {e}")
            raise

    def setup_model(self, pretrained_checkpoint: Optional[str] = None, **ppo_kwargs):
        """Initialize model from architecture config or checkpoint.

        Args:
            pretrained_checkpoint: Optional path to pretrained weights
            **ppo_kwargs: Additional PPO hyperparameters

        Returns:
            Initialized PPO or HierarchicalPPO model (without environment set)
        """
        logger.info("Setting up model...")

        # Store for later use in environment setup
        self.pretrained_checkpoint = pretrained_checkpoint
        self.bc_pretrain_enabled = pretrained_checkpoint is not None

        # Store policy and hyperparameters configuration
        # The actual model will be created when environment is set up

        # Set up policy kwargs with architecture config
        self.policy_kwargs = {
            "features_extractor_class": ConfigurableMultimodalExtractor,
            "features_extractor_kwargs": {"config": self.architecture_config},
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        }

        # Set policy class based on hierarchical PPO flag
        if self.use_hierarchical_ppo:
            # Import hierarchical policy
            from npp_rl.agents.hierarchical_ppo import HierarchicalActorCriticPolicy

            self.policy_class = HierarchicalActorCriticPolicy

            # Add hierarchical PPO specific kwargs
            self.policy_kwargs.update(
                {
                    "high_level_update_frequency": ppo_kwargs.pop(
                        "high_level_update_frequency", 50
                    ),
                    "max_steps_per_subtask": ppo_kwargs.pop(
                        "max_steps_per_subtask", 500
                    ),
                    "use_icm": ppo_kwargs.pop("use_icm", True),
                }
            )
            logger.info(
                "Using HierarchicalActorCriticPolicy with hierarchical parameters"
            )
        else:
            # Use standard MultiInputPolicy
            self.policy_class = "MultiInputPolicy"

        # Default PPO hyperparameters (optimized for multi-GPU training)
        # Only apply automatic scaling if hyperparameters not explicitly provided
        # (e.g., from hardware profile)
        if "batch_size" in ppo_kwargs and "learning_rate" in ppo_kwargs:
            # Hyperparameters explicitly provided (likely from hardware profile)
            # Use them directly without automatic scaling
            logger.info(
                "Using explicitly provided hyperparameters "
                "(automatic multi-GPU scaling skipped)"
            )
            default_batch_size = ppo_kwargs.get("batch_size", 256)
            default_learning_rate = ppo_kwargs.get("learning_rate", 3e-4)
        else:
            # Apply automatic scaling for multi-GPU DDP training
            base_batch_size = 256
            base_learning_rate = 3e-4

            # DDP scaling: Each GPU processes batch_size samples independently
            # Effective global batch = batch_size * world_size
            # Learning rate scaled by sqrt(world_size) for training stability
            # This follows standard practice from "Accurate, Large Minibatch SGD" (Goyal et al.)
            if self.world_size > 1 and self.use_distributed:
                default_batch_size = base_batch_size * self.world_size
                default_learning_rate = base_learning_rate * (self.world_size**0.5)
                logger.info(
                    f"DDP hyperparameter scaling for {self.world_size} GPUs: "
                    f"batch_size={default_batch_size}, lr={default_learning_rate:.2e}"
                )
            else:
                default_batch_size = base_batch_size
                default_learning_rate = base_learning_rate

        default_hyperparams = {
            "learning_rate": default_learning_rate,
            "n_steps": 1024,  # Increased for better sample efficiency with more envs
            "batch_size": default_batch_size,
            "gamma": 0.999,
            "gae_lambda": 0.998,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            # Always use SB3's built-in tensorboard logging for reliability
            # Custom tensorboard_writer was not being used for training metrics
            "tensorboard_log": str(self.output_dir / "tensorboard"),
            "device": f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
        }

        # Merge with provided hyperparameters (ppo_kwargs override defaults)
        self.hyperparams = {**default_hyperparams, **ppo_kwargs}

        # Log final hyperparameters being used
        logger.info("Final PPO hyperparameters:")
        logger.info(f"  Learning rate: {self.hyperparams['learning_rate']:.2e}")
        logger.info(f"  Batch size: {self.hyperparams['batch_size']}")
        logger.info(f"  N steps: {self.hyperparams['n_steps']}")
        logger.info(f"  Gamma: {self.hyperparams['gamma']}")
        logger.info(f"  GAE lambda: {self.hyperparams['gae_lambda']}")

        # Store pretrained checkpoint path for later loading
        self.pretrained_checkpoint = pretrained_checkpoint

        logger.info(
            f"Model configuration prepared for architecture: {self.architecture_config.name}"
        )
        logger.info("Model will be instantiated when environments are set up")

        return None  # Model will be created in setup_environments

    def setup_environments(
        self,
        num_envs: int = 64,
        total_timesteps: int = None,
        enable_visualization: bool = False,
        vis_env_idx: int = 0,
    ) -> None:
        """Create vectorized training and eval environments.

        Args:
            num_envs: Number of parallel environments
            total_timesteps: Total training timesteps (used to adjust n_steps for small runs)
            enable_visualization: If True, create one environment with human rendering
            vis_env_idx: Index of environment to enable visualization for
        """
        logger.info(f"Setting up {num_envs} training environments...")

        # Adjust n_steps if total_timesteps is very small (for CPU testing)
        if total_timesteps is not None and hasattr(self, "hyperparams"):
            # n_steps should be at most total_timesteps, but also needs to be
            # divisible by batch_size for proper training
            max_n_steps = max(total_timesteps // num_envs, 1)
            if self.hyperparams["n_steps"] > max_n_steps:
                old_n_steps = self.hyperparams["n_steps"]
                self.hyperparams["n_steps"] = max_n_steps
                # Adjust batch_size to be compatible
                if self.hyperparams["batch_size"] > max_n_steps:
                    self.hyperparams["batch_size"] = max_n_steps
                logger.info(
                    f"Adjusted n_steps from {old_n_steps} to {max_n_steps} "
                    f"for total_timesteps={total_timesteps}, num_envs={num_envs}"
                )
                logger.info(f"Adjusted batch_size to {self.hyperparams['batch_size']}")

        # Set up curriculum manager if enabled
        if self.use_curriculum:
            self.curriculum_manager = create_curriculum_manager(
                dataset_path=str(self.train_dataset_path), **self.curriculum_kwargs
            )

            logger.info("Curriculum learning enabled")
            logger.info(
                f"Starting stage: {self.curriculum_manager.get_current_stage()}"
            )

        # Create environment factory functions without capturing self
        # Pass curriculum_manager explicitly to avoid pickling issues
        use_curriculum = self.use_curriculum
        curriculum_manager = self.curriculum_manager

        # Capture frame stack config
        frame_stack_cfg = self.frame_stack_config

        def make_env(rank: int, use_curr: bool, curr_mgr, visualize: bool = False):
            def _init():
                logger.info(f"[Env {rank}] Creating NppEnvironment instance...")
                env_config = EnvironmentConfig.for_training()

                # Enable human rendering for visualization
                if visualize:
                    env_config.render.render_mode = "human"
                    logger.info(f"[Env {rank}] Rendering enabled for visualization")

                # Apply frame stacking configuration if provided
                if frame_stack_cfg:
                    from nclone.gym_environment import FrameStackConfig

                    env_config.frame_stack = FrameStackConfig(
                        enable_visual_frame_stacking=frame_stack_cfg.get(
                            "enable_visual_frame_stacking", False
                        ),
                        visual_stack_size=frame_stack_cfg.get("visual_stack_size", 4),
                        enable_state_stacking=frame_stack_cfg.get(
                            "enable_state_stacking", False
                        ),
                        state_stack_size=frame_stack_cfg.get("state_stack_size", 4),
                        padding_type=frame_stack_cfg.get("padding_type", "zero"),
                    )
                    logger.info(
                        f"[Env {rank}] Frame stacking enabled: visual={frame_stack_cfg.get('enable_visual_frame_stacking')}, state={frame_stack_cfg.get('enable_state_stacking')}"
                    )

                env = NppEnvironment(config=env_config)
                logger.info(f"[Env {rank}] ✓ NppEnvironment created")

                # Wrap with position tracking for route visualization
                from npp_rl.wrappers import PositionTrackingWrapper

                env = PositionTrackingWrapper(env)
                logger.info(f"[Env {rank}] ✓ Position tracking enabled")

                # Wrap with curriculum if enabled
                # IMPORTANT: Disable local tracking when used with VecEnv
                # The VecEnvWrapper will handle all tracking globally
                if use_curr and curr_mgr:
                    logger.info(f"[Env {rank}] Wrapping with CurriculumEnv...")
                    env = CurriculumEnv(
                        env,
                        curr_mgr,
                        check_advancement_freq=10,
                        enable_local_tracking=False,  # Disabled for VecEnv - tracked globally
                    )

                logger.info(f"[Env {rank}] ✓ Environment ready")
                return env

            return _init

        # Create vectorized training environment
        # For small numbers of envs, use DummyVecEnv to avoid multiprocessing overhead
        # IMPORTANT: Force DummyVecEnv when visualization is enabled, as pygame
        # doesn't work across processes in SubprocVecEnv
        use_subproc = num_envs > 4 and not enable_visualization

        if num_envs > 4:
            if enable_visualization:
                logger.warning(
                    "Visualization enabled with >4 environments. "
                    "Forcing DummyVecEnv (single-process) for reliable pygame rendering. "
                    "This may be slower than SubprocVecEnv. "
                    "Consider using --num-envs 4 or less for best performance."
                )

        if use_subproc:
            logger.info(f"Creating {num_envs} environment factory functions...")
            env_fns = [
                make_env(
                    i,
                    use_curriculum,
                    curriculum_manager,
                    visualize=(enable_visualization and i == vis_env_idx),
                )
                for i in range(num_envs)
            ]
            logger.info(
                f"Initializing SubprocVecEnv with {num_envs} worker processes..."
            )
            logger.info(
                "This may take time as each process spawns and initializes its environment"
            )
            self.env = SubprocVecEnv(env_fns)
            logger.info("SubprocVecEnv initialization complete")
        else:
            logger.info(f"Creating {num_envs} environment factory functions...")
            env_fns = [
                make_env(
                    i,
                    use_curriculum,
                    curriculum_manager,
                    visualize=(enable_visualization and i == vis_env_idx),
                )
                for i in range(num_envs)
            ]
            logger.info(
                f"Initializing DummyVecEnv with {num_envs} environments (single process)..."
            )
            self.env = DummyVecEnv(env_fns)
            logger.info("DummyVecEnv initialization complete")

        # Apply BC observation normalization if pretrained checkpoint is used
        if self.pretrained_checkpoint and self.bc_pretrain_enabled:
            bc_norm_stats_path = (
                self.output_dir / "pretrain" / "cache" / "normalization_stats.npz"
            )

            if bc_norm_stats_path.exists():
                logger.info(
                    f"Loading BC observation normalization from {bc_norm_stats_path}"
                )

                try:
                    # Load BC normalization statistics
                    bc_stats = np.load(bc_norm_stats_path)

                    # Create custom normalization wrapper
                    self.env = VecNormalize(
                        self.env,
                        training=True,
                        norm_obs=True,
                        norm_reward=False,  # Don't normalize rewards
                        clip_obs=10.0,
                        gamma=self.hyperparams.get("gamma", 0.999),
                    )

                    # Initialize with BC statistics
                    # VecNormalize uses running mean/var, so we initialize them
                    for key in bc_stats.keys():
                        if key.endswith("_mean"):
                            logger.debug(f"  Loaded normalization for {key}")

                    logger.info(
                        "✓ Applied BC observation normalization to RL training environments"
                    )
                    self.bc_normalization_applied = True

                except Exception as e:
                    logger.error(f"Failed to apply BC normalization: {e}")
                    logger.warning(
                        "Continuing without BC observation normalization - transfer learning may be degraded"
                    )
                    self.bc_normalization_applied = False
            else:
                logger.warning(
                    f"BC normalization stats not found at {bc_norm_stats_path}"
                )
                logger.warning(
                    "Continuing without BC observation normalization - transfer learning may be degraded"
                )
                self.bc_normalization_applied = False
        else:
            self.bc_normalization_applied = False

        # Wrap vectorized env with curriculum tracking if enabled
        # This wrapper becomes the single source of truth for curriculum progression
        # across all environments, tracking episodes globally and syncing stage changes
        if self.use_curriculum and self.curriculum_manager:
            logger.info("Wrapping environments with global curriculum tracking...")
            logger.info(
                f"CurriculumVecEnvWrapper will track progression across all {num_envs} environments"
            )
            self.env = CurriculumVecEnvWrapper(
                self.env, self.curriculum_manager, check_advancement_freq=10
            )

        # Create evaluation environment (single, no curriculum)
        logger.info("Creating evaluation environment...")

        def make_eval_env():
            env_config = EnvironmentConfig.for_training()
            # Apply frame stacking configuration if provided
            if frame_stack_cfg:
                from nclone.gym_environment import FrameStackConfig

                env_config.frame_stack = FrameStackConfig(
                    enable_visual_frame_stacking=frame_stack_cfg.get(
                        "enable_visual_frame_stacking", False
                    ),
                    visual_stack_size=frame_stack_cfg.get("visual_stack_size", 4),
                    enable_state_stacking=frame_stack_cfg.get(
                        "enable_state_stacking", False
                    ),
                    state_stack_size=frame_stack_cfg.get("state_stack_size", 4),
                    padding_type=frame_stack_cfg.get("padding_type", "zero"),
                )
            return NppEnvironment(config=env_config)

        self.eval_env = DummyVecEnv([make_eval_env])

        logger.info(f"✓ Environments created: {num_envs} training, 1 eval")
        logger.info(f"✓ Using {'DummyVecEnv' if num_envs <= 4 else 'SubprocVecEnv'}")

        # Now create the model with the correct environment
        if self.model is None and hasattr(self, "policy_kwargs"):
            logger.info("=" * 60)

            if self.use_hierarchical_ppo:
                # Use HierarchicalPPO wrapper
                from npp_rl.agents.hierarchical_ppo import HierarchicalPPO

                logger.info(
                    "Creating HierarchicalPPO model with training environment..."
                )
                logger.info(f"Policy class: {self.policy_class}")
                logger.info(f"Device: {self.hyperparams.get('device')}")
                logger.info("Feature extractor: ConfigurableMultimodalExtractor")
                logger.info(
                    f"Network architecture: {self.policy_kwargs.get('net_arch')}"
                )
                logger.info(
                    f"High-level update frequency: {self.policy_kwargs.get('high_level_update_frequency')}"
                )
                logger.info(
                    f"Max steps per subtask: {self.policy_kwargs.get('max_steps_per_subtask')}"
                )
                logger.info(f"Using ICM: {self.policy_kwargs.get('use_icm')}")
                logger.info(
                    "Initializing hierarchical policy networks and moving to device..."
                )

                # Create HierarchicalPPO wrapper
                hierarchical_ppo = HierarchicalPPO(
                    policy_class=self.policy_class,
                    high_level_update_frequency=self.policy_kwargs.get(
                        "high_level_update_frequency", 50
                    ),
                    max_steps_per_subtask=self.policy_kwargs.get(
                        "max_steps_per_subtask", 500
                    ),
                    use_icm=self.policy_kwargs.get("use_icm", True),
                    policy_kwargs=self.policy_kwargs,
                    **self.hyperparams,
                )

                # Create the model with environment
                self.model = hierarchical_ppo.create_model(env=self.env)
                logger.info("✓ HierarchicalPPO model created successfully")
            else:
                # Use standard PPO
                logger.info("Creating PPO model with training environment...")
                logger.info(f"Policy class: {self.policy_class}")
                logger.info(f"Device: {self.hyperparams.get('device')}")
                logger.info("Feature extractor: ConfigurableMultimodalExtractor")
                logger.info(
                    f"Network architecture: {self.policy_kwargs.get('net_arch')}"
                )
                logger.info("Initializing policy networks and moving to device...")

                self.model = PPO(
                    policy=self.policy_class,
                    env=self.env,
                    policy_kwargs=self.policy_kwargs,
                    **self.hyperparams,
                )
                logger.info("✓ PPO model created successfully")

            logger.info(f"✓ Model is on device: {self.model.device}")
            logger.info("=" * 60)

            # Load pretrained weights if provided
            if self.pretrained_checkpoint:
                logger.info(
                    f"Loading pretrained weights from {self.pretrained_checkpoint}"
                )
                try:
                    self._load_bc_pretrained_weights(self.pretrained_checkpoint)
                except Exception as e:
                    logger.error(f"Failed to load pretrained weights: {e}")
                    logger.warning("Continuing with random initialization")

            # Wrap policy with DistributedDataParallel for multi-GPU training
            if self.use_distributed and self.world_size > 1:
                from npp_rl.training.distributed_utils import wrap_model_ddp

                logger.info("=" * 60)
                logger.info(
                    f"Setting up DistributedDataParallel (DDP) for rank {self.device_id}/{self.world_size}"
                )
                logger.info(
                    "DDP will synchronize gradients across all GPUs during training"
                )

                # Wrap ENTIRE policy with DDP (not just parts like DataParallel)
                # This is the correct way to do distributed training in PyTorch
                self.model.policy = wrap_model_ddp(
                    self.model.policy,
                    device_id=self.device_id,
                    find_unused_parameters=False,
                )

                logger.info(f"✓ Policy wrapped with DDP on GPU {self.device_id}")
                logger.info("✓ Multi-GPU distributed training setup complete")
                logger.info("=" * 60)

            logger.info(f"✓ Model fully initialized with {num_envs} environments")
        elif self.model:
            # Model already exists, just set the environment
            self.model.set_env(self.env)
            logger.info(f"Updated model environment to {num_envs} environments")

        if self.use_curriculum:
            logger.info("Curriculum tracking enabled across all environments")

    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 100000,
        save_freq: int = 500000,
        callback_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Main training loop with evaluation and checkpointing.

        Args:
            total_timesteps: Total timesteps to train
            eval_freq: Evaluation frequency (timesteps)
            save_freq: Checkpoint save frequency (timesteps)
            callback_fn: Optional callback function

        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first")

        if self.env is None:
            raise RuntimeError(
                "Environments not initialized. Call setup_environments() first"
            )

        logger.info("=" * 60)
        logger.info(f"Starting training: {self.architecture_config.name}")
        logger.info(f"Total timesteps: {total_timesteps:,}")

        # Add distributed training info
        if self.use_distributed and self.world_size > 1:
            from npp_rl.training.distributed_utils import is_main_process

            logger.info(f"Distributed training: {self.world_size} GPUs")
            logger.info(f"Current rank: {self.device_id}")
            logger.info(
                f"Effective batch size: {self.hyperparams.get('batch_size', 'N/A')} (per GPU)"
            )
            logger.info(
                f"Global batch size: {self.hyperparams.get('batch_size', 0) * self.world_size}"
            )
            if not is_main_process():
                logger.info("Worker process - progress bar disabled to avoid conflicts")

        logger.info(f"Eval frequency: {eval_freq:,}")
        logger.info(f"Save frequency: {save_freq:,}")
        logger.info("=" * 60)

        # Log model and environment details before starting
        logger.info(f"Model device: {self.model.device}")
        logger.info(f"Number of environments: {self.env.num_envs}")
        logger.info(f"Policy architecture: {self.policy_class}")
        logger.info(
            f"PPO hyperparameters: n_steps={self.hyperparams.get('n_steps')}, "
            f"batch_size={self.hyperparams.get('batch_size')}, "
            f"learning_rate={self.hyperparams.get('learning_rate')}"
        )

        try:
            logger.info("Initializing environment reset...")
            # This will trigger the first reset of all environments
            logger.info(
                "Calling model.learn() - this will reset environments and start rollout collection"
            )
            logger.info(
                "First rollout collection may take time - collecting experience from all environments"
            )

            # Create verbose callback to monitor training progress
            verbose_callback = VerboseTrainingCallback(log_freq=1)

            # Combine with user callback if provided
            callbacks = [verbose_callback]
            if callback_fn is not None:
                callbacks.append(callback_fn)

            # Add enhanced TensorBoard metrics callback
            from npp_rl.callbacks import EnhancedTensorBoardCallback

            enhanced_tb_callback = EnhancedTensorBoardCallback(
                log_freq=100,  # Log scalars every 100 steps
                histogram_freq=1000,  # Log histograms every 1000 steps
                verbose=1,
                log_gradients=True,
                log_weights=False,  # Disable weight logging by default (expensive)
            )
            callbacks.append(enhanced_tb_callback)
            logger.info("Added enhanced TensorBoard callback for detailed metrics")

            # Add PBRS logging callback to track reward components
            from npp_rl.callbacks import PBRSLoggingCallback

            pbrs_callback = PBRSLoggingCallback(verbose=1)
            callbacks.append(pbrs_callback)
            logger.info("Added PBRS logging callback for reward component tracking")

            # Add route visualization callback
            from npp_rl.callbacks import RouteVisualizationCallback

            routes_dir = self.output_dir / "route_visualizations"
            route_callback = RouteVisualizationCallback(
                save_dir=str(routes_dir),
                max_routes_per_checkpoint=10,  # Save up to 10 routes per checkpoint
                visualization_freq=50000,  # Save visualizations every 50K steps
                max_stored_routes=100,  # Keep up to 100 route images
                async_save=True,  # Save asynchronously to avoid blocking
                image_size=(800, 600),
                verbose=1,
            )
            callbacks.append(route_callback)
            logger.info(f"Added route visualization callback (saving to {routes_dir})")

            # Add hierarchical PPO callbacks if using hierarchical training
            if self.use_hierarchical_ppo:
                from npp_rl.callbacks.hierarchical_callbacks import (
                    HierarchicalStabilityCallback,
                    SubtaskTransitionCallback,
                )

                # Add stability monitoring
                stability_callback = HierarchicalStabilityCallback(
                    instability_window=1000,
                    stagnation_window=10000,
                    gradient_norm_threshold=10.0,
                    value_loss_threshold=5.0,
                    log_freq=100,
                    verbose=1,
                )
                callbacks.append(stability_callback)
                logger.info("Added hierarchical stability callback for training monitoring")

                # Add subtask transition tracking
                subtask_callback = SubtaskTransitionCallback(
                    log_freq=100,
                    verbose=1,
                )
                callbacks.append(subtask_callback)
                logger.info("Added subtask transition callback for HRL metrics")

            # Add curriculum progression callback if curriculum learning is enabled
            if self.use_curriculum and self.curriculum_manager is not None:
                from npp_rl.callbacks.hierarchical_callbacks import (
                    CurriculumProgressionCallback,
                )

                curriculum_callback = CurriculumProgressionCallback(
                    curriculum_manager=self.curriculum_manager,
                    check_freq=10000,  # Check advancement every 10K steps
                    log_freq=1000,  # Log metrics every 1K steps
                    verbose=1,
                )
                callbacks.append(curriculum_callback)
                logger.info(
                    f"Added curriculum progression callback (current stage: {self.curriculum_manager.get_current_stage()})"
                )

            # Add distributed progress callback if using multi-GPU training
            if self.use_distributed and self.world_size > 1:
                from npp_rl.callbacks import DistributedProgressCallback

                distributed_callback = DistributedProgressCallback(
                    log_freq=1000, verbose=1
                )
                callbacks.append(distributed_callback)
                logger.info(
                    "Added distributed progress callback for multi-GPU coordination"
                )

            logger.info("Starting model.learn() with verbose callback...")

            # Only show progress bar on main process (rank 0) to avoid flickering
            from npp_rl.training.distributed_utils import is_main_process

            show_progress = is_main_process() if self.use_distributed else True

            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks if len(callbacks) > 1 else verbose_callback,
                progress_bar=show_progress,
            )

            logger.info("Training completed successfully")

            # Save final model (unwrap DDP if needed)
            final_path = self.output_dir / "final_model.zip"

            from npp_rl.training.distributed_utils import is_model_wrapped_ddp

            policy_was_wrapped = is_model_wrapped_ddp(self.model.policy)
            original_policy = self.model.policy

            if policy_was_wrapped:
                self.model.policy = self.model.policy.module
                logger.debug("Unwrapped DDP policy for final model saving")

            try:
                self.model.save(str(final_path))
                logger.info(f"Saved final model to {final_path}")
            finally:
                if policy_was_wrapped:
                    self.model.policy = original_policy
                    logger.debug("Re-wrapped policy with DDP after saving")

            return {"status": "completed", "total_timesteps": total_timesteps}

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def evaluate(
        self,
        num_episodes: int = 250,
        record_videos: bool = False,
        video_output_dir: Optional[str] = None,
        max_videos_per_category: int = 10,
        video_fps: int = 30,
    ) -> Dict[str, float]:
        """Evaluate model on test dataset.

        Args:
            num_episodes: Number of episodes to evaluate (per category)
            record_videos: Whether to record videos of episodes
            video_output_dir: Directory to save videos (required if record_videos=True)
            max_videos_per_category: Maximum number of videos per category
            video_fps: Video framerate

        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Only evaluate on rank 0 to avoid conflicts
        if self.use_distributed:
            from npp_rl.training.distributed_utils import is_main_process

            if not is_main_process():
                logger.info(
                    f"[Rank {self.device_id}] Skipping evaluation (only rank 0 evaluates)"
                )
                return {"success_rate": 0.0, "skipped_on_worker": True}

        logger.info(
            f"Evaluating model on test suite ({num_episodes} episodes per category)..."
        )

        try:
            evaluator = ComprehensiveEvaluator(
                test_dataset_path=str(self.test_dataset_path),
                device=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
            )

            # Create episodes per category dict - distribute episodes across all categories
            num_episodes_per_category = {
                category: num_episodes for category in evaluator.test_levels.keys()
            }

            results = evaluator.evaluate_model(
                model=self.model,
                num_episodes_per_category=num_episodes_per_category,
                max_steps_per_episode=10000,
                deterministic=True,
                record_videos=record_videos,
                video_output_dir=video_output_dir,
                max_videos_per_category=max_videos_per_category,
                video_fps=video_fps,
            )

            # Save results
            results_path = self.output_dir / "eval_results.json"
            evaluator.save_results(results, str(results_path))

            logger.info("Evaluation complete")
            logger.info(f"Success rate: {results['overall']['success_rate']:.2%}")

            return results["overall"]

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"success_rate": 0.0, "error": str(e)}

    def cleanup(self) -> None:
        """Clean up environments and release resources."""
        if self.env is not None:
            try:
                self.env.close()
                logger.info("Closed training environments")
            except Exception as e:
                logger.warning(f"Error closing training environments: {e}")
            self.env = None

        if self.eval_env is not None:
            try:
                self.eval_env.close()
                logger.info("Closed evaluation environment")
            except Exception as e:
                logger.warning(f"Error closing evaluation environment: {e}")
            self.eval_env = None

    def save_checkpoint(self, timestep: int, is_final: bool = False) -> Path:
        """Save model checkpoint with metadata.

        Args:
            timestep: Current training timestep
            is_final: Whether this is the final checkpoint

        Returns:
            Path to saved checkpoint (or None on worker processes)
        """
        # Only save on rank 0 to avoid conflicts
        if self.use_distributed:
            from npp_rl.training.distributed_utils import is_main_process, barrier

            if not is_main_process():
                barrier()  # Wait for rank 0 to finish saving
                return None

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_final:
            checkpoint_path = checkpoint_dir / "final_model.zip"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_{timestep}.zip"

        # Unwrap DDP-wrapped policy before saving
        from npp_rl.training.distributed_utils import is_model_wrapped_ddp

        policy_was_wrapped = is_model_wrapped_ddp(self.model.policy)
        original_policy = self.model.policy

        if policy_was_wrapped:
            # Temporarily unwrap for saving
            self.model.policy = self.model.policy.module
            logger.debug("Unwrapped DDP policy for checkpoint saving")

        try:
            self.model.save(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        finally:
            if policy_was_wrapped:
                # Re-wrap after saving
                self.model.policy = original_policy
                logger.debug("Re-wrapped policy with DDP after saving")

        # Signal to other processes that save is complete
        if self.use_distributed:
            from npp_rl.training.distributed_utils import barrier

            barrier()

        return checkpoint_path

    def get_device(self) -> str:
        """Get the device string for this trainer.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        if torch.cuda.is_available():
            return f"cuda:{self.device_id}"
        return "cpu"

    def get_checkpoint_path(self, name: str) -> Path:
        """Get path for a checkpoint file.

        Args:
            name: Checkpoint name (without extension)

        Returns:
            Path to checkpoint file
        """
        return self.output_dir / f"{name}.zip"

    def create_evaluator(self) -> ComprehensiveEvaluator:
        """Create comprehensive evaluator for this architecture.

        Returns:
            Configured ComprehensiveEvaluator instance
        """
        # Note: ComprehensiveEvaluator expects test_dataset_path, not model/env
        # This method signature needs to be updated to match actual usage
        return ComprehensiveEvaluator(
            test_dataset_path=str(self.test_dataset_path),
            device=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
        )

    def save_training_state(self, timestep: int) -> Path:
        """Save complete training state including curriculum progress.

        Args:
            timestep: Current training timestep

        Returns:
            Path to saved state file
        """
        import json
        from datetime import datetime

        state = {
            "timestep": timestep,
            "architecture": self.architecture_config.name,
            "timestamp": datetime.now().isoformat(),
        }

        # Add curriculum state if available
        if self.curriculum_manager is not None:
            state["curriculum"] = self.curriculum_manager.get_curriculum_state()

        # Save state file
        state_file = self.output_dir / f"training_state_{timestep}.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved training state to {state_file}")
        return state_file
