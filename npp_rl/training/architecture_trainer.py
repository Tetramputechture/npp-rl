"""Architecture trainer for single architecture training runs.

Handles training for a specific architecture configuration including
setup, training loop, evaluation, and checkpointing.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from nclone.gym_environment.reward_calculation.reward_config import RewardConfig
from nclone.gym_environment.constants import (
    GAME_STATE_CHANNELS,
    REACHABILITY_FEATURES_DIM,
)
from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor

from npp_rl.training.bc_weight_loader import BCWeightLoader
from npp_rl.training.environment_factory import EnvironmentFactory
from npp_rl.training.callback_factory import CallbackFactory
from npp_rl.agents.masked_ppo import MaskedPPO

logger = logging.getLogger(__name__)


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
        use_curriculum: bool = False,
        curriculum_kwargs: Optional[Dict[str, Any]] = None,
        use_distributed: bool = False,
        frame_stack_config: Optional[Dict[str, Any]] = None,
        frame_skip_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = 1.0,
        enable_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        debug_mode: bool = False,
        production_mode: bool = True,
        single_level_path: Optional[str] = None,
        use_path_predictor: bool = False,
        path_predictor_checkpoint: Optional[str] = None,
        path_predictor_update_freq: int = 1000,
        runtime_profiler: Optional[Any] = None,
        enable_env_profiling: bool = False,
    ):
        """Initialize architecture trainer.

        NOTE: PBRS is ALWAYS enabled in base environment. No enable_pbrs flag.
        Graph building is automatically configured for PBRS requirements.

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
            use_curriculum: Enable curriculum learning
            curriculum_kwargs: Curriculum manager configuration
            use_distributed: Enable DistributedDataParallel mode for multi-GPU
            frame_stack_config: Frame stacking configuration dict with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str
            frame_skip_config: Frame skip configuration dict with keys:
                - enable: bool (default: False)
                - skip: int (default: 4, recommended for N++ based on input buffers)
                - accumulate_rewards: bool (default: True)
            pbrs_gamma: Discount factor for PBRS (always enabled)
        """
        self.architecture_config = architecture_config
        # Resolve paths to absolute to ensure consistent resolution regardless of working directory
        # (important for distributed training where worker processes may have different CWDs)
        # Expand user home directory (~) before resolving
        self.train_dataset_path = Path(os.path.expanduser(train_dataset_path)).resolve()
        self.test_dataset_path = Path(os.path.expanduser(test_dataset_path)).resolve()
        self.output_dir = Path(output_dir)
        self.device_id = device_id
        self.world_size = world_size
        self.tensorboard_writer = tensorboard_writer
        self.use_mixed_precision = use_mixed_precision
        self.use_curriculum = use_curriculum
        self.curriculum_kwargs = curriculum_kwargs or {}
        self.use_distributed = use_distributed
        self.frame_stack_config = frame_stack_config or {}
        self.frame_skip_config = frame_skip_config or {}
        self.pbrs_gamma = pbrs_gamma
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.debug_mode = debug_mode
        self.production_mode = production_mode
        self.single_level_path = single_level_path
        self.use_path_predictor = use_path_predictor
        self.path_predictor_checkpoint = path_predictor_checkpoint
        self.path_predictor_update_freq = path_predictor_update_freq
        self.runtime_profiler = runtime_profiler
        self.enable_env_profiling = enable_env_profiling
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate PBRS gamma matches reward constants for policy invariance
        from nclone.gym_environment.reward_calculation.reward_constants import (
            PBRS_GAMMA,
        )

        if abs(self.pbrs_gamma - PBRS_GAMMA) > 1e-6:
            logger.warning(
                f"PBRS gamma mismatch! "
                f"Trainer: {self.pbrs_gamma}, "
                f"Reward constants: {PBRS_GAMMA}. "
                f"This violates PBRS policy invariance guarantee."
            )

        # Training state
        self.model = None
        self.env = None
        self.eval_env = None
        self.curriculum_manager = None
        self.environment_factory = None
        self.callback_factory = None
        self.pretrained_checkpoint = None
        self.policy_kwargs = None
        self.policy_class = None
        self.hyperparams = None
        self.navigation_system = None  # For path predictor integration

        # Initialize reward configuration with curriculum-aware lifecycle
        # Total timesteps will be updated when train() is called
        self.reward_config = RewardConfig(total_timesteps=10_000_000)

        logger.info(f"Initialized trainer for architecture: {architecture_config.name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Train dataset: {self.train_dataset_path}")
        logger.info(f"Test dataset: {self.test_dataset_path}")
        logger.info(f"Device: cuda:{device_id}")
        logger.info(f"Curriculum learning: {use_curriculum}")
        logger.info("PBRS: ALWAYS ENABLED (mandatory)")
        logger.info(
            f"  PBRS gamma: {pbrs_gamma} (must match PPO gamma and reward constants)"
        )
        logger.info(f"Reward Config: {self.reward_config}")
        if frame_stack_config:
            logger.info(f"Frame stacking: {frame_stack_config}")
        if frame_skip_config and frame_skip_config.get("enable", False):
            skip = frame_skip_config.get("skip", 4)
            logger.info(f"Frame skip: ENABLED (skip={skip} frames)")
            logger.info(
                f"  → {skip}-frame skip is within all N++ input buffers (jump/floor/wall: 5 frames, launch pad: 4 frames)"
            )
            logger.info(
                f"  → Expected: {(1 - 1 / skip) * 100:.0f}% reduction in agent decisions"
            )
        if self.single_level_path:
            logger.info(f"Single level mode: {self.single_level_path}")
            logger.info("  → Training and evaluation will use this single level file")
        if use_path_predictor:
            logger.info("Path predictor enabled")
            if path_predictor_checkpoint:
                logger.info(f"  Checkpoint: {path_predictor_checkpoint}")
            logger.info(f"  Update frequency: {path_predictor_update_freq} steps")

    @classmethod
    def from_hyperparameter_dict(
        cls,
        architecture_config: ArchitectureConfig,
        hyperparameters: Dict[str, Any],
        train_dataset_path: str,
        test_dataset_path: str,
        output_dir: str,
        experiment_name: str,
        device_id: int = 0,
        use_curriculum: Optional[bool] = None,
        **kwargs,
    ) -> "ArchitectureTrainer":
        """
        Create ArchitectureTrainer from Optuna hyperparameter dictionary.

        Args:
            architecture_config: Architecture configuration
            hyperparameters: Dict of sampled hyperparameters from Optuna trial
            train_dataset_path: Path to training dataset
            test_dataset_path: Path to test dataset
            output_dir: Output directory for checkpoints/logs
            experiment_name: Experiment name
            device_id: CUDA device ID
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured ArchitectureTrainer instance
        """
        # Create output directory with experiment name
        output_path = Path(output_dir) / experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        trainer = cls(
            architecture_config=architecture_config,
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            output_dir=output_path,
            device_id=device_id,
            use_curriculum=use_curriculum or False,
            **kwargs,
        )

        # Store hyperparameters for use in setup_model()
        trainer._optuna_hyperparameters = hyperparameters

        return trainer

    def setup_model(self, pretrained_checkpoint: Optional[str] = None, **ppo_kwargs):
        """Initialize model configuration from architecture config.

        Args:
            pretrained_checkpoint: Optional path to pretrained weights
            **ppo_kwargs: Additional PPO hyperparameters

        Returns:
            None (model will be created when environments are set up)
        """
        logger.info("Setting up model configuration...")

        # Log detailed architecture information
        self._log_architecture_details()

        # Store pretrained checkpoint path for later loading
        self.pretrained_checkpoint = pretrained_checkpoint

        # Set up policy kwargs with architecture config
        # Build net_arch from Optuna params if available
        if hasattr(self, "_optuna_hyperparameters"):
            optuna_params = self._optuna_hyperparameters
            if "net_arch_depth" in optuna_params and "net_arch_width" in optuna_params:
                net_arch = [optuna_params["net_arch_width"]] * optuna_params[
                    "net_arch_depth"
                ]
                default_net_arch = {"pi": net_arch, "vf": net_arch}
            else:
                default_net_arch = {"pi": [256, 256, 128], "vf": [256, 256, 128]}
        else:
            default_net_arch = {"pi": [256, 256, 128], "vf": [256, 256, 128]}

        # Get debug_mode and production_mode from kwargs if available (passed from ArchitectureTrainer)
        debug_mode = getattr(self, "debug_mode", False)
        production_mode = getattr(self, "production_mode", True)

        self.policy_kwargs = {
            "features_extractor_class": ConfigurableMultimodalExtractor,
            "features_extractor_kwargs": {
                "config": self.architecture_config,
                "frame_stack_config": self.frame_stack_config,
                "debug_mode": debug_mode and not production_mode,
            },
            "net_arch": default_net_arch,
            "optimizer_kwargs": {"eps": 1e-5},  # PPO standard from openai/baselines
        }

        # Override features_dim if specified in Optuna params
        if hasattr(self, "_optuna_hyperparameters"):
            optuna_params = self._optuna_hyperparameters
            if "features_dim" in optuna_params:
                # Update architecture config's features_dim (create modified config)
                from dataclasses import replace

                self.architecture_config = replace(
                    self.architecture_config, features_dim=optuna_params["features_dim"]
                )
                # Update features_extractor_kwargs
                self.policy_kwargs["features_extractor_kwargs"]["config"] = (
                    self.architecture_config
                )

        # Always use DeepResNet with dueling (no objective attention needed)
        from npp_rl.agents.deep_resnet_actor_critic_policy import (
            DeepResNetActorCriticPolicy,
        )

        self.policy_class = DeepResNetActorCriticPolicy

        # Use deeper network architecture for attention config
        if self.architecture_config.name == "attention" and not hasattr(
            self, "_optuna_hyperparameters"
        ):
            # Override default architecture with deeper network
            default_net_arch = {
                "pi": [512, 512, 384, 256, 256],  # 5-layer policy network
                "vf": [512, 384, 256],  # 3-layer value network
            }
            self.policy_kwargs["net_arch"] = default_net_arch

        # Add DeepResNetActorCriticPolicy kwargs
        self.policy_kwargs.update(
            {
                "share_features_extractor": True,  # OPTIMIZED: Shared extractors for 1.3-1.5x speedup
                "activation_fn": nn.SiLU,  # Modern activation
                "use_residual": ppo_kwargs.pop("use_residual", True),
                "use_layer_norm": ppo_kwargs.pop("use_layer_norm", True),
                "dropout": ppo_kwargs.pop("dropout", 0.1),
                "dueling": True,  # Keep dueling decomposition
            }
        )
        logger.info("Using DeepResNetActorCriticPolicy (Deep ResNet + Dueling)")
        logger.info("  - Deep ResNet MLP: 5-layer policy, 3-layer value")
        logger.info("  - Dueling architecture (always enabled)")
        logger.info(f"  - Residual connections: {self.policy_kwargs['use_residual']}")
        logger.info(f"  - LayerNorm: {self.policy_kwargs['use_layer_norm']}")
        logger.info("  - Shared feature extractors for policy/value (optimized)")

        # Configure hyperparameters
        self._configure_hyperparameters(ppo_kwargs)

        logger.info(
            f"Model configuration prepared for architecture: {self.architecture_config.name}"
        )
        logger.info("Model will be instantiated when environments are set up")

        return None

    def _log_architecture_details(self) -> None:
        """Log comprehensive architecture configuration details."""
        logger.info("=" * 80)
        logger.info("ARCHITECTURE CONFIGURATION DETAILS")
        logger.info("=" * 80)

        config = self.architecture_config

        # Basic info
        logger.info(f"Architecture Name: {config.name}")
        logger.info(f"Description: {config.description}")

        # Modalities
        logger.info("\n--- Modalities ---")
        modalities = config.modalities
        logger.info(f"  Player Frame: {'✓' if modalities.use_player_frame else '✗'}")
        logger.info(f"  Global View: {'✓' if modalities.use_global_view else '✗'}")
        logger.info(f"  Graph: {'✓' if modalities.use_graph else '✗'}")
        logger.info(f"  Game State: {'✓' if modalities.use_game_state else '✗'}")
        logger.info(f"  Reachability: {'✓' if modalities.use_reachability else '✗'}")

        # Visual processing
        if modalities.use_player_frame or modalities.use_global_view:
            logger.info("\n--- Visual Processing ---")
            visual = config.visual
            if modalities.use_player_frame:
                logger.info("  Player Frame:")
                logger.info(f"    Channels: {visual.player_frame_channels}")
                logger.info(f"    Output Dim: {visual.player_frame_output_dim}")
            if modalities.use_global_view:
                logger.info("  Global View:")
                logger.info(f"    Channels: {visual.global_channels}")
                logger.info(f"    Output Dim: {visual.global_output_dim}")
            logger.info(f"  CNN Dropout: {visual.cnn_dropout}")

        # Graph processing
        if modalities.use_graph:
            logger.info("\n--- Graph Processing ---")
            graph = config.graph
            logger.info(f"  Architecture: {graph.architecture.value}")
            logger.info(f"  Hidden Dim: {graph.hidden_dim}")
            logger.info(f"  Output Dim: {graph.output_dim}")
            logger.info(f"  Num Layers: {graph.num_layers}")
            if hasattr(graph, "num_heads") and graph.num_heads:
                logger.info(f"  Num Heads: {graph.num_heads}")
            if hasattr(graph, "use_type_embeddings"):
                logger.info(
                    f"  Type Embeddings: {'✓' if graph.use_type_embeddings else '✗'}"
                )
            if hasattr(graph, "use_edge_features"):
                logger.info(
                    f"  Edge Features: {'✓' if graph.use_edge_features else '✗'}"
                )
            logger.info(f"  Dropout: {getattr(graph, 'dropout', 'N/A')}")

        # State processing
        if modalities.use_game_state:
            logger.info("\n--- State Processing ---")
            state = config.state
            logger.info(f"  Game State Dim: {GAME_STATE_CHANNELS}")
            logger.info(f"  Reachability Dim: {REACHABILITY_FEATURES_DIM}")
            logger.info(f"  Hidden Dim: {state.hidden_dim}")
            logger.info(f"  Output Dim: {state.output_dim}")
            if hasattr(state, "use_attentive_state_mlp"):
                logger.info(
                    f"  Attentive State MLP: {'✓ ENABLED' if state.use_attentive_state_mlp else '✗ DISABLED'}"
                )
                if state.use_attentive_state_mlp:
                    logger.info(
                        "    → Using multi-head attention over state components"
                    )
                    logger.info(
                        "    → Components: physics (29), objectives (15), mines (8), progress (3), sequential (3)"
                    )

        # Fusion
        logger.info("\n--- Multimodal Fusion ---")
        fusion = config.fusion
        logger.info(f"  Fusion Type: {fusion.fusion_type.value}")
        if fusion.fusion_type.value in ["multi_head", "single_head"]:
            logger.info(f"  Attention Heads: {fusion.num_attention_heads}")
        logger.info(f"  Dropout: {fusion.dropout}")
        if fusion.fusion_type.value == "multi_head":
            logger.info("    → Enhanced fusion with modality embeddings and FFN")

        # Feature dimensions
        logger.info("\n--- Feature Dimensions ---")
        logger.info(f"  Final Features Dim: {config.features_dim}")

        # Frame stacking
        if self.frame_stack_config:
            logger.info("\n--- Frame Stacking ---")
            if self.frame_stack_config.get("enable_visual_frame_stacking", False):
                stack_size = self.frame_stack_config.get("visual_stack_size", 4)
                logger.info(f"  Visual Stacking: ✓ ({stack_size} frames)")
            else:
                logger.info("  Visual Stacking: ✗")
            if self.frame_stack_config.get("enable_state_stacking", False):
                stack_size = self.frame_stack_config.get("state_stack_size", 4)
                logger.info(f"  State Stacking: ✓ ({stack_size} states)")
            else:
                logger.info("  State Stacking: ✗")

        logger.info("=" * 80)

    def _format_learning_rate_for_logging(self, lr: Any) -> str:
        """Safely format learning rate for logging, handling callable schedules.

        Args:
            lr: Learning rate value (can be float, int, or callable schedule function)

        Returns:
            String representation of learning rate suitable for logging
        """
        if callable(lr):
            # Learning rate is a schedule function
            # Try to extract initial LR value by calling with progress_remaining=1.0
            # This works for warmup schedules and other schedules that accept progress_remaining
            try:
                initial_lr = lr(1.0)
                # Check function name to provide more context
                func_name = getattr(lr, "__name__", "")
                if "warmup" in func_name.lower():
                    return f"schedule (warmup, initial={initial_lr:.2e})"
                elif "linear" in func_name.lower():
                    return f"schedule (linear, initial={initial_lr:.2e})"
                else:
                    return f"schedule function (initial={initial_lr:.2e})"
            except Exception:
                # Function call failed, just report it's a schedule
                func_name = getattr(lr, "__name__", "")
                if "warmup" in func_name.lower():
                    return "schedule (warmup)"
                elif "linear" in func_name.lower():
                    return "schedule (linear)"
                else:
                    return "schedule function"
        else:
            # Learning rate is a numeric value
            try:
                return f"{lr:.2e}"
            except (ValueError, TypeError):
                return str(lr)

    def _configure_hyperparameters(self, ppo_kwargs: Dict[str, Any]) -> None:
        """Configure PPO hyperparameters with automatic scaling for multi-GPU.

        Args:
            ppo_kwargs: User-provided PPO hyperparameters
        """
        # Only apply automatic scaling if hyperparameters not explicitly provided
        if "batch_size" in ppo_kwargs and "learning_rate" in ppo_kwargs:
            # Use explicitly provided hyperparameters
            logger.info(
                "Using explicitly provided hyperparameters "
                "(automatic multi-GPU scaling skipped)"
            )
            default_batch_size = ppo_kwargs.get(
                "batch_size", 256
            )  # INCREASED from 128 for better GPU utilization
            default_learning_rate = ppo_kwargs.get("learning_rate", 3e-4)
        else:
            # Apply automatic scaling for multi-GPU DDP training
            # Architecture-specific batch size: attention needs smaller batches for more updates
            base_batch_size = (
                128 if self.architecture_config.name == "attention" else 256
            )
            base_learning_rate = 3e-4

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

        # Default PPO hyperparameters (memory-optimized for scaling)
        # n_steps reduced from 1024 to 512 for additional memory savings (allows more envs)
        # batch_size increased from 128 to 256 for better GPU utilization
        # n_epochs: 10 for standard PPO (deep networks need fewer epochs to avoid overfitting)

        # Architecture-specific gradient clipping
        # Attention architecture has 5-6 level attention (optional Temporal → AttentiveStateMLP → GCN
        # → MultiHeadFusion → ObjectiveAttentionPolicy → Dueling) with ~15-18M parameters, requiring higher max_grad_norm
        default_max_grad_norm = (
            2.5 if self.architecture_config.name == "attention" else 1.0
        )

        # Architecture-specific entropy coefficient for exploration
        # Attention architecture needs stronger exploration pressure due to complex action space
        default_ent_coef = (
            0.05 if self.architecture_config.name == "attention" else 0.01
        )

        # Architecture-specific clip range for policy updates
        # Attention architecture benefits from larger updates due to high-dimensional feature space
        default_clip_range = (
            0.3 if self.architecture_config.name == "attention" else 0.2
        )

        # Architecture-specific value function coefficient
        # Attention architecture benefits from stronger value learning to handle complex state space
        default_vf_coef = 1.0 if self.architecture_config.name == "attention" else 0.5

        default_hyperparams = {
            "learning_rate": default_learning_rate,
            "n_steps": 1024,
            "batch_size": default_batch_size,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": default_clip_range,
            "clip_range_vf": None,  # Disable value clipping (recommended by SB3/OpenAI)
            "ent_coef": default_ent_coef,
            "vf_coef": default_vf_coef,
            "max_grad_norm": default_max_grad_norm,
            "tensorboard_log": str(self.output_dir / "tensorboard"),
            "device": f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
        }

        # Learning rate warmup for deep attention architectures
        # Store schedule separately to apply AFTER model creation
        # (SB3 has issues logging callable learning rates during __init__)
        self._lr_schedule = None
        self._total_training_steps = None  # Will be set during train() call
        if self.architecture_config.name == "attention":
            # Warmup: 0.1x LR for first 25% of training, then decay to 1e-5
            # Note: total_steps will be determined dynamically during train() call
            # to avoid hardcoding and ensure schedule matches actual training length

            # Extract base learning rate (might already be callable from hardware profile)
            lr_value = default_hyperparams["learning_rate"]
            if callable(lr_value):
                # Already a schedule - extract initial value
                try:
                    base_lr = lr_value(1.0)  # Get LR at start (progress_remaining=1.0)
                except Exception:
                    base_lr = 3e-4  # Fallback default
                logger.info(
                    "Attention architecture: Replacing existing LR schedule with warmup schedule"
                )
            else:
                base_lr = lr_value

            def create_warmup_lr_schedule(total_steps: int):
                """Create LR schedule with warmup (25% of training) then linear decay.

                Args:
                    total_steps: Total training steps (determined at runtime)

                Returns:
                    Learning rate schedule function
                """
                warmup_steps = int(0.25 * total_steps)  # 25% warmup

                def warmup_lr_schedule(progress_remaining: float) -> float:
                    """Custom LR schedule with warmup then linear decay.

                    Args:
                        progress_remaining: Progress from 1.0 (start) to 0.0 (end)

                    Returns:
                        Current learning rate
                    """
                    # progress_remaining goes from 1.0 (start) to 0.0 (end)
                    progress = 1.0 - progress_remaining
                    current_step = progress * total_steps

                    if current_step < warmup_steps:
                        # Warmup phase: linear ramp from 0.1x to 1.0x
                        warmup_progress = current_step / warmup_steps
                        return base_lr * (0.1 + 0.9 * warmup_progress)
                    else:
                        # Decay phase: linear decay to 1e-5
                        decay_progress = (current_step - warmup_steps) / (
                            total_steps - warmup_steps
                        )
                        return base_lr * (1.0 - decay_progress) + 1e-5 * decay_progress

                return warmup_lr_schedule

            # Store schedule creator to apply after we know total_timesteps
            self._lr_schedule_creator = create_warmup_lr_schedule
            self._base_lr = base_lr
            # Set base_lr back to numeric for model initialization
            default_hyperparams["learning_rate"] = base_lr
            logger.info(
                f"Attention architecture: Will use LR warmup schedule with dynamic total_steps. "
                f"base_lr={base_lr:.2e}, warmup=25% of training, decay to 1e-5"
            )

        # If Optuna hyperparameters provided, override defaults
        if hasattr(self, "_optuna_hyperparameters"):
            optuna_params = self._optuna_hyperparameters

            # Override PPO hyperparameters
            for key in [
                "learning_rate",
                "n_steps",
                "batch_size",
                "gamma",
                "gae_lambda",
                "clip_range",
                "clip_range_vf",
                "ent_coef",
                "vf_coef",
                "max_grad_norm",
                "n_epochs",
            ]:
                if key in optuna_params:
                    default_hyperparams[key] = optuna_params[key]

            # Handle learning rate schedule
            if "lr_schedule" in optuna_params:
                if optuna_params["lr_schedule"] == "linear":
                    from stable_baselines3.common.utils import get_linear_fn

                    # Extract base learning rate (might already be callable)
                    lr_value = default_hyperparams["learning_rate"]
                    if callable(lr_value):
                        try:
                            base_lr = lr_value(1.0)
                        except Exception:
                            base_lr = 3e-4
                    else:
                        base_lr = lr_value

                    # Store schedule to apply after model creation
                    self._lr_schedule = get_linear_fn(base_lr, 1e-6, 1.0)
                    # Set base_lr back to numeric for initialization
                    default_hyperparams["learning_rate"] = base_lr
                    logger.info(f"Using linear LR schedule: {base_lr:.2e} -> 1e-6")

        # Handle callable learning rate from ppo_kwargs (e.g., from hardware profile)
        # Extract it before merging to avoid SB3 format errors
        if "learning_rate" in ppo_kwargs and callable(ppo_kwargs["learning_rate"]):
            lr_callable = ppo_kwargs["learning_rate"]
            # Extract numeric base value
            try:
                base_lr_from_kwargs = lr_callable(1.0)
            except Exception:
                base_lr_from_kwargs = 3e-4

            # If we don't already have a custom schedule (warmup/optuna), use this one
            if self._lr_schedule is None:
                self._lr_schedule = lr_callable
                logger.info(
                    f"Using LR schedule from ppo_kwargs (initial={base_lr_from_kwargs:.2e})"
                )
            else:
                logger.info(
                    "Custom LR schedule already set, ignoring ppo_kwargs schedule"
                )

            # Replace callable with numeric value for model initialization
            ppo_kwargs = {**ppo_kwargs, "learning_rate": base_lr_from_kwargs}

        # Filter out architecture-specific parameters that are not valid PPO hyperparameters
        # These are used to configure the policy/feature extractors but should not be passed to PPO.__init__()
        architecture_specific_params = {
            "net_arch_depth",
            "net_arch_width",
            "features_dim",
            "num_envs",
            "lr_schedule",
            "policy_class",  # Used to select policy class, not a PPO param
            "gnn_num_layers",
            "gnn_hidden_dim",
            "gnn_num_heads",
            "cnn_base_channels",
            "cnn_num_layers",
            "use_residual",
            "use_layer_norm",
            "dropout",
            "dueling",  # DeepResNet policy-specific param, not a PPO param
            "use_curriculum",  # Trainer-level parameter, not a PPO param
        }
        filtered_ppo_kwargs = {
            k: v for k, v in ppo_kwargs.items() if k not in architecture_specific_params
        }

        # Merge with provided hyperparameters (filtered ppo_kwargs takes precedence)
        self.hyperparams = {**default_hyperparams, **filtered_ppo_kwargs}

        # Log final hyperparameters
        logger.info("Final PPO hyperparameters:")
        # Check if we have a stored LR schedule (will be applied after model creation)
        if hasattr(self, "_lr_schedule") and self._lr_schedule is not None:
            lr_str = self._format_learning_rate_for_logging(self._lr_schedule)
        else:
            lr = self.hyperparams["learning_rate"]
            lr_str = self._format_learning_rate_for_logging(lr)
        logger.info(f"  Learning rate: {lr_str}")
        logger.info(f"  Batch size: {self.hyperparams['batch_size']}")
        logger.info(f"  N steps: {self.hyperparams['n_steps']}")
        logger.info(f"  N epochs: {self.hyperparams.get('n_epochs', 10)}")
        logger.info(f"  Gamma: {self.hyperparams['gamma']}")
        logger.info(f"  GAE lambda: {self.hyperparams['gae_lambda']}")

        # Log memory optimization details
        logger.info("")
        logger.info("Memory-Optimized Hyperparameters (for scaling to 32-64 envs):")
        n_steps = self.hyperparams["n_steps"]
        batch_size = self.hyperparams["batch_size"]
        n_epochs = self.hyperparams.get("n_epochs", 10)
        gradient_updates = (n_steps // batch_size) * n_epochs
        logger.info(f"  Rollout buffer size: {n_steps} steps")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient updates per rollout: {gradient_updates}")

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
        logger.info("=" * 60)
        logger.info(f"Setting up {num_envs} training environments...")
        logger.info(f"Frame stacking config: {self.frame_stack_config}")

        # Add warning if both curriculum and custom map are enabled
        if self.use_curriculum and self.single_level_path:
            logger.warning("=" * 60)
            logger.warning("Both curriculum learning and custom_map_path are enabled.")
            logger.warning(f"Custom map '{self.single_level_path}' will take priority.")
            logger.warning(
                "Curriculum logic will be bypassed - all environments will load the same custom map."
            )
            logger.warning("=" * 60)

        try:
            # Adjust n_steps if total_timesteps is very small (for CPU testing)
            if total_timesteps is not None:
                self._adjust_hyperparams_for_small_runs(total_timesteps, num_envs)

            # Set up curriculum manager if enabled (skip if using custom map)
            if self.use_curriculum and not self.single_level_path:
                from npp_rl.training.curriculum_factory import (
                    create_curriculum_for_parallel_training,
                )

                # Automatically select best curriculum implementation based on number of environments
                self.curriculum_manager, _ = create_curriculum_for_parallel_training(
                    dataset_path=str(self.train_dataset_path),
                    num_parallel_envs=num_envs,
                    **self.curriculum_kwargs,
                )
                logger.info("Curriculum learning enabled with optimized implementation")
                logger.info(f"Number of environments: {num_envs}")
                logger.info(
                    f"Starting stage: {self.curriculum_manager.get_current_stage()}"
                )
            elif self.use_curriculum and self.single_level_path:
                logger.info("=" * 60)
                logger.info(
                    "Curriculum learning disabled: single_level_path takes precedence"
                )
                logger.info(f"All environments will use: {self.single_level_path}")
                logger.info("=" * 60)
                self.use_curriculum = False
                self.curriculum_manager = None

            # Create environment factory
            logger.info("Creating environment factory...")
            self.environment_factory = EnvironmentFactory(
                use_curriculum=self.use_curriculum,
                curriculum_manager=self.curriculum_manager,
                frame_stack_config=self.frame_stack_config,
                frame_skip_config=self.frame_skip_config,
                pbrs_gamma=self.pbrs_gamma,
                output_dir=self.output_dir,
                pretrained_checkpoint=self.pretrained_checkpoint,
                test_dataset_path=str(self.test_dataset_path),
                architecture_config=self.architecture_config,
                reward_config=self.reward_config,  # Pass curriculum-aware reward config
                custom_map_path=self.single_level_path,  # Pass single level path if specified
                enable_profiling=self.enable_env_profiling,
            )

            # Create training environment
            logger.info(f"Creating {num_envs} training environments...")
            self.env = self.environment_factory.create_training_env(
                num_envs=num_envs,
                gamma=self._get_hyperparameter("gamma", 0.99),
            )
            logger.info(f"✓ Training environment created: {type(self.env).__name__}")

            # Create evaluation environment
            logger.info("Creating evaluation environment...")
            self.eval_env = self.environment_factory.create_eval_env()
            logger.info(f"✓ Eval environment created: {type(self.eval_env).__name__}")

            # Create callback factory
            logger.info("Creating callback factory...")
            self.callback_factory = CallbackFactory(
                output_dir=self.output_dir,
                use_curriculum=self.use_curriculum,
                curriculum_manager=self.curriculum_manager,
                use_distributed=self.use_distributed,
                world_size=self.world_size,
                enable_early_stopping=self.enable_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
            )

            # Create the model with the environment
            logger.info("Creating model with environment...")
            self._create_model()

            # Initialize navigation system with path predictor if enabled
            if self.use_path_predictor:
                logger.info("Initializing generalized navigation system...")
                self._initialize_navigation_system()

            logger.info(f"✓ Model fully initialized with {num_envs} environments")
            logger.info("=" * 60)

        except Exception as e:
            print("!" * 60)
            print("CRITICAL: Environment setup failed!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Num envs: {num_envs}")
            print(f"Frame stack config: {self.frame_stack_config}")
            # print("Full traceback:", exc_info=True)
            print("!" * 60)
            raise

    def _get_hyperparameter(self, key: str, default: Any = None) -> Any:
        """Safely get hyperparameter value from either hyperparams or _optuna_hyperparameters.

        This helper is needed because during Optuna trials, setup_environments() is called
        before setup_model(), so self.hyperparams may not be set yet but
        self._optuna_hyperparameters will contain the sampled values.

        Args:
            key: Hyperparameter key to retrieve
            default: Default value if key not found

        Returns:
            Hyperparameter value or default
        """
        if self.hyperparams is not None:
            return self.hyperparams.get(key, default)
        elif (
            hasattr(self, "_optuna_hyperparameters")
            and self._optuna_hyperparameters is not None
        ):
            return self._optuna_hyperparameters.get(key, default)
        else:
            return default

    def _get_hyperparameters_dict(self) -> Dict[str, Any]:
        """Get the full hyperparameters dictionary.

        Returns either self.hyperparams or self._optuna_hyperparameters,
        depending on which is available. Used for unpacking hyperparams
        when creating the model.

        Returns:
            Hyperparameters dictionary, or empty dict if none available
        """
        if self.hyperparams is not None:
            return self.hyperparams
        elif (
            hasattr(self, "_optuna_hyperparameters")
            and self._optuna_hyperparameters is not None
        ):
            return self._optuna_hyperparameters
        else:
            return {}

    def _adjust_hyperparams_for_small_runs(
        self, total_timesteps: int, num_envs: int
    ) -> None:
        """Adjust hyperparameters for small test runs.

        Args:
            total_timesteps: Total training timesteps
            num_envs: Number of environments
        """
        # Get hyperparams from either self.hyperparams or _optuna_hyperparameters
        # (during Optuna trials, hyperparams may not be set yet)
        if self.hyperparams is not None:
            hyperparams = self.hyperparams
        elif (
            hasattr(self, "_optuna_hyperparameters")
            and self._optuna_hyperparameters is not None
        ):
            hyperparams = self._optuna_hyperparameters
        else:
            # No hyperparameters available yet, skip adjustment
            return

        # Skip if n_steps not in hyperparams yet
        if "n_steps" not in hyperparams:
            return

        max_n_steps = max(total_timesteps // num_envs, 1)
        if hyperparams["n_steps"] > max_n_steps:
            old_n_steps = hyperparams["n_steps"]
            hyperparams["n_steps"] = max_n_steps
            # Adjust batch_size to be compatible
            if "batch_size" in hyperparams and hyperparams["batch_size"] > max_n_steps:
                hyperparams["batch_size"] = max_n_steps
            logger.info(
                f"Adjusted n_steps from {old_n_steps} to {max_n_steps} "
                f"for total_timesteps={total_timesteps}, num_envs={num_envs}"
            )
            logger.info(f"Adjusted batch_size to {hyperparams['batch_size']}")

    def _create_model(self) -> None:
        """Create PPO instance."""
        logger.info("=" * 60)
        logger.info("Creating model...")

        try:
            self._create_standard_model()

            logger.info(f"✓ Model is on device: {self.model.device}")
            logger.info("=" * 60)

            # MEMORY PROFILING: Record snapshot after model creation
            if self.runtime_profiler is not None:
                self.runtime_profiler.record_memory_snapshot("after_model_creation")
                logger.info("✓ Memory snapshot recorded: after_model_creation")

            # Update model's observation space to match frame-stacked environment
            # This ensures evaluation correctly detects frame stacking configuration
            if self.frame_stack_config and (
                self.frame_stack_config.get("enable_visual_frame_stacking", False)
                or self.frame_stack_config.get("enable_state_stacking", False)
            ):
                logger.info(
                    "Updating model observation space to match frame-stacked environment..."
                )
                self.model.observation_space = self.env.observation_space
                self.model.policy.observation_space = self.env.observation_space
                logger.info("✓ Model observation space updated to match environment")

            # Load pretrained weights if provided
            if self.pretrained_checkpoint:
                self._load_pretrained_weights()

            # Wrap policy with DistributedDataParallel if using multi-GPU
            if self.use_distributed and self.world_size > 1:
                self._wrap_model_ddp()

        except Exception as e:
            print("!" * 60)
            print("CRITICAL: Model creation failed!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Architecture: {self.architecture_config.name}")
            print(f"Policy class: {self.policy_class}")
            print(f"Device: {self._get_hyperparameter('device', 'unknown')}")
            # print("Full traceback:", exc_info=True)
            print("!" * 60)
            raise

    def _create_standard_model(self) -> None:
        """Create standard PPO model."""
        logger.info("Creating PPO model with training environment...")
        logger.info(f"Policy class: {self.policy_class}")
        logger.info(f"Device: {self._get_hyperparameter('device')}")
        logger.info("Feature extractor: ConfigurableMultimodalExtractor")
        logger.info(f"Network architecture: {self.policy_kwargs.get('net_arch')}")
        logger.info("Initializing policy networks and moving to device...")

        # Suppress benign SB3 warning about separate feature extractors
        # When share_features_extractor=False, SB3 warns that "features_extractor will be ignored"
        # but this is misleading - the features_extractor_class/kwargs ARE used to create
        # two separate instances (policy and value), which is exactly what we want.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Provided features_extractor will be ignored because the features extractor is not shared",
                category=UserWarning,
            )
            # Use MaskedPPO instead of standard PPO to ensure action_mask is properly preserved
            # This prevents masked action selection bugs in vectorized environments
            # Use sparse rollout buffer for ~95% memory reduction on graph observations
            self.model = MaskedPPO(
                policy=self.policy_class,
                env=self.env,
                policy_kwargs=self.policy_kwargs,
                debug=self.debug_mode,
                # Use standard DictRolloutBuffer (default) - sparse logic removed
                **self._get_hyperparameters_dict(),
            )
        logger.info("✓ PPO model created successfully")

        # Apply learning rate schedule if one was configured
        # (must be done AFTER model creation to avoid SB3 format errors)
        if hasattr(self, "_lr_schedule") and self._lr_schedule is not None:
            self.model.learning_rate = self._lr_schedule
            logger.info("✓ Applied custom learning rate schedule to model")

        # Note: Mixed precision training (use_mixed_precision flag) is not yet
        # supported by stable-baselines3 PPO. To enable it, would require custom
        # PPO implementation with AMP integration. Current training uses FP32.
        if self.use_mixed_precision:
            print(
                "Mixed precision requested but not supported by stable-baselines3 PPO. "
                "Training will use FP32. To enable mixed precision, implement custom "
                "PPO with AMPHelper integration."
            )

    def _load_pretrained_weights(self) -> None:
        """Load BC pretrained weights into model."""
        logger.info("=" * 60)
        logger.info(f"Loading pretrained weights from {self.pretrained_checkpoint}")
        logger.info(f"Architecture: {self.architecture_config.name}")
        if self.frame_stack_config:
            logger.info(f"Frame stacking config: {self.frame_stack_config}")

        try:
            weight_loader = BCWeightLoader(
                model=self.model,
                architecture_name=self.architecture_config.name,
                frame_stack_config=self.frame_stack_config,
            )
            weight_loader.load_weights(self.pretrained_checkpoint)
            logger.info("=" * 60)
        except Exception as e:
            print("!" * 60)
            print("CRITICAL: Failed to load pretrained weights!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Checkpoint path: {self.pretrained_checkpoint}")
            print(f"Architecture: {self.architecture_config.name}")
            print(f"Frame stack config: {self.frame_stack_config}")
            # print("Full traceback:", exc_info=True)
            print("!" * 60)
            print(
                "⚠️  Continuing with RANDOM INITIALIZATION - training will start from scratch!"
            )
            print("⚠️  BC pretraining benefits will be LOST!")
            print("!" * 60)

    def _wrap_model_ddp(self) -> None:
        """Wrap model policy with DistributedDataParallel."""
        from npp_rl.training.distributed_utils import wrap_model_ddp

        logger.info("=" * 60)
        logger.info(
            f"Setting up DistributedDataParallel (DDP) for rank {self.device_id}/{self.world_size}"
        )
        logger.info("DDP will synchronize gradients across all GPUs during training")

        self.model.policy = wrap_model_ddp(
            self.model.policy,
            device_id=self.device_id,
            find_unused_parameters=False,
        )

        logger.info(f"✓ Policy wrapped with DDP on GPU {self.device_id}")
        logger.info("✓ Multi-GPU distributed training setup complete")
        logger.info("=" * 60)

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

        # Update reward config with actual total_timesteps for curriculum
        self.reward_config.total_timesteps = total_timesteps
        logger.info(f"Reward configuration updated: {self.reward_config}")

        # Apply learning rate schedule with actual total_timesteps for attention architecture
        # This fixes the bug where hardcoded total_steps caused negative LR after 2M steps
        if (
            hasattr(self, "_lr_schedule_creator")
            and self._lr_schedule_creator is not None
        ):
            # Create the schedule with actual total_timesteps
            self._lr_schedule = self._lr_schedule_creator(total_timesteps)
            self.model.learning_rate = self._lr_schedule
            warmup_steps = int(0.25 * total_timesteps)
            logger.info(
                f"Applied LR warmup schedule for attention architecture: "
                f"total_steps={total_timesteps:,}, warmup_steps={warmup_steps:,}, "
                f"base_lr={self._base_lr:.2e}"
            )

        self._log_training_start(total_timesteps, eval_freq, save_freq)

        try:
            # Create callbacks
            callbacks = self.callback_factory.create_callbacks(
                user_callback=callback_fn
            )

            logger.info("Starting model.learn() with callbacks...")

            # Only show progress bar on main process (rank 0)
            from npp_rl.training.distributed_utils import is_main_process

            show_progress = is_main_process() if self.use_distributed else True

            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=show_progress,
            )

            logger.info("Training completed successfully")

            # Save final model
            self._save_final_model()

            return {"status": "completed", "total_timesteps": total_timesteps}

        except Exception as e:
            import traceback

            # print(f"Training failed: {e}", exc_info=True)
            print("=" * 60)
            print("TRAINING FAILURE DETAILS:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"  Total timesteps attempted: {total_timesteps}")
            print(f"  Model device: {self.model.device if self.model else 'N/A'}")
            print(f"  Num environments: {self.env.num_envs if self.env else 'N/A'}")
            print("\n  Stack trace:")
            print("  " + "  ".join(traceback.format_exc().splitlines(True)))
            print("=" * 60)
            return {"status": "failed", "error": str(e), "error_type": type(e).__name__}

    def update_reward_config(self, timesteps_trained: int, success_rate: float) -> None:
        """Update reward configuration based on training progress.

        This method should be called by evaluation callbacks to update reward
        component weights and enable/disable curriculum-controlled components.

        The reward system will automatically transition through phases:
        - Early (0-1M): Strong PBRS guidance (2.0), no time penalty
        - Mid (1M-3M): Moderate PBRS (1.0), optional time penalty if success >50%
        - Late (3M+): Light PBRS (0.5), full time penalty if successful

        NOTE: This method updates the config object, but environments must be
        recreated or have their reward calculators updated to reflect changes.
        For proper integration, environments should receive self.reward_config
        at creation time and have their RewardCalculator.update_config() called
        during evaluation callbacks.

        Args:
            timesteps_trained: Current total timesteps trained
            success_rate: Recent evaluation success rate (0.0-1.0)
        """
        old_phase = self.reward_config.training_phase
        self.reward_config.update(timesteps_trained, success_rate)

        # Log phase transitions
        if self.reward_config.training_phase != old_phase:
            logger.info(
                f"\n{'=' * 60}\n"
                f"REWARD CURRICULUM TRANSITION\n"
                f"Phase: {old_phase} → {self.reward_config.training_phase}\n"
                f"Timesteps: {timesteps_trained:,}\n"
                f"Success Rate: {success_rate:.1%}\n"
                f"Active Components:\n"
                f"  PBRS Weight: {self.reward_config.pbrs_objective_weight:.2f}\n"
                f"  Time Penalty: {self.reward_config.time_penalty_per_step:.4f}/step\n"
                f"  Normalization Scale: {self.reward_config.pbrs_normalization_scale:.2f}\n"
                f"{'=' * 60}\n"
            )

        # TODO: Call env.reward_calculator.update_config() on all environments
        # This would require environment access and proper reward calculator exposure

    def _log_training_start(
        self, total_timesteps: int, eval_freq: int, save_freq: int
    ) -> None:
        """Log training start information.

        Args:
            total_timesteps: Total timesteps to train
            eval_freq: Evaluation frequency
            save_freq: Save frequency
        """
        logger.info("=" * 60)
        logger.info(f"Starting training: {self.architecture_config.name}")
        logger.info(f"Total timesteps: {total_timesteps:,}")

        # Add distributed training info
        if self.use_distributed and self.world_size > 1:
            from npp_rl.training.distributed_utils import is_main_process

            logger.info(f"Distributed training: {self.world_size} GPUs")
            logger.info(f"Current rank: {self.device_id}")
            logger.info(
                f"Effective batch size: {self._get_hyperparameter('batch_size', 'N/A')} (per GPU)"
            )
            logger.info(
                f"Global batch size: {self._get_hyperparameter('batch_size', 0) * self.world_size}"
            )
            if not is_main_process():
                logger.info("Worker process - progress bar disabled to avoid conflicts")

        logger.info(f"Eval frequency: {eval_freq:,}")
        logger.info(f"Save frequency: {save_freq:,}")
        logger.info("=" * 60)

        # Log model and environment details
        logger.info(f"Model device: {self.model.device}")
        logger.info(f"Number of environments: {self.env.num_envs}")
        logger.info(f"Policy architecture: {self.policy_class}")
        # Get effective learning rate (schedule or numeric)
        if hasattr(self, "_lr_schedule") and self._lr_schedule is not None:
            lr_str = self._format_learning_rate_for_logging(self._lr_schedule)
        else:
            lr_str = self._format_learning_rate_for_logging(
                self._get_hyperparameter("learning_rate")
            )
        logger.info(
            f"PPO hyperparameters: n_steps={self._get_hyperparameter('n_steps')}, "
            f"batch_size={self._get_hyperparameter('batch_size')}, "
            f"learning_rate={lr_str}"
        )

        logger.info("Initializing environment reset...")
        logger.info(
            "Calling model.learn() - this will reset environments and start rollout collection"
        )
        logger.info(
            "First rollout collection may take time - collecting experience from all environments"
        )

    def _save_final_model(self) -> None:
        """Save final trained model."""
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

    def evaluate(
        self,
        num_episodes: int = 250,
        record_videos: bool = False,
        video_output_dir: Optional[str] = None,
        max_videos_per_category: int = 10,
        video_fps: int = 30,
        timeout_per_episode: float = 200.0,
        categories_to_evaluate: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test dataset.

        Args:
            num_episodes: Number of episodes to evaluate (per category)
            record_videos: Whether to record videos of episodes
            video_output_dir: Directory to save videos (required if record_videos=True)
            max_videos_per_category: Maximum number of videos per category
            video_fps: Video framerate
            timeout_per_episode: Timeout in seconds per episode (default: 200.0)

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

        # Handle single level evaluation mode
        if self.single_level_path:
            logger.info(
                f"Single level mode: Evaluating {num_episodes} episodes on {self.single_level_path}"
            )
            logger.info("  → Test dataset will NOT be used")
            return self._evaluate_single_level(
                num_episodes=num_episodes,
                record_videos=record_videos,
                video_output_dir=video_output_dir,
                max_videos_per_category=max_videos_per_category,
                video_fps=video_fps,
                timeout_per_episode=timeout_per_episode,
            )

        logger.info(
            f"Evaluating model on test suite ({num_episodes} episodes per category)..."
        )

        try:
            evaluator = ComprehensiveEvaluator(
                test_dataset_path=str(self.test_dataset_path),
                device=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
            )

            # Create episodes per category dict
            num_episodes_per_category = {
                category: num_episodes for category in evaluator.test_levels.keys()
            }

            # replace num_episodes_per_category with 0 for categories not in categories_to_evaluate
            if categories_to_evaluate is not None:
                for category in num_episodes_per_category.keys():
                    if category not in categories_to_evaluate:
                        num_episodes_per_category[category] = 0

            results = evaluator.evaluate_model(
                model=self.model,
                num_episodes_per_category=num_episodes_per_category,
                max_steps_per_episode=10000,
                deterministic=True,
                timeout_per_episode=timeout_per_episode,
                record_videos=record_videos,
                video_output_dir=video_output_dir,
                max_videos_per_category=max_videos_per_category,
                video_fps=video_fps,
                frame_stack_config=self.frame_stack_config,
            )

            # Save results
            results_path = self.output_dir / "eval_results.json"
            evaluator.save_results(results, str(results_path))

            logger.info("Evaluation complete")
            logger.info(f"Success rate: {results['overall']['success_rate']:.2%}")

            return results["overall"]

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {"success_rate": 0.0, "error": str(e)}

    def _evaluate_single_level(
        self,
        num_episodes: int,
        record_videos: bool = False,
        video_output_dir: Optional[str] = None,
        max_videos_per_category: int = 10,
        video_fps: int = 30,
        timeout_per_episode: float = 200.0,
    ) -> Dict[str, float]:
        """Evaluate model on a single level file.

        Args:
            num_episodes: Number of episodes to run on the single level
            record_videos: Whether to record videos
            video_output_dir: Directory to save videos
            max_videos_per_category: Maximum videos to record
            video_fps: Video framerate
            timeout_per_episode: Timeout per episode in seconds

        Returns:
            Evaluation metrics dictionary
        """
        import time
        import numpy as np
        from pathlib import Path
        from stable_baselines3.common.vec_env import DummyVecEnv
        from nclone.gym_environment.npp_environment import NppEnvironment
        from nclone.gym_environment.config import EnvironmentConfig
        from nclone.gym_environment.frame_stack_wrapper import FrameStackWrapper
        from npp_rl.utils.video_recorder import create_video_recorder
        from npp_rl.training.distributed_utils import unwrap_policy_for_inference

        logger.info(
            f"Running {num_episodes} episodes on single level: {self.single_level_path}"
        )

        successes = []
        episode_steps = []
        episode_rewards = []
        videos_recorded = 0

        # Detect visual modalities
        visual_modalities = ComprehensiveEvaluator.detect_visual_modalities(self.model)
        uses_visual = (
            visual_modalities["uses_player_frame"]
            or visual_modalities["uses_global_view"]
        )

        for episode_idx in range(num_episodes):
            logger.debug(f"Episode {episode_idx + 1}/{num_episodes}")
            env = None
            video_recorder = None

            try:
                # Create environment for this episode
                def make_env():
                    config = EnvironmentConfig.for_evaluation(
                        test_dataset_path=str(self.test_dataset_path),
                        custom_map_path=self.single_level_path,
                    )

                    # Disable visual observations if model doesn't use them
                    if not uses_visual and not record_videos:
                        config.enable_visual_observations = False
                    elif record_videos:
                        config.render.render_mode = "rgb_array"
                        config.enable_visual_observations = True

                    env = NppEnvironment(config=config)

                    # Apply frame stacking if configured
                    if self.frame_stack_config and (
                        self.frame_stack_config.get(
                            "enable_visual_frame_stacking", False
                        )
                        or self.frame_stack_config.get("enable_state_stacking", False)
                    ):
                        env = FrameStackWrapper(
                            env,
                            visual_stack_size=self.frame_stack_config.get(
                                "visual_stack_size", 4
                            ),
                            state_stack_size=self.frame_stack_config.get(
                                "state_stack_size", 4
                            ),
                            enable_visual_stacking=self.frame_stack_config.get(
                                "enable_visual_frame_stacking", False
                            ),
                            enable_state_stacking=self.frame_stack_config.get(
                                "enable_state_stacking", False
                            ),
                            padding_type=self.frame_stack_config.get(
                                "padding_type", "zero"
                            ),
                        )

                    return env

                env = DummyVecEnv([make_env])
                obs = env.reset()
                obs = ComprehensiveEvaluator._filter_observation_for_model(
                    obs, self.model
                )

                # Initialize video recorder if needed
                should_record = (
                    record_videos
                    and video_output_dir is not None
                    and videos_recorded < max_videos_per_category
                )

                if should_record:
                    video_path = (
                        Path(video_output_dir)
                        / f"single_level_ep{episode_idx:03d}_temp.mp4"
                    )
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    video_recorder = create_video_recorder(
                        output_path=str(video_path),
                        fps=video_fps,
                    )
                    if video_recorder:
                        video_recorder.start_recording()
                        frame = env.render()
                        if frame is not None:
                            if isinstance(frame, (list, tuple)):
                                frame = frame[0]
                            if hasattr(frame, "ndim") and frame.ndim == 3:
                                video_recorder.record_frame(frame)

                # Run episode
                done = False
                steps = 0
                episode_reward = 0
                start_time = time.time()
                max_steps = 10000

                while not done and steps < max_steps:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout_per_episode:
                        logger.warning(
                            f"Episode {episode_idx + 1} timed out after {elapsed_time:.1f}s"
                        )
                        break

                    # Get action from model
                    with unwrap_policy_for_inference(self.model):
                        action, _ = self.model.predict(obs, deterministic=True)

                    # Step environment
                    obs, reward, done, info = env.step(action)
                    obs = ComprehensiveEvaluator._filter_observation_for_model(
                        obs, self.model
                    )
                    done = done[0]
                    reward = reward[0]
                    info = info[0]

                    # Record frame if needed
                    if video_recorder and video_recorder.is_recording:
                        try:
                            frame = env.render()
                            if frame is not None:
                                if isinstance(frame, (list, tuple)):
                                    frame = frame[0]
                                if hasattr(frame, "ndim") and frame.ndim == 3:
                                    video_recorder.record_frame(frame)
                        except Exception:
                            pass

                    episode_reward += reward
                    steps += 1

                # Record results
                success = info.get("is_success", False) if done else False
                successes.append(1 if success else 0)
                episode_steps.append(steps)
                episode_rewards.append(episode_reward)

                # Save video
                if video_recorder and video_recorder.is_recording:
                    video_recorder.stop_recording(save=True)
                    success_label = "success" if success else "failure"
                    final_path = (
                        Path(video_output_dir)
                        / f"single_level_ep{episode_idx:03d}_{success_label}.mp4"
                    )
                    temp_path = (
                        Path(video_output_dir)
                        / f"single_level_ep{episode_idx:03d}_temp.mp4"
                    )
                    if temp_path.exists():
                        temp_path.rename(final_path)
                        videos_recorded += 1

                env.close()

            except Exception as e:
                logger.error(f"Failed to evaluate episode {episode_idx + 1}: {e}")
                import traceback

                traceback.print_exc()
                successes.append(0)
                episode_steps.append(10000)
                episode_rewards.append(0)
            finally:
                if video_recorder and video_recorder.is_recording:
                    try:
                        video_recorder.stop_recording(save=False)
                    except Exception:
                        pass
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

        # Calculate metrics
        success_rate = np.mean(successes) if len(successes) > 0 else 0.0
        avg_steps = np.mean(episode_steps) if len(episode_steps) > 0 else 0.0
        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        std_steps = np.std(episode_steps) if len(episode_steps) > 0 else 0.0

        results = {
            "success_rate": float(success_rate),
            "avg_steps": float(avg_steps),
            "std_steps": float(std_steps),
            "avg_reward": float(avg_reward),
            "total_episodes": len(successes),
            "episode_steps": [int(s) for s in episode_steps],
            "episode_rewards": [float(r) for r in episode_rewards],
            "successes": [int(s) for s in successes],
        }

        # Save results
        results_path = self.output_dir / "eval_results.json"
        import json

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Single level evaluation complete")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Average steps: {avg_steps:.1f} ± {std_steps:.1f}")
        logger.info(f"Average reward: {avg_reward:.2f}")

        return results

    def cleanup(self) -> None:
        """Clean up environments and release resources."""
        if self.env is not None:
            try:
                self.env.close()
                logger.info("Closed training environments")
            except Exception as e:
                print(f"Error closing training environments: {e}")
            self.env = None

        if self.eval_env is not None:
            try:
                self.eval_env.close()
                logger.info("Closed evaluation environment")
            except Exception as e:
                print(f"Error closing evaluation environment: {e}")
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
            self.model.policy = self.model.policy.module
            logger.debug("Unwrapped DDP policy for checkpoint saving")

        try:
            self.model.save(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        finally:
            if policy_was_wrapped:
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
        return ComprehensiveEvaluator(
            test_dataset_path=str(self.test_dataset_path),
            device=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
        )

    def _initialize_navigation_system(self) -> None:
        """Initialize generalized navigation system with path predictor.

        Creates and configures the navigation system for reward shaping and
        online path learning during RL training.
        """
        from npp_rl.training.generalized_navigation_system import (
            GeneralizedNavigationSystem,
            GeneralizedNavigationConfig,
        )

        # Create navigation system config
        nav_config = GeneralizedNavigationConfig(
            num_path_candidates=4,
            max_waypoints=20,
            graph_feature_dim=256,
            tile_pattern_dim=64,
            entity_feature_dim=32,
            path_predictor_checkpoint=self.path_predictor_checkpoint,
        )

        # Create navigation system
        self.navigation_system = GeneralizedNavigationSystem(
            config=nav_config,
            demonstration_data=None,  # No demonstrations for now
            save_dir=str(self.output_dir / "navigation_system"),
            env=self.env.envs[0] if hasattr(self.env, "envs") else self.env,
        )

        logger.info("✓ Generalized navigation system initialized")

        # Log statistics
        stats = self.navigation_system.get_system_statistics()
        logger.info(f"  Path predictor: {stats.get('path_predictor_stats', {})}")
        logger.info(
            f"  Pattern database: {stats.get('patterns_in_database', 0)} patterns"
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
